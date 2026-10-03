"""
BFGS & Learned Hessian Inversion Engine for RainAI.

Provides Quasi-Newton second-order optimization and learned inverse Hessian
preconditioning for neural audio synthesis, physical loss surfaces, and continuous flow matching.

Mathematical Basis:
The Broyden–Fletcher–Goldfarb–Shanno (BFGS) update iteratively builds an approximation
of the inverse Hessian matrix H_k ≈ (∇² L)⁻¹ using only parameter displacements (s_k)
and gradient differences (y_k), satisfying the secant equation H_{k+1} y_k = s_k:

    H_{k+1} = (I - ρ_k s_k y_kᵀ) H_k (I - ρ_k y_k s_kᵀ) + ρ_k s_k s_kᵀ, where ρ_k = 1 / (y_kᵀ s_k)

Limited-Memory BFGS (L-BFGS) maintains a circular buffer of the m most recent pairs
and computes H_k v via the two-loop recursion in O(m·d) time and memory.
"""

from typing import List, Tuple, Optional, Callable, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BFGSCurvatureTracker:
    """
    Online Limited-Memory BFGS Curvature Tracker & Inverse Hessian Vector Product (iHVP) engine.
    Stores past displacement (s_k) and gradient difference (y_k) pairs to evaluate H_k v
    without ever storing or inverting full d×d Hessian matrices.
    """

    def __init__(self, history_size: int = 10, eps: float = 1e-8):
        self.history_size = max(1, history_size)
        self.eps = eps
        self.s_history: List[torch.Tensor] = []
        self.y_history: List[torch.Tensor] = []
        self.rho_history: List[float] = []

    def reset(self):
        """Clears all curvature history."""
        self.s_history.clear( )
        self.y_history.clear()
        self.rho_history.clear()

    def update(self, s: torch.Tensor, y: torch.Tensor) -> bool:
        """
        Records a new displacement s = x_{k+1} - x_k and gradient difference y = g_{k+1} - g_k.
        Enforces positive definiteness (curvature condition yᵀ s > 0).
        """
        s_flat = s.detach().flatten()
        y_flat = y.detach().flatten()

        ys = float(torch.dot(y_flat, s_flat).item())
        if ys <= self.eps:
            # Skip update if curvature condition is violated to maintain positive definiteness
            return False

        rho = 1.0 / ys
        self.s_history.append(s_flat)
        self.y_history.append(y_flat)
        self.rho_history.append(rho)

        if len(self.s_history) > self.history_size:
            self.s_history.pop(0)
            self.y_history.pop(0)
            self.rho_history.pop(0)

        return True

    def apply_inverse_hessian(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes H_k v using the L-BFGS two-loop recursion.
        If no history exists, returns v (Identity initial Hessian H₀ = I).
        """
        if not self.s_history:
            return v

        orig_shape = v.shape
        q = v.detach().flatten().clone()
        m = len(self.s_history)
        alphas = [0.0] * m

        # Loop 1: Backward pass over stored history
        for i in reversed(range(m)):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]

            alpha_i = rho_i * float(torch.dot(s_i, q).item())
            alphas[i] = alpha_i
            q = q - alpha_i * y_i

        # Initial inverse Hessian scaling: γ_k = (s_{k-1}ᵀ y_{k-1}) / (y_{k-1}ᵀ y_{k-1})
        s_last = self.s_history[-1]
        y_last = self.y_history[-1]
        yy = float(torch.dot(y_last, y_last).item())
        gamma = (float(torch.dot(s_last, y_last).item()) / (yy + self.eps)) if yy > self.eps else 1.0

        r = gamma * q

        # Loop 2: Forward pass over stored history
        for i in range(m):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]
            alpha_i = alphas[i]

            beta_i = rho_i * float(torch.dot(y_i, r).item())
            r = r + s_i * (alpha_i - beta_i)

        return r.view(orig_shape)


class LearnedHessianInverter(nn.Module):
    """
    Parametric Neural Inverse Hessian Operator (Learned Metric Tensor).
    Represents the local inverse Hessian H⁻¹(θ, c) as a low-rank plus diagonal positive-definite operator:
        H⁻¹(c) v = D(c) ⊙ v + U(c) (V(c)ᵀ v)
    where D(c) > 0 is diagonal curvature scaling and U, V are low-rank metric modulations.
    """

    def __init__(
        self,
        param_dim: int,
        cond_dim: int = 16,
        rank: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.cond_dim = cond_dim
        self.rank = rank

        # Predicts positive diagonal inverse curvature D(c)
        self.diag_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_dim),
        )

        # Predicts low-rank basis factors U and V
        self.u_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_dim * rank),
        )
        self.v_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_dim * rank),
        )

    def forward(self, v: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Preconditions gradient/velocity vector v by the learned inverse Hessian metric tensor.
        v: (B, param_dim) or (param_dim,)
        cond: (B, cond_dim) or (cond_dim,) conditioning state (latents / acoustic controls)
        """
        is_1d = (v.dim() == 1)
        if is_1d:
            v = v.unsqueeze(0)

        batch_size = v.size(0)
        device = v.device

        if cond is None:
            cond = torch.zeros((batch_size, self.cond_dim), device=device)
        elif cond.dim() == 1:
            cond = cond.unsqueeze(0).expand(batch_size, -1)

        # 1. Positive diagonal scaling D = Softplus(raw_diag) + 1e-4
        raw_diag = self.diag_net(cond)
        diag = F.softplus(raw_diag) + 1e-4  # (B, param_dim)
        d_v = diag * v

        # 2. Low-rank update: U (Vᵀ v)
        u_factors = self.u_net(cond).view(batch_size, self.param_dim, self.rank)  # (B, D, R)
        v_factors = self.v_net(cond).view(batch_size, self.param_dim, self.rank)  # (B, D, R)

        # Vᵀ v -> (B, R)
        vt_v = torch.bmm(v.unsqueeze(1), v_factors).squeeze(1)  # (B, R)
        # U (Vᵀ v) -> (B, D)
        low_rank = torch.bmm(u_factors, vt_v.unsqueeze(-1)).squeeze(-1)  # (B, D)

        preconditioned = d_v + 0.1 * low_rank

        return preconditioned.squeeze(0) if is_1d else preconditioned


class HybridBFGSOptimizer:
    """
    Hybrid First-to-Second-Order Optimization Coordinator.
    Trains with AdamW for global parameter discovery, then transitions smoothly
    to L-BFGS for quadratic second-order convergence on physical acoustic manifolds.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        bfgs_lr: float = 0.5,
        history_size: int = 10,
        switch_epoch: int = 5,
        max_iter: int = 10,
    ):
        self.params = list(params)
        self.switch_epoch = switch_epoch
        self.current_epoch = 1

        self.adamw = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        self.lbfgs = torch.optim.LBFGS(
            self.params,
            lr=bfgs_lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    @property
    def is_bfgs_active(self) -> bool:
        return self.current_epoch >= self.switch_epoch

    def zero_grad(self):
        if self.is_bfgs_active:
            self.lbfgs.zero_grad()
        else:
            self.adamw.zero_grad()

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        if self.is_bfgs_active:
            if closure is None:
                raise ValueError("L-BFGS optimization requires a closure callback.")
            return self.lbfgs.step(closure)
        else:
            if closure is not None:
                loss = closure()
            return self.adamw.step()


def create_optimizer(
    params,
    optimizer_type: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    history_size: int = 10,
    max_iter: int = 10,
    switch_epoch: int = 5,
) -> Union[torch.optim.Optimizer, HybridBFGSOptimizer]:
    """
    Factory function instantiating either first-order (AdamW), second-order Quasi-Newton (L-BFGS),
    or hybrid first-to-second-order optimizers.
    """
    opt_name = (optimizer_type or "adamw").lower().strip()
    param_list = list(params)

    if opt_name in ("lbfgs", "bfgs"):
        return torch.optim.LBFGS(
            param_list,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )
    elif opt_name in ("hybrid", "hybrid_bfgs"):
        return HybridBFGSOptimizer(
            param_list,
            lr=lr,
            weight_decay=weight_decay,
            bfgs_lr=min(lr * 2.0, 1.0),
            history_size=history_size,
            switch_epoch=switch_epoch,
            max_iter=max_iter,
        )
    else:
        return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay)
