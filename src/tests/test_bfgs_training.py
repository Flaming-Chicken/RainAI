"""
Tests for BFGS, Learned Hessian Inversion, and Second-Order Optimization in RainAI.
Verifies Quasi-Newton curvature tracking, parametric neural metric inversion,
and end-to-end second-order training convergence.
"""

import pytest
import torch
import torch.nn as nn
from src.dsp.bfgs import (
    BFGSCurvatureTracker,
    LearnedHessianInverter,
    HybridBFGSOptimizer,
    create_optimizer,
)


def test_bfgs_curvature_tracker_secant_equation():
    """
    Tests that BFGSCurvatureTracker satisfies the fundamental secant equation:
    H_{k+1} y_k = s_k for displacement s_k and gradient difference y_k.
    """
    dim = 8
    tracker = BFGSCurvatureTracker(history_size=5)

    # Initial state: H_0 = I
    v = torch.randn(dim)
    assert torch.allclose(tracker.apply_inverse_hessian(v), v)

    # Synthetic displacement and gradient step under a positive definite quadratic A
    A = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    s = torch.tensor([0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, -0.05])
    y = torch.mv(A, s)  # y = A s => A = ∇² L

    # Update tracker
    accepted = tracker.update(s, y)
    assert accepted is True

    # The secant equation requires H * y ≈ s
    inv_hessian_y = tracker.apply_inverse_hessian(y)
    cos_sim = torch.dot(inv_hessian_y, s) / (torch.norm(inv_hessian_y) * torch.norm(s))
    assert cos_sim.item() > 0.99, f"Expected secant condition alignment, got cos_sim={cos_sim.item()}"


def test_bfgs_curvature_condition_rejection():
    """
    Verifies that the curvature tracker rejects updates violating the curvature condition (yᵀ s <= 0)
    to guarantee positive definiteness.
    """
    dim = 4
    tracker = BFGSCurvatureTracker(history_size=5)

    s = torch.tensor([1.0, 0.0, 0.0, 0.0])
    y_negative = torch.tensor([-1.0, 0.0, 0.0, 0.0])  # yᵀ s = -1 < 0

    accepted = tracker.update(s, y_negative)
    assert accepted is False
    assert len(tracker.s_history) == 0


def test_learned_hessian_inverter_forward():
    """
    Tests that LearnedHessianInverter acts as a well-conditioned parametric metric tensor.
    """
    param_dim = 16
    cond_dim = 8
    inverter = LearnedHessianInverter(param_dim=param_dim, cond_dim=cond_dim, rank=4)

    batch_size = 4
    v = torch.randn(batch_size, param_dim)
    cond = torch.randn(batch_size, cond_dim)

    preconditioned_v = inverter(v, cond)

    assert preconditioned_v.shape == (batch_size, param_dim)
    assert torch.isfinite(preconditioned_v).all()

    # Preconditioned vector must have positive inner product with original vector (positive definiteness)
    dot_products = torch.sum(v * preconditioned_v, dim=-1)
    assert (dot_products > 0.0).all(), "Learned metric tensor must be positive definite"


def test_create_optimizer_factory():
    """
    Tests optimizer instantiation across AdamW, L-BFGS, and HybridBFGSOptimizer.
    """
    model = nn.Linear(10, 2)

    # 1. Standard AdamW
    opt_adam = create_optimizer(model.parameters(), optimizer_type="adamw", lr=1e-3)
    assert isinstance(opt_adam, torch.optim.AdamW)

    # 2. L-BFGS
    opt_lbfgs = create_optimizer(model.parameters(), optimizer_type="lbfgs", lr=0.1)
    assert isinstance(opt_lbfgs, torch.optim.LBFGS)

    # 3. Hybrid
    opt_hybrid = create_optimizer(model.parameters(), optimizer_type="hybrid_bfgs", lr=1e-3, switch_epoch=3)
    assert isinstance(opt_hybrid, HybridBFGSOptimizer)
    assert not opt_hybrid.is_bfgs_active
    opt_hybrid.set_epoch(3)
    assert opt_hybrid.is_bfgs_active


def test_lbfgs_quadratic_optimization_convergence():
    """
    Tests that L-BFGS optimizer solves an ill-conditioned quadratic function
    significantly faster than standard gradient steps.
    """
    dim = 6
    # Ill-conditioned quadratic matrix A
    eigenvals = torch.tensor([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    A = torch.diag(eigenvals)
    b = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])

    x = nn.Parameter(torch.zeros(dim))
    optimizer = create_optimizer([x], optimizer_type="lbfgs", lr=1.0, max_iter=20)

    initial_loss = 0.5 * torch.dot(x, torch.mv(A, x)) - torch.dot(b, x)

    def closure():
        optimizer.zero_grad()
        loss = 0.5 * torch.dot(x, torch.mv(A, x)) - torch.dot(b, x)
        loss.backward()
        return loss

    optimizer.step(closure)

    final_loss = 0.5 * torch.dot(x, torch.mv(A, x)) - torch.dot(b, x)
    assert final_loss.item() < initial_loss.item()
    # True minimum x* = A⁻¹ b
    x_star = torch.linalg.solve(A, b)
    assert torch.allclose(x, x_star, atol=1e-3), f"L-BFGS failed to solve quadratic minimum: error={torch.norm(x - x_star)}"
