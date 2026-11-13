# Data Bibliography & Provenance Register

**RainAI · Neural Spatial Soundscape Synthesis & Acoustic Dataset Provenance**

This register provides formal accounting, academic literature citations, licensing terms, and cryptographic verification metadata for all audio corpora, acoustic impulse responses, and physics formulations utilized in training RainAI models.

---

## 1. Training Audio: Sources and Licenses

RainAI ships **no audio, datasets or weights** in git. Training audio is
downloaded by `rainai_ingest` from the sources listed in
[`source_registry.json`](source_registry.json), and each item's license is read
from the platform's own metadata at download time.

**Allowed licenses, and nothing else:**
- CC0 1.0 and public domain (including the Public Domain Mark)
- CC BY 3.0 / 4.0, with creator, title, source URL, license URL and changes recorded
- audio you record or generate yourself (for example `synth_rain`), and
  contributions under the RainAI contributor grant (see
  [CONTRIBUTING_DATA.md](CONTRIBUTING_DATA.md))

**Excluded:** anything NonCommercial, NoDerivatives or ShareAlike; platform
licenses (Mixkit, Videvo, Pixabay, BigSoundBank, Zapsplat and similar);
YouTube or AudioSet audio; ESC-50 (CC BY-NC); datasets whose card states a
license but whose files have no per-item provenance; any file with no
recorded source.

| Source | How the license is read | Notes |
|---|---|---|
| [Freesound](https://freesound.org) | `license` field of each sound (API search filtered to CC0 and Attribution) | Needs `FREESOUND_API_KEY`; downloads the HQ preview |
| [Wikimedia Commons](https://commons.wikimedia.org) | `extmetadata.LicenseShortName` per file | Commons rain audio is mostly CC BY-SA, which is rejected |
| [Internet Archive](https://archive.org) (Radio Aporee) | `licenseurl` per item | PD Mark and CC BY 3.0 items only |

**Credits.** Every downloaded file gets a record in `data/rain/ledger.jsonl`
(source, item URL, creator, title, license, license URL, retrieval date,
SHA-256 and the processing applied). `data/rain/ATTRIBUTIONS.txt` and the app's
attribution data are generated from that ledger and published with each model
release; they are never committed. Files used in training that have no ledger
record are reported and should be removed.

**History.** Before 13 Nov 2026 the repository committed audio from sources
that don't meet this standard (ESC-50, Mixkit, Videvo, BigSoundBank,
CC BY-SA files, YouTube-derived AudioSet clips, and files with no recorded
source). That audio and the models trained on it were removed from the history.

---

## 2. Theoretical Architecture & Loss Formulations

### 2.1 State Space Duality (Mamba-2)
- **Primary Citation**:
  - Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. arXiv preprint arXiv:2405.21060.
- **Implementation**: Structured state space selective scan with Mixture of Experts (MoE) routing for autoregressive latent trajectories.

### 2.2 Differentiable Digital Signal Processing (DDSP)
- **Primary Citation**:
  - Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). *DDSP: Differentiable Digital Signal Processing*. International Conference on Learning Representations (ICLR).
- **Role**: Differentiable harmonic additive synthesizers and filtered noise models for droplet cavitation.

### 2.3 First-Order Ambisonics (B-Format)
- **Primary Citation**:
  - Gerzon, M. A. (1973). *Periphony: With-Height Sound Reproduction*. Journal of the Audio Engineering Society, 21(1), 2–10.
- **Channels**: 4-channel spherical harmonic decomposition ($W$: omnidirectional pressure, $X, Y, Z$: directional velocity gradients).

---

## 3. Cryptographic Provenance Manifest

Processed acoustic chunks and spectral features are hashed with SHA-256 and tracked in [`data/processed/manifest.json`](data/processed/manifest.json):

```json
{
  "dataset_version": "v1.0.0-spatial-foliage",
  "sampling_rate": 48000,
  "channels": 4,
  "total_duration_hours": 128.4,
  "manifest_sha256": "3a8c1f9d45e0...",
  "license_tier_distribution": {
    "CC0_PublicDomain": "84.2%",
    "CC_BY_Attribution": "12.6%",
    "CC_BY_SA": "3.2%"
  }
}
```

---

## 4. Community Data Contributions & Ingestion Specification

Guidelines, manifest schemas, and acoustic quality validation for adding new rain sources are specified in [`CONTRIBUTING_DATA.md`](CONTRIBUTING_DATA.md) and [`docs/data_contribution_guide.md`](docs/data_contribution_guide.md).

Contributed sources are screened using the CLI tool:
```bash
cargo run -p utilities --bin rainai_contribute -- validate <manifest_or_audio_dir>
cargo run -p utilities --bin rainai_contribute -- import-dir <audio_dir>
```
All approved assets are cryptographically logged with SHA-256 hashes in [`data/rain/manifest_provenance.json`](data/rain/manifest_provenance.json) and attributed in [`data/rain/ATTRIBUTIONS.txt`](data/rain/ATTRIBUTIONS.txt).

