# RainAI Data Contribution & Acoustic Ingestion Specification

## 1. Architecture Overview

RainAI uses a high-efficiency, pure-Rust data ingestion and acoustic preprocessing pipeline designed for reproducible dataset management, zero quality loss, and frictionless community contributions.

```
 Community Contributor (Web UI / CLI)
        |
        +---> Web Audio API / WASM DSP Pre-Screen (RMS & Clipping in browser)
        +---> In-Browser Vorbis/ID3 Tag Injection
        +---> Lossless PCM -> FLAC Level 8 / Native Lossy Preservation
        |
        v
 [Cloudflare R2 Staging Bucket] (Zero-SQL Document & Audio Object Store)
    ├── staging/quarantine/ (Unlicensed or DSP-failed submissions)
    ├── staging/approved/   (Vetted community audio payloads)
    └── staging/urls/       (Remote YouTube/Cloud pointers, lazy pull)
        |
        v
 Maintainer CLI (crates/utilities)
    ├── cargo run -p utilities --bin rainai_contribute -- triage
    └── cargo run -p utilities --bin rainai_contribute -- pull-approved
        |
        v
 Data/raw/ (dev branch only via Git LFS) & sources.json
        |
        v
 [rainai_upmix] ──> 4-Channel Ambisonics (FOA B-Format: W, Y, Z, X)
        |
        v
 [rainai_features] ──> 16-Band Sub-band Energies & Conditioning Tensors
        |
        v
 [rainai_train_candle] ──> Spatial VAE + Mamba-2 SSD MoE Training
```

---

## 2. Objective Quality Screening Metrics

When an audio file is evaluated, Symphonia streaming decodes the waveform block-by-block with constant $O(1)$ memory consumption and evaluates:

### 2.1 RMS Energy & Dynamic Range
$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^N x[n]^2}$$
- **Threshold**: $\text{RMS} \ge 0.001$
- **Purpose**: Eliminates muted microphone recordings, uncalibrated audio interfaces, and digital dropout buffers while accommodating distant, gentle rainfall.

### 2.2 Sample Clipping Ratio
$$\text{Clipping Ratio} = \frac{\# \{n : |x[n]| \ge 0.999\}}{N}$$
- **Threshold**: $\text{Clipping Ratio} \le 5.0\%$
- **Purpose**: Rejects severely distorted recordings where rainfall droplet impacts exceed microphone preamp headroom.

### 2.3 Wiener Spectral Flatness
$$\text{SFM} = \frac{\exp\left(\frac{1}{K}\sum_{k=1}^K \ln |X[k]|^2\right)}{\frac{1}{K}\sum_{k=1}^K |X[k]|^2}$$
- **Threshold**: $\text{SFM} \ge 0.02$
- **Purpose**: Rejects narrow-band electrical interference, 50/60 Hz ground loop hum, or pure tones. Natural rainfall acoustic texture exhibits broadband stochastic noise.

---

## 3. Storage Optimization & Zero Quality Degradation

1. **Uncompressed PCM (WAV, AIFF):** Converted client-side to **FLAC Level 8** (100% bit-for-bit lossless, reducing storage by 40–60%).
2. **Native Lossy (Opus, OGG, AAC, MP3):** Preserved in their native containers with zero transcoding to prevent generational degradation.
3. **Video Recordings (MP4, MKV):** Audio track extracted client-side; video stream discarded prior to upload.
4. **Remote URLs:** Cataloged as verified pointers; pulled lazily on-demand during local ML training passes to spare Git repository bloat.

---

## 4. Licensing & Ethical Governance

RainAI operates a comprehensive provenance and attribution architecture. All audio assets admitted into the training corpus are **permanently attributed upon ingestion** in [`data/rain/ATTRIBUTIONS.txt`](../data/rain/ATTRIBUTIONS.txt) and [`DATA_BIBLIOGRAPHY.md`](../DATA_BIBLIOGRAPHY.md). Because attribution is permanently secured at dataset registration, **model inference outputs are never blocked at runtime**. 

Explainable AI (XAI) runtime output tracing is provided as an **optional transparency feature** across a 4-tier hierarchy rather than a blocking operational constraint.

### 4.1 Supported License Tiers & Compatible Licenses

| Tier | Compatible Licenses | Training Ingestion Policy | XAI Runtime Attribution |
|---|---|---|---|
| **Tier 1: Public Domain** | `CC0 1.0 Universal`, `Public Domain`, `Unlicense`, `WTFPL`, `ODC-PDDL`, US Government unconstrained | Unconditional approved ingestion | Optional provenance tracing; unconstrained generation |
| **Tier 2: RainAI / Spodeian Proprietary** | `RainAI-FC-Proprietary-License` (Default) | Approved commercial & non-commercial training grant | Optional contributor credit and transparency telemetry |
| **Tier 3: Permissive Attribution** | `CC-BY 4.0/3.0/2.0`, `MIT`, `Apache-2.0`, `BSD-2/3-Clause`, `ISC`, `ODC-By` | Approved with permanent attribution registered in `ATTRIBUTIONS.txt` | Optional dynamic output attribution mapping; generation never fails |
| **Tier 4: Commercial Share-Alike** | `CC-BY-SA 4.0/3.0` | Approved with derivative copyleft attribution registered in model manifest | Permanent attribution in model cards and weights documentation |
| **Quarantine: Unknown / Unspecified** | `Unknown`, `Unspecified`, `Pending Discovery` | **Immediately Quarantined** to `staging/quarantine/` | Never enters training corpus until license is verified |
| **Restricted: Ineligible** | Non-Commercial (`-NC`), No-Derivatives (`-ND`), All Rights Reserved | **Rejected** | Prohibited from ingestion and quarantine promotion |

### 4.2 The Default Proprietary License (`RainAI-FC-Proprietary-License`)
This is the default option for user submissions, providing a broad perpetual grant for RainAI, Spodeian, and Flaming Chicken projects:

> *"By submitting this data and metadata, I grant Spodeian, Flaming Chicken, and their respective affiliates, successors, and assigns a worldwide, non-exclusive, royalty-free, perpetual, irrevocable, and sublicensable right to use, reproduce, modify, adapt, publish, translate, create derivative works from, distribute, and publicly display this data for any purpose, including commercial and non-commercial applications. This explicitly includes, without limitation, the right to use the data to train, test, and validate machine learning models for the RainAI project and any other current or future projects. I represent and warrant that I own or have the necessary rights to grant this license."*

**UI Transparency Copy:**
> *"While this license allows us to use your data freely to build RainAI, our system is designed for transparency. We track the metadata of all contributions, meaning you will always be credited when your specific data directly influences our explainable AI's outputs."*

### 4.3 Immediate Quarantine for Unknown Licenses
Contributors who do not know the license of an audio file or remote stream may select `Unknown / Unspecified`.
- Audio and metadata JSON descriptors are staged **immediately to `staging/quarantine/`**.
- Automated ingestion background workers inspect the quarantined payload to scrape and extract license headers from Vorbis comments, ID3 tags, or remote web pages.
- If a valid compatible license (e.g. CC0, CC-BY, MIT) is discovered and verified, maintainers can promote the asset to `staging/approved/` using `rainai_contribute triage`. Otherwise, the file remains safely isolated.

### 4.4 Mandatory Rights Warranty
All contributors must affirm before staging:
> *"I represent and warrant that I own or have the necessary rights to grant this license."*

---

## 5. Multi-Tier Failsafe Fallback

If cloud staging or upload limits are reached, contributors can email files and URLs directly to:
- **Email:** `spodeian@proton.me`
- Pre-formatted mailto links with tags, license, and checksum are automatically generated in the web application modal.

---

## 6. Maintainer CLI Commands

```bash
# Inspect quarantined records and failure diagnostics
cargo run -p utilities --bin rainai_contribute -- triage [DIR]

# Promote approved staging records to Data/raw/ on the dev branch
cargo run -p utilities --bin rainai_contribute -- pull-approved [DIR]

# Audit dataset tag distribution
cargo run -p utilities --bin rainai_contribute -- quota
```
