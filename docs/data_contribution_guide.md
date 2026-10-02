# RainAI Data Contribution & Acoustic Ingestion Specification

## 1. Architecture Overview

RainAI uses a zero-Python, pure-Rust data ingestion and acoustic preprocessing pipeline designed for reproducible dataset management and federated community contributions.

```
 Community Contributor
  (Field Recordist / Researcher)
       |
       |  WAV files + Manifest (.json)
       v
 [rainai_contribute]
   ├── Ethical License Verifier (CC0, CC-BY, Public Domain)
   ├── Acoustic Quality Screener (RMS, Clipping, Flatness, Cavitation >4kHz)
   ├── Surface Taxonomy Mapper (9 Canonical Surfaces)
   └── Cryptographic Hasher (SHA-256)
       |
       v
  data/rain/ (Raw Staged Audio)
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

When an audio file is submitted or discovered, `validate_audio_file` decodes the PCM waveform and evaluates the following DSP criteria:

### 2.1 RMS Energy & Dynamic Range
$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^N x[n]^2}$$
- **Threshold**: $\text{RMS} \ge 0.003$
- **Purpose**: Eliminates muted microphone recordings, uncalibrated audio interfaces, and digital dropout buffers.

### 2.2 Sample Clipping Ratio
$$\text{Clipping Ratio} = \frac{\# \{n : |x[n]| \ge 0.999\}}{N}$$
- **Threshold**: $\text{Clipping Ratio} \le 1.5\%$
- **Purpose**: Rejects distorted recordings where rainfall droplet impacts exceed microphone preamp headroom.

### 2.3 Wiener Spectral Flatness
$$\text{SFM} = \frac{\exp\left(\frac{1}{K}\sum_{k=1}^K \ln |X[k]|^2\right)}{\frac{1}{K}\sum_{k=1}^K |X[k]|^2}$$
- **Threshold**: $\text{SFM} \ge 0.04$
- **Purpose**: Rejects narrow-band electrical interference, 50/60 Hz ground loop hum, or pure tones. Natural rainfall acoustic texture exhibits high spectral entropy and broadband flatness.

### 2.4 High-Frequency Cavitation Ratio
$$\text{HF Ratio} = \frac{\sum_{f_k \ge 4000\text{ Hz}} |X[k]|^2}{\sum_{k=1}^K |X[k]|^2}$$
- **Threshold**: $\text{HF Ratio} \ge 0.03$
- **Purpose**: Rain droplet impact and sub-surface bubble cavitation produce characteristic acoustic acoustic energy in the 4 kHz to 16 kHz band (Minnaert / Pumphrey-Crum frequency range). Muffled or severely bandlimited files are rejected.

---

## 3. Quota Auditing & Shannon Diversity Entropy

To prevent class imbalance during neural model training (e.g. over-representing pavement while tin roofs or canvas tents are starved of data), the system monitors **Shannon Diversity Entropy**:

$$H = -\sum_{i=1}^9 p_i \ln(p_i)$$
$$\text{Normalized Diversity} = \frac{H}{\ln(9)} \in [0.0, 1.0]$$

Contributors can query current deficits using:
```bash
cargo run -p utilities --bin rainai_contribute -- quota
```

Surfaces flagged as underrepresented should be prioritized by field recordists.

---

## 4. Provenance Register & Licensing Contract

Every contributed file generates an immutable entry in:
1. `data/rain/ATTRIBUTIONS.txt`: Plaintext attribution log with recordist name, license, source URL/path, and SHA-256 hash.
2. `data/rain/manifest_provenance.json`: JSON catalog tracking audio quality metrics, file size, surface category, and timestamp.
3. `sources.json`: Workspace-wide source catalog feeding the training and synthesis pipelines.

All contributions must be released under an open commercial-friendly license (`CC0`, `CC-BY`, or `Public Domain`).
