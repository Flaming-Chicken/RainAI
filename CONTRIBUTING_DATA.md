# Contributing Rain Audio Data to RainAI

Welcome to the **RainAI Acoustic Dataset Contribution Guide**! RainAI synthesizes realistic, immersive, 3D spatial soundscapes using neural models (Spatial VAE + Mamba-2 SSD Mixture-of-Experts) trained on real-world precipitation acoustics.

To capture the vast diversity of global rain textures—from tropical monsoon deluges on tin roofs to high-altitude pine needle mists—we welcome audio contributions from field recordists, acoustic researchers, sound designers, and audio enthusiasts.

---

## 1. Zero-Friction Web Contribution

The easiest way to contribute is directly through the RainAI Web Application:
1. Open the RainAI Studio interface.
2. Click the **🌧 Contribute Data** button in the top navigation bar.
3. Select your local audio/video file or paste a remote URL (YouTube, Freesound, Google Drive).
4. The browser will automatically:
   - Run Web Audio API DSP checks (RMS & clipping) directly on your device.
   - Compress uncompressed PCM/WAV to **FLAC Level 8** (100% bit-for-bit lossless, zero quality drop).
   - Preserve existing lossy files (Opus, AAC, MP3, OGG) without generational transcode artifacts.
   - Demux audio from video files (MP4, MKV) and discard video tracks locally to save upload bandwidth.
   - Embed your chosen license and tags directly into the container's Vorbis/ID3 comments.
5. Click **Stage Submission to R2**.

---

## 2. Audio Technical Standards

| Parameter | Requirement | Rationale |
|---|---|---|
| **Format** | FLAC Level 8, WAV, AIFF, Opus, AAC, MP3, or OGG | Zero quality loss (lossless compressed with FLAC Level 8; native preservation for lossy) |
| **Sample Rate** | $\ge 44.1\text{ kHz}$ (48 kHz preferred) | Pipeline standardizes to 48 kHz |
| **Channels** | Mono (1ch), Stereo (2ch), Binaural (2ch), or FOA/HOA (4ch/16ch) | Spatial upmixing projects to FOA B-Format |
| **Duration** | Any length (short droplets to multi-hour ambient field captures) | Pipeline segments dynamically |
| **RMS Energy** | $\ge 0.001$ | Rejects digital silence while admitting subtle, distant rainfall |
| **Clipping** | $\le 5.0\%$ clipped samples | Rejects severely distorted recordings |
| **Spectral Flatness** | $\ge 0.02$ | Eliminates pure tones and 50/60Hz ground loop hum |

---

## 3. Licensing Options & Contributor Warranty

RainAI features an **Explainable AI (XAI)** architecture that natively tracks data provenance, mapping model outputs back to input attribution metadata. In RainAI, all training data is **permanently attributed upon ingestion** into [`data/rain/ATTRIBUTIONS.txt`](data/rain/ATTRIBUTIONS.txt) and project documentation, ensuring that model generation is **never blocked at runtime**. Real-time XAI explainability operates as an optional feature.

### 3.1 `RainAI-FC-Proprietary-License` (Default / Recommended)
This is the default option when submitting audio data:
> *"By submitting this data and metadata, I grant Spodeian, Flaming Chicken, and their respective affiliates, successors, and assigns a worldwide, non-exclusive, royalty-free, perpetual, irrevocable, and sublicensable right to use, reproduce, modify, adapt, publish, translate, create derivative works from, distribute, and publicly display this data for any purpose, including commercial and non-commercial applications. This explicitly includes, without limitation, the right to use the data to train, test, and validate machine learning models for the RainAI project and any other current or future projects. I represent and warrant that I own or have the necessary rights to grant this license."*

**Our Transparency Commitment:**
> *"While this license allows us to use your data freely to build RainAI, our system is designed for transparency. We track the metadata of all contributions, meaning you will always be credited when your specific data directly influences our explainable AI's outputs."*

### 3.2 Compatible Open Licenses
Contributors may choose from a wide range of compatible open licenses:
- **`CC0 1.0 Universal` / Public Domain:** Unconstrained dedication (includes `Unlicense`, `WTFPL`, `ODC-PDDL`).
- **`CC-BY 4.0` (Permissive Attribution):** Commercial attribution (also covers `CC-BY 3.0/2.0`, `MIT`, `Apache-2.0`, `BSD`, `ISC`, `ODC-By`). Permanently attributed upon ingestion.
- **`CC-BY-SA 4.0` (Commercial Share-Alike):** Copyleft commercial compatibility with derivative attribution.
- **`Custom`:** Other verified open content licenses.

### 3.3 `Unknown / Unspecified` (Immediate Quarantine)
If you do not know the exact license of a recording or stream:
- Select **`Unknown / Unspecified`**.
- Your submission will be placed **immediately into `staging/quarantine/`**.
- Automated processing workers will scan the audio file and source URL to discover and scrape any embedded license metadata (Vorbis comments, ID3 tags, web metadata) before maintainers review for promotion.

### 3.4 Contributor Warranty
Before staging any audio or URL, you must confirm:
> *"I represent and warrant that I own or have the necessary rights to grant this license."*

*Note: Submissions containing Non-Commercial (`-NC`) or No-Derivatives (`-ND`) clauses are ineligible for commercial neural training and are automatically quarantined or rejected.*

---

## 4. Multi-Tier Failsafe Fallback

If you have a large multi-gigabyte recording library, private cloud folder, or experience any upload connectivity issues, you can email us directly:
- **Email:** `spodeian@proton.me`
- **Subject:** `RainAI Audio Contribution`
- **Include:** Recordist name, license grant, tags (e.g. `tin_roof, car_hood, heavy_downpour`), and download link or attachments.

---

## 5. Developer CLI Workflows

For terminal users and pipeline maintainers:

```bash
# Check current tag distribution
cargo run -p utilities --bin rainai_contribute -- quota

# Acoustically validate a manifest or local directory
cargo run -p utilities --bin rainai_contribute -- validate ./my_recordings/

# Ingest a directory on the dev branch
cargo run -p utilities --bin rainai_contribute -- import-dir ./my_recordings/ \
  --surface tin_roof \
  --rate heavy_rain \
  --author "Recordist Name" \
  --license "RainAI-FC-Proprietary-License"

# Inspect quarantined records
cargo run -p utilities --bin rainai_contribute -- triage

# Promote approved staging records to Data/raw/ on the dev branch
cargo run -p utilities --bin rainai_contribute -- pull-approved
```
