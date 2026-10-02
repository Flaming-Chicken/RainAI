# Contributing Rain Audio Data to RainAI

Welcome to the **RainAI Acoustic Dataset Contribution Guide**! RainAI synthesizes realistic, immersive, 3D spatial soundscapes using neural models (Spatial VAE + Mamba-2 SSD Mixture-of-Experts) trained on real-world precipitation acoustics.

To capture the vast diversity of global rain textures—from tropical monsoon deluges on tin roofs to high-altitude pine needle mists—we welcome audio contributions from field recordists, acoustic researchers, sound designers, and audio enthusiasts.

This guide explains how to format, validate, and submit your raw rain recordings.

---

## 1. Quick Start

### 1.1 Step 1: Check Underrepresented Surfaces
Before recording or contributing, check which physical impact surfaces are currently needed most:
```bash
cargo run -p utilities --bin rainai_contribute -- quota
```
This prints the dataset's current balance across the 9 canonical surfaces and highlights any surface deficits.

### 1.2 Step 2: Generate a Manifest Template
Generate an annotated JSON template:
```bash
cargo run -p utilities --bin rainai_contribute -- template --out my_rain_contribution.json
```

### 1.3 Step 3: Validate Your Recordings
Verify that your audio files pass objective acoustic quality screening and ethical license checks:
```bash
# Validate your manifest:
cargo run -p utilities --bin rainai_contribute -- validate my_rain_contribution.json

# Or validate a folder of WAV files directly:
cargo run -p utilities --bin rainai_contribute -- validate ./my_recordings/
```

### 1.4 Step 4: Import into the Local Dataset
Import your recordings directly into the training pipeline:
```bash
cargo run -p utilities --bin rainai_contribute -- import-dir ./my_recordings/ \
  --surface tin_roof \
  --rate heavy_rain \
  --author "Your Name <your.email@example.org>" \
  --license "CC0 1.0 Universal"
```

---

## 2. Audio Technical Requirements

To ensure high training fidelity, all submitted audio must meet these standards:

| Parameter | Requirement | Rationale |
|---|---|---|
| **Format** | Standard uncompressed WAV (`.wav`) | Lossless PCM preservation |
| **Sample Rate** | 44.1 kHz, 48.0 kHz, or 96.0 kHz (48 kHz preferred) | Pipeline standardizes to 48 kHz |
| **Bit Depth** | 16-bit or 24-bit PCM, or 32-bit float | High dynamic range |
| **Channels** | Mono (1ch), Stereo (2ch), Binaural (2ch), or FOA/HOA (4ch/16ch) | Spatial upmixing projects to FOA/HOA |
| **Minimum Duration**| $\ge 2.0$ seconds (5.0s – 30.0s recommended) | Pipeline slices 5.0s training chunks |
| **RMS Energy** | $\ge 0.003$ | Eliminates silence and digital dropouts |
| **Clipping** | $< 1.5\%$ clipped samples | Rejects distorted, overdriven recordings |
| **Noise Floor** | No intrusive speech, sirens, dog barks, or heavy engine hum | Pure environmental rain texture |

---

## 3. The 9 Canonical Surface Categories

Raindrop cavitation sound depends heavily on the resonant impedance and damping of the physical impact surface. Every contributed recording must be categorized into one of our 9 canonical surfaces:

1. **`asphalt`**: Wet highways, tarmac, roadways, parking lots, coarse porous bitumen.
2. **`pavement`**: Concrete sidewalks, cobblestones, granite pavers, courtyards, brick plazas.
3. **`tin_roof`**: Corrugated iron roofs, metal awnings, zinc sheds, gutters, downspouts.
4. **`canvas_tent`**: Nylon tents, rainflies, fabric umbrellas, tarps, awnings, gazebos.
5. **`foliage`**: Tree canopies, deciduous leaves, pine needle understories, bamboo, forest underbrush.
6. **`wood_deck`**: Cedar decking, boardwalks, wooden benches, lumber pallets, timber shingles.
7. **`glass`**: Window panes, skylights, car windshields, conservatory glass, greenhouse panels.
8. **`puddle_shallow`**: Thin standing water puddles ($< 3\text{ cm}$), gravel splashes, storm drains.
9. **`water_deep`**: Lakes, ponds, rivers, oceans, swimming pools, submerged hydrophone cavitation.

---

## 4. Permitted Open Licenses

All training data in RainAI must be commercially viable and publicly shareable. We enforce strict licensing verification via `LicenseVerifier`:

- **Approved**:
  - `CC0 1.0 Universal` (Public Domain Dedication)
  - `Public Domain` / US Government Unconstrained
  - `CC-BY 4.0` (Creative Commons Attribution 4.0 International)
  - `CC-BY 3.0` / `CC-BY 2.0`
  - `CC-BY-SA 4.0` (Creative Commons Attribution-ShareAlike)
- **Strictly Rejected**:
  - Any license with **NonCommercial** (`-NC`) clauses (e.g. `CC-BY-NC`).
  - Any license with **NoDerivatives** (`-ND`) clauses (e.g. `CC-BY-ND`).
  - Proprietary / All Rights Reserved material without a formal licensing grant.

---

## 5. Contribution Manifest Schema

A contribution manifest is a simple JSON file specifying your recordings:

```json
{
  "manifest_version": "1.0",
  "dataset_name": "Pacific Northwest Rainforest Spring Deluge",
  "contributor_name": "Alex Smith <alex@example.org>",
  "contributor_contact": "https://github.com/alexsmith",
  "default_license": "CC0 1.0 Universal",
  "default_surface": null,
  "sources": [
    {
      "id": "pnw_shed_tin_roof_01",
      "file_path": "recordings/tin_roof_heavy_rain.wav",
      "url": null,
      "surface": "tin_roof",
      "precipitation_rate": "heavy_rain",
      "environment": "Backyard shed surrounded by cedar trees",
      "microphone_setup": "stereo_ortf",
      "sample_rate": 48000,
      "license": "CC0 1.0 Universal",
      "author": "Alex Smith",
      "notes": "Recorded with Zoom H5 and pair of matched cardioid capsules",
      "sha256": null
    },
    {
      "id": "pnw_cedar_canopy_drizzle_02",
      "file_path": "recordings/cedar_canopy_drizzle.wav",
      "url": null,
      "surface": "foliage",
      "precipitation_rate": "drizzle",
      "environment": "Temperate rainforest understory",
      "microphone_setup": "binaural_in_ear",
      "sample_rate": 48000,
      "license": "CC-BY 4.0",
      "author": "Alex Smith",
      "notes": "In-ear binaural mics mounted on windshield baffle",
      "sha256": null
    }
  ]
}
```

### Valid Values for Fields:
- **`surface`**: `"asphalt"`, `"pavement"`, `"tin_roof"`, `"canvas_tent"`, `"foliage"`, `"wood_deck"`, `"glass"`, `"puddle_shallow"`, `"water_deep"`.
- **`precipitation_rate`**: `"drizzle"`, `"light_rain"`, `"moderate_rain"`, `"heavy_rain"`, `"violent_storm"`.
- **`microphone_setup`**: `"mono"`, `"stereo_spaced"`, `"stereo_ortf"`, `"binaural_in_ear"`, `"ambisonic_foa"`, `"ambisonic_hoa"`, `"hydrophone"`, `"contact_mic"`.

---

## 6. How Your Contributed Data is Used

Once imported:
1. **Provenance Tracking**: Your name, license, and file SHA-256 hash are recorded in `data/rain/ATTRIBUTIONS.txt` and `manifest_provenance.json`.
2. **Ambisonic Spatial Upmixing**: Files are upmixed to 4-channel First-Order Ambisonics ($W, Y, Z, X$) in 5.0-second training chunks via `rainai_upmix`.
3. **Sub-Band Feature Extraction**: Spectral features, RMS energy, and 16-band log mel energies are computed via `rainai_features`.
4. **Candle Neural Training**: The neural network learns the physical droplet cavitation dynamics from your audio via `rainai_train_candle`.
5. **Real-Time Synthesis**: End users in the browser and desktop app experience your rain textures in spatial 3D audio.

Thank you for helping make RainAI the most realistic physical acoustic rainfall synthesis engine!
