# Sound Designer & Studio Engineering Manual

## 1. Introduction

`RainAI` provides studio sound designers and acoustic engineers with real-time controls over physical rain acoustics, room geometry, 3D Ambisonic spatialization, and live room noise calibration.

---

## 2. Room Acoustics & Material Selection

### 2.1 Geometry Inspector
In the **Advanced Studio Inspector**:
- **Width ($X$), Length ($Y$), Height ($Z$)**: Sets room boundary dimensions in meters. Changing geometry immediately updates early reflection arrival times and room volume.
- **Surface Materials**: Configure boundary materials individually:
  - **Floor**: Select `PineTimber` for domestic warmth or `Concrete` for urban echo.
  - **Ceiling**: Select `Plasterboard` for residential spaces or `SheetMetal` for industrial rain-on-roof ambience.
  - **Front/Back/Side Walls**: Add `Glass` for window reflections or `Fabric` for heavy acoustic damping.

### 2.2 Reverberation Time ($T_{60}$)
The Sabine $T_{60}$ readout indicates the time required for acoustic reflections to decay by 60 dB. Domestic rooms typically range from $0.3\text{ s}$ to $0.6\text{ s}$, while large halls or warehouses exceed $2.0\text{ s}$.

---

## 3. Spatial Monitoring & Export Formats

RainAI supports 4 primary listening formats:

1. **Headphones (Binaural HRTF)**:
   - High-fidelity binaural rendering using spherical filter decomposition and SOFA HRTF interpolation.
   - Includes real-time 3D head-tracking across Yaw, Pitch, and Roll.
2. **Phase-Correct Stereo Speakers**:
   - Mid/Side and Cardioid stereo speaker downmixing with cross-feed compensation.
3. **ITU 7.1.4 Immersive Surround**:
   - 12 discrete channels feeding 7 ear-level speakers, 1 LFE subwoofer, and 4 height/ceiling speakers.
4. **16-Channel 3rd-Order Ambisonics (HOA)**:
   - Full ACN/SN3D format export for VR, game engines (Unreal, Unity), and spatial digital audio workstations (DAWs).

---

## 4. Live Microphone Environmental Calibration

### 4.1 Ambient Room Noise Masking
The `AmbientNoiseMasker` continuously monitors environmental background noise via your microphone input:
- Computes real-time RMS and spectral balance across sub-bass, mid, and high frequencies.
- Automatically adjusts rain droplet density, velocity, and output gain to mask external distractions (e.g. traffic, HVAC rumble, office chatter).

### 4.2 Standing Wave Resonance Cancellation
The `AcousticSceneAdapter` utilizes Recursive Least Squares (RLS) tracking to detect physical standing waves in the room (such as 63 Hz and 125 Hz room modes), automatically deploying dynamic parametric notch filters to prevent unnatural acoustic boominess.
