//! Canonical physical surface classification and acoustic modal profiles for RainAI.
//!
//! Provides the single source of truth for the 9-class physical material taxonomy across
//! dataset ingestion, quota balancing, neural conditioning vectors, and physical acoustic synthesis.

use serde::{Deserialize, Serialize};

/// Canonical 9-class physical surface classification for rain acoustic modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalSurface {
    Asphalt,
    Pavement,
    TinRoof,
    CanvasTent,
    Foliage,
    WoodDeck,
    Glass,
    PuddleShallow,
    WaterDeep,
}

impl CanonicalSurface {
    /// Array containing all 9 canonical surfaces in deterministic order.
    pub const ALL: [Self; 9] = [
        Self::Asphalt,
        Self::Pavement,
        Self::TinRoof,
        Self::CanvasTent,
        Self::Foliage,
        Self::WoodDeck,
        Self::Glass,
        Self::PuddleShallow,
        Self::WaterDeep,
    ];

    /// Deterministic 0..9 integer index for categorical indexing.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Self::Asphalt => 0,
            Self::Pavement => 1,
            Self::TinRoof => 2,
            Self::CanvasTent => 3,
            Self::Foliage => 4,
            Self::WoodDeck => 5,
            Self::Glass => 6,
            Self::PuddleShallow => 7,
            Self::WaterDeep => 8,
        }
    }

    /// Offset in the 64-dimensional conditioning vector (indices 10..19).
    #[inline]
    pub const fn condition_index(self) -> usize {
        10 + self.index()
    }

    /// Canonical snake_case string representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asphalt => "asphalt",
            Self::Pavement => "pavement",
            Self::TinRoof => "tin_roof",
            Self::CanvasTent => "canvas_tent",
            Self::Foliage => "foliage",
            Self::WoodDeck => "wood_deck",
            Self::Glass => "glass",
            Self::PuddleShallow => "puddle_shallow",
            Self::WaterDeep => "water_deep",
        }
    }

    /// Maps heterogeneous source category tags or file names to one of the 9 canonical physical surfaces.
    pub fn from_tag(tag: &str) -> Self {
        let s = tag.trim().to_lowercase();
        if s.contains("asphalt")
            || s.contains("highway")
            || s.contains("street")
            || s.contains("traffic")
            || s.contains("tarmac")
            || s.contains("roadway")
            || s.contains("driveway")
        {
            Self::Asphalt
        } else if s.contains("pavement")
            || s.contains("cobble")
            || s.contains("granite")
            || s.contains("sidewalk")
            || s.contains("courtyard")
            || s.contains("flagstone")
            || s.contains("brick")
            || s.contains("concrete")
            || s.contains("plaza")
        {
            Self::Pavement
        } else if s.contains("tin")
            || s.contains("roof")
            || s.contains("metal")
            || s.contains("iron")
            || s.contains("awning")
            || s.contains("downpipe")
            || s.contains("downspout")
            || s.contains("zinc")
            || s.contains("aluminum")
            || s.contains("corrugated")
            || s.contains("shed")
            || s.contains("gutter")
        {
            Self::TinRoof
        } else if s.contains("tent")
            || s.contains("canvas")
            || s.contains("umbrella")
            || s.contains("gazebo")
            || s.contains("rainfly")
            || s.contains("tarp")
            || s.contains("parasol")
            || s.contains("bimini")
            || s.contains("nylon")
            || s.contains("fabric")
        {
            Self::CanvasTent
        } else if s.contains("foliage")
            || s.contains("leaves")
            || s.contains("leaf")
            || s.contains("forest")
            || s.contains("canopy")
            || s.contains("pine")
            || s.contains("needle")
            || s.contains("bamboo")
            || s.contains("moss")
            || s.contains("jungle")
            || s.contains("fern")
            || s.contains("vegetation")
            || s.contains("tree")
            || s.contains("woods")
        {
            Self::Foliage
        } else if s.contains("wood")
            || s.contains("deck")
            || s.contains("boardwalk")
            || s.contains("patio")
            || s.contains("bench")
            || s.contains("cedar")
            || s.contains("shingle")
            || s.contains("timber")
            || s.contains("plank")
            || s.contains("lumber")
        {
            Self::WoodDeck
        } else if s.contains("glass")
            || s.contains("window")
            || s.contains("skylight")
            || s.contains("windshield")
            || s.contains("conservatory")
            || s.contains("pane")
            || s.contains("glazing")
            || s.contains("sunroof")
        {
            Self::Glass
        } else if s.contains("puddle")
            || s.contains("splash")
            || s.contains("drain")
            || s.contains("shallow")
            || s.contains("gravel")
            || s.contains("plop")
            || s.contains("runoff")
        {
            Self::PuddleShallow
        } else if s.contains("deep")
            || s.contains("water")
            || s.contains("lake")
            || s.contains("pond")
            || s.contains("hydrophone")
            || s.contains("ocean")
            || s.contains("river")
            || s.contains("sea")
            || s.contains("stream")
            || s.contains("reservoir")
            || s.contains("cavitation")
        {
            Self::WaterDeep
        } else {
            Self::Pavement
        }
    }

    /// Backward-compatible alias for `from_tag`.
    #[inline]
    pub fn from_category_tag(tag: &str) -> Self {
        Self::from_tag(tag)
    }

    /// Returns the acoustic modal resonator profile for physical droplet synthesis.
    pub fn modal_profile(self) -> ModalSurfaceProfile {
        match self {
            Self::TinRoof => ModalSurfaceProfile {
                freq1: 1250.0,
                freq2: 2550.0,
                damp1: 180.0,
                damp2: 270.0,
                amp1: 0.5,
                amp2: 0.25,
            },
            Self::CanvasTent => ModalSurfaceProfile {
                freq1: 400.0,
                freq2: 850.0,
                damp1: 450.0,
                damp2: 600.0,
                amp1: 0.3,
                amp2: 0.15,
            },
            Self::Glass => ModalSurfaceProfile {
                freq1: 3100.0,
                freq2: 5200.0,
                damp1: 220.0,
                damp2: 350.0,
                amp1: 0.45,
                amp2: 0.2,
            },
            Self::WoodDeck => ModalSurfaceProfile {
                freq1: 650.0,
                freq2: 1300.0,
                damp1: 320.0,
                damp2: 480.0,
                amp1: 0.4,
                amp2: 0.2,
            },
            Self::WaterDeep | Self::PuddleShallow => ModalSurfaceProfile {
                freq1: 800.0,
                freq2: 1800.0,
                damp1: 60.0,
                damp2: 120.0,
                amp1: 0.6,
                amp2: 0.3,
            },
            Self::Asphalt | Self::Pavement => ModalSurfaceProfile {
                freq1: 950.0,
                freq2: 2100.0,
                damp1: 380.0,
                damp2: 520.0,
                amp1: 0.35,
                amp2: 0.18,
            },
            Self::Foliage => ModalSurfaceProfile {
                freq1: 520.0,
                freq2: 1150.0,
                damp1: 420.0,
                damp2: 580.0,
                amp1: 0.32,
                amp2: 0.16,
            },
        }
    }
}

impl std::fmt::Display for CanonicalSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Physical modal vibration parameters for plate and membrane acoustic resonance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModalSurfaceProfile {
    pub freq1: f32,
    pub freq2: f32,
    pub damp1: f32,
    pub damp2: f32,
    pub amp1: f32,
    pub amp2: f32,
}

/// 7-dimensional continuous physical material properties replacing rigid categorical surface bins
/// in the continuous neural conditioning manifold and physical DSP simulation.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct PhysicalMaterialProperties {
    /// Surface structural rigidity (0.0 soft foliage/canvas -> 1.0 hardened tin/glass)
    pub hardness: f32,
    /// Modal resonant pitch center (0.0 deep rumble / 200 Hz -> 1.0 high ping / 8 kHz)
    pub resonance_freq: f32,
    /// Structural loss factor (0.0 prolonged metallic chime -> 1.0 dead absorbent thud)
    pub damping: f32,
    /// Energy transmission vs reflection boundary (0.0 absorbent -> 1.0 reflective barrier)
    pub acoustic_impedance: f32,
    /// Microscopic surface roughness scattering (0.0 mirror glass -> 1.0 gravel/bark)
    pub roughness: f32,
    /// Standing water layer depth (0.0 dry surface -> 0.5 damp film -> 1.0 deep cavitation puddle)
    pub water_depth: f32,
    /// Cavity and enclosure hollowness (0.0 solid earth/pavement -> 1.0 hollow drum/tent/attic)
    pub cavity_hollowness: f32,
}

impl Default for PhysicalMaterialProperties {
    fn default() -> Self {
        Self {
            hardness: 0.5,
            resonance_freq: 0.5,
            damping: 0.5,
            acoustic_impedance: 0.5,
            roughness: 0.5,
            water_depth: 0.2,
            cavity_hollowness: 0.2,
        }
    }
}

impl PhysicalMaterialProperties {
    #[inline]
    pub const fn to_array(&self) -> [f32; 7] {
        [
            self.hardness,
            self.resonance_freq,
            self.damping,
            self.acoustic_impedance,
            self.roughness,
            self.water_depth,
            self.cavity_hollowness,
        ]
    }
}

impl CanonicalSurface {
    /// Returns the continuous acoustic material properties for this canonical surface.
    pub const fn material_properties(self) -> PhysicalMaterialProperties {
        match self {
            Self::TinRoof => PhysicalMaterialProperties {
                hardness: 0.95,
                resonance_freq: 0.75, // ~5.0 kHz
                damping: 0.15,        // high ring
                acoustic_impedance: 0.90,
                roughness: 0.10,
                water_depth: 0.05,
                cavity_hollowness: 0.70, // corrugated sheet cavity
            },
            Self::CanvasTent => PhysicalMaterialProperties {
                hardness: 0.25,
                resonance_freq: 0.22, // ~750 Hz
                damping: 0.75,        // damped fabric
                acoustic_impedance: 0.30,
                roughness: 0.40,
                water_depth: 0.10,
                cavity_hollowness: 0.90, // hollow tent drum
            },
            Self::Foliage => PhysicalMaterialProperties {
                hardness: 0.15,
                resonance_freq: 0.35, // ~1.5 kHz
                damping: 0.85,        // soft leaves
                acoustic_impedance: 0.20,
                roughness: 0.65,
                water_depth: 0.20,
                cavity_hollowness: 0.10,
            },
            Self::WoodDeck => PhysicalMaterialProperties {
                hardness: 0.60,
                resonance_freq: 0.38, // ~1.8 kHz
                damping: 0.50,        // moderate resonance
                acoustic_impedance: 0.60,
                roughness: 0.50,
                water_depth: 0.15,
                cavity_hollowness: 0.45,
            },
            Self::Glass => PhysicalMaterialProperties {
                hardness: 0.98,
                resonance_freq: 0.85, // ~6.2 kHz
                damping: 0.10,        // crisp ring
                acoustic_impedance: 0.95,
                roughness: 0.02,
                water_depth: 0.05,
                cavity_hollowness: 0.15,
            },
            Self::Pavement | Self::Asphalt => PhysicalMaterialProperties {
                hardness: 0.85,
                resonance_freq: 0.45, // ~2.2 kHz
                damping: 0.75,        // rigid but heavily damped
                acoustic_impedance: 0.80,
                roughness: 0.80,
                water_depth: 0.25,
                cavity_hollowness: 0.02,
            },
            Self::PuddleShallow => PhysicalMaterialProperties {
                hardness: 0.20,
                resonance_freq: 0.28, // ~1.0 kHz
                damping: 0.90,
                acoustic_impedance: 0.35,
                roughness: 0.10,
                water_depth: 0.65, // shallow water layer
                cavity_hollowness: 0.05,
            },
            Self::WaterDeep => PhysicalMaterialProperties {
                hardness: 0.05,
                resonance_freq: 0.15, // ~400 Hz cavitation
                damping: 0.98,
                acoustic_impedance: 0.10,
                roughness: 0.05,
                water_depth: 1.00, // deep body of water
                cavity_hollowness: 0.02,
            },
        }
    }
}
