use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use utilities::ingest::{
    AttributionPolicyOutcome, CONTRIBUTOR_WARRANTY_STATEMENT, DownloadItem, LicenseTier,
    LicenseVerifier, ProvenanceManifest, ProvenanceRecord, RAINAI_FC_PROPRIETARY_LICENSE_TEXT,
    RAINAI_FC_TRANSPARENCY_NOTE, TagBalanceQuota, analyze_pcm_samples, compute_file_sha256,
    evaluate_attribution_policy,
};

#[test]
fn test_license_verifier_logic() {
    // Approved tiers
    let (ok, tier, _) = LicenseVerifier::verify("RainAI-FC-Proprietary-License");
    assert!(ok);
    assert_eq!(tier, LicenseTier::ProjectProprietary);

    let (ok, tier, _) = LicenseVerifier::verify("CC0 1.0 Universal");
    assert!(ok);
    assert_eq!(tier, LicenseTier::PublicDomain);

    let (ok, tier, _) = LicenseVerifier::verify("Public Domain");
    assert!(ok);
    assert_eq!(tier, LicenseTier::PublicDomain);

    let (ok, tier, _) = LicenseVerifier::verify("CC-BY 4.0");
    assert!(ok);
    assert_eq!(tier, LicenseTier::AttributionOnly);

    let (ok, tier, _) = LicenseVerifier::verify("CC-BY-SA 4.0");
    assert!(ok);
    assert_eq!(tier, LicenseTier::ShareAlike);

    // Permissive compatible open licenses
    let (ok, tier, _) = LicenseVerifier::verify("MIT");
    assert!(ok);
    assert_eq!(tier, LicenseTier::AttributionOnly);

    let (ok, tier, _) = LicenseVerifier::verify("Apache 2.0");
    assert!(ok);
    assert_eq!(tier, LicenseTier::AttributionOnly);

    let (ok, tier, _) = LicenseVerifier::verify("Unlicense");
    assert!(ok);
    assert_eq!(tier, LicenseTier::PublicDomain);

    let (ok, tier, _) = LicenseVerifier::verify("ODC-By");
    assert!(ok);
    assert_eq!(tier, LicenseTier::AttributionOnly);

    // Unknown tier -> Immediately quarantined pending discovery / scraping
    let (ok, tier, reason) = LicenseVerifier::verify("Unknown");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Unknown);
    assert!(reason.contains("Quarantined"));

    let (ok, tier, _) = LicenseVerifier::verify("unspecified");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Unknown);

    // Rejected tiers (NonCommercial / NoDerivs)
    let (ok, tier, reason) = LicenseVerifier::verify("CC-BY-NC 4.0");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Restricted);
    assert!(reason.contains("NonCommercial"));

    let (ok, tier, _) = LicenseVerifier::verify("CC-BY-ND 3.0");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Restricted);

    let (ok, tier, _) = LicenseVerifier::verify("All Rights Reserved Proprietary");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Restricted);
}

#[test]
fn test_tag_balance_quota_tracking() {
    let mut quota = TagBalanceQuota::new();
    assert_eq!(quota.total_samples(), 0);

    for tag in ["asphalt", "pavement", "tin_roof", "glass", "foliage"] {
        let tag_vec = vec![tag.to_string()];
        quota.record(&tag_vec);
        quota.record(&tag_vec);
    }

    assert_eq!(quota.total_samples(), 10);
    assert_eq!(quota.counts.get("tin_roof"), Some(&2));
    assert_eq!(quota.counts.get("glass"), Some(&2));
}

#[test]
fn test_acoustic_quality_metrics_rain_vs_silence() {
    // 1. Pure silence
    let silence = vec![0.0f32; 48000];
    let metrics_silence = analyze_pcm_samples(&silence, 48000, 1);
    assert_eq!(metrics_silence.rms_energy, 0.0);
    assert!(!metrics_silence.is_valid_rain_texture);

    // 2. Synthetic stochastic rain audio (white noise with random transient droplet spikes)
    let mut rain_sim = Vec::with_capacity(48000);
    let mut seed: u64 = 12345;
    for i in 0..48000 {
        // LCG PRNG
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let rand_f = ((seed >> 32) as i32 as f32) / (i32::MAX as f32);
        let noise = rand_f * 0.08;
        // Occasional droplet spike
        let spike = if i % 1200 == 0 { 0.35 } else { 0.0 };
        rain_sim.push(noise + spike);
    }

    let metrics_rain = analyze_pcm_samples(&rain_sim, 48000, 1);
    assert!(metrics_rain.rms_energy > 0.01);
    assert!(
        metrics_rain.spectral_entropy > 0.6,
        "Entropy was {}",
        metrics_rain.spectral_entropy
    );
    assert!(metrics_rain.clipping_ratio < 0.001);
    assert!(metrics_rain.is_valid_rain_texture);
}

#[test]
fn test_provenance_manifest_serialization() {
    let mut tags_dist = HashMap::new();
    tags_dist.insert("canvas_tent".to_string(), 5);
    tags_dist.insert("tin_roof".to_string(), 6);

    let manifest = ProvenanceManifest {
        generated_at_utc: "2026-09-17T12:00:00Z".to_string(),
        total_sources: 11,
        tags_distribution: tags_dist,
        records: vec![ProvenanceRecord {
            filename: "test_rain.wav".to_string(),
            source_url: "https://example.com/test.wav".to_string(),
            source_platform: "TestPlatform".to_string(),
            tags: vec!["tin_roof".to_string()],
            license: Some("CC0".to_string()),
            license_tier: LicenseTier::PublicDomain,
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
            file_size_bytes: 48000,
            quality: None,
            descriptions: vec!["Test soundscape".to_string()],
            alternate_licenses: Vec::new(),
            contributors: vec!["Tester".to_string()],
        }],
    };

    let serialized = serde_json::to_string(&manifest).expect("Serialization failed");
    let deserialized: ProvenanceManifest =
        serde_json::from_str(&serialized).expect("Deserialization failed");
    assert_eq!(deserialized.total_sources, 11);
    assert_eq!(deserialized.records.len(), 1);
    assert_eq!(deserialized.records[0].tags, vec!["tin_roof".to_string()]);
}

#[test]
fn test_compute_file_sha256() {
    let temp_dir = std::env::temp_dir();
    let test_file = temp_dir.join("rainai_test_sha.txt");
    std::fs::write(&test_file, b"RainAI Audio Ingestion Diversity Test")
        .expect("Failed to write test file");

    let hash = compute_file_sha256(&test_file).expect("Failed to compute SHA256");
    assert_eq!(hash.len(), 64);
    let _ = std::fs::remove_file(test_file);
}

#[test]
fn test_sources_json_catalog_integrity() {
    let candidates = [
        PathBuf::from("sources.json"),
        PathBuf::from("../../sources.json"),
        PathBuf::from("../sources.json"),
    ];
    let path = candidates
        .iter()
        .find(|p| p.exists())
        .expect("sources.json not found in search paths");
    let content = std::fs::read_to_string(path).expect("Failed to read sources.json");
    let items: Vec<DownloadItem> =
        serde_json::from_str(&content).expect("Failed to parse sources.json");

    assert!(
        items.len() >= 120,
        "Expected at least 120 sources, found {}",
        items.len()
    );

    let mut urls = HashSet::new();
    let mut filenames = HashSet::new();
    let mut category_counts = HashMap::new();

    for item in &items {
        // 1. Uniqueness
        assert!(
            urls.insert(&item.url),
            "Duplicate URL in sources.json: {}",
            item.url
        );
        assert!(
            filenames.insert(&item.filename),
            "Duplicate filename in sources.json: {}",
            item.filename
        );

        // 2. Ethical Licensing
        let (approved, tier, reason) = LicenseVerifier::verify(&item.license);
        assert!(
            approved,
            "Unapproved license for item {}: {} ({})",
            item.filename, item.license, reason
        );
        assert_ne!(tier, LicenseTier::Restricted);

        *category_counts.entry(item.category.clone()).or_insert(0) += 1;
    }

    assert!(!category_counts.is_empty());
}

#[test]
fn test_license_policy_and_warranty_constants() {
    // 1. Full proprietary legal grant text checks
    assert!(RAINAI_FC_PROPRIETARY_LICENSE_TEXT.contains("Spodeian"));
    assert!(RAINAI_FC_PROPRIETARY_LICENSE_TEXT.contains("Flaming Chicken"));
    assert!(RAINAI_FC_PROPRIETARY_LICENSE_TEXT.contains("commercial and non-commercial"));
    assert!(
        RAINAI_FC_PROPRIETARY_LICENSE_TEXT
            .contains("train, test, and validate machine learning models")
    );
    assert!(RAINAI_FC_PROPRIETARY_LICENSE_TEXT.contains("perpetual, irrevocable"));
    assert!(RAINAI_FC_PROPRIETARY_LICENSE_TEXT.contains("represent and warrant"));

    // 2. Transparency note checks
    assert!(RAINAI_FC_TRANSPARENCY_NOTE.contains("transparency"));
    assert!(RAINAI_FC_TRANSPARENCY_NOTE.contains("explainable AI"));
    assert!(RAINAI_FC_TRANSPARENCY_NOTE.contains("credited"));

    // 3. Contributor warranty check
    assert_eq!(
        CONTRIBUTOR_WARRANTY_STATEMENT,
        "I represent and warrant that I own or have the necessary rights to grant this license."
    );
}

#[test]
fn test_attribution_policy_and_tiers() {
    // Requires attribution
    assert!(LicenseTier::AttributionOnly.requires_attribution());
    assert!(LicenseTier::ShareAlike.requires_attribution());
    assert!(!LicenseTier::PublicDomain.requires_attribution());
    assert!(!LicenseTier::ProjectProprietary.requires_attribution());

    // Approved for raw training corpus
    assert!(LicenseTier::PublicDomain.is_approved());
    assert!(LicenseTier::AttributionOnly.is_approved());
    assert!(LicenseTier::ShareAlike.is_approved());
    assert!(LicenseTier::ProjectProprietary.is_approved());
    assert!(!LicenseTier::Unknown.is_approved());
    assert!(!LicenseTier::Restricted.is_approved());

    // Immediate quarantine check
    assert!(LicenseTier::Unknown.is_quarantined_pending_discovery());
    assert!(!LicenseTier::PublicDomain.is_quarantined_pending_discovery());
}

#[test]
fn test_evaluate_attribution_policy_routing() {
    // 1. Valid attribution metadata present -> Attributed (runtime XAI link)
    let res_cc = evaluate_attribution_policy(
        LicenseTier::AttributionOnly,
        Some("Alice Recordist"),
        "CC-BY 4.0",
    );
    assert_eq!(
        res_cc,
        AttributionPolicyOutcome::Attributed {
            contributor: "Alice Recordist".to_string(),
            license: "CC-BY 4.0".to_string(),
            tier: LicenseTier::AttributionOnly,
        }
    );

    let res_prop = evaluate_attribution_policy(
        LicenseTier::ProjectProprietary,
        Some("Bob Recordist"),
        "RainAI-FC-Proprietary-License",
    );
    assert_eq!(
        res_prop,
        AttributionPolicyOutcome::Attributed {
            contributor: "Bob Recordist".to_string(),
            license: "RainAI-FC-Proprietary-License".to_string(),
            tier: LicenseTier::ProjectProprietary,
        }
    );

    // 2. Missing runtime attribution on CC-BY 4.0 -> NEVER BLOCKS output!
    // Falls back to permanent dataset-level attribution in ATTRIBUTIONS.txt
    let res_cc_missing =
        evaluate_attribution_policy(LicenseTier::AttributionOnly, None, "CC-BY 4.0");
    match res_cc_missing {
        AttributionPolicyOutcome::GlobalTrainingAttributed { ref note, tier } => {
            assert_eq!(tier, LicenseTier::AttributionOnly);
            assert!(note.contains("Permanent dataset-level attribution active"));
            assert!(note.contains("output served unconditionally"));
        }
        _ => panic!("Expected GlobalTrainingAttributed for missing runtime CC-BY attribution"),
    }

    // 3. Missing runtime attribution on Proprietary -> GlobalTrainingAttributed (safely served)
    let res_prop_missing = evaluate_attribution_policy(
        LicenseTier::ProjectProprietary,
        None,
        "RainAI-FC-Proprietary-License",
    );
    match res_prop_missing {
        AttributionPolicyOutcome::GlobalTrainingAttributed { ref note, tier } => {
            assert_eq!(tier, LicenseTier::ProjectProprietary);
            assert!(note.contains("Permanent dataset-level attribution active"));
            assert!(note.contains("output served unconditionally"));
        }
        _ => panic!("Expected GlobalTrainingAttributed for proprietary license"),
    }

    // 4. Missing attribution on CC0 Public Domain -> Unconstrained (safely served)
    let res_cc0_missing = evaluate_attribution_policy(LicenseTier::PublicDomain, None, "CC0 1.0");
    match res_cc0_missing {
        AttributionPolicyOutcome::Unconstrained { ref note, tier } => {
            assert_eq!(tier, LicenseTier::PublicDomain);
            assert!(note.contains("unconstrained generation"));
        }
        _ => panic!("Expected Unconstrained for CC0"),
    }
}

#[test]
fn test_provenance_record_reconciliation_multi_license() {
    let mut rec1 = ProvenanceRecord {
        filename: "storm_001.flac".to_string(),
        source_url: "https://example.org/storm.flac".to_string(),
        source_platform: "Archive".to_string(),
        tags: vec!["heavy_rain".to_string()],
        license: Some("Unknown".to_string()),
        license_tier: LicenseTier::Unknown,
        sha256: "aabbcc112233".to_string(),
        file_size_bytes: 120000,
        quality: None,
        descriptions: vec!["Distant thunder with steady rain".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Anonymous".to_string()],
    };

    let rec2 = ProvenanceRecord {
        filename: "storm_001.flac".to_string(),
        source_url: "local://recontributed.flac".to_string(),
        source_platform: "Community Contribution".to_string(),
        tags: vec!["heavy_rain".to_string(), "thunder".to_string()],
        license: Some("RainAI-FC-Proprietary-License".to_string()),
        license_tier: LicenseTier::ProjectProprietary,
        sha256: "aabbcc112233".to_string(),
        file_size_bytes: 120000,
        quality: None,
        descriptions: vec!["Acoustic rumble and low-frequency resonance".to_string()],
        alternate_licenses: Vec::new(),
        contributors: vec!["Recordist_Jane".to_string()],
    };

    rec1.reconcile_with(rec2);

    assert_eq!(rec1.tags, vec!["heavy_rain", "thunder"]);
    assert_eq!(
        rec1.descriptions,
        vec![
            "Distant thunder with steady rain",
            "Acoustic rumble and low-frequency resonance"
        ]
    );
    assert_eq!(rec1.contributors, vec!["Anonymous", "Recordist_Jane"]);
    // License promoted from Unknown to ProjectProprietary!
    assert_eq!(
        rec1.license.as_deref(),
        Some("RainAI-FC-Proprietary-License")
    );
    assert_eq!(rec1.license_tier, LicenseTier::ProjectProprietary);
}
