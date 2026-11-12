//! Autonomous Training Ecosystem Test Suite:
//! - Persistent Utility Ledger & Historical Score Caching
//! - Zero-Veto Multiplicative Utility Formulation
//! - Rolling Eviction with Ledger Retention
//! - Muon, Lion, and MuonLionHybrid Optimizers
//! - Catalogue Exhaustion Detection & High-Yield Replay Buffer Sampling

use candle_core::{DType, Device, Tensor, Var};
use std::{collections::HashMap, fs};
use utilities::{
    autopilot::{detect_catalogue_exhaustion, sample_high_yield_replay_batch},
    candle::optimizers::{
        LionConfig, LionOptimizer, MuonConfig, MuonLionHybrid, MuonLionHybridConfig, MuonOptimizer,
        newton_schulz5, tensor_sign,
    },
    data_worker::{
        DataChunkUtilityRecord, DataUtilityLedger, DatabaseHealthWorker,
        compute_geometric_utility_score, compute_retraining_priority, ledger_path_for_manifest,
    },
    features::AudioMetadata,
};

#[test]
fn test_zero_veto_and_geometric_utility_scoring() {
    // 1. Zero-Veto Property: Corrupt or silent audio (Q <= 1e-4) MUST have 0 utility
    // even with massive reconstruction loss (e.g., L = 500.0)
    let corrupt_q = 0.0f32;
    let massive_loss = 500.0f32;
    let u_corrupt = compute_geometric_utility_score(corrupt_q, massive_loss, 1.5, 1.5, 0, 0.0);
    assert_eq!(
        u_corrupt, 0.0,
        "Zero-veto must strictly zero out utility on corrupt audio regardless of loss"
    );

    let priority_corrupt = compute_retraining_priority(corrupt_q, massive_loss, 1.5, 1.5);
    assert_eq!(
        priority_corrupt, 0.0,
        "Zero-veto must strictly zero out retraining priority on corrupt audio"
    );

    // 2. High fidelity audio with healthy dynamics
    let clean_q = 0.85f32;
    let moderate_loss = 1.2f32;
    let u_clean_fresh = compute_geometric_utility_score(clean_q, moderate_loss, 1.0, 1.0, 0, 0.0);
    assert!(
        u_clean_fresh > 0.5,
        "Fresh clean audio should receive high utility score (got {:.3})",
        u_clean_fresh
    );

    // 3. Satiation Discount: As training count increases, utility decreases
    let u_clean_trained_5 =
        compute_geometric_utility_score(clean_q, moderate_loss, 1.0, 1.0, 5, 0.0);
    let u_clean_trained_20 =
        compute_geometric_utility_score(clean_q, moderate_loss, 1.0, 1.0, 20, 0.0);
    assert!(
        u_clean_trained_5 > u_clean_trained_20,
        "Utility must monotonically decay with exposure count (got {:.3} vs {:.3})",
        u_clean_trained_5,
        u_clean_trained_20
    );

    // 4. Retraining Priority: Difficult boundary cases (high loss) with high acoustic fidelity (high Q)
    let easy_loss = 0.05f32;
    let hard_loss = 3.5f32;
    let priority_easy = compute_retraining_priority(clean_q, easy_loss, 1.0, 1.0);
    let priority_hard = compute_retraining_priority(clean_q, hard_loss, 1.0, 1.0);
    assert!(
        priority_hard > priority_easy,
        "Hard boundary samples must have higher retraining priority than easy samples (got {:.3} vs {:.3})",
        priority_hard,
        priority_easy
    );
}

#[test]
fn test_persistent_utility_ledger_lifecycle() {
    let temp_dir =
        std::env::temp_dir().join(format!("rainai_ledger_test_{}", rand::random::<u64>()));
    fs::create_dir_all(&temp_dir).unwrap();
    let ledger_path = temp_dir.join("data_utility_ledger.json");

    let mut ledger = DataUtilityLedger::load_or_create(&ledger_path);
    assert_eq!(ledger.records.len(), 0);

    let record1 = DataChunkUtilityRecord {
        chunk_id: "rain_tin_roof_001".to_string(),
        sha256: "abcdef123456".to_string(),
        source_id: "tin_roof.wav".to_string(),
        surface_tag: "tin_roof".to_string(),
        material_properties: [0.9, 0.8, 0.1, 0.95, 0.2, 0.0, 0.7],
        acoustic_quality_score: 0.88,
        loss_ema: 2.5,
        gradient_norm_ema: 1.2,
        times_trained: 1,
        last_trained_timestamp: 1000,
        information_novelty: 1.2,
        current_utility_score: 1.5,
        retraining_priority: 2.1,
        is_cached_locally: true,
    };

    let record2 = DataChunkUtilityRecord {
        chunk_id: "rain_pavement_002".to_string(),
        sha256: "789012abcdef".to_string(),
        source_id: "pavement.wav".to_string(),
        surface_tag: "pavement".to_string(),
        material_properties: [0.85, 0.1, 0.9, 0.9, 0.8, 0.1, 0.0],
        acoustic_quality_score: 0.60,
        loss_ema: 0.1,
        gradient_norm_ema: 0.05,
        times_trained: 10,
        last_trained_timestamp: 2000,
        information_novelty: 0.3,
        current_utility_score: 0.08,
        retraining_priority: 0.12,
        is_cached_locally: true,
    };

    ledger.upsert_record(record1);
    ledger.upsert_record(record2);
    assert_eq!(ledger.records.len(), 2);
    assert_eq!(ledger.total_samples_ever_indexed, 2);

    // Save and reload
    ledger.save_to_file(&ledger_path).unwrap();
    let mut reloaded = DataUtilityLedger::load_or_create(&ledger_path);
    assert_eq!(reloaded.records.len(), 2);
    assert!(reloaded.records.contains_key("rain_tin_roof_001"));

    // Test training feedback update
    reloaded.update_training_feedback("rain_tin_roof_001", 1.8, 0.9, 1500, 1.2);
    let updated = reloaded.records.get("rain_tin_roof_001").unwrap();
    assert_eq!(updated.times_trained, 2);
    assert_eq!(updated.last_trained_timestamp, 1500);

    // Test eviction marking
    reloaded.mark_evicted("rain_pavement_002");
    assert!(
        !reloaded
            .records
            .get("rain_pavement_002")
            .unwrap()
            .is_cached_locally
    );

    // Test top retraining candidates
    let candidates = reloaded.top_retraining_candidates(1);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].chunk_id, "rain_tin_roof_001");

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_rolling_pruning_with_ledger_retention() {
    let temp_dir = std::env::temp_dir().join(format!(
        "rainai_prune_ledger_test_{}",
        rand::random::<u64>()
    ));
    fs::create_dir_all(&temp_dir).unwrap();

    let processed_dir = temp_dir.join("processed");
    fs::create_dir_all(&processed_dir).unwrap();
    let manifest_path = temp_dir.join("manifest.json");
    let ledger_path = ledger_path_for_manifest(&manifest_path);

    // Create 2 chunks: one low quality, one high quality
    let meta_low = AudioMetadata {
        path: processed_dir
            .join("low_q.wav")
            .to_string_lossy()
            .to_string(),
        filename: "low_q.wav".to_string(),
        sample_rate: 48000,
        channels: 2,
        duration_secs: 5.0,
        rms_energy: 0.0001,
        rain_rate: 1.0,
        droplet_density: 0.05,
        drops_per_second: 10.0,
        high_freq_ratio: 0.01,
        spectral_centroid: 300.0,
        spectral_rolloff: 600.0,
        spectral_flatness: 0.95,
        surface_tag: "pavement".to_string(),
    };

    let meta_high = AudioMetadata {
        path: processed_dir
            .join("high_q.wav")
            .to_string_lossy()
            .to_string(),
        filename: "high_q.wav".to_string(),
        sample_rate: 48000,
        channels: 2,
        duration_secs: 5.0,
        rms_energy: 0.06,
        rain_rate: 45.0,
        droplet_density: 0.6,
        drops_per_second: 600.0,
        high_freq_ratio: 0.50,
        spectral_centroid: 4800.0,
        spectral_rolloff: 8800.0,
        spectral_flatness: 0.35,
        surface_tag: "pavement".to_string(),
    };

    let file_low = processed_dir.join("low_q.wav");
    let file_high = processed_dir.join("high_q.wav");
    fs::write(&file_low, vec![0u8; 1024]).unwrap();
    fs::write(&file_high, vec![0u8; 1024]).unwrap();

    let mut manifest = HashMap::new();
    manifest.insert("chunk_low".to_string(), meta_low);
    manifest.insert("chunk_high".to_string(), meta_high);
    let f = fs::File::create(&manifest_path).unwrap();
    serde_json::to_writer_pretty(f, &manifest).unwrap();

    // Quotas: pavement heavily overrepresented
    let mut counts = HashMap::new();
    counts.insert("pavement".to_string(), 100);
    counts.insert("glass".to_string(), 5);
    let (_, quotas) = utilities::autopilot::SurfaceEntropyAuditor::audit(&counts);

    // Enforce 1500 bytes ceiling (total is 2048 bytes)
    let evicted = DatabaseHealthWorker::enforce_rolling_quota_with_ceiling(
        &processed_dir,
        &manifest_path,
        &quotas,
        1500,
    );
    assert_eq!(evicted, 1);

    // Verify low quality WAV was evicted from disk
    assert!(
        !file_low.exists(),
        "Low quality chunk WAV must be deleted from disk"
    );
    assert!(
        file_high.exists(),
        "High quality chunk WAV must be preserved"
    );

    // Verify ledger preserved metadata and marked low_q as not cached locally
    let ledger = DataUtilityLedger::load_or_create(&ledger_path);
    assert!(ledger.records.contains_key("chunk_low"));
    let low_record = ledger.records.get("chunk_low").unwrap();
    assert!(
        !low_record.is_cached_locally,
        "Evicted chunk must be retained in ledger with is_cached_locally = false"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_muon_lion_optimizers() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // 1. Test tensor_sign
    let test_t = Tensor::new(&[-2.5f32, 0.0, 1.7, -0.01, 10.0], &device)?;
    let sign_t = tensor_sign(&test_t)?;
    let sign_vals = sign_t.to_vec1::<f32>()?;
    assert!((sign_vals[0] - (-1.0)).abs() < 1e-3);
    assert!(sign_vals[1].abs() < 1e-3);
    assert!((sign_vals[2] - 1.0).abs() < 1e-3);
    assert!((sign_vals[3] - (-1.0)).abs() < 1e-3);
    assert!((sign_vals[4] - 1.0).abs() < 1e-3);

    // 2. Test Newton-Schulz 5-step matrix orthogonalization
    // Create random 2D matrix (4 x 8)
    let g = Tensor::new(
        &[
            [1.0f32, 0.5, -0.3, 0.8, -1.2, 0.4, 0.1, -0.9],
            [-0.2, 1.4, 0.7, -0.5, 0.3, -0.8, 1.1, 0.0],
            [0.6, -0.1, 1.3, 0.2, -0.4, 0.9, -0.7, 0.5],
            [-0.8, 0.3, -0.6, 1.5, 0.1, -0.2, 0.4, 1.2],
        ],
        &device,
    )?;
    let ortho = newton_schulz5(&g, 5, 1e-7)?;
    assert_eq!(ortho.dims(), &[4, 8]);

    // Gram matrix ortho * ortho^T should approximate Identity scaled
    let gram = ortho.matmul(&ortho.t()?)?;
    let eye = Tensor::eye(4, DType::F32, &device)?;
    let diff = (gram - eye)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    assert!(
        diff < 1.0,
        "Newton-Schulz iteration must orthogonalize matrix (Frobenius residual: {:.4})",
        diff
    );

    // 3. Test LionOptimizer step
    let var_1d = Var::new(&[1.0f32, -2.0, 3.0], &device)?;
    let mut lion = LionOptimizer::new(vec![var_1d.clone()], LionConfig::default())?;

    // Create grads via backprop
    let loss_1d = (var_1d.as_tensor() * &Tensor::new(&[0.5f32, -0.8, 0.2], &device)?)?.sum_all()?;
    let grads = loss_1d.backward()?;

    lion.step(&grads)?;
    let updated_vals = var_1d.as_tensor().to_vec1::<f32>()?;
    // Parameter should have stepped in opposite direction of sign(grad)
    assert!(updated_vals[0] < 1.0); // grad > 0 -> param decreased
    assert!(updated_vals[1] > -2.0); // grad < 0 -> param increased

    // 4. Test MuonOptimizer step on 2D parameter
    let var_2d = Var::new(
        &[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        &device,
    )?;
    let mut muon = MuonOptimizer::new(vec![var_2d.clone()], MuonConfig::default())?;
    let loss_2d = var_2d.as_tensor().sum_all()?;
    let grads_2d = loss_2d.backward()?;

    muon.step(&grads_2d)?;
    let updated_2d = var_2d.as_tensor().to_vec2::<f32>()?;
    assert!(updated_2d[0][0] < 1.0);

    // 5. Test MuonLionHybrid
    let var_mat = Var::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device)?;
    let var_bias = Var::new(&[0.5f32, -0.5], &device)?;
    let mut hybrid = MuonLionHybrid::new(
        vec![var_mat.clone(), var_bias.clone()],
        MuonLionHybridConfig::default(),
    )?;

    let hybrid_loss = (&var_mat.as_tensor().sum_all()? + &var_bias.as_tensor().sum_all()?)?;
    let hybrid_grads = hybrid_loss.backward()?;

    hybrid.set_learning_rate(0.001);
    hybrid.step(&hybrid_grads)?;

    assert!(var_mat.as_tensor().to_vec2::<f32>()?[0][0] < 1.0);
    assert!(var_bias.as_tensor().to_vec1::<f32>()?[0] < 0.5);

    Ok(())
}

#[test]
fn test_catalogue_exhaustion_and_replay_sampling() {
    let temp_dir =
        std::env::temp_dir().join(format!("rainai_exhaust_test_{}", rand::random::<u64>()));
    fs::create_dir_all(&temp_dir).unwrap();

    let sources_path = temp_dir.join("sources.json");
    let manifest_path = temp_dir.join("manifest.json");
    let ledger_path = ledger_path_for_manifest(&manifest_path);

    // 1. Ingested 1 of 2 sources -> Not exhausted
    let sources_json = r#"[
        {"filename": "rain_sample_01.mp3", "category": "tin_roof", "license": "CC0"},
        {"filename": "rain_sample_02.mp3", "category": "pavement", "license": "CC-BY"}
    ]"#;
    fs::write(&sources_path, sources_json).unwrap();

    let manifest_json = r#"{
        "chunk_01": {"filename": "rain_sample_01_chunk001.wav", "surface_tag": "tin_roof"}
    }"#;
    fs::write(&manifest_path, manifest_json).unwrap();

    let status1 = detect_catalogue_exhaustion(&sources_path, &manifest_path);
    assert_eq!(status1.total_sources, 2);
    assert_eq!(status1.ingested_sources, 1);
    assert!(
        !status1.is_exhausted,
        "Should not be exhausted when 1 source is missing"
    );

    // 2. Both sources ingested -> Catalogue is exhausted
    let manifest_all_json = r#"{
        "chunk_01": {"filename": "rain_sample_01_chunk001.wav", "surface_tag": "tin_roof"},
        "chunk_02": {"filename": "rain_sample_02_chunk001.wav", "surface_tag": "pavement"}
    }"#;
    fs::write(&manifest_path, manifest_all_json).unwrap();

    // Populate ledger with high-uncertainty and low-uncertainty records
    let mut ledger = DataUtilityLedger::default();
    ledger.upsert_record(DataChunkUtilityRecord {
        chunk_id: "chunk_01".to_string(),
        sha256: "hash1".to_string(),
        source_id: "rain_sample_01.mp3".to_string(),
        surface_tag: "tin_roof".to_string(),
        material_properties: [0.9; 7],
        acoustic_quality_score: 0.9,
        loss_ema: 2.8,
        gradient_norm_ema: 1.5,
        times_trained: 3,
        last_trained_timestamp: 100,
        information_novelty: 1.5,
        current_utility_score: 1.8,
        retraining_priority: 3.2,
        is_cached_locally: true,
    });
    ledger.upsert_record(DataChunkUtilityRecord {
        chunk_id: "chunk_02".to_string(),
        sha256: "hash2".to_string(),
        source_id: "rain_sample_02.mp3".to_string(),
        surface_tag: "pavement".to_string(),
        material_properties: [0.5; 7],
        acoustic_quality_score: 0.7,
        loss_ema: 0.05,
        gradient_norm_ema: 0.01,
        times_trained: 15,
        last_trained_timestamp: 200,
        information_novelty: 0.2,
        current_utility_score: 0.05,
        retraining_priority: 0.1,
        is_cached_locally: true,
    });
    ledger.save_to_file(&ledger_path).unwrap();

    let status2 = detect_catalogue_exhaustion(&sources_path, &manifest_path);
    assert!(
        status2.is_exhausted,
        "Catalogue should now be reported as exhausted"
    );
    assert_eq!(status2.total_sources, 2);
    assert_eq!(status2.ingested_sources, 2);
    assert_eq!(status2.high_yield_replay_candidates, 1);

    // Test targeted replay batch sampling
    let replay_batch = sample_high_yield_replay_batch(&ledger_path, 1);
    assert_eq!(replay_batch.len(), 1);
    assert_eq!(replay_batch[0].chunk_id, "chunk_01");
    assert_eq!(replay_batch[0].surface_tag, "tin_roof");

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_seed_canonical_data_utility_ledger() {
    let prov_candidates = [
        std::path::PathBuf::from("data/rain/manifest_provenance.json"),
        std::path::PathBuf::from("../../data/rain/manifest_provenance.json"),
        std::path::PathBuf::from("../data/rain/manifest_provenance.json"),
    ];
    let found = prov_candidates.iter().find(|p| p.exists());
    assert!(
        found.is_some(),
        "manifest_provenance.json must exist in repository"
    );
    let prov_path = found.unwrap();
    let ledger = DataUtilityLedger::seed_from_provenance_or_manifest(None, Some(prov_path))
        .expect("Should seed ledger from provenance");
    assert!(
        ledger.records.len() >= 30,
        "Must index all canonical sources (got {})",
        ledger.records.len()
    );
    let target_path = prov_path.parent().unwrap().join("data_utility_ledger.json");
    ledger
        .save_to_file(&target_path)
        .expect("Should save ledger to git-tracked location");
    assert!(target_path.exists());
}
