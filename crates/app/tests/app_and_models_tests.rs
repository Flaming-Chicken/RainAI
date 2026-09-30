use app::TemplateApp;
use shared::{Priority, ItemCollection, export_to_json, import_from_json};


#[test]
fn test_template_app_initialization() {
    let app = TemplateApp::default();
    assert_eq!(app.state.collection.total_count(), 3);
    assert_eq!(app.state.collection.completed_count(), 0);
}

#[test]
fn test_item_collection_defaults_and_operations() {
    let mut collection = ItemCollection::default();
    assert_eq!(collection.total_count(), 3);
    assert_eq!(collection.completed_count(), 0);

    let id = collection.add("New Task", "Task Description", Priority::High);
    assert_eq!(collection.total_count(), 4);

    collection.toggle(id);
    assert_eq!(collection.completed_count(), 1);
}

#[test]
fn test_json_roundtrip() {
    let collection = ItemCollection::default();
    let json = export_to_json(&collection).expect("Failed to export JSON");
    let imported = import_from_json(&json).expect("Failed to import JSON");
    assert_eq!(imported.items.len(), collection.items.len());
}

#[test]
fn test_rain_view_custom_ir_file_picker_and_session_state() {
    let mut rain_view = app::RainView::default();
    assert!(rain_view.custom_ir_meta.is_none());
    assert!(rain_view.custom_ir_status.is_none());

    let temp_dir = std::env::temp_dir().join(format!(
        "rainai_app_cas_{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));

    // Synthetic impulse response JSON
    let ir_json = r#"{"left_ir": [1.0, 0.5, 0.25], "right_ir": [0.25, 0.5, 1.0]}"#;
    let res = rain_view.load_custom_ir_bytes("custom_cathedral.sofa", ir_json.as_bytes(), Some(&temp_dir));
    assert!(res.is_ok());

    assert!(rain_view.custom_ir_meta.is_some());
    let meta = rain_view.custom_ir_meta.as_ref().unwrap();
    assert_eq!(meta.name, "custom_cathedral.sofa");
    assert_eq!(meta.format, "sofa");
    assert_eq!(meta.sample_count, 3);

    assert!(rain_view.custom_ir_status.is_some());
    let status = rain_view.custom_ir_status.as_ref().unwrap();
    assert!(status.contains("custom_cathedral.sofa"));

    // Check PersistentSessionState serialization round-trip
    let session = app::storage_manager::PersistentSessionState {
        flow_solver: rain_view.flow_solver,
        active_preset_name: "Gentle Summer Rain".to_string(),
        master_volume: 0.6,
        decode_mode: rain_view.decode_mode,
        noise_masking_enabled: rain_view.noise_masking_enabled,
        noise_masking_threshold_db: -40.0,
        hrtf_profile: rain_view.hrtf_profile.clone(),
        webgpu_fp16_enabled: rain_view.webgpu_fp16,
        show_advanced_inspector: rain_view.show_advanced_inspector,
        custom_ir_hash: Some(meta.sha256_hash.clone()),
    };

    let serialized = serde_json::to_string(&session).expect("serialize session");
    let deserialized: app::storage_manager::PersistentSessionState = serde_json::from_str(&serialized).expect("deserialize session");
    assert_eq!(deserialized.custom_ir_hash, Some(meta.sha256_hash.clone()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}
