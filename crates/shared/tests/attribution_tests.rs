use shared::attribution::{
    AttributionRecordInput, BinaryAttributionDictionary, compile_binary_attribution_dictionary,
};

#[test]
fn test_binary_attribution_roundtrip() {
    let records = vec![
        AttributionRecordInput {
            sha256_hex: "33d42b8fdf937be8447280852779b790898fb575bb49cc00233f6eafce8d9235"
                .to_string(),
            contributor: "Alice Recordist".to_string(),
            license: "CC-BY-4.0".to_string(),
            license_tier: 4,
            surface: "canvas_tent".to_string(),
        },
        AttributionRecordInput {
            sha256_hex: "78522c77fe791a2aed8221cb387699ae4c4c9acc1361d89b82cad1e99fd0ec52"
                .to_string(),
            contributor: "Bob Engineer".to_string(),
            license: "RainAI-FC-Proprietary-License".to_string(),
            license_tier: 6,
            surface: "tin_roof".to_string(),
        },
        AttributionRecordInput {
            sha256_hex: "111122223333444455556666777788889999aaaabbbbccccddddeeeeffff0000"
                .to_string(),
            contributor: "Charlie Field".to_string(),
            license: "CC0 1.0".to_string(),
            license_tier: 5,
            surface: "foliage".to_string(),
        },
    ];

    let bytes = compile_binary_attribution_dictionary(records);
    let dict = BinaryAttributionDictionary::new(&bytes)
        .expect("Valid binary attribution dictionary should parse");

    assert_eq!(dict.len(), 3);
    assert!(!dict.is_empty());

    // 1. Test lookup by full hex
    let alice = dict
        .lookup_by_hex("33d42b8fdf937be8447280852779b790898fb575bb49cc00233f6eafce8d9235")
        .expect("Alice should be found");
    assert_eq!(alice.contributor, "Alice Recordist");
    assert_eq!(alice.license, "CC-BY-4.0");
    assert_eq!(alice.license_tier, 4);
    assert_eq!(alice.surface, "canvas_tent");

    // 2. Test lookup by prefix
    let bob = dict
        .lookup_by_hex("78522c77fe791a2aed8221cb387699ae")
        .expect("Bob should be found by 32-hex prefix");
    assert_eq!(bob.contributor, "Bob Engineer");
    assert_eq!(bob.license, "RainAI-FC-Proprietary-License");
    assert_eq!(bob.license_tier, 6);
    assert_eq!(bob.surface, "tin_roof");

    // 3. Test lookup of non-existent
    assert!(
        dict.lookup_by_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            .is_none()
    );

    // 4. Test iterator
    let items: Vec<_> = dict.iter().collect();
    assert_eq!(items.len(), 3);
    // Entries should be sorted by prefix (Charlie: 1111..., Alice: 33d4..., Bob: 7852...)
    assert_eq!(items[0].contributor, "Charlie Field");
    assert_eq!(items[1].contributor, "Alice Recordist");
    assert_eq!(items[2].contributor, "Bob Engineer");
}

#[test]
fn test_corrupt_or_truncated_dictionary() {
    assert!(BinaryAttributionDictionary::new(&[]).is_none());
    assert!(BinaryAttributionDictionary::new(b"RATT").is_none()); // header too short
    assert!(BinaryAttributionDictionary::new(b"NOPE000000").is_none()); // bad magic
}
