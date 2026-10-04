//! Zero-Copy Binary Attribution Dictionary
//!
//! Provides high-efficiency, zero-heap-allocation lookup of contributor credits,
//! licenses, and surface acoustic provenance by SHA-256 hash or surface profile.
//!
//! Implemented without unsafe code, supporting WASM, mobile, and desktop runtimes.

use serde::{Deserialize, Serialize};

/// Magic bytes identifying a RainAI Binary Attribution Dictionary ("RATT").
pub const DICTIONARY_MAGIC: &[u8; 4] = b"RATT";

/// Format version.
pub const DICTIONARY_VERSION: u16 = 1;

/// Fixed byte size of each entry in the index table.
pub const ENTRY_SIZE: usize = 36;

/// Fixed byte size of the file header (4 magic + 2 version + 4 entry count).
pub const HEADER_SIZE: usize = 10;

/// Plain input record used to compile binary dictionaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionRecordInput {
    pub sha256_hex: String,
    pub contributor: String,
    pub license: String,
    pub license_tier: u8,
    pub surface: String,
}

/// Zero-allocation borrowed reference to an attribution entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributionEntryRef<'a> {
    pub sha_prefix: &'a [u8; 16],
    pub license_tier: u8,
    pub contributor: &'a str,
    pub license: &'a str,
    pub surface: &'a str,
}

/// Zero-copy reader over a contiguous binary attribution slice.
#[derive(Debug, Clone, Copy)]
pub struct BinaryAttributionDictionary<'a> {
    data: &'a [u8],
    num_entries: usize,
    string_table_offset: usize,
}

impl<'a> BinaryAttributionDictionary<'a> {
    /// Mounts and validates a binary attribution dictionary slice.
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() < HEADER_SIZE {
            return None;
        }

        if &data[0..4] != DICTIONARY_MAGIC {
            return None;
        }

        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != DICTIONARY_VERSION {
            return None;
        }

        let num_entries = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;
        let string_table_offset = HEADER_SIZE + num_entries * ENTRY_SIZE;

        if data.len() < string_table_offset {
            return None;
        }

        Some(Self {
            data,
            num_entries,
            string_table_offset,
        })
    }

    /// Number of records in the dictionary.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_entries
    }

    /// Whether the dictionary contains zero records.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// Retrieves an entry by 0-based index.
    pub fn get(&self, index: usize) -> Option<AttributionEntryRef<'a>> {
        if index >= self.num_entries {
            return None;
        }

        let offset = HEADER_SIZE + index * ENTRY_SIZE;
        let entry_slice = &self.data[offset..offset + ENTRY_SIZE];

        let sha_prefix: &'a [u8; 16] = entry_slice[0..16].try_into().ok()?;
        let license_tier = entry_slice[16];
        // entry_slice[17] is reserved flags

        let contrib_off = u32::from_le_bytes(entry_slice[18..22].try_into().ok()?) as usize;
        let contrib_len = u16::from_le_bytes(entry_slice[22..24].try_into().ok()?) as usize;

        let lic_off = u32::from_le_bytes(entry_slice[24..28].try_into().ok()?) as usize;
        let lic_len = u16::from_le_bytes(entry_slice[28..30].try_into().ok()?) as usize;

        let surf_off = u32::from_le_bytes(entry_slice[30..34].try_into().ok()?) as usize;
        let surf_len = u16::from_le_bytes(entry_slice[34..36].try_into().ok()?) as usize;

        let contributor = self.read_string(contrib_off, contrib_len)?;
        let license = self.read_string(lic_off, lic_len)?;
        let surface = self.read_string(surf_off, surf_len)?;

        Some(AttributionEntryRef {
            sha_prefix,
            license_tier,
            contributor,
            license,
            surface,
        })
    }

    #[inline]
    fn read_string(&self, offset: usize, len: usize) -> Option<&'a str> {
        let start = self.string_table_offset + offset;
        let end = start + len;
        if end > self.data.len() {
            return None;
        }
        std::str::from_utf8(&self.data[start..end]).ok()
    }

    /// Performs an O(log N) binary search for a record matching the 16-byte SHA-256 prefix.
    pub fn lookup_by_prefix(&self, target_prefix: &[u8; 16]) -> Option<AttributionEntryRef<'a>> {
        let mut low = 0;
        let mut high = self.num_entries;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = HEADER_SIZE + mid * ENTRY_SIZE;
            let prefix: &[u8; 16] = (&self.data[offset..offset + 16]).try_into().ok()?;

            match prefix.cmp(target_prefix) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return self.get(mid),
            }
        }

        None
    }

    /// Looks up a record by full or partial hexadecimal SHA-256 string.
    pub fn lookup_by_hex(&self, hex_str: &str) -> Option<AttributionEntryRef<'a>> {
        let clean = hex_str.trim();
        if clean.len() < 32 {
            return None;
        }

        let mut prefix = [0u8; 16];
        let bytes = clean.as_bytes();
        let (chunks, _) = bytes[..32].as_chunks::<2>();
        for (i, chunk) in chunks.iter().enumerate() {
            let byte_str = std::str::from_utf8(chunk).ok()?;
            prefix[i] = u8::from_str_radix(byte_str, 16).ok()?;
        }

        self.lookup_by_prefix(&prefix)
    }

    /// Returns an iterator yielding all entries in the dictionary.
    pub fn iter(&self) -> AttributionIter<'a> {
        AttributionIter {
            dict: *self,
            current: 0,
        }
    }
}

/// Iterator over entries in `BinaryAttributionDictionary`.
#[derive(Debug, Clone, Copy)]
pub struct AttributionIter<'a> {
    dict: BinaryAttributionDictionary<'a>,
    current: usize,
}

impl<'a> Iterator for AttributionIter<'a> {
    type Item = AttributionEntryRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.dict.len() {
            let item = self.dict.get(self.current);
            self.current += 1;
            item
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.dict.len().saturating_sub(self.current);
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for AttributionIter<'a> {}

/// Compiles a collection of attribution records into a zero-copy binary format.
pub fn compile_binary_attribution_dictionary(mut records: Vec<AttributionRecordInput>) -> Vec<u8> {
    // 1. Sort records by 16-byte SHA-256 prefix for binary search
    records.sort_by(|a, b| {
        let p_a = decode_hex_prefix_16(&a.sha256_hex);
        let p_b = decode_hex_prefix_16(&b.sha256_hex);
        p_a.cmp(&p_b)
    });

    let num_entries = records.len();
    let mut string_table = Vec::new();

    // Intermediate struct to hold index offsets
    struct EntryData {
        sha_prefix: [u8; 16],
        license_tier: u8,
        contrib_offset: u32,
        contrib_len: u16,
        lic_offset: u32,
        lic_len: u16,
        surf_offset: u32,
        surf_len: u16,
    }

    let mut index_entries = Vec::with_capacity(num_entries);

    for rec in &records {
        let sha_prefix = decode_hex_prefix_16(&rec.sha256_hex);

        let contrib_offset = string_table.len() as u32;
        let contrib_bytes = rec.contributor.as_bytes();
        string_table.extend_from_slice(contrib_bytes);
        let contrib_len = contrib_bytes.len() as u16;

        let lic_offset = string_table.len() as u32;
        let lic_bytes = rec.license.as_bytes();
        string_table.extend_from_slice(lic_bytes);
        let lic_len = lic_bytes.len() as u16;

        let surf_offset = string_table.len() as u32;
        let surf_bytes = rec.surface.as_bytes();
        string_table.extend_from_slice(surf_bytes);
        let surf_len = surf_bytes.len() as u16;

        index_entries.push(EntryData {
            sha_prefix,
            license_tier: rec.license_tier,
            contrib_offset,
            contrib_len,
            lic_offset,
            lic_len,
            surf_offset,
            surf_len,
        });
    }

    // 2. Assemble output buffer
    let mut out = Vec::with_capacity(HEADER_SIZE + num_entries * ENTRY_SIZE + string_table.len());

    // Header
    out.extend_from_slice(DICTIONARY_MAGIC);
    out.extend_from_slice(&DICTIONARY_VERSION.to_le_bytes());
    out.extend_from_slice(&(num_entries as u32).to_le_bytes());

    // Index Table
    for entry in index_entries {
        out.extend_from_slice(&entry.sha_prefix);
        out.push(entry.license_tier);
        out.push(0); // reserved flags
        out.extend_from_slice(&entry.contrib_offset.to_le_bytes());
        out.extend_from_slice(&entry.contrib_len.to_le_bytes());
        out.extend_from_slice(&entry.lic_offset.to_le_bytes());
        out.extend_from_slice(&entry.lic_len.to_le_bytes());
        out.extend_from_slice(&entry.surf_offset.to_le_bytes());
        out.extend_from_slice(&entry.surf_len.to_le_bytes());
    }

    // String Table
    out.extend_from_slice(&string_table);

    out
}

fn decode_hex_prefix_16(hex: &str) -> [u8; 16] {
    let clean = hex.trim();
    let mut prefix = [0u8; 16];
    if clean.len() < 32 {
        return prefix;
    }
    let bytes = clean.as_bytes();
    let (chunks, _) = bytes[..32].as_chunks::<2>();
    for (i, chunk) in chunks.iter().enumerate() {
        if let Ok(s) = std::str::from_utf8(chunk) {
            if let Ok(b) = u8::from_str_radix(s, 16) {
                prefix[i] = b;
            }
        }
    }
    prefix
}
