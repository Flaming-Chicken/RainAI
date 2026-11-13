//! Source registry, downloader adapters and provenance ledger (PLAN.md D2).
//!
//! RainAI ships no audio. `source_registry.json` (committed) lists *sources*
//! (a platform plus a query or collection), never individual files. Each
//! adapter asks its platform for candidate items and reads every item's license
//! from the platform's own metadata; [`LicenseVerifier`] then decides whether
//! the item may be downloaded. Every downloaded file gets one [`LedgerRecord`]
//! in `data/rain/ledger.jsonl`, and the attribution file, the provenance
//! manifest and the local catalog (`data/sources.json`) are all generated from
//! that ledger. None of them are committed.

use crate::ingest::{
    DownloadItem, LicenseTier, LicenseVerifier, ProvenanceManifest, ProvenanceRecord,
    chrono_lite_timestamp,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Processing applied to every ledgered file before training; recorded so
/// CC BY credits can state the changes made, as the license requires.
pub const STANDARD_CHANGES: &str = "decoded, resampled to 48 kHz, chunked, and spatialised to first-order ambisonics for model training";

/// One configured source in `source_registry.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "adapter", rename_all = "snake_case")]
pub enum SourceSpec {
    /// Freesound text search. Needs `FREESOUND_API_KEY`; downloads the HQ preview.
    Freesound {
        id: String,
        category: String,
        query: String,
        #[serde(default = "default_max_items")]
        max_items: usize,
    },
    /// Audio files in one Wikimedia Commons category.
    WikimediaCommons {
        id: String,
        category: String,
        commons_category: String,
        #[serde(default = "default_max_items")]
        max_items: usize,
    },
    /// Internet Archive advanced search; each item's `licenseurl` is checked.
    InternetArchive {
        id: String,
        category: String,
        query: String,
        #[serde(default = "default_max_items")]
        max_items: usize,
    },
}

fn default_max_items() -> usize {
    25
}

impl SourceSpec {
    pub fn id(&self) -> &str {
        match self {
            Self::Freesound { id, .. }
            | Self::WikimediaCommons { id, .. }
            | Self::InternetArchive { id, .. } => id,
        }
    }
    pub fn category(&self) -> &str {
        match self {
            Self::Freesound { category, .. }
            | Self::WikimediaCommons { category, .. }
            | Self::InternetArchive { category, .. } => category,
        }
    }
    pub fn adapter_name(&self) -> &'static str {
        match self {
            Self::Freesound { .. } => "Freesound",
            Self::WikimediaCommons { .. } => "Wikimedia Commons",
            Self::InternetArchive { .. } => "Internet Archive",
        }
    }
}

/// The committed list of sources.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceRegistry {
    pub sources: Vec<SourceSpec>,
}

impl SourceRegistry {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let reg: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let mut seen = HashSet::new();
        for s in &reg.sources {
            if !seen.insert(s.id()) {
                bail!("duplicate source id '{}' in {}", s.id(), path.display());
            }
        }
        Ok(reg)
    }
}

/// An item a source offers, with the license as the platform reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub item_url: String,
    pub download_url: String,
    pub creator: String,
    pub title: String,
    pub license: String,
    pub license_url: String,
    /// File extension for the downloaded file, without the dot.
    pub ext: String,
}

/// One downloaded file and everything needed to credit it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LedgerRecord {
    pub file: String,
    pub sha256: String,
    pub source_id: String,
    pub platform: String,
    pub category: String,
    pub item_url: String,
    pub download_url: String,
    pub creator: String,
    pub title: String,
    pub license: String,
    pub license_url: String,
    pub license_tier: LicenseTier,
    pub retrieved_utc: String,
    pub changes: String,
}

impl LedgerRecord {
    /// A record for a public contribution that passed `check_contribution`.
    #[allow(clippy::too_many_arguments)]
    pub fn contribution(
        file: &str,
        sha256: &str,
        contributor: &str,
        title: &str,
        license: &str,
        tier: LicenseTier,
        surface: &str,
        intake: &str,
    ) -> Self {
        Self {
            file: file.to_string(),
            sha256: sha256.to_string(),
            source_id: format!("contribution-{intake}"),
            platform: "RainAI contribution".to_string(),
            category: surface.to_string(),
            item_url: format!(
                "contribution://{}/{}",
                contributor.replace(' ', "_"),
                sha256
            ),
            download_url: String::new(),
            creator: contributor.to_string(),
            title: title.to_string(),
            license: license.to_string(),
            license_url: String::new(),
            license_tier: tier,
            retrieved_utc: chrono_lite_timestamp(),
            changes: STANDARD_CHANGES.to_string(),
        }
    }
}

/// Append-only JSON-lines ledger.
pub struct Ledger;

impl Ledger {
    pub fn load(path: &Path) -> Result<Vec<LedgerRecord>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(path)?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, l)| {
                serde_json::from_str(l)
                    .with_context(|| format!("{} line {}", path.display(), i + 1))
            })
            .collect()
    }

    pub fn append(path: &Path, rec: &LedgerRecord) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(f, "{}", serde_json::to_string(rec)?)?;
        Ok(())
    }
}

/// `ATTRIBUTIONS.txt`, generated from the ledger.
pub fn render_attributions(records: &[LedgerRecord]) -> String {
    let mut out = String::from(
        "RainAI training data attributions\n\
         Generated from data/rain/ledger.jsonl. Do not edit or commit.\n\n",
    );
    let mut sorted: Vec<&LedgerRecord> = records.iter().collect();
    sorted.sort_by(|a, b| (&a.platform, &a.title).cmp(&(&b.platform, &b.title)));
    for r in sorted {
        let license = if r.license_url.is_empty() || r.license_url == r.license {
            r.license.clone()
        } else {
            format!("{} <{}>", r.license, r.license_url)
        };
        out.push_str(&format!(
            "\"{}\" by {} ({}), {}. Source: {}. Changes: {}.\n",
            r.title, r.creator, r.platform, license, r.item_url, r.changes
        ));
    }
    out
}

/// The local catalog the TUI and data worker read (`data/sources.json`).
pub fn to_catalog(records: &[LedgerRecord]) -> Vec<DownloadItem> {
    records
        .iter()
        .map(|r| DownloadItem {
            url: r.item_url.clone(),
            filename: r.file.clone(),
            category: r.category.clone(),
            license: r.license.clone(),
            source_platform: r.platform.clone(),
            media_type: "audio".to_string(),
            ingest_method: "ledger".to_string(),
        })
        .collect()
}

/// `manifest_provenance.json`, generated from the ledger.
pub fn to_provenance(records: &[LedgerRecord], data_dir: &Path) -> ProvenanceManifest {
    let mut tags_distribution: HashMap<String, usize> = HashMap::new();
    let recs = records
        .iter()
        .map(|r| {
            *tags_distribution.entry(r.category.clone()).or_insert(0) += 1;
            ProvenanceRecord {
                filename: r.file.clone(),
                source_url: r.item_url.clone(),
                source_platform: r.platform.clone(),
                tags: vec![r.category.clone()],
                license: Some(r.license.clone()),
                license_tier: r.license_tier,
                sha256: r.sha256.clone(),
                file_size_bytes: std::fs::metadata(data_dir.join(&r.file))
                    .map(|m| m.len())
                    .unwrap_or(0),
                quality: None,
                descriptions: Vec::new(),
                alternate_licenses: Vec::new(),
                contributors: vec![r.creator.clone()],
            }
        })
        .collect::<Vec<_>>();
    ProvenanceManifest {
        generated_at_utc: chrono_lite_timestamp(),
        total_sources: recs.len(),
        tags_distribution,
        records: recs,
    }
}

// ---------------------------------------------------------------------------
// Platform response parsers (pure, so they're unit-tested without network).
// ---------------------------------------------------------------------------

fn str_at<'a>(v: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
    let mut cur = v;
    for k in path {
        cur = cur.get(*k)?;
    }
    cur.as_str()
}

/// Strips HTML tags and collapses whitespace (Commons' `Artist` field is HTML).
pub fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn ext_of(url: &str, fallback: &str) -> String {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    path.rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .filter(|e| e.len() <= 5 && e.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or_else(|| fallback.to_string())
}

/// Freesound `search/text` response (fields `id,name,username,license,url,previews`).
pub fn parse_freesound(json: &serde_json::Value) -> Vec<Candidate> {
    json.get("results")
        .and_then(|r| r.as_array())
        .into_iter()
        .flatten()
        .filter_map(|r| {
            let download_url = str_at(r, &["previews", "preview-hq-mp3"])?.to_string();
            let license_url = r.get("license")?.as_str()?.to_string();
            Some(Candidate {
                item_url: r.get("url")?.as_str()?.to_string(),
                download_url,
                creator: r.get("username")?.as_str()?.to_string(),
                title: r.get("name")?.as_str()?.to_string(),
                license: license_url.clone(),
                license_url,
                ext: "mp3".to_string(),
            })
        })
        .collect()
}

/// Commons `query` response with `prop=imageinfo&iiprop=url|mime|extmetadata`.
pub fn parse_wikimedia(json: &serde_json::Value) -> Vec<Candidate> {
    let Some(pages) = json.get("query").and_then(|q| q.get("pages")) else {
        return Vec::new();
    };
    let iter: Box<dyn Iterator<Item = &serde_json::Value>> = match pages {
        serde_json::Value::Object(m) => Box::new(m.values()),
        serde_json::Value::Array(a) => Box::new(a.iter()),
        _ => return Vec::new(),
    };
    let mut out: Vec<Candidate> = iter
        .filter_map(|p| {
            let info = p.get("imageinfo")?.as_array()?.first()?;
            // Commons labels Ogg Vorbis/Opus as application/ogg.
            let mime = info.get("mime")?.as_str()?;
            if !(mime.starts_with("audio/") || mime == "application/ogg") {
                return None;
            }
            let meta = info.get("extmetadata")?;
            let download_url = info.get("url")?.as_str()?.to_string();
            let title = str_at(meta, &["ObjectName", "value"])
                .map(strip_html)
                .unwrap_or_else(|| {
                    p.get("title")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .trim_start_matches("File:")
                        .to_string()
                });
            Some(Candidate {
                item_url: info.get("descriptionurl")?.as_str()?.to_string(),
                ext: ext_of(&download_url, "ogg"),
                download_url,
                creator: str_at(meta, &["Artist", "value"])
                    .map(strip_html)
                    .unwrap_or_default(),
                title,
                license: str_at(meta, &["LicenseShortName", "value"])?.to_string(),
                license_url: str_at(meta, &["LicenseUrl", "value"])
                    .unwrap_or("")
                    .to_string(),
            })
        })
        .collect();
    out.sort_by(|a, b| a.item_url.cmp(&b.item_url));
    out
}

/// Internet Archive `advancedsearch` docs: `(identifier, title, creator, licenseurl)`.
pub fn parse_ia_search(json: &serde_json::Value) -> Vec<(String, String, String, String)> {
    let first = |v: &serde_json::Value| -> Option<String> {
        match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(a) => a.first()?.as_str().map(str::to_string),
            _ => None,
        }
    };
    json.get("response")
        .and_then(|r| r.get("docs"))
        .and_then(|d| d.as_array())
        .into_iter()
        .flatten()
        .filter_map(|d| {
            Some((
                d.get("identifier")?.as_str()?.to_string(),
                d.get("title").and_then(first).unwrap_or_default(),
                d.get("creator").and_then(first).unwrap_or_default(),
                d.get("licenseurl").and_then(first)?,
            ))
        })
        .collect()
}

/// Picks the best audio file from an Internet Archive `metadata` response.
pub fn pick_ia_audio_file(json: &serde_json::Value) -> Option<String> {
    const PREFERRED: [&str; 5] = ["Flac", "24bit Flac", "WAVE", "Ogg Vorbis", "VBR MP3"];
    let files = json.get("files")?.as_array()?;
    PREFERRED.iter().find_map(|fmt| {
        files
            .iter()
            .find(|f| f.get("format").and_then(|v| v.as_str()) == Some(*fmt))
            .and_then(|f| f.get("name")?.as_str().map(str::to_string))
    })
}

fn url_encode_path(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                (b as char).to_string()
            }
            _ => format!("%{:02X}", b),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Network adapters.
// ---------------------------------------------------------------------------

async fn get_json(client: &reqwest::Client, url: &str) -> Result<serde_json::Value> {
    let resp = client.get(url).send().await?.error_for_status()?;
    let json: serde_json::Value = resp.json().await?;
    // Some APIs (Internet Archive search) report failures as 200 + {"error": ...}.
    if let Some(err) = json.get("error") {
        bail!("{url}: API error: {err}");
    }
    Ok(json)
}

/// Asks a source for candidate items. A missing API key is an error the caller logs.
pub async fn fetch_candidates(
    client: &reqwest::Client,
    spec: &SourceSpec,
) -> Result<Vec<Candidate>> {
    match spec {
        SourceSpec::Freesound {
            query, max_items, ..
        } => {
            let key = std::env::var("FREESOUND_API_KEY").context(
                "FREESOUND_API_KEY is not set (get one at https://freesound.org/apiv2/apply)",
            )?;
            let url = reqwest::Url::parse_with_params(
                "https://freesound.org/apiv2/search/text/",
                &[
                    ("query", query.as_str()),
                    (
                        "filter",
                        "license:(\"Creative Commons 0\" OR \"Attribution\")",
                    ),
                    ("fields", "id,name,username,license,url,previews"),
                    ("page_size", &max_items.min(&150).to_string()),
                    ("token", &key),
                ],
            )?;
            Ok(parse_freesound(&get_json(client, url.as_str()).await?)
                .into_iter()
                .take(*max_items)
                .collect())
        }
        SourceSpec::WikimediaCommons {
            commons_category,
            max_items,
            ..
        } => {
            let title = format!("Category:{commons_category}");
            let url = reqwest::Url::parse_with_params(
                "https://commons.wikimedia.org/w/api.php",
                &[
                    ("action", "query"),
                    ("format", "json"),
                    ("generator", "categorymembers"),
                    ("gcmtitle", title.as_str()),
                    ("gcmtype", "file"),
                    ("gcmlimit", &max_items.min(&500).to_string()),
                    ("prop", "imageinfo"),
                    ("iiprop", "url|mime|extmetadata"),
                ],
            )?;
            Ok(parse_wikimedia(&get_json(client, url.as_str()).await?)
                .into_iter()
                .take(*max_items)
                .collect())
        }
        SourceSpec::InternetArchive {
            query, max_items, ..
        } => {
            let url = reqwest::Url::parse_with_params(
                "https://archive.org/advancedsearch.php",
                &[
                    ("q", query.as_str()),
                    ("fl[]", "identifier"),
                    ("fl[]", "title"),
                    ("fl[]", "creator"),
                    ("fl[]", "licenseurl"),
                    // Most items fail the license check, so ask for more than we keep.
                    ("rows", &(max_items * 5).to_string()),
                    ("output", "json"),
                ],
            )?;
            let mut out = Vec::new();
            for (id, title, creator, license_url) in
                parse_ia_search(&get_json(client, url.as_str()).await?)
            {
                // Check the license before fetching the item's file list.
                if !LicenseVerifier::verify(&license_url).0 {
                    continue;
                }
                let meta = get_json(client, &format!("https://archive.org/metadata/{id}")).await?;
                let Some(name) = pick_ia_audio_file(&meta) else {
                    continue;
                };
                let download_url = format!(
                    "https://archive.org/download/{id}/{}",
                    url_encode_path(&name)
                );
                out.push(Candidate {
                    item_url: format!("https://archive.org/details/{id}"),
                    ext: ext_of(&name, "flac"),
                    download_url,
                    creator,
                    title,
                    license: license_url.clone(),
                    license_url,
                });
                if out.len() >= *max_items {
                    break;
                }
            }
            Ok(out)
        }
    }
}

/// A filesystem-safe, stable name: `<source_id>/<slug>_<hash8>.<ext>`.
pub fn ledger_filename(source_id: &str, cand: &Candidate) -> String {
    use sha2::{Digest, Sha256};
    let slug: String = cand
        .title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|s| !s.is_empty())
        .take(8)
        .collect::<Vec<_>>()
        .join("_");
    let h = Sha256::digest(cand.item_url.as_bytes());
    let short: String = h.iter().take(4).map(|b| format!("{b:02x}")).collect();
    let slug = if slug.is_empty() {
        "item".to_string()
    } else {
        slug
    };
    format!("{source_id}/{slug}_{short}.{}", cand.ext)
}

/// Where the pipeline keeps its local, git-ignored data.
#[derive(Debug, Clone)]
pub struct DataPaths {
    /// `data/rain`: downloaded files, ledger, attributions, provenance.
    pub rain_dir: PathBuf,
    /// `data/sources.json`: local catalog for the TUI and data worker.
    pub catalog: PathBuf,
}

impl DataPaths {
    pub fn under(root: &Path) -> Self {
        Self {
            rain_dir: root.join("data").join("rain"),
            catalog: root.join("data").join("sources.json"),
        }
    }
    /// Paths for a `.../data/rain` directory (the catalog sits next to it).
    pub fn from_rain_dir(rain_dir: &Path) -> Self {
        Self {
            rain_dir: rain_dir.to_path_buf(),
            catalog: rain_dir
                .parent()
                .unwrap_or(Path::new("."))
                .join("sources.json"),
        }
    }
    pub fn ledger(&self) -> PathBuf {
        self.rain_dir.join("ledger.jsonl")
    }
    pub fn attributions(&self) -> PathBuf {
        self.rain_dir.join("ATTRIBUTIONS.txt")
    }
    pub fn provenance(&self) -> PathBuf {
        self.rain_dir.join("manifest_provenance.json")
    }
}

/// Regenerates the attribution file, provenance manifest and local catalog
/// from the ledger. Returns the number of ledger records.
pub fn regenerate_outputs(paths: &DataPaths) -> Result<usize> {
    let records = Ledger::load(&paths.ledger())?;
    std::fs::create_dir_all(&paths.rain_dir)?;
    crate::ingest::atomic_write(
        &paths.attributions(),
        render_attributions(&records).as_bytes(),
    )?;
    crate::ingest::atomic_write(
        &paths.provenance(),
        serde_json::to_string_pretty(&to_provenance(&records, &paths.rain_dir))?.as_bytes(),
    )?;
    crate::ingest::atomic_write(
        &paths.catalog,
        serde_json::to_string_pretty(&to_catalog(&records))?.as_bytes(),
    )?;
    Ok(records.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn registry_round_trips_and_rejects_duplicate_ids() {
        let dir = std::env::temp_dir().join(format!("rainai-reg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("r.json");
        std::fs::write(
            &p,
            r#"{"sources":[{"adapter":"freesound","id":"fs","category":"c","query":"rain"},
                          {"adapter":"wikimedia_commons","id":"wm","category":"c","commons_category":"Sounds of rain","max_items":5}]}"#,
        )
        .unwrap();
        let reg = SourceRegistry::load(&p).unwrap();
        assert_eq!(reg.sources.len(), 2);
        assert_eq!(reg.sources[0].id(), "fs");
        assert!(matches!(
            reg.sources[0],
            SourceSpec::Freesound { max_items: 25, .. }
        ));
        std::fs::write(
            &p,
            r#"{"sources":[{"adapter":"freesound","id":"x","category":"c","query":"a"},
                          {"adapter":"freesound","id":"x","category":"c","query":"b"}]}"#,
        )
        .unwrap();
        assert!(SourceRegistry::load(&p).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn freesound_parser_reads_creator_and_license_url() {
        let c = parse_freesound(&json!({"results":[
            {"id":1,"name":"Rain on tin","username":"alice","license":"https://creativecommons.org/licenses/by/4.0/",
             "url":"https://freesound.org/people/alice/sounds/1/","previews":{"preview-hq-mp3":"https://cdn.freesound.org/p/1-hq.mp3"}},
            {"id":2,"name":"No preview","username":"bob","license":"x","url":"u"}
        ]}));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].creator, "alice");
        assert_eq!(c[0].ext, "mp3");
        assert!(LicenseVerifier::verify(&c[0].license).0);
    }

    #[test]
    fn wikimedia_parser_keeps_audio_and_strips_html() {
        let c = parse_wikimedia(&json!({"query":{"pages":{
            "1":{"title":"File:Rain.ogg","imageinfo":[{"url":"https://upload.wikimedia.org/a/Rain.ogg","descriptionurl":"https://commons.wikimedia.org/wiki/File:Rain.ogg","mime":"audio/ogg",
                "extmetadata":{"LicenseShortName":{"value":"CC BY-SA 4.0"},"LicenseUrl":{"value":"https://creativecommons.org/licenses/by-sa/4.0"},
                               "Artist":{"value":"<a href=\"//x\">Some  User</a>"}}}]},
            "2":{"title":"File:Pic.jpg","imageinfo":[{"url":"u","descriptionurl":"d","mime":"image/jpeg","extmetadata":{"LicenseShortName":{"value":"CC0"}}}]}
        }}}));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].creator, "Some User");
        assert_eq!(c[0].title, "Rain.ogg");
        // ShareAlike is parsed but rejected by the verifier.
        assert!(!LicenseVerifier::verify(&c[0].license).0);
    }

    #[test]
    fn internet_archive_parsers() {
        let docs = parse_ia_search(&json!({"response":{"docs":[
            {"identifier":"a","title":"Rain","creator":["Ann"],"licenseurl":"https://creativecommons.org/publicdomain/mark/1.0/"},
            {"identifier":"b","title":"No license"}
        ]}}));
        assert_eq!(
            docs,
            vec![(
                "a".into(),
                "Rain".into(),
                "Ann".into(),
                "https://creativecommons.org/publicdomain/mark/1.0/".into()
            )]
        );
        let pick = pick_ia_audio_file(&json!({"files":[
            {"name":"x.mp3","format":"VBR MP3"},{"name":"x.flac","format":"Flac"},{"name":"x.png","format":"PNG"}]}));
        assert_eq!(pick.as_deref(), Some("x.flac"));
        assert_eq!(url_encode_path("a b/ü.flac"), "a%20b/%C3%BC.flac");
    }

    #[test]
    fn ledger_drives_attributions_and_catalog() {
        let dir = std::env::temp_dir().join(format!("rainai-ledger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = DataPaths::under(&dir);
        let rec = LedgerRecord {
            file: "fs/rain_on_tin_0a1b2c3d.mp3".into(),
            sha256: "00".into(),
            source_id: "fs".into(),
            platform: "Freesound".into(),
            category: "surface_metal".into(),
            item_url: "https://freesound.org/people/alice/sounds/1/".into(),
            download_url: "https://cdn/p.mp3".into(),
            creator: "alice".into(),
            title: "Rain on tin".into(),
            license: "https://creativecommons.org/licenses/by/4.0/".into(),
            license_url: "https://creativecommons.org/licenses/by/4.0/".into(),
            license_tier: LicenseTier::AttributionOnly,
            retrieved_utc: "2026-11-13T00:00:00Z".into(),
            changes: STANDARD_CHANGES.into(),
        };
        Ledger::append(&paths.ledger(), &rec).unwrap();
        assert_eq!(regenerate_outputs(&paths).unwrap(), 1);
        let attr = std::fs::read_to_string(paths.attributions()).unwrap();
        assert!(attr.contains("\"Rain on tin\" by alice (Freesound)"));
        assert!(attr.contains("Changes: decoded"));
        let cat: Vec<DownloadItem> =
            serde_json::from_str(&std::fs::read_to_string(&paths.catalog).unwrap()).unwrap();
        assert_eq!(cat[0].filename, rec.file);
        assert_eq!(Ledger::load(&paths.ledger()).unwrap(), vec![rec]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ledger_filenames_are_stable_and_safe() {
        let c = Candidate {
            item_url: "https://x/1".into(),
            download_url: "https://x/1.mp3".into(),
            creator: "a".into(),
            title: "Heavy Rain (on a Tin Roof!)".into(),
            license: "CC0".into(),
            license_url: String::new(),
            ext: "mp3".into(),
        };
        let a = ledger_filename("fs", &c);
        assert!(a.starts_with("fs/heavy_rain_on_a_tin_roof_"));
        assert!(a.ends_with(".mp3"));
        assert_eq!(a, ledger_filename("fs", &c));
    }
}
