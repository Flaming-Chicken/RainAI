# Metadata & Contribution Audit (13 Nov 2026)

An audit of how RainAI records where audio came from, who made it and how it's
licensed, across every path audio enters the corpus. It was done for PLAN.md D2
(only verifiably commercially safe sources) and the decision that public
contributions are allowed if they carry enough information to be lawful and
useful; anything else is triaged.

Paths audited: the downloader (`rainai_ingest`), the local drop folder
(`data/input/`), `rainai_contribute import-dir` and `import-manifest`, the web
upload worker (`crates/web/_worker.js`) and its local pull
(`data_worker::sync_edge_staging_buffer`), training-time attribution
(`candle/dataset.rs`), the attribution export (`export-attributions`) and the
in-app Contribute dialog.

## What a contribution must have

Checked by `contribute::check_contribution` (Rust) and `checkContribution`
(worker); a contribution missing any of these goes to triage with the reasons:

| Field | Why |
|---|---|
| Allowed license: CC0 / public domain, CC BY 3.0/4.0, or the RainAI contributor grant | Lawful use (D2). ShareAlike, NC, ND, platform and software licenses are rejected. |
| Named contributor (not "Anonymous", "Unknown", "RainAI Community") | Credit, and someone accountable for the warranty |
| `rights_warranty: true` | The contributor confirmed they own the recording |
| An uploaded recording, not a link | A link (YouTube, Drive, someone's Freesound page) can't be shown to be the contributor's own |
| A recognised surface (`asphalt`, `pavement`, `tin_roof`, `canvas_tent`, `foliage`, `wood_deck`, `glass`, `puddle_shallow`, `water_deep`, or a tag that maps to one) | Useful for training; unknown tags used to become `pavement` silently |
| A title, for CC BY | CC BY credit needs creator, title, source and license |
| Audio that passes the quality screening | Useful for training (existing DSP checks) |
| No conflicting record for the same audio | See finding 3 |

Accepted contributions get a record in `data/rain/ledger.jsonl`, like
downloads, and `ATTRIBUTIONS.txt` / `manifest_provenance.json` /
`data/sources.json` are regenerated from it.

## Findings

Severity: **High** = could put unlawful or uncredited audio into training, or
lets an outsider change or destroy data. **Medium** = wrong or lost metadata.
**Low** = cosmetic or latent.

### Fixed in this change

1. **High · The web worker had no maintainer authentication.** `approved-list`,
   `quarantine-list` and `blob/<sha>` were public (anyone could list every
   submission and download quarantined audio, which may be someone else's
   copyrighted work), and `ack-ingested` let anyone delete records and blobs.
   → All four need `Authorization: Bearer <MAINTAINER_TOKEN>` and fail closed
   if the secret is unset. The local pull sends `RAINAI_MAINTAINER_TOKEN`.
2. **High · Uploads could skip review.** `upload-blob` had no bot check, took
   its destination folder from a client header (`X-Target-Status: approved`),
   didn't validate the SHA-256 or extension in the object key, and never
   checked the bytes matched the hash. → A blob can only be uploaded for a
   record that already went through `submit-record`; the folder comes from that
   record; the SHA-256 and extension are validated; the bytes are hashed and
   must match.
3. **High · Duplicate audio took the "best" license.** When identical audio was
   submitted twice, the worker and `reconcile_with` kept the most permissive
   license, ranking the contributor grant highest. So anyone could re-submit a
   CC BY or quarantined NC recording claiming the RainAI grant or CC0 and have
   it approved, dropping the real creator's terms. Two tests asserted exactly
   this. → Same audio with a different license or contributor is now a
   conflict and goes to triage; the first recorded license stands.
4. **High · Turnstile failed open.** Without `TURNSTILE_SECRET_KEY` the worker
   used Cloudflare's always-pass test secret. → It now refuses submissions
   (503) until the secret is set.
5. **High · Contributions were approved on license alone.** No contributor,
   warranty or surface was required anywhere: the worker approved on license
   plus a client-reported `dsp_passed`; `import-dir` with no `--license`
   imported files anyway (tier Restricted, license empty); `import-manifest`
   copied files with no provenance at all and defaulted the author to
   "Anonymous"; manifest validation accepted link-only entries with made-up
   quality metrics (RMS 0.05, entropy 0.8). → One completeness check on every
   path (table above). The worker's `URL_ONLY` status is gone: links are
   triaged.
6. **High · Contributions would have lost their credit.** Ingest now
   regenerates `ATTRIBUTIONS.txt`, `manifest_provenance.json` and
   `data/sources.json` from the ledger, but contribution paths wrote those
   files directly and never the ledger, so the next ingest would erase every
   contribution's credit. The local pull from the worker ignored the record's
   license and contributor altogether. → Every path writes a ledger record.
7. **Medium · Two surface taxonomies.** `shared::surface::CanonicalSurface` has
   `asphalt` but not `pine_needles`; `autopilot::CANONICAL_SURFACES` (written
   by hand) had `pine_needles` but not `asphalt`, so asphalt quotas were never
   tracked. → `CANONICAL_SURFACES` is derived from the enum.
8. **Medium · Unknown surfaces became pavement.** `CanonicalSurface::from_tag`
   returns `Pavement` for any unrecognised tag ("ambient", typos), silently
   mislabelling training data. → New `try_from_tag` returns `None`;
   contributions must use a recognised surface. `from_tag` keeps its fallback
   for existing callers.
9. **Medium · The attribution export invented credits.** Records with no
   contributor were exported as "RainAI Community", with surface "ambient",
   and records with Unknown or Restricted licenses were exported too. → Only
   records with an allowed license and (where attribution is required) a
   named contributor are exported.
10. **Medium · The worker stored whatever the client sent.** The R2 sidecar
    spread the request body (`...body`) into the stored record. → Only known
    fields are stored. Internal error messages are no longer returned to
    clients.
11. **Medium · The drop folder took no warranty.** → Sidecars carry
    `rights_warranty: true` (JSON, TOML, YAML or `key: value` text), and
    `import-dir` needs `--rights-warranty`.

### Not fixed here (recorded in PLAN.md)

12. **High · The in-app Contribute dialog does nothing.** It collects a file
    or URL, license, author and the warranty, then queues a
    `TaskKind::DataContribution` task that nothing handles. Either wire it to
    the worker (`submit-record` then `upload-blob`, sending `title`,
    `surface` and `rights_warranty`) or remove it.
13. **Medium · Seven overlapping provenance structs.** `DownloadItem`, the
    TUI's own copy `SourceItem`, `ProvenanceRecord`, `RainSourceEntry`,
    `ExtractedAudioMetadata`, `LedgerRecord` and `AttributionRecordInput`
    disagree on names and types (`author` / `contributors` / `creator`;
    `category` / `tags` / `surface_tag`; license as `Option<String>` or
    `String`), and only `LedgerRecord` has the title, license URL, date and
    changes CC BY needs. The ledger is now the source of truth; the others
    should become views of it.
14. **Medium · `dsp_passed` is client-reported.** The worker still uses it to
    triage, but it can't be trusted; the local pull re-runs the screening
    before training, which is the real check.
15. **Low · Attribution tiers in the app.** `rain_view` colours tier 3
    (ShareAlike) as CC BY. ShareAlike is no longer exported, so it's latent.
16. **Low · `contributor_contact` in manifests** is personal data. It's only
    stored locally, but shouldn't be copied into anything published.
17. **Low · D1 schema.** New columns are added with `ALTER TABLE` on start-up;
    rows staged before this change have no `surface`, `title` or warranty and
    will be triaged on their next update.

## Deployment notes (PLAN.md step 7)

The worker has never been deployed (no Cloudflare credentials), so none of the
issues above were exploitable. Before deploying, set the secrets
`TURNSTILE_SECRET_KEY` and `MAINTAINER_TOKEN`, and set
`RAINAI_MAINTAINER_TOKEN` to the same token wherever the local pull runs.
