/**
 * End-to-End Test for RainAI Cloudflare Edge Staging Worker (`crates/web/worker.js`)
 * 
 * Verifies:
 * 1. Health check diagnostics
 * 2. Unapproved/unknown license routing to staging quarantine
 * 3. Binary audio upload to staging bucket
 * 4. Maintainer triage quarantine listing
 * 5. Re-contribution deduplication, metadata reconciliation, and auto-promotion to approved
 * 6. Automatic audio payload migration from quarantine to approved
 */

import worker from "../worker.js";

// In-memory mock for Cloudflare R2Bucket
class MockR2Bucket {
  constructor() {
    this.store = new Map();
  }

  async put(key, body, options = {}) {
    let content = body;
    if (typeof body === "string") {
      content = body;
    } else if (body instanceof ArrayBuffer || body instanceof Uint8Array) {
      content = Buffer.from(body);
    }
    this.store.set(key, {
      body: content,
      httpMetadata: options.httpMetadata || {},
      customMetadata: options.customMetadata || {},
      async json() {
        return JSON.parse(content.toString());
      },
      async text() {
        return content.toString();
      },
    });
    return { key };
  }

  async get(key) {
    return this.store.get(key) || null;
  }

  async delete(key) {
    this.store.delete(key);
  }

  async list(options = {}) {
    const prefix = options.prefix || "";
    const objects = [];
    for (const [key, val] of this.store.entries()) {
      if (key.startsWith(prefix)) {
        objects.push({ key, customMetadata: val.customMetadata });
      }
    }
    return { objects };
  }
}

// In-memory mock for Cloudflare D1Database
class MockD1Database {
  constructor() {
    this.rows = new Map();
  }

  prepare(query) {
    const db = this;
    return {
      _params: [],
      bind(...params) {
        this._params = params;
        return this;
      },
      async first() {
        if (query.includes("SELECT * FROM records WHERE sha256 = ?")) {
          const sha = this._params[0];
          return db.rows.get(sha) || null;
        }
        return null;
      },
      async all() {
        if (query.includes("WHERE status = 'QUARANTINE'")) {
          const results = Array.from(db.rows.values()).filter((r) => r.status === "QUARANTINE");
          return { results };
        }
        if (query.includes("WHERE status = 'APPROVED'")) {
          const results = Array.from(db.rows.values()).filter((r) => r.status === "APPROVED");
          return { results };
        }
        return { results: Array.from(db.rows.values()) };
      },
      async run() {
        if (query.includes("INSERT INTO records")) {
          const [
            sha256, filename, status, license, license_tier, license_rank,
            license_approved, dsp_passed, author, tags_json, descriptions_json,
            contributors_json, alternate_licenses_json, file_size_bytes, quarantine_reason
          ] = this._params;
          db.rows.set(sha256, {
            sha256,
            filename,
            status,
            license,
            license_tier,
            license_rank,
            license_approved,
            dsp_passed,
            author,
            tags_json,
            descriptions_json,
            contributors_json,
            alternate_licenses_json,
            file_size_bytes,
            quarantine_reason,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
          return { success: true };
        }
        if (query.includes("UPDATE records SET file_size_bytes = ?")) {
          const [size, sha] = this._params;
          const rec = db.rows.get(sha);
          if (rec) {
            rec.file_size_bytes = size;
            rec.updated_at = new Date().toISOString();
          }
          return { success: true };
        }
        if (query.includes("DELETE FROM records WHERE sha256 = ?")) {
          const sha = this._params[0];
          db.rows.delete(sha);
          return { success: true };
        }
        return { success: true };
      },
    };
  }
}

// In-memory mock for Cloudflare Pages / Workers env.ASSETS
class MockAssets {
  constructor() {
    this.files = new Map([
      ["/index.html", "<!DOCTYPE html><html><body><canvas id='egui_canvas'></canvas></body></html>"],
      ["/attributions.bin", Buffer.from("RATT\x01\x00\x00\x00\x00\x00")],
      ["/pkg/web_bg.wasm", Buffer.from("\x00asm\x01\x00\x00\x00")],
    ]);
  }

  async fetch(request) {
    const url = new URL(request.url);
    const path = url.pathname === "/" ? "/index.html" : url.pathname;
    if (this.files.has(path)) {
      const content = this.files.get(path);
      const isHtml = path.endsWith(".html");
      const isWasm = path.endsWith(".wasm");
      return new Response(content, {
        status: 200,
        headers: {
          "Content-Type": isHtml
            ? "text/html; charset=utf-8"
            : isWasm
            ? "application/wasm"
            : "application/octet-stream",
        },
      });
    }
    return new Response("Not Found", { status: 404 });
  }
}

async function runE2ETests() {
  console.log("=== RainAI Edge Worker E2E Test Suite ===");
  const bucket = new MockR2Bucket();
  const assets = new MockAssets();
  const env = { DATA_BUCKET: bucket, STAGING_BUCKET: bucket, ASSETS: assets };
  let passed = 0;
  let failed = 0;

  function assert(condition, message) {
    if (!condition) {
      console.error(`  FAIL: ${message}`);
      failed++;
      throw new Error(`Assertion failed: ${message}`);
    } else {
      console.log(`  PASS: ${message}`);
      passed++;
    }
  }

  // 1. Health Check
  {
    const req = new Request("https://staging.rainai.app/api/contribute/health", { method: "GET" });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Health check returns 200 OK");
    const json = await res.json();
    assert(json.status === "healthy", "Health status is 'healthy'");
    assert(json.r2_bound === true, "R2 bucket is confirmed bound");
  }

  // 2. Submit initial contribution with Unknown license -> Quarantine
  const testSha = "3a7b9c1d0e4f2a5b6c8d7e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b";
  {
    const payload = {
      sha256: testSha,
      filename: "remote_tin_roof.flac",
      cf_turnstile_response: '1x00000000000000000000AA',
      license: "Unknown", // Ineligible / unverified license
      license_approved: false,
      dsp_passed: true,
      tags: ["tin_roof", "shed"],
      descriptions: ["Gentle rain pitter-patter on corrugated metal roof"],
      author: "Alice",
      contributors: ["Alice"],
    };
    const req = new Request("https://staging.rainai.app/api/contribute/submit-record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Submit record returns 200");
    const json = await res.json();
    assert(json.status === "QUARANTINE", "Initial submission with Unknown license routes to QUARANTINE");
    assert(json.object_key === `ingest/quarantine/${testSha}.json`, "Stored under ingest/quarantine prefix");
  }

  // 3. Upload binary audio payload into quarantine
  {
    const dummyAudio = Buffer.from("RIFF....WAVEfmt ....data....test_audio_flac_content");
    const req = new Request(`https://staging.rainai.app/api/contribute/upload-blob/${testSha}`, {
      method: "PUT",
      headers: {
        "X-Target-Status": "quarantine",
        "X-File-Extension": "flac",
        "Content-Type": "audio/flac",
      },
      body: dummyAudio,
    });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Upload audio blob to quarantine returns 200");
    const json = await res.json();
    assert(json.object_key === `ingest/quarantine/${testSha}.flac`, "Audio blob saved at ingest/quarantine/...");
  }

  // 4. Quarantine triage listing
  {
    const req = new Request("https://staging.rainai.app/api/contribute/quarantine-list", { method: "GET" });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Quarantine list returns 200");
    const json = await res.json();
    assert(json.total === 1, "Quarantine contains exactly 1 item");
    assert(json.records[0].sha256 === testSha, "Quarantined record matches expected SHA-256");
    assert(json.records[0].quarantine_reason === "UNAPPROVED_OR_MISSING_LICENSE", "Quarantine reason accurately tagged");
  }

  // 5. Re-contribution with Approved License -> Content-Addressed Reconciliation & Auto-Promotion
  {
    const payload = {
      sha256: testSha,
      filename: "remote_tin_roof.flac",
      cf_turnstile_response: '1x00000000000000000000AA',
      license: "RainAI-FC-Proprietary-License", // Project Proprietary License (Rank 6)
      license_approved: true,
      dsp_passed: true,
      tags: ["tin_roof", "monsoon_storm"],
      descriptions: [
        "Resonant metallic high-frequency droplet splatter texture",
        "Gentle rain pitter-patter on corrugated metal roof", // Duplicate line should deduplicate
      ],
      author: "Bob",
      contributors: ["Bob"],
    };
    const req = new Request("https://staging.rainai.app/api/contribute/submit-record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Re-submit record returns 200");
    const json = await res.json();
    assert(json.status === "APPROVED", "Re-contributed record automatically promoted to APPROVED");
    assert(json.reconciled === true, "Deduplication reconciliation flag is true");
    assert(json.promoted === true, "Auto-promotion flag is true");
    assert(json.predominant_license === "RainAI-FC-Proprietary-License", "Predominant license resolved correctly");
    assert(json.object_key === `ingest/approved/${testSha}.json`, "Stored under ingest/approved prefix");
  }

  // 6. Verify Quarantine Was Cleaned & Audio Blob Was Automatically Moved
  {
    const oldQuarantineJson = await bucket.get(`ingest/quarantine/${testSha}.json`);
    assert(oldQuarantineJson === null, "Old quarantine JSON sidecar purged");

    const oldQuarantineBlob = await bucket.get(`ingest/quarantine/${testSha}.flac`);
    assert(oldQuarantineBlob === null, "Old quarantine audio blob purged");

    const approvedAudio = await bucket.get(`ingest/approved/${testSha}.flac`);
    assert(approvedAudio !== null, "Audio blob automatically migrated to ingest/approved/ prefix");

    const approvedJsonObj = await bucket.get(`ingest/approved/${testSha}.json`);
    assert(approvedJsonObj !== null, "Approved JSON metadata sidecar exists");
    const approvedData = await approvedJsonObj.json();

    // Verify merged metadata fields
    assert(approvedData.tags.includes("tin_roof") && approvedData.tags.includes("monsoon_storm") && approvedData.tags.includes("shed"), "Tags merged uniquely");
    assert(approvedData.descriptions.length === 2, "Parallel descriptions preserved without duplicates (length 2)");
    assert(approvedData.contributors.includes("Alice") && approvedData.contributors.includes("Bob"), "Contributors merged (Alice & Bob)");
    assert(approvedData.alternate_licenses.includes("Unknown"), "Alternate licenses array includes previously granted Unknown");
  }

  // 7. Approved list verification
  {
    const req = new Request("https://staging.rainai.app/api/contribute/approved-list", { method: "GET" });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Approved list returns 200");
    const json = await res.json();
    assert(json.total === 1, "Approved queue contains 1 promoted item");
    assert(json.records[0].sha256 === testSha, "Approved record SHA matches");
  }

  // 8. Ephemeral Acknowledgment & Purge in R2
  {
    const req = new Request("https://staging.rainai.app/api/contribute/ack-ingested", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sha256_list: [testSha] }),
    });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Ack-ingested purge returns 200");
    const json = await res.json();
    assert(json.purged === 1, "Purged 1 record");
    const purgedJson = await bucket.get(`ingest/approved/${testSha}.json`);
    assert(purgedJson === null, "Approved JSON sidecar purged after ack");
    const purgedBlob = await bucket.get(`ingest/approved/${testSha}.flac`);
    assert(purgedBlob === null, "Approved audio blob purged after ack");
  }

  console.log("\n--- Testing D1 SQLite Transaction Coordinator Engine ---");
  const d1Db = new MockD1Database();
  const d1Bucket = new MockR2Bucket();
  const d1Env = { DB: d1Db, DATA_BUCKET: d1Bucket };

  // 9. D1 Health Check with DB bound
  {
    const req = new Request("https://staging.rainai.app/api/contribute/health", { method: "GET" });
    const res = await worker.fetch(req, d1Env, {});
    assert(res.status === 200, "D1 Health check returns 200 OK");
    const json = await res.json();
    assert(json.d1_bound === true, "D1 is confirmed bound");
    assert(json.r2_bound === true, "R2 is confirmed bound");
  }

  // 10. D1 Submit record to Quarantine
  const d1Sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  {
    const payload = {
      sha256: d1Sha,
      filename: "quarantine_sample.wav",
      cf_turnstile_response: '1x00000000000000000000AA',
      license: "Unknown",
      license_approved: false,
      dsp_passed: true,
      tags: ["raw_rain"],
      descriptions: ["Field recording in woods"],
      author: "Charlie",
    };
    const req = new Request("https://staging.rainai.app/api/contribute/submit-record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await worker.fetch(req, d1Env, {});
    assert(res.status === 200, "D1 Submit record returns 200");
    const json = await res.json();
    assert(json.status === "QUARANTINE", "D1 routes Unknown to QUARANTINE");

    // Verify row in D1
    const d1Row = await d1Db.prepare("SELECT * FROM records WHERE sha256 = ?").bind(d1Sha).first();
    assert(d1Row !== null, "Record persisted in D1 database");
    assert(d1Row.status === "QUARANTINE", "D1 record status is QUARANTINE");
  }

  // 11. D1 Quarantine listing
  {
    const req = new Request("https://staging.rainai.app/api/contribute/quarantine-list", { method: "GET" });
    const res = await worker.fetch(req, d1Env, {});
    assert(res.status === 200, "D1 Quarantine list returns 200");
    const json = await res.json();
    assert(json.total === 1, "D1 has 1 quarantined record");
    assert(json.records[0].sha256 === d1Sha, "Quarantined record matches d1Sha");
    assert(json.records[0].tags.includes("raw_rain"), "Tags parsed accurately from JSON");
  }

  // 12. D1 Reconcile & Auto-promote via UPSERT
  {
    const payload = {
      sha256: d1Sha,
      filename: "quarantine_sample.wav",
      cf_turnstile_response: '1x00000000000000000000AA',
      license: "CC-BY-4.0",
      license_approved: true,
      dsp_passed: true,
      tags: ["raw_rain", "forest_canopy"],
      descriptions: ["Gentle rain under dense forest canopy"],
      author: "Charlie & Dave",
    };
    const req = new Request("https://staging.rainai.app/api/contribute/submit-record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await worker.fetch(req, d1Env, {});
    assert(res.status === 200, "D1 Reconcile returns 200");
    const json = await res.json();
    assert(json.status === "APPROVED", "D1 record promoted to APPROVED");
    assert(json.reconciled === true, "D1 record reconciled");
    assert(json.promoted === true, "D1 record marked promoted");

    // Check row in D1
    const d1Row = await d1Db.prepare("SELECT * FROM records WHERE sha256 = ?").bind(d1Sha).first();
    assert(d1Row.status === "APPROVED", "D1 row status updated to APPROVED");
    assert(d1Row.license === "CC-BY-4.0", "D1 row license upgraded");
  }

  // 13. D1 Approved list
  {
    const req = new Request("https://staging.rainai.app/api/contribute/approved-list", { method: "GET" });
    const res = await worker.fetch(req, d1Env, {});
    assert(res.status === 200, "D1 Approved list returns 200");
    const json = await res.json();
    assert(json.total === 1, "D1 has 1 approved record");
    assert(json.records[0].sha256 === d1Sha, "Approved record SHA matches");
  }

  // 14. D1 Ephemeral Sync Purge
  {
    const req = new Request("https://staging.rainai.app/api/contribute/ack-ingested", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sha256_list: [d1Sha] }),
    });
    const res = await worker.fetch(req, d1Env, {});
    assert(res.status === 200, "D1 Ack-ingested returns 200");
    const json = await res.json();
    assert(json.purged === 1, "Purged 1 record from D1");

    const d1Row = await d1Db.prepare("SELECT * FROM records WHERE sha256 = ?").bind(d1Sha).first();
    assert(d1Row === null, "Record successfully purged from D1 middle database");
  }

  console.log("\n--- Testing Cloudflare Pages / Workers Static Asset & Security Engine ---");

  // 15. Root HTML Entry Point & COOP/COEP Headers for SharedArrayBuffer
  {
    const req = new Request("https://rainai.app/", { method: "GET" });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Root request returns 200 OK");
    const text = await res.text();
    assert(text.includes("egui_canvas"), "Serves HTML entrypoint containing egui_canvas");
    assert(res.headers.get("Cross-Origin-Opener-Policy") === "same-origin", "COOP header set to 'same-origin'");
    assert(res.headers.get("Cross-Origin-Embedder-Policy") === "require-corp", "COEP header set to 'require-corp'");
  }

  // 16. Zero-Copy Binary Attribution Dictionary with Immutable Caching
  {
    const req = new Request("https://rainai.app/attributions.bin", { method: "GET" });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "Attributions binary returns 200 OK");
    assert(res.headers.get("Cache-Control").includes("immutable"), "Cache-Control is immutable");
    assert(res.headers.get("Cross-Origin-Resource-Policy") === "cross-origin", "CORP header set to 'cross-origin'");
    const buf = await res.arrayBuffer();
    assert(buf.byteLength > 0, "Binary attribution dictionary body returned");
  }

  // 17. WebAssembly Binary Delivery
  {
    const req = new Request("https://rainai.app/pkg/web_bg.wasm", { method: "GET" });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "WASM binary returns 200 OK");
    assert(res.headers.get("Content-Type") === "application/wasm", "Content-Type is application/wasm");
    assert(res.headers.get("Cache-Control").includes("immutable"), "WASM caching is immutable");
  }

  // 18. Single-Page Application (SPA) Client Routing Fallback
  {
    const req = new Request("https://rainai.app/presets/attic_roof", {
      method: "GET",
      headers: { "Sec-Fetch-Mode": "navigate" },
    });
    const res = await worker.fetch(req, env, {});
    assert(res.status === 200, "SPA deep route navigation returns 200 via index.html fallback");
    const text = await res.text();
    assert(text.includes("egui_canvas"), "SPA fallback serves root HTML application");
    assert(res.headers.get("Cross-Origin-Opener-Policy") === "same-origin", "SPA fallback preserves COOP header");
  }

  console.log("\n--- Testing Edge Rate Limiting & Zero-Cost Quota Guards ---");

  // 19. Payload Size Protection: 413 for oversized metadata and audio blobs
  {
    const bigMetaReq = new Request("https://staging.rainai.app/api/contribute/submit-record", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": (600 * 1024).toString(), // 600 KB > 512 KB
      },
      body: JSON.stringify({}),
    });
    const bigMetaRes = await worker.fetch(bigMetaReq, env, {});
    assert(bigMetaRes.status === 413, "Metadata record > 512KB rejected with 413 Payload Too Large");

    const bigBlobReq = new Request(`https://staging.rainai.app/api/contribute/upload-blob/oversized_sha`, {
      method: "PUT",
      headers: {
        "Content-Type": "audio/flac",
        "Content-Length": (55 * 1024 * 1024).toString(), // 55 MB > 50 MB
      },
      body: Buffer.from("big"),
    });
    const bigBlobRes = await worker.fetch(bigBlobReq, env, {});
    assert(bigBlobRes.status === 413, "Audio blob > 50MB rejected with 413 Payload Too Large");
  }

  // 20. Rate Limiting: 429 Too Many Requests when write threshold exceeded (15/min)
  {
    const spamIp = "198.51.100.42";
    let reached429 = false;
    for (let i = 0; i < 20; i++) {
      const dummyReq = new Request(`https://staging.rainai.app/api/contribute/ack-ingested`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "CF-Connecting-IP": spamIp,
        },
        body: JSON.stringify({ sha256_list: [] }),
      });
      const res = await worker.fetch(dummyReq, env, {});
      if (res.status === 429) {
        reached429 = true;
        assert(res.headers.get("Retry-After") === "60", "Rate limit response includes Retry-After header");
        const json = await res.json();
        assert(json.category === "write", "Rate limit category identified as write");
        break;
      }
    }
    assert(reached429, "Client exceeding 15 writes/min blocked with 429 Too Many Requests");
  }

  // 21. Audio Blob Download Endpoint for Local Maintenance & Git Sync
  {
    const sampleSha = "downloadable_sample_sha_123456789";
    const sampleAudio = Buffer.from("RIFF....WAVEfmt ....test_downloadable_audio_stream");
    await bucket.put(`ingest/approved/${sampleSha}.flac`, sampleAudio, {
      httpMetadata: { contentType: "audio/flac" },
    });

    const downloadReq = new Request(`https://staging.rainai.app/api/contribute/blob/${sampleSha}`, {
      method: "GET",
      headers: { "CF-Connecting-IP": "192.0.2.1" },
    });
    const downloadRes = await worker.fetch(downloadReq, env, {});
    assert(downloadRes.status === 200, "Audio blob retrieval returns 200 OK");
    assert(downloadRes.headers.get("X-SHA256") === sampleSha, "Audio blob X-SHA256 header matches");
    const downloadedBuf = Buffer.from(await downloadRes.arrayBuffer());
    assert(downloadedBuf.equals(sampleAudio), "Downloaded binary audio matches uploaded bytes exactly");
  }

  // 22. List Capping and Pagination Limits
  {
    const listReq = new Request("https://staging.rainai.app/api/contribute/approved-list?limit=25&offset=0", {
      method: "GET",
      headers: { "CF-Connecting-IP": "192.0.2.2" },
    });
    const listRes = await worker.fetch(listReq, env, {});
    assert(listRes.status === 200, "Approved list with pagination returns 200 OK");
    const json = await listRes.json();
    assert(json.limit === 25, "List pagination respects requested limit of 25");
    assert(json.offset === 0, "List pagination respects offset of 0");
  }

  console.log(`\nResults: ${passed} passed, ${failed} failed.\n`);
  if (failed > 0) {
    process.exit(1);
  }
}

runE2ETests().catch((err) => {
  console.error("Unhandled test exception:", err);
  process.exit(1);
});
