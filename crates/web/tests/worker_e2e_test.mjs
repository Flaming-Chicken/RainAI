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

async function runE2ETests() {
  console.log("=== RainAI Edge Worker E2E Test Suite ===");
  const bucket = new MockR2Bucket();
  const env = { DATA_BUCKET: bucket, STAGING_BUCKET: bucket };
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

  console.log(`\nResults: ${passed} passed, ${failed} failed.\n`);
  if (failed > 0) {
    process.exit(1);
  }
}

runE2ETests().catch((err) => {
  console.error("Unhandled test exception:", err);
  process.exit(1);
});
