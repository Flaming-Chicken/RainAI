/**
 * RainAI Cloudflare Edge Worker
 * 
 * Provides unified, serverless, zero-SQL staging and triage for community rain audio contributions.
 * Directly leverages Cloudflare R2 object storage for both audio payloads and metadata JSON sidecars.
 */

function getLicenseRank(lic) {
  if (!lic) return { rank: 2, tier: "Unknown", approved: false };
  const clean = lic.trim().toLowerCase();
  if (
    clean === "" ||
    clean === "unknown" ||
    clean === "unspecified" ||
    clean.includes("pending") ||
    clean === "none" ||
    clean === "to be scraped"
  ) {
    return { rank: 2, tier: "Unknown", approved: false };
  }
  if (
    clean.includes("rainai-fc") ||
    clean.includes("rainai-proprietary") ||
    clean.includes("spodeian-permission") ||
    clean.includes("flaming chicken") ||
    clean.includes("spodeian")
  ) {
    return { rank: 6, tier: "ProjectProprietary", approved: true };
  }
  if (
    clean.includes("nc") ||
    clean.includes("noncommercial") ||
    clean.includes("nd") ||
    clean.includes("noderivatives")
  ) {
    return { rank: 1, tier: "Restricted", approved: false };
  }
  if (clean.includes("cc-by-sa")) {
    return { rank: 3, tier: "ShareAlike", approved: true };
  }
  if (
    clean.includes("cc-by") ||
    clean.includes("attribution") ||
    clean.includes("mit") ||
    clean.includes("apache") ||
    clean.includes("bsd") ||
    clean.includes("isc") ||
    clean.includes("odc-by") ||
    clean.includes("mixkit")
  ) {
    return { rank: 4, tier: "AttributionOnly", approved: true };
  }
  if (
    clean.includes("cc0") ||
    clean.includes("public domain") ||
    clean.includes("unlicense") ||
    clean.includes("wtfpl") ||
    clean.includes("pddl") ||
    clean.includes("open access") ||
    clean.includes("nps natural sound")
  ) {
    return { rank: 5, tier: "PublicDomain", approved: true };
  }
  return { rank: 2, tier: "Unknown", approved: false };
}

let dbInitialized = false;
async function ensureDbSchema(db) {
  if (!db || dbInitialized) return;
  try {
    await db.exec(`
      CREATE TABLE IF NOT EXISTS records (
        sha256 TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('QUARANTINE', 'APPROVED', 'URL_ONLY')),
        license TEXT NOT NULL,
        license_tier TEXT NOT NULL,
        license_rank INTEGER NOT NULL DEFAULT 2,
        license_approved INTEGER NOT NULL DEFAULT 0,
        dsp_passed INTEGER NOT NULL DEFAULT 0,
        author TEXT,
        tags_json TEXT NOT NULL DEFAULT '[]',
        descriptions_json TEXT NOT NULL DEFAULT '[]',
        contributors_json TEXT NOT NULL DEFAULT '[]',
        alternate_licenses_json TEXT NOT NULL DEFAULT '[]',
        file_size_bytes INTEGER NOT NULL DEFAULT 0,
        quarantine_reason TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
      CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
    `);
    dbInitialized = true;
  } catch (err) {
    console.error("D1 schema initialization error:", err);
  }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // CORS headers for browser WASM client
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, X-SHA256, X-License, X-Tags, Authorization",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // 0. Static Asset Passthrough (Cloudflare Pages / Workers Assets)
    if (!url.pathname.startsWith("/api/")) {
      if (env.ASSETS) {
        let assetResponse = await env.ASSETS.fetch(request);

        // SPA navigation fallback: if 404 on an HTML navigation request, serve index.html
        if (assetResponse.status === 404 && (request.mode === "navigate" || !url.pathname.includes("."))) {
          const indexReq = new Request(new URL("/index.html", request.url), request);
          assetResponse = await env.ASSETS.fetch(indexReq);
        }

        const response = new Response(assetResponse.body, assetResponse);
        response.headers.set("Cross-Origin-Opener-Policy", "same-origin");
        response.headers.set("Cross-Origin-Embedder-Policy", "require-corp");

        // Set optimized Cache-Control headers
        if (url.pathname.endsWith(".wasm") || url.pathname.endsWith(".bin")) {
          response.headers.set("Cache-Control", "public, max-age=31536000, immutable");
          response.headers.set("Cross-Origin-Resource-Policy", "cross-origin");
        } else if (url.pathname === "/" || url.pathname.endsWith(".html") || url.pathname.endsWith("/sw.js")) {
          response.headers.set("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0");
        }

        return response;
      }

      return new Response("Not Found", { status: 404, headers: corsHeaders });
    }

    try {
      if (env.DB) {
        await ensureDbSchema(env.DB);
      }

      // 1. Health & Quota Diagnostics
      if (url.pathname === "/api/contribute/health" && request.method === "GET") {
        return new Response(
          JSON.stringify({
            status: "healthy",
            engine: "RainAI Edge Worker",
            d1_bound: !!env.DB,
            r2_bound: !!env.DATA_BUCKET,
            timestamp: new Date().toISOString(),
          }),
          {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      }

      // 2. Submit Contribution Metadata Record (D1 Transaction / R2 Sidecar)
      if (url.pathname === "/api/contribute/submit-record" && request.method === "POST") {
        const body = await request.json();

        // Turnstile Verification
        const turnstileToken = body.cf_turnstile_response || request.headers.get("X-Turnstile-Token");
        if (!turnstileToken) {
          return new Response(JSON.stringify({ error: "Missing Turnstile verification token" }), {
            status: 403,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }

        const secretKey = env.TURNSTILE_SECRET_KEY || "1x0000000000000000000000000000000AA";
        const formData = new FormData();
        formData.append("secret", secretKey);
        formData.append("response", turnstileToken);
        formData.append("remoteip", request.headers.get("CF-Connecting-IP") || "");

        const turnstileReq = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
          method: "POST",
          body: formData,
        });
        const turnstileRes = await turnstileReq.json();

        if (!turnstileRes.success) {
          return new Response(JSON.stringify({ error: "Turnstile verification failed", details: turnstileRes["error-codes"] }), {
            status: 403,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }

        const sha256 = body.sha256 || `url_${Date.now()}`;
        const isUrlOnly = !!body.is_url_only;

        let existingRecord = null;
        let existingKey = null;

        // Try reading existing from D1 first if bound
        if (env.DB && sha256) {
          const d1Row = await env.DB.prepare(
            "SELECT * FROM records WHERE sha256 = ?"
          ).bind(sha256).first();
          if (d1Row) {
            existingRecord = {
              sha256: d1Row.sha256,
              filename: d1Row.filename,
              status: d1Row.status,
              license: d1Row.license,
              tags: JSON.parse(d1Row.tags_json || "[]"),
              descriptions: JSON.parse(d1Row.descriptions_json || "[]"),
              contributors: JSON.parse(d1Row.contributors_json || "[]"),
              alternate_licenses: JSON.parse(d1Row.alternate_licenses_json || "[]"),
              dsp_passed: d1Row.dsp_passed === 1,
              license_approved: d1Row.license_approved === 1,
              author: d1Row.author,
              file_size_bytes: d1Row.file_size_bytes || 0,
            };
            existingKey = `ingest/${d1Row.status.toLowerCase()}/${sha256}.json`;
          }
        }

        // Fallback to checking R2 if not found in D1
        if (!existingRecord && env.DATA_BUCKET && sha256) {
          const checkKeys = [
            `ingest/quarantine/${sha256}.json`,
            `ingest/approved/${sha256}.json`,
            `ingest/urls/${sha256}.json`,
          ];
          for (const k of checkKeys) {
            const obj = await env.DATA_BUCKET.get(k);
            if (obj) {
              existingRecord = await obj.json();
              existingKey = k;
              break;
            }
          }
        }

        // Reconcile fields if existing record was found
        const mergedTags = Array.from(new Set([
          ...(existingRecord?.tags || []),
          ...(body.tags || [])
        ]));

        const existingDescs = existingRecord?.descriptions || (existingRecord?.notes ? [existingRecord.notes] : []);
        const incomingDescs = body.descriptions || (body.notes ? [body.notes] : []);
        const mergedDescriptions = Array.from(new Set([
          ...existingDescs,
          ...incomingDescs
        ].map(s => s && s.trim()).filter(Boolean)));

        const existingContribs = existingRecord?.contributors || (existingRecord?.author ? [existingRecord.author] : []);
        const incomingContribs = body.contributors || (body.author ? [body.author] : []);
        const mergedContributors = Array.from(new Set([
          ...existingContribs,
          ...incomingContribs
        ].map(s => s && s.trim()).filter(Boolean)));

        // Multi-license aggregation and predominant license selection
        const candidateLicenses = Array.from(new Set([
          ...(existingRecord?.alternate_licenses || []),
          existingRecord?.license,
          ...(body.alternate_licenses || []),
          body.license
        ].filter(Boolean)));

        let bestLic = body.license || "Unknown";
        let bestRankInfo = getLicenseRank(bestLic);

        for (const lic of candidateLicenses) {
          const info = getLicenseRank(lic);
          if (info.rank > bestRankInfo.rank) {
            bestRankInfo = info;
            bestLic = lic;
          }
        }

        const alternateLicenses = candidateLicenses.filter(l => l !== bestLic);
        const licenseApproved = bestRankInfo.approved;
        const dspPassed = (existingRecord?.dsp_passed !== false) && (body.dsp_passed !== false);

        let targetPrefix = `ingest/approved`;
        let targetStatus = "APPROVED";
        if (!licenseApproved || !dspPassed) {
          targetPrefix = `ingest/quarantine`;
          targetStatus = "QUARANTINE";
        } else if (isUrlOnly) {
          targetPrefix = `ingest/urls`;
          targetStatus = "URL_ONLY";
        }

        const objectKey = `${targetPrefix}/${sha256}.json`;
        const wasPromoted = (existingRecord?.status === "QUARANTINE" || (existingKey && existingKey.startsWith(`ingest/quarantine/`))) && targetStatus === "APPROVED";

        const quarantineReason = (!licenseApproved)
          ? "UNAPPROVED_OR_MISSING_LICENSE"
          : (!dspPassed ? "ACOUSTIC_DSP_SCREENING_FAILED" : null);

        // 2a. Update D1 if bound (Atomic UPSERT with ACID guarantees)
        if (env.DB) {
          const insertSql = `
            INSERT INTO records (
              sha256, filename, status, license, license_tier, license_rank,
              license_approved, dsp_passed, author, tags_json, descriptions_json,
              contributors_json, alternate_licenses_json, file_size_bytes, quarantine_reason,
              updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(sha256) DO UPDATE SET
              filename = excluded.filename,
              status = excluded.status,
              license = excluded.license,
              license_tier = excluded.license_tier,
              license_rank = excluded.license_rank,
              license_approved = excluded.license_approved,
              dsp_passed = excluded.dsp_passed,
              author = excluded.author,
              tags_json = excluded.tags_json,
              descriptions_json = excluded.descriptions_json,
              contributors_json = excluded.contributors_json,
              alternate_licenses_json = excluded.alternate_licenses_json,
              file_size_bytes = CASE WHEN excluded.file_size_bytes > 0 THEN excluded.file_size_bytes ELSE records.file_size_bytes END,
              quarantine_reason = excluded.quarantine_reason,
              updated_at = datetime('now');
          `;
          await env.DB.prepare(insertSql).bind(
            sha256,
            body.filename || existingRecord?.filename || "unknown_audio",
            targetStatus,
            bestLic,
            bestRankInfo.tier,
            bestRankInfo.rank,
            licenseApproved ? 1 : 0,
            dspPassed ? 1 : 0,
            body.author || existingRecord?.author || null,
            JSON.stringify(mergedTags),
            JSON.stringify(mergedDescriptions),
            JSON.stringify(mergedContributors),
            JSON.stringify(alternateLicenses),
            body.file_size_bytes || existingRecord?.file_size_bytes || 0,
            quarantineReason
          ).run();
        }

        // 2b. Update R2 metadata sidecar and migrate blob if R2 is bound
        const metadataPayload = {
          ...existingRecord,
          ...body,
          tags: mergedTags,
          descriptions: mergedDescriptions,
          contributors: mergedContributors,
          license: bestLic,
          alternate_licenses: alternateLicenses,
          license_approved: licenseApproved,
          dsp_passed: dspPassed,
          staged_at: existingRecord?.staged_at || new Date().toISOString(),
          last_updated_at: new Date().toISOString(),
          target_prefix: targetPrefix,
          reconciled: !!existingRecord,
          promoted: wasPromoted,
          quarantine_reason: quarantineReason,
        };

        if (env.DATA_BUCKET) {
          await env.DATA_BUCKET.put(objectKey, JSON.stringify(metadataPayload, null, 2), {
            httpMetadata: { contentType: "application/json" },
            customMetadata: {
              license: bestLic,
              sha256: sha256,
              tags: mergedTags.join(","),
              promoted: wasPromoted ? "true" : "false",
            },
          });

          // If promoted from quarantine to approved, cleanup old quarantine json and promote audio blob
          if (wasPromoted && existingKey && existingKey !== objectKey) {
            await env.DATA_BUCKET.delete(existingKey);

            // Move any audio blob associated with this sha256
            for (const ext of ["flac", "wav", "m4a", "opus", "mp3"]) {
              const oldBlobKey = `ingest/quarantine/${sha256}.${ext}`;
              const blobObj = await env.DATA_BUCKET.get(oldBlobKey);
              if (blobObj) {
                const newBlobKey = `ingest/approved/${sha256}.${ext}`;
                await env.DATA_BUCKET.put(newBlobKey, blobObj.body, {
                  httpMetadata: blobObj.httpMetadata,
                });
                await env.DATA_BUCKET.delete(oldBlobKey);
              }
            }
          }
        }

        return new Response(
          JSON.stringify({
            success: true,
            object_key: objectKey,
            status: targetStatus,
            reconciled: !!existingRecord,
            promoted: wasPromoted,
            predominant_license: bestLic,
          }),
          {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      }

      // 3. Direct Binary Audio Blob Upload to R2
      if (url.pathname.startsWith("/api/contribute/upload-blob/") && request.method === "PUT") {
        const sha256 = url.pathname.replace("/api/contribute/upload-blob/", "");
        const targetStatus = request.headers.get("X-Target-Status") || "approved";
        const fileExt = request.headers.get("X-File-Extension") || "flac";
        const objectKey = `ingest/${targetStatus}/${sha256}.${fileExt}`;

        if (env.DATA_BUCKET) {
          await env.DATA_BUCKET.put(objectKey, request.body, {
            httpMetadata: {
              contentType: request.headers.get("Content-Type") || "audio/flac",
            },
          });
        }

        // If D1 is bound, update file_size_bytes if known
        const contentLength = parseInt(request.headers.get("Content-Length") || "0", 10);
        if (env.DB && contentLength > 0 && sha256) {
          await env.DB.prepare(
            "UPDATE records SET file_size_bytes = ?, updated_at = datetime('now') WHERE sha256 = ?"
          ).bind(contentLength, sha256).run();
        }

        return new Response(
          JSON.stringify({ success: true, object_key: objectKey }),
          {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      }

      // 4. Maintainer Triage: List Quarantined Records
      if (url.pathname === "/api/contribute/quarantine-list" && request.method === "GET") {
        if (env.DB) {
          const sql = "SELECT * FROM records WHERE status = 'QUARANTINE' ORDER BY updated_at DESC";
          const stmt = env.DB.prepare(sql);
          const res = await stmt.all();
          const records = (res.results || []).map((r) => ({
            sha256: r.sha256,
            filename: r.filename,
            status: r.status,
            license: r.license,
            license_tier: r.license_tier,
            license_rank: r.license_rank,
            license_approved: r.license_approved === 1,
            dsp_passed: r.dsp_passed === 1,
            author: r.author,
            tags: JSON.parse(r.tags_json || "[]"),
            descriptions: JSON.parse(r.descriptions_json || "[]"),
            contributors: JSON.parse(r.contributors_json || "[]"),
            alternate_licenses: JSON.parse(r.alternate_licenses_json || "[]"),
            file_size_bytes: r.file_size_bytes,
            quarantine_reason: r.quarantine_reason,
            created_at: r.created_at,
            updated_at: r.updated_at,
          }));
          return new Response(JSON.stringify({ records, total: records.length }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        if (!env.DATA_BUCKET) {
          return new Response(JSON.stringify({ records: [] }), { headers: corsHeaders });
        }

        const listed = await env.DATA_BUCKET.list({ prefix: `ingest/quarantine/` });
        const jsonKeys = listed.objects.filter((obj) => obj.key.endsWith(".json"));
        const records = [];

        for (const item of jsonKeys) {
          const obj = await env.DATA_BUCKET.get(item.key);
          if (obj) {
            const data = await obj.json();
            records.push(data);
          }
        }

        return new Response(JSON.stringify({ records, total: records.length }), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      // 5. Maintainer Pull-Approved: List Approved Records
      if (url.pathname === "/api/contribute/approved-list" && request.method === "GET") {
        if (env.DB) {
          const sql = "SELECT * FROM records WHERE status = 'APPROVED' ORDER BY updated_at DESC";
          const stmt = env.DB.prepare(sql);
          const res = await stmt.all();
          const records = (res.results || []).map((r) => ({
            sha256: r.sha256,
            filename: r.filename,
            status: r.status,
            license: r.license,
            license_tier: r.license_tier,
            license_rank: r.license_rank,
            license_approved: r.license_approved === 1,
            dsp_passed: r.dsp_passed === 1,
            author: r.author,
            tags: JSON.parse(r.tags_json || "[]"),
            descriptions: JSON.parse(r.descriptions_json || "[]"),
            contributors: JSON.parse(r.contributors_json || "[]"),
            alternate_licenses: JSON.parse(r.alternate_licenses_json || "[]"),
            file_size_bytes: r.file_size_bytes,
            quarantine_reason: r.quarantine_reason,
            created_at: r.created_at,
            updated_at: r.updated_at,
          }));
          return new Response(JSON.stringify({ records, total: records.length }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        if (!env.DATA_BUCKET) {
          return new Response(JSON.stringify({ records: [] }), { headers: corsHeaders });
        }

        const listed = await env.DATA_BUCKET.list({ prefix: `ingest/approved/` });
        const jsonKeys = listed.objects.filter((obj) => obj.key.endsWith(".json"));
        const records = [];

        for (const item of jsonKeys) {
          const obj = await env.DATA_BUCKET.get(item.key);
          if (obj) {
            const data = await obj.json();
            records.push(data);
          }
        }

        return new Response(JSON.stringify({ records, total: records.length }), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      // 6. Ephemeral Sync Purge: Acknowledge and Remove Records Committed to Git/LFS
      if (url.pathname === "/api/contribute/ack-ingested" && request.method === "POST") {
        const body = await request.json();
        const shas = body.sha256_list || (body.sha256 ? [body.sha256] : []);
        let purgedCount = 0;

        for (const sha of shas) {
          if (env.DB) {
            await env.DB.prepare("DELETE FROM records WHERE sha256 = ?").bind(sha).run();
          }
          if (env.DATA_BUCKET) {
            const cleanupKeys = [
              `ingest/approved/${sha}.json`,
              `ingest/quarantine/${sha}.json`,
              `ingest/urls/${sha}.json`,
            ];
            for (const ext of ["flac", "wav", "m4a", "opus", "mp3"]) {
              cleanupKeys.push(`ingest/approved/${sha}.${ext}`);
              cleanupKeys.push(`ingest/quarantine/${sha}.${ext}`);
              cleanupKeys.push(`ingest/blobs/${sha}.${ext}`);
            }
            for (const key of cleanupKeys) {
              await env.DATA_BUCKET.delete(key);
            }
          }
          purgedCount++;
        }

        return new Response(
          JSON.stringify({ success: true, purged: purgedCount }),
          {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      }

      return new Response("Not Found", { status: 404, headers: corsHeaders });
    } catch (err) {
      return new Response(JSON.stringify({ error: err.message }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
  },
};
