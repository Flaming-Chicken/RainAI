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

    try {
      // 1. Health & Quota Diagnostics
      if (url.pathname === "/api/contribute/health" && request.method === "GET") {
        return new Response(
          JSON.stringify({
            status: "healthy",
            engine: "RainAI Edge Worker",
            r2_bound: !!env.DATA_BUCKET,
            environment: env.ENVIRONMENT || "unknown",
            timestamp: new Date().toISOString(),
          }),
          {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      }

      // 2. Submit Contribution Metadata Record (JSON Sidecar)
      if (url.pathname === "/api/contribute/submit-record" && request.method === "POST") {
        const body = await request.json();
        const sha256 = body.sha256 || `url_${Date.now()}`;
        const isUrlOnly = !!body.is_url_only;

        let existingRecord = null;
        let existingKey = null;

        if (env.DATA_BUCKET && sha256) {
          const checkKeys = [
            `/quarantine/${sha256}.json`,
            `/approved/${sha256}.json`,
            `/urls/${sha256}.json`,
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
        if (!licenseApproved || !dspPassed) {
          targetPrefix = `ingest/quarantine`;
        } else if (isUrlOnly) {
          targetPrefix = `ingest/urls`;
        }

        const objectKey = `${targetPrefix}/${sha256}.json`;
        const wasPromoted = existingKey && existingKey.startsWith(`ingest/quarantine/`) && targetPrefix === `ingest/approved`;

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
          quarantine_reason: (!licenseApproved) 
            ? "UNAPPROVED_OR_MISSING_LICENSE" 
            : (!dspPassed ? "ACOUSTIC_DSP_SCREENING_FAILED" : null),
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
          if (wasPromoted && existingKey !== objectKey) {
            await env.DATA_BUCKET.delete(existingKey);

            // Move any audio blob associated with this sha256
            for (const ext of ["flac", "wav", "m4a", "opus", "mp3"]) {
              const oldBlobKey = `/quarantine/${sha256}.${ext}`;
              const blobObj = await env.DATA_BUCKET.get(oldBlobKey);
              if (blobObj) {
                const newBlobKey = `/approved/${sha256}.${ext}`;
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
            status: targetPrefix.replace(`ingest/`, "").toUpperCase(),
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
        const objectKey = `/${targetStatus}/${sha256}.${fileExt}`;

        if (env.DATA_BUCKET) {
          await env.DATA_BUCKET.put(objectKey, request.body, {
            httpMetadata: {
              contentType: request.headers.get("Content-Type") || "audio/flac",
            },
          });
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

      return new Response("Not Found", { status: 404, headers: corsHeaders });
    } catch (err) {
      return new Response(JSON.stringify({ error: err.message }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
  },
};
