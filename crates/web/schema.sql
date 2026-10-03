-- RainAI Ephemeral Ingestion & Quarantine Database Schema
-- Provides ACID transactions for metadata, licensing, tags, and status transitions.
-- Designed for zero-waste space efficiency (no redundant histories or audit tables).

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
    environment TEXT NOT NULL DEFAULT 'production',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_environment ON records(environment);
