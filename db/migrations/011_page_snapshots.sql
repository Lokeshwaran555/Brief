-- Migration 011 — page-hash snapshots for the page_hash_scout.
--
-- Stores the most-recent content hash per monitored URL so the scout
-- can detect when a competitor marketing page actually changes
-- (vs noise from analytics pixels, cache busters, etc.).
--
-- Idempotent: run with `\i 011_page_snapshots.sql` from psql or paste
-- into the Supabase SQL editor.

create table if not exists page_snapshots (
    url             text primary key,
    content_hash    text not null,
    raw_text_preview text,
    first_seen_at   timestamptz not null default now(),
    last_seen_at    timestamptz not null default now(),
    last_changed_at timestamptz
);

create index if not exists page_snapshots_last_changed_idx
    on page_snapshots (last_changed_at desc nulls last);
