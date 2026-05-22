-- Sobha MDI · Intelligence layer
-- The new architectural split (per Codex discussion 2026-04-24):
--   signals    = triggers / leads (existing)
--   events     = canonical market moves reconstructed from many signals
--   evidence   = every source/artifact backing an event
--   facts      = structured extracted fields
--   implications = Sobha-specific analysis + decision pointers
--
-- Apply after 003_brief_pointers.sql. Idempotent.

create table if not exists market_events (
  id              bigserial primary key,
  slug            text unique,                 -- stable human-readable id e.g. 'modon-tara-park-final-phase-2026-04-24'
  event_type      text,                         -- dynamic LLM-classified: launch | phase-release | price-change | partnership | land-acq | financing | commission | delay | hiring | other-free-text
  developer_slug  text,
  project_name    text,
  location        text,
  event_date      date,
  status          text default 'investigating', -- investigating | ready | stale | dismissed
  confidence      numeric(3,2),
  headline        text,
  summary         text,
  seed_signal_id  bigint references signals(id) on delete set null,
  first_seen_at   timestamptz default now(),
  last_updated_at timestamptz default now()
);
create index if not exists market_events_dev_idx    on market_events (developer_slug, event_date desc);
create index if not exists market_events_status_idx on market_events (status, last_updated_at desc);
create index if not exists market_events_type_idx   on market_events (event_type);

create table if not exists event_evidence (
  id          bigserial primary key,
  event_id    bigint references market_events(id) on delete cascade,
  source      text not null,                    -- 'gnews' | 'youtube:channel_x' | 'instagram:handle' | 'tavily' | 'bayut' | 'dubizzle' | 'forum:pf-blog' | ...
  source_url  text,
  title       text,
  excerpt     text,
  fetched_at  timestamptz default now(),
  raw_json    jsonb
);
create index if not exists event_evidence_event_idx on event_evidence (event_id, fetched_at desc);

create table if not exists event_facts (
  id          bigserial primary key,
  event_id    bigint references market_events(id) on delete cascade unique,
  facts       jsonb not null,                   -- full structured fact dict (units, unit_mix, psf, payment_plan, partners, etc)
  field_confidence jsonb,                       -- {field_name: 0.0-1.0}
  unresolved  jsonb,                            -- list of open questions
  extracted_at timestamptz default now(),
  model       text
);

create table if not exists event_implications (
  id            bigserial primary key,
  event_id      bigint references market_events(id) on delete cascade unique,
  tldr          text,                           -- 1-line so-what for Sobha
  angles        jsonb,                          -- list of {angle, reasoning}
  pointers      jsonb,                          -- list of imperative action bullets
  benchmarks    jsonb,                          -- comparisons vs Sobha MIS (PSF delta, unit-mix overlap, payment-plan delta)
  watch_next    jsonb,                          -- what to monitor over next 7-30 days
  generated_at  timestamptz default now(),
  model         text
);
