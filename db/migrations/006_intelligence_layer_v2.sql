-- Sobha MDI · intelligence layer v2
-- Implements the next-step sequence from Codex's review:
--   1. Typed evidence (event_evidence.evidence_type)
--   2. Investigation run history (event_runs)
--   3. Event state machine (market_events.run_count + state values)
--   4. Projects canonical layer (projects + market_events.project_id)
--   5. Drawer UX restructure — UI only, no schema
--   6. Signal novelty (signals.novelty)
--
-- Apply after 005_signal_freshness.sql. Idempotent.

-- ─── 1. Typed evidence ───────────────────────────────────────────────────
alter table event_evidence
  add column if not exists evidence_type text;
-- Values: official_dev | portal | broker | social | video | forum | news
--         | regulatory | search | internal
create index if not exists event_evidence_type_idx
  on event_evidence (event_id, evidence_type);

-- ─── 2. Investigation run history ────────────────────────────────────────
create table if not exists event_runs (
  id                bigserial primary key,
  event_id          bigint references market_events(id) on delete cascade,
  signal_id         bigint references signals(id) on delete set null,
  run_at            timestamptz default now(),
  facts             jsonb,
  implications      jsonb,
  evidence_count    int,
  unresolved_count  int,
  duration_s        numeric,
  ok                boolean,
  trigger_source    text   -- 'manual' | 'auto' | 'sweep' | 're-investigate'
);
create index if not exists event_runs_event_idx on event_runs (event_id, run_at desc);

-- ─── 3. Event state machine ──────────────────────────────────────────────
-- status values now used: seeded | investigating | ready | stale | conflicted
alter table market_events
  add column if not exists run_count int default 0;

-- ─── 4. Projects canonical layer ────────────────────────────────────────
create table if not exists projects (
  id                  bigserial primary key,
  slug                text unique,                 -- canonical: <developer_slug>-<project_slug>
  developer_slug      text,
  developer_name      text,
  project_name        text,
  location            text,
  micro_market        text,
  project_type        text,                        -- tower | villa | branded-residence | mixed-use | other
  branded             boolean default false,
  brand_partner       text,                        -- e.g. 'Mercedes-Benz' when branded=true
  status              text default 'active',       -- active | sold-out | handover | dormant
  first_seen_at       timestamptz default now(),
  last_updated_at     timestamptz default now(),

  -- Latest known facts (denormalised so the dashboard can sort/filter
  -- without joining event_facts on every event)
  units_total         int,
  unit_mix            text[],
  starting_price_aed  bigint,
  starting_psf_aed    numeric(10,2),
  payment_plan        text,
  handover_date       text,
  target_segment      text,
  partners            text[],

  fact_confidence     jsonb,                       -- per-field confidence
  event_count         int default 0,               -- how many market_events roll up
  raw                 jsonb                         -- catch-all for additional fields
);
create index if not exists projects_dev_idx     on projects (developer_slug, last_updated_at desc);
create index if not exists projects_market_idx  on projects (location, project_type);
create index if not exists projects_branded_idx on projects (branded) where branded = true;

-- Link events → project so dashboards can group by canonical project
alter table market_events
  add column if not exists project_id bigint references projects(id) on delete set null;
create index if not exists market_events_project_idx
  on market_events (project_id, last_updated_at desc);

-- ─── 6. Signal novelty ──────────────────────────────────────────────────
-- 0.0 = very common (we've seen many like this), 1.0 = brand new
alter table signals
  add column if not exists novelty numeric(3,2);
create index if not exists signals_novelty_idx
  on signals (novelty desc) where archived = false and novelty is not null;
