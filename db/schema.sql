-- Sobha MDI · agents store
-- Apply once: paste into https://supabase.com/dashboard/project/radgoeohjtwjwsrtkcgw/sql/new
-- Week-1 scope: intel_raw + intel_scored + signals. pgvector/embeddings deferred.

-- ─── 1. raw scraped items ────────────────────────────────────────────────────
create table if not exists intel_raw (
  id          bigserial primary key,
  source      text not null,          -- 'gnews:pre-launch' | 'bayut:listings' | ...
  source_url  text,
  title       text,
  summary     text,
  raw_json    jsonb,                  -- preserve full scraper payload
  fetched_at  timestamptz default now(),
  dedup_key   text unique,            -- sha256(title || source_domain) — unique so upsert ON CONFLICT works
  processed   boolean default false
);
create index if not exists intel_raw_fetched_at_idx on intel_raw (fetched_at desc);
create index if not exists intel_raw_processed_idx  on intel_raw (processed) where processed = false;

-- ─── 2. LLM-scored items ─────────────────────────────────────────────────────
create table if not exists intel_scored (
  id           bigserial primary key,
  raw_id       bigint references intel_raw(id) on delete cascade,
  keep         boolean,
  category     text,
  decision_tag text,                  -- pricing | capital | launch | risk | none
  priority     text,                  -- high | medium | low
  confidence   numeric(3,2),
  headline     text,
  dek          text,
  entities     text[],                -- ['Emaar', 'Creek Harbour']
  reason       text,
  model        text,                  -- 'meta/llama-3.3-70b-instruct'
  scored_at    timestamptz default now()
);
create index if not exists intel_scored_kept_idx on intel_scored (priority, confidence desc) where keep = true;
create index if not exists intel_scored_raw_idx  on intel_scored (raw_id);

-- ─── 3. canonical deduplicated signals ───────────────────────────────────────
-- Week-1: title-hash dedup only. pgvector upgrade in a later pass.
create table if not exists signals (
  id            bigserial primary key,
  cluster_key   text unique,          -- sha256(normalized title)
  first_seen_at timestamptz not null, -- the MD's edge
  last_seen_at  timestamptz not null,
  source_count  int default 1,
  dev_slug      text,                 -- emaar | damac | aldar | ...
  category      text,
  decision_tag  text,
  priority      text,
  confidence    numeric(3,2),
  headline      text,
  dek           text,
  entities      text[],
  urls          text[],               -- every URL that confirmed this signal
  client_score  numeric(5,3),         -- recomputed nightly
  archived      boolean default false
);
create index if not exists signals_dev_idx    on signals (dev_slug, priority, last_seen_at desc);
create index if not exists signals_score_idx  on signals (client_score desc) where archived = false;
create index if not exists signals_latest_idx on signals (last_seen_at desc) where archived = false;

-- ─── 4. per-developer pulse (computed nightly) ───────────────────────────────
create table if not exists dev_pulse (
  dev_slug          text primary key,
  day               date not null,
  signal_count_14d  int,
  high_priority_14d int,
  baseline_14d      numeric,          -- 90-day rolling avg
  z_score           numeric,
  label             text,             -- quiet | low | normal | elevated | high | surge
  updated_at        timestamptz default now()
);

-- ─── 5. user state ───────────────────────────────────────────────────────────
create table if not exists watchlist (
  user_id    text,
  dev_slug   text,
  starred_at timestamptz default now(),
  primary key (user_id, dev_slug)
);

create table if not exists signal_feedback (
  user_id   text,
  signal_id bigint references signals(id),
  verdict   text,                     -- useful | noise
  given_at  timestamptz default now(),
  primary key (user_id, signal_id)
);

-- ─── 6. daily morning brief (one row per calendar day, GST) ─────────────────
-- A-daily crew writes narrative; C flow writes top_signals + synthesis.
-- Dashboard reads this table to render the 04:00 GST brief hero block.
create table if not exists daily_briefs (
  id              bigserial primary key,
  day             date unique not null,
  narrative       text,                 -- 250-word MD-voice narrative (crew output)
  top_signals     jsonb,                -- list of {headline, dek, url, priority, confidence}
  synthesis       jsonb,                -- {markets: [...], regulatory: [...], competitors: [...], dubai: [...], global: [...]}
  signal_count    int,
  generated_at    timestamptz default now(),
  generator_model text
);
create index if not exists daily_briefs_day_idx on daily_briefs (day desc);
