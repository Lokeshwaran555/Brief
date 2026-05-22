-- Sobha MDI · cross-signal pattern persistence
--
-- Stores every pattern detected per week so the dashboard can build a
-- historical view ('Capital cluster appeared 3 weeks running',
-- 'Meydan activity peaked week of 2026-04-18', etc.).
--
-- Keyed by (key, week_start_iso) so a re-run of the detector on the
-- same week is idempotent — it overwrites the same row instead of
-- producing duplicates.
--
-- The detector is deterministic (Phase 3 shipped 4 detectors — dev
-- surge, topic cluster, micro-market, tech keyword) so the same
-- inputs produce the same key. The LLM-prose layer (Phase 4.2)
-- mutates `sobha_implication` + `action` over time as the prompt
-- evolves — those land on this row too.
--
-- Apply after 009_ceo_scans.sql. Idempotent.

create table if not exists market_patterns (
  id            bigserial primary key,
  key           text not null,         -- stable detector id, e.g. 'dev_surge:emaar:7d'
  week_start    date not null,         -- Monday of the week the pattern landed
  title         text,
  stance        text,                  -- opportunity | risk | watch
  claim         text,                  -- deterministic detector claim
  region        text,                  -- dubai | abu_dhabi | usa | australia | other | NULL
  evidence_count int default 0,
  evidence_signal_ids jsonb,           -- array of signal ids
  metric        jsonb,                 -- detector-specific metric dict
  sobha_implication text,              -- LLM-prose layer (Phase 4.2)
  action        text,                  -- LLM-prose layer
  detected_at   timestamptz default now(),
  unique(key, week_start)
);

create index if not exists market_patterns_week_idx on market_patterns(week_start desc);
create index if not exists market_patterns_key_idx on market_patterns(key);
create index if not exists market_patterns_region_idx on market_patterns(region) where region is not null;
