-- Sobha MDI · daily metrics time-series store (Round-2 batch, 2026-04-29)
--
-- The signals table is event-based — one row per news/social event the
-- system surfaces. But many of the things the dashboard needs to show are
-- TIME SERIES, not events: EIBOR daily fix, Brent close, MAG-7 prices,
-- ABS dwelling approvals month, Sobha Sukuk yield, Baltic Dry Index, etc.
--
-- This table is the parallel store for those. Pure derivation from scouts:
-- market_snapshot writes ticker closes; eibor_scout writes EIBOR fixings;
-- materials_scout writes BDI/SCFI; etc.
--
-- The dashboard's Markets & Capital + Construction Econ pages can render
-- charts from this table without bolting pgvector or extra LLM calls onto
-- numeric data. The signals table stays focused on narrative events.
--
-- Apply after 016_signal_quality.sql. Idempotent.

create table if not exists metrics_daily (
  id            bigserial primary key,
  -- Identity. metric_key is the canonical lookup ("eibor_3m", "brent",
  -- "us_10y", "spy", "abs_nsw_approvals", "bdi", "scfi", "sobha_sukuk_2030_yield").
  metric_key    text not null,
  -- Bucket — drives chart-rendering on which themed page:
  --   markets / construction / global / dubai / capital_flow / tech
  bucket        text,
  -- Captured-at — UTC moment of measurement. For daily fixes use
  -- midnight UTC of the trading day so dedup stays clean.
  captured_at   timestamptz not null,
  -- Numeric value. Stored as numeric for arbitrary precision (basis
  -- points, large unit counts, AED amounts).
  value_num     numeric,
  -- Free-text value when the metric isn't numeric (e.g. status:
  -- "open" / "closed" / "pending" for the Suez / Red Sea status badge).
  value_text    text,
  -- Source — usually a scout source-prefix or "yahoo:close", "cbuae:eibor".
  source        text not null,
  -- Optional unit + display hint for the dashboard.
  unit          text,             -- "USD/bbl", "%", "bps", "AED", "containers"
  -- Anything else the writer wants to remember (% change, raw quote, etc.).
  raw_json      jsonb not null default '{}'::jsonb
);

-- Lookups by metric_key + recent → drive the ~30d sparkline render.
create index if not exists metrics_daily_key_time_idx
  on metrics_daily (metric_key, captured_at desc);

-- Bucket filter for the per-page chart strip.
create index if not exists metrics_daily_bucket_time_idx
  on metrics_daily (bucket, captured_at desc) where bucket is not null;

-- Dedup: one row per (metric_key, captured_at) — re-running a scout for
-- the same trading day overwrites, never duplicates.
create unique index if not exists metrics_daily_unique
  on metrics_daily (metric_key, captured_at);
