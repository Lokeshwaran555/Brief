-- Sobha MDI · signal freshness
-- Adds the source publish date to `signals` so the dashboard can
-- display "published 47 weeks ago" separately from "we first saw it
-- today". Also lets us filter stale-content-that-got-re-ingested at
-- query time.
--
-- Apply after 004_intelligence_layer.sql. Idempotent.

alter table signals
  add column if not exists source_published_at timestamptz;

create index if not exists signals_source_published_idx
  on signals (source_published_at desc)
  where archived = false;
