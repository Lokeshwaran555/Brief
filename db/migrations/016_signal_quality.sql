-- Sobha MDI · cross-source corroboration scoring (Round-2 batch, 2026-04-29)
--
-- New column on signals: corroboration_count. Tracks how many INDEPENDENT
-- scout sources have produced a signal in the same semantic cluster (same
-- cluster_key OR pgvector-near). Lets the brief weight a 3-source confirmed
-- launch > a 1-source rumor.
--
-- Computed at upsert time (tools/supabase_tool.py:upsert_signal already
-- does the cluster-key + pgvector lookup). When upsert merges into an
-- existing signal, we increment corroboration_count if the new raw came
-- from a different scout source-prefix than the prior one.
--
-- Apply after 015_analyses.sql. Idempotent.

alter table signals
  add column if not exists corroboration_count integer not null default 1;

-- Highly-corroborated stories — the brief weights these higher.
create index if not exists signals_corroboration_idx
  on signals (corroboration_count desc) where corroboration_count >= 2;

-- Track which scout sources have contributed to each signal so re-runs
-- of the same scout don't keep incrementing corroboration. JSONB array
-- of source-prefix strings: ["proptech:funding", "instagram:stake_uae"].
alter table signals
  add column if not exists corroborating_sources jsonb not null default '[]'::jsonb;

create index if not exists signals_corroborating_sources_gin
  on signals using gin (corroborating_sources);
