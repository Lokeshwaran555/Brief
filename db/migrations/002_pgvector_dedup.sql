-- Sobha MDI · pgvector semantic dedup for signals
-- Apply after db/schema.sql. Idempotent.
-- Paste into Supabase SQL editor.

create extension if not exists vector;

-- Embedding column. NVIDIA nv-embedqa-e5-v5 → 1024 dims.
alter table signals
  add column if not exists embedding vector(1024);

-- IVF index for fast cosine lookups. lists=100 is fine up to ~100k rows;
-- bump later if signals table grows past that.
create index if not exists signals_embedding_cos_idx
  on signals using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- RPC: nearest-neighbour lookup over recent un-archived signals.
-- Returns up to `match_count` rows where cosine similarity >= `match_threshold`,
-- restricted to the last `window_days` so old/stale clusters don't pull
-- new signals back in.
create or replace function match_signal(
  query_embedding vector(1024),
  match_threshold float default 0.86,
  match_count int default 1,
  window_days int default 30
)
returns table (
  id bigint,
  cluster_key text,
  headline text,
  similarity float
)
language sql
stable
as $$
  select
    s.id,
    s.cluster_key,
    s.headline,
    1 - (s.embedding <=> query_embedding) as similarity
  from signals s
  where s.embedding is not null
    and s.archived = false
    and s.last_seen_at >= now() - (window_days || ' days')::interval
    and 1 - (s.embedding <=> query_embedding) >= match_threshold
  order by s.embedding <=> query_embedding asc
  limit match_count;
$$;
