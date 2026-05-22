-- Sobha MDI · saved analyses (Phase N5, 2026-04-29)
--
-- Every "Analyze ↗" click on the dashboard is persisted. The MD can
-- re-open a past analysis from a Saved-Analyses rail, and the UI shows
-- what new evidence has landed against the same signal_ids since.
--
-- Apply after 014_watchlist.sql. Idempotent.

create table if not exists analyses (
  id              bigserial primary key,
  user_id         text not null default 'md',
  -- What was analyzed.
  scope           text not null check (scope in ('bullet','bucket','section','thread')),
  signal_ids      bigint[] not null default '{}',
  -- The /api/analyze response — bullets, data_gaps, stance.
  bullets         jsonb,    -- [{text, signal_ids}]
  data_gaps       jsonb,    -- [str]
  stance          text,     -- opportunity | risk | watch | mixed
  signal_count    integer,
  -- Optional MD label — derived from the bucket name or thread anchor
  -- when the click happened, MD can rename later.
  label           text,
  -- Lifecycle.
  created_at      timestamptz not null default now(),
  archived_at     timestamptz
);

create index if not exists analyses_user_created_idx
  on analyses (user_id, created_at desc);

create index if not exists analyses_signal_ids_gin
  on analyses using gin (signal_ids);
