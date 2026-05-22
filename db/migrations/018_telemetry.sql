-- Sobha MDI · MD interaction telemetry (Round-2 batch, 2026-04-29)
--
-- Closes the feedback loop on the dashboard. Every meaningful MD action
-- (Analyze click, pin, source-link click, thread-card open, source-↗ tap)
-- becomes a row here. Used by the classifier-priority feedback work to
-- learn implicit relevance — what does the MD actually engage with?
--
-- Apply after 017_metrics_daily.sql. Idempotent.

create table if not exists telemetry (
  id            bigserial primary key,
  user_id       text not null default 'md',
  -- Action taxonomy. Keep narrow on purpose — only the events that
  -- carry real preference signal. Don't log every page-load tick.
  event         text not null check (event in (
    'analyze_click',
    'pin',
    'unpin',
    'thread_open',
    'bullet_source_click',
    'signal_source_click',
    'page_view',
    'banner_dismiss'
  )),
  -- Target identity — flexible on purpose so we can log a signal_id for
  -- pin events, a bucket name for analyze_click, a thread anchor for
  -- thread_open, etc.
  target_kind   text,                  -- 'signal' | 'thread' | 'bucket' | 'page' | 'analysis'
  target_id     text,                  -- string id; cast at read time
  -- Optional context blob — bucket name, signal_ids array, scope, etc.
  context       jsonb not null default '{}'::jsonb,
  -- When it happened.
  created_at    timestamptz not null default now()
);

-- Time-series scan (for cohort + recent-activity views).
create index if not exists telemetry_time_idx
  on telemetry (created_at desc);

-- Per-target rollup (for "this signal got 3 analyze clicks → boost priority").
create index if not exists telemetry_target_idx
  on telemetry (target_kind, target_id, created_at desc)
  where target_kind is not null;

-- Per-event rollup (for "how many analyze clicks today" velocity).
create index if not exists telemetry_event_time_idx
  on telemetry (event, created_at desc);
