-- Sobha MDI · pinned watchlist (Phase N4, 2026-04-29)
--
-- The MD can pin signals / patterns / threads / analyses from MD Scan.
-- Pinned items stick until dismissed and accumulate new evidence on
-- their card with each ingest cycle. Persists across devices since
-- the MD opens the dashboard from phone + laptop.
--
-- Apply after 013_friday_conversations.sql. Idempotent.

create table if not exists watchlist (
  id            bigserial primary key,
  -- Identity. user_id is currently always 'md' (single-user app);
  -- multi-user support is a future migration if/when we get there.
  user_id       text not null default 'md',
  -- What's pinned: a signal, a market_pattern, an active thread anchor,
  -- or a saved analysis (joins to N5's analyses table).
  target_kind   text not null check (target_kind in ('signal','pattern','thread','analysis')),
  target_id     text not null,            -- signal id, pattern id, thread anchor, analysis id
  -- Display fields cached at pin-time so the rail can render even when
  -- the source row archives or pattern_id rotates.
  label         text,                     -- short title for the rail card
  note          text,                     -- optional MD note
  -- Lifecycle.
  pinned_at     timestamptz not null default now(),
  dismissed_at  timestamptz,              -- null until user un-pins
  last_seen_at  timestamptz not null default now()
);

-- One row per (user, kind, target). Re-pinning the same item updates
-- the existing row (dismissed_at → null, last_seen_at → now).
create unique index if not exists watchlist_user_target_uidx
  on watchlist (user_id, target_kind, target_id);

-- Active-only filter for the rail.
create index if not exists watchlist_active_idx
  on watchlist (user_id, dismissed_at) where dismissed_at is null;
