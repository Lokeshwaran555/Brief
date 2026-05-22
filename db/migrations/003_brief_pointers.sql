-- Sobha MDI · daily brief pointers (TL;DR + action bullets)
-- Apply after 002_pgvector_dedup.sql. Idempotent.

alter table daily_briefs
  add column if not exists pointers jsonb;

-- pointers shape: {"tldr": "≤25-word headline", "actions": ["…", "…", "…"]}
