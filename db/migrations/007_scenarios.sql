-- Sobha MDI · scenario-framed implications
--
-- Replaces the single "angles" deterministic frame with 2-3 scenarios
-- per investigation: opportunity / risk / watch reads, each with
-- falsification fields (what_supports / what_would_confirm /
-- what_would_reject) so the MD gets choices, not fake certainty.
--
-- `angles` column kept for backward compat — daily_brief_flow still
-- reads it; new investigations populate both (angles is derived from
-- scenarios) until that path is cut over.
--
-- Apply after 006_intelligence_layer_v2. Idempotent.

alter table event_implications
  add column if not exists scenarios jsonb;

-- scenarios shape (per item):
-- {
--   "stance": "opportunity" | "risk" | "watch",
--   "claim": "one-sentence read of what this move could mean",
--   "what_supports": "what in the facts/evidence backs this read",
--   "what_would_confirm": "what new data point would prove it true",
--   "what_would_reject": "what would falsify it",
--   "sobha_exposure": "which Sobha project / segment / channel is in play",
--   "action": "imperative recommended action (Pull / Brief / Benchmark / ...)"
-- }
