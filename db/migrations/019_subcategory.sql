-- Migration 019 — LLM-driven subcategory routing (round 7, 2026-04-30).
--
-- Adds a `subcategory` field to signals so the classifier LLM can do
-- fine-grained routing (proptech vs contech vs frontier_ai vs
-- materials_cost vs materials_innovation vs regulatory_visa vs
-- regulatory_dev vs ...). Replaces the keyword-filter post-process
-- on themed pages with a single AND clause:
--   WHERE category=tech AND subcategory IN (proptech, frontier_ai)
--
-- One signal, one page — no duplications (stakeholder doctrine
-- 2026-04-30).

ALTER TABLE signals ADD COLUMN IF NOT EXISTS subcategory TEXT;
CREATE INDEX IF NOT EXISTS idx_signals_subcategory ON signals(subcategory);
CREATE INDEX IF NOT EXISTS idx_signals_cat_subcat ON signals(category, subcategory);

-- Same on intel_scored so the classifier output persists.
ALTER TABLE intel_scored ADD COLUMN IF NOT EXISTS subcategory TEXT;

COMMENT ON COLUMN signals.subcategory IS 'LLM-assigned fine-grained routing key. NULL on legacy rows pre-2026-04-30; new rows get one of: proptech, contech, frontier_ai, materials_cost, materials_innovation, re_software, regulatory_visa, regulatory_dev, competitor_ma, competitor_launch, capital_sukuk, capital_listed, capital_fx, geopolitics_re_impact, hpi_prime, hnwi_migration, lifestyle, other. See tools/nvidia_llm.py:INTEL_SYSTEM for full taxonomy.';
