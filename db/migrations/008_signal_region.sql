-- Sobha MDI · region tagging on signals
--
-- Drives the MD Scan dashboard's region tabs (Dubai / Abu Dhabi /
-- USA / Australia / Other) plus the dynamic 'Other' split when 3+
-- items from one country cluster.
--
-- `region` is one of: dubai | abu_dhabi | usa | australia | other
-- `country_code` is ISO 3166-1 alpha-2 (US, AE, AU, IN, PL, ...)
--
-- Scouts can stamp these directly when they know the source country
-- (gdelt_scout, region-tuned gnews queries). Otherwise the classifier
-- prompt v3 emits them per signal during scoring.
--
-- Apply after 007_scenarios.sql. Idempotent.

alter table signals
  add column if not exists region text;

alter table signals
  add column if not exists country_code text;

-- Region tab rendering filters by `region`; index helps the dashboard
-- and ceo_scan_flow stay fast.
create index if not exists signals_region_idx on signals(region) where region is not null;

-- Country code feeds the dynamic-split behaviour for 'other' region.
create index if not exists signals_country_code_idx on signals(country_code) where country_code is not null;
