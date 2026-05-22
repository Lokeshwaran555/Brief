-- Sobha MDI · CEO scan store
--
-- One row per (region, cadence, scan_day). Output is the full
-- dashboard payload — list of dynamic sections with claims, fact
-- rows, recommended actions. Frontend reads via /api/ceo-scan.
--
-- Apply after 008_signal_region.sql. Idempotent.

create table if not exists ceo_scans (
  id            bigserial primary key,
  region        text not null,        -- dubai | abu_dhabi | usa | australia | other
  cadence       text not null,        -- 'daily' | 'weekly'
  scan_day      date not null,
  output        jsonb not null,       -- {stamp, sections: [{key, heading, claim, rows, action}, ...]}
  signal_count  int default 0,        -- how many signals fed this scan
  generated_at  timestamptz default now(),
  model         text,
  unique(region, cadence, scan_day)
);

create index if not exists ceo_scans_region_idx on ceo_scans(region, cadence, scan_day desc);
