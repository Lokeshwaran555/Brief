-- Migration 012 — Sobha portfolio (the missing context for every LLM stage).
--
-- Source: ~/Downloads/Sobha_MIS_Consolidated_Mar2026.xlsx
-- Sheets: 'Finance MIS Input' (46 projects) + 'Dev MIS Input' (42 projects).
-- Joined on project_name; Dev sheet is the authoritative shape (units/area),
-- Finance sheet adds sales/pricing/gp%.
--
-- This table backs `tools/sobha_context.py` which feeds every LLM stage:
-- classifier, brief synth, CEO scan, pattern prose, investigation
-- implications, deep-dive crew. Without it, the implication writer hard-codes
-- {"status": "MIS not yet wired"} and scenarios reference Sobha generically.
--
-- Idempotent: paste into Supabase SQL editor or `\i 012_sobha_projects.sql`.

create table if not exists sobha_projects (
    project_name           text primary key,
    community              text not null,
    region                 text not null default 'dubai',
    status                 text not null,
    launch_date            date,

    -- Development MIS columns (from 📊 Dev MIS Input sheet)
    total_units            integer,
    resi_units             integer,
    retail_other           integer,
    plot_area_sqft         numeric,
    gfa_sqft               numeric,
    bua_sqft               numeric,
    saleable_area_sqft     numeric,
    far                    numeric,
    sa_per_gfa             numeric,
    sa_per_bua             numeric,

    -- Finance MIS columns (from 💰 Finance MIS Input sheet)
    project_topline_aed    numeric,
    sa_launched_sqft       numeric,
    total_sales_aed        numeric,
    sold_sa_sqft           numeric,
    sold_pct               numeric,            -- 0..1
    avg_sold_psf_aed       numeric,
    unsold_sa_sqft         numeric,
    unsold_pct             numeric,            -- 0..1
    unsold_value_aed       numeric,
    construction_pct       numeric,            -- 0..1
    gp_pct_inception       numeric,
    gp_pct_fy2026_ytd      numeric,
    ytd_collection_aed     numeric,
    revenue_recognized_aed numeric,

    -- Raw row dumps for forensic / future reference
    raw_dev_json           jsonb,
    raw_finance_json       jsonb,

    updated_at             timestamptz not null default now()
);

create index if not exists sobha_projects_community_idx on sobha_projects (community);
create index if not exists sobha_projects_region_idx    on sobha_projects (region);
create index if not exists sobha_projects_status_idx    on sobha_projects (status);
