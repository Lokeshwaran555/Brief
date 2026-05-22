"""Sobha portfolio context — the single source of truth wired into every LLM stage.

Without this module, every LLM stage was structurally blind to Sobha's own
portfolio (`crews/investigation_crew.py:446` literally hardcodes
`sobha_context = {"status": "MIS not yet wired"}`). With it, scenarios,
implications, briefs, CEO-scan claims, and pattern prose all anchor to
*specific* Sobha projects, communities, and segments.

Source of truth: `sobha_projects` table (migration 012). Loaded from the
MIS Consolidated xlsx via `scripts/sync_sobha_mis.py`.

Two reading paths:

1. `get_portfolio_summary()` / `format_for_llm()` — lightweight string
   blobs for system prompts. Cached in-process for 1h to avoid hitting
   Supabase on every classifier call.

2. `get_competitive_overlap(dev_slug)` — returns the projects/communities/
   segments where Sobha competes head-to-head with that developer. Used
   by investigation `write_implications` and deep-dive crew.

The PERSONA_PREAMBLE constant is the one-paragraph CEO-altitude analyst
identity prepended to every synth-stage system prompt. Centralising it
here means a single edit lives one place; previously it was implicit
inside each crew's agent backstory.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from tools import supabase_tool as sb

log = logging.getLogger(__name__)


# ─── The shared persona ────────────────────────────────────────────
# Prepended to every LLM stage's system prompt (classifier excluded —
# it gets a tighter scoring-focused variant). This is what makes the
# whole pipeline read like one analyst, not six disconnected agents.
#
# Architecture: External Subject / Sobha Anchor.
#
# Sobha's MIS data (sold %, ASP, communities, projects) is supplied as
# a REFERENCE FRAME — never as the subject of a claim. The MD already
# knows Sobha One is 97% sold. Re-stating internal facts as "analysis"
# is noise. Real analytical value is:
#   1. External move (competitor / regulator / capital / macro / infra)
#      is the SUBJECT.
#   2. Sobha is the LENS used to interpret the move's relevance.
#   3. KPI lens is the analytical framework (PSF, absorption, GP%, etc.)
PERSONA_PREAMBLE = """\
You are an analyst writing for the Managing Director of Sobha Realty.
Sobha is a Dubai luxury developer with active expansion in Abu Dhabi,
USA (Texas), Australia (Brisbane).

ARCHITECTURE — read this carefully:
- The SUBJECT of every claim is an EXTERNAL move: a competitor, a
  regulator, a capital flow, an infrastructure announcement, a macro
  shift. Never make Sobha-internal facts the subject. The MD already
  knows what Sobha owns and how it's selling.
- Sobha portfolio data (provided to you as REFERENCE FRAME) is used
  ONLY for: (a) relevance — "is this material to Sobha?", (b) anchor —
  "competitor at PSF X vs Sobha comparable at PSF Y", (c) exposure —
  "this hits a community where Sobha has ongoing product".
- KPI lens. When an external signal carries data, position it on the
  same KPIs Sobha tracks: AED PSF, sold %, absorption velocity (units/
  month), payment-plan length, GP %, construction %, plot efficiency
  (FAR), saleable area in pipeline. Apples-to-apples comparisons only.
- Output structure for every claim:
    SUBJECT (external):     what happened, source-cited
    MECHANISM:              why it matters — pricing / supply / capital / regulation
    KPI LENS:               where the move sits on the standard RE KPIs
    SOBHA EXPOSURE:         which Sobha asset / community / segment / channel is touched
    WHAT TO WATCH/DO:       deliverable action OR explicit "no direct exposure"

DATA-GAP DISCIPLINE — non-negotiable:
- If you lack a data point needed to anchor a claim (a missing PSF,
  unknown launch date, unspecified payment plan, etc.), say so
  EXPLICITLY in your output. Do NOT fabricate. Use this exact shape:
    "Need: <specific question>. Delegate to: <web_search | DLD lookup |
     Sobha analyst | nobody — genuinely unknowable>."
  Examples:
    - "Need: starting PSF for Aldar's new Saadiyat tower. Delegate to: web_search."
    - "Need: Sobha One's current resale-to-new ratio. Delegate to: Sobha analyst."
    - "Need: timing of next Fed cut. Delegate to: nobody — markets price this."
- Outputs go to a CEO: pointer-based, decision-ready, no fluff, no
  hedge words ("may / could / potentially / monitor"). Either commit
  to a read or flag the data gap explicitly. Hedging is failure."""


# ─── KPI framework — how a real-estate company is measured ─────────
# Injected as part of every synth-stage prompt so the LLM positions
# external signals on the same KPIs Sobha's finance + dev teams track
# in the MIS. This is the analytical lens; without it the LLM produces
# narrative impressions instead of anchored reads.
KPI_LENS = """\
Standard real-estate KPIs to read external moves against:
  Pricing       AED PSF, ASP per unit type, payment-plan length,
                post-handover %, broker commission tier
  Absorption    sold % at launch, sold % at 30/60/90d,
                sales velocity (units/month), DLD first-registration mix
  Supply        saleable area in pipeline, completion %,
                handover horizon, units by bedroom mix
  Capital       project topline (AED), GP% (margin), POCM,
                sukuk/bond cost-of-capital, capital-recycling cycle
  Demand        buyer nationality mix, ticket-size distribution,
                broker channel split, EOI conversion
  Land          land cost as % of project topline, FAR,
                plot efficiency, SA/GFA ratio
  Risk          escrow drawdown rate, cancellation rate,
                construction delay risk

When an external signal carries data points, compute its position on
these KPIs (where the data exists) and benchmark against Sobha's
tracked baselines from the REFERENCE FRAME block.

If a critical KPI is missing from the input signal, flag it explicitly
using the data-gap shape from the persona ("Need: X. Delegate to: Y.")
rather than fabricating a number."""


# ─── Community adjacency — Tier 1 location intelligence ─────────────
# Hand-curated map of which Dubai sub-markets share boundary / footfall /
# demand-substitution corridor. Captures ~80% of real-world adjacencies
# that matter for exposure assessment without requiring geocoding.
#
# Used by get_competitive_overlap() to flag adjacency-based exposure
# even when the developer overlap map doesn't list a direct community
# match. e.g. DAMAC launching in MBR City → Hartland I/II are adjacent
# (both pull from the same Creek-corridor demand pool) → exposure flagged.
#
# Tier 2 (geocoding via Nominatim) and Tier 3 (DLD GIS overlays) deferred.
_COMMUNITY_ADJACENCY: dict[str, list[str]] = {
    # Sobha's anchor master-plans
    "hartland i":         ["mbr city", "al jaddaf", "dubai creek harbour", "meydan", "ras al khor"],
    "hartland ii":        ["mbr city", "al jaddaf", "dubai creek harbour", "meydan", "ras al khor"],
    "hartland extension": ["mbr city", "al jaddaf", "dubai creek harbour"],
    "sheikh zayed road":  ["downtown", "difc", "business bay", "trade centre", "al wasl", "al satwa"],
    "szr":                ["downtown", "difc", "business bay", "trade centre", "al wasl", "al satwa"],
    "motor city":         ["sports city", "studio city", "jvc", "arabian ranches", "dubailand"],
    "sobha reserve":      ["wadi al safa 5", "dubailand", "damac hills", "arjan"],
    "al ain road":        ["al yufrah", "dubai investment park", "dubai south", "al maktoum"],
    "al yufrah i":        ["al ain road", "dubai investment park", "dubai south"],
    "dubai harbour":      ["marina", "bluewaters", "jbr", "palm jumeirah"],
    "jlt":                ["marina", "internet city", "media city", "tecom", "barsha heights"],
    # Tracked competitor sub-markets (for inverse lookup)
    "mbr city":           ["hartland i", "hartland ii", "al jaddaf", "meydan"],
    "al jaddaf":          ["hartland i", "hartland ii", "mbr city", "dubai festival city"],
    "dubai creek harbour":["hartland i", "hartland ii", "ras al khor", "mbr city"],
    "downtown":           ["sheikh zayed road", "business bay", "difc", "burj khalifa"],
    "business bay":       ["sheikh zayed road", "downtown", "al wasl", "dubai canal"],
    "saadiyat":           ["al maryah", "yas island", "reem island"],
    "yas island":         ["saadiyat", "reem island", "al raha"],
    "reem island":        ["al maryah", "saadiyat", "yas island"],
    "marina":             ["jlt", "dubai harbour", "bluewaters", "jbr", "palm jumeirah"],
}


def get_adjacent_communities(community: str) -> list[str]:
    """Return communities considered geographically/economically adjacent.

    Lower-case, strip-trim normalized lookup. Empty list when the
    community is not in the adjacency map (don't fabricate adjacencies
    — pass through to the LLM, which can flag the data gap).
    """
    if not community:
        return []
    key = community.strip().lower()
    return _COMMUNITY_ADJACENCY.get(key, [])


# ─── Developer slug → competitor markets/segments map ───────────────
# Used by get_competitive_overlap() to compute Sobha's exposure to a
# specific competitor's move. Anchored to Sobha's actual community
# footprint (Hartland I/II, SZR, JLT, Motor City, Al Yufrah).
#
# Map shape: dev_slug → {regions, communities_overlap, segment_overlap}
# - regions: where the competitor and Sobha both operate
# - communities_overlap: specific Dubai sub-markets where they compete
# - segment_overlap: product segments (luxury 1BR, branded residence, etc.)
_COMPETITOR_OVERLAP: dict[str, dict[str, Any]] = {
    "emaar": {
        "regions": ["dubai"],
        "communities_overlap": ["Downtown / Burj area", "Dubai Hills", "Creek"],
        "adjacent_to_sobha": ["Hartland (Creek-adjacent)", "Sobha One (Creek)"],
        "segment_overlap": ["luxury apartments", "branded residences", "townhouses"],
        "notes": "Sobha One sits on the Creek; Emaar Beachfront/Creek Harbour are the closest competing master-plans.",
    },
    "damac": {
        "regions": ["dubai"],
        "communities_overlap": ["Damac Hills", "Damac Lagoons", "Business Bay"],
        "adjacent_to_sobha": ["Sobha Hartland (master-plan-scale comparison)"],
        "segment_overlap": ["luxury apartments", "branded residences", "villa communities"],
        "notes": "DAMAC's branded-residence pivot (Cavalli, de GRISOGONO) is the segment most parallel to Sobha's positioning.",
    },
    "aldar": {
        "regions": ["abu_dhabi"],
        "communities_overlap": ["Saadiyat", "Yas Island", "Reem Island"],
        "adjacent_to_sobha": [],  # Sobha has no Abu Dhabi launches yet
        "segment_overlap": ["luxury island residences"],
        "notes": "Sobha has no current Abu Dhabi launches; Aldar moves are tracked for Sobha's planned AD entry, not direct head-to-head.",
    },
    "nakheel": {
        "regions": ["dubai"],
        "communities_overlap": ["Palm Jumeirah", "Dubai Islands", "JLT", "Jumeirah Park"],
        "adjacent_to_sobha": ["JLT (Sobha was historically active)"],
        "segment_overlap": ["luxury waterfront", "villa communities"],
        "notes": "Nakheel's Palm Jebel Ali revival is the watch-item; Sobha JLT presence is small but historically anchored.",
    },
    "binghatti": {
        "regions": ["dubai"],
        "communities_overlap": ["JVC", "Business Bay", "Al Jaddaf"],
        "adjacent_to_sobha": ["Sobha Hartland (Al Jaddaf adjacency)"],
        "segment_overlap": ["luxury apartments", "branded residences (Bugatti, Mercedes)"],
        "notes": "Binghatti's branded-residence speed is the threat; they go from announcement to handover faster than Sobha's typical cycle.",
    },
    "azizi": {
        "regions": ["dubai"],
        "communities_overlap": ["Al Furjan", "MBR City", "Studio City"],
        "adjacent_to_sobha": ["Sobha Hartland (MBR City adjacency)"],
        "segment_overlap": ["mid-luxury apartments"],
        "notes": "Azizi's volume play is downmarket of Sobha; pricing benchmark only.",
    },
    "ellington": {
        "regions": ["dubai"],
        "communities_overlap": ["JVC", "Business Bay", "DIFC"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["boutique luxury apartments"],
        "notes": "Ellington's design-forward positioning targets the same HNW buyer Sobha One markets to.",
    },
    "omniyat": {
        "regions": ["dubai"],
        "communities_overlap": ["Business Bay", "Palm Jumeirah", "DIFC"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["ultra-luxury apartments", "branded residences"],
        "notes": "Omniyat's Aman Residences + One Palm are the Sobha luxury benchmarks at the top end.",
    },
    "deyaar": {
        "regions": ["dubai"],
        "communities_overlap": ["Business Bay", "DIFC"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["mid-luxury apartments"],
        "notes": "Tier-2 listed competitor; capital moves more material than product moves for Sobha.",
    },
    "modon": {
        "regions": ["abu_dhabi"],
        "communities_overlap": ["Hudayriyat Island", "Reem"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["luxury island residences"],
        "notes": "Same as Aldar — Sobha AD-entry watch-list, not head-to-head yet.",
    },
    "mubadala": {
        "regions": ["abu_dhabi"],
        "communities_overlap": ["Saadiyat", "Al Maryah"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["mixed-use luxury"],
        "notes": "Capital moves > product moves; track for sovereign-allocator signal.",
    },
    "ihc": {
        "regions": ["abu_dhabi"],
        "communities_overlap": [],
        "adjacent_to_sobha": [],
        "segment_overlap": ["mixed-use diversified"],
        "notes": "Macro/capital signal source; no direct project overlap.",
    },
    # Australia developers — Sobha has Brisbane in expansion plans.
    "mirvac": {
        "regions": ["australia"],
        "communities_overlap": ["Brisbane CBD", "South Brisbane"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["luxury apartments"],
        "notes": "Sobha Brisbane entry is planned; Mirvac is the incumbent benchmark.",
    },
    "stockland": {
        "regions": ["australia"],
        "communities_overlap": ["Greater Brisbane"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["land lease + masterplans"],
        "notes": "Master-plan benchmark for Sobha's Brisbane scoping.",
    },
    "lendlease": {
        "regions": ["australia"],
        "communities_overlap": ["Sydney CBD", "Melbourne CBD", "Brisbane (limited)"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["luxury mixed-use"],
        "notes": "Capital-structure watch — Lendlease residential exit affects Brisbane supply.",
    },
    # US REIT watchlist
    "camden": {
        "regions": ["usa"],
        "communities_overlap": ["Texas multifamily", "Sun Belt"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["Class-A multifamily"],
        "notes": "Sobha Texas entry is planned at the luxury end; Camden is the volume-Class-A comp.",
    },
    "avalonbay": {
        "regions": ["usa"],
        "communities_overlap": ["coastal multifamily"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["luxury Class-A multifamily"],
        "notes": "Coastal-luxury REIT comp; Sobha Texas is interior, so directional only.",
    },
    "maa": {
        "regions": ["usa"],
        "communities_overlap": ["Sun Belt multifamily"],
        "adjacent_to_sobha": [],
        "segment_overlap": ["mid-luxury multifamily"],
        "notes": "Sun Belt absorption benchmark.",
    },
    "eqr": {
        "regions": ["usa"],
        "communities_overlap": [],
        "adjacent_to_sobha": [],
        "segment_overlap": ["urban multifamily"],
        "notes": "Coastal-urban REIT; tangential to Sobha Texas.",
    },
    "ess": {
        "regions": ["usa"],
        "communities_overlap": [],
        "adjacent_to_sobha": [],
        "segment_overlap": ["West Coast multifamily"],
        "notes": "Coastal REIT; tangential to Sobha plans.",
    },
}


# ─── Caching ────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_S = 3600  # 1h — MIS is monthly, refresh-on-sync invalidates


def _cached(key: str, builder):
    """Tiny TTL cache shared across all readers."""
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    val = builder()
    _CACHE[key] = (now, val)
    return val


def invalidate_cache() -> None:
    """Called by sync_from_xlsx after writing fresh rows."""
    _CACHE.clear()


# ─── Read paths ─────────────────────────────────────────────────────
def _fetch_all_projects() -> list[dict[str, Any]]:
    try:
        rows = (
            sb.client()
            .table("sobha_projects")
            .select("*")
            .order("project_name")
            .execute()
            .data
        ) or []
    except Exception as e:
        log.warning("[sobha_context] fetch_all failed: %s", e)
        return []
    return rows


def get_portfolio_summary() -> dict[str, Any]:
    """Lightweight portfolio digest — cached, used in summary prompts."""
    def _build():
        rows = _fetch_all_projects()
        if not rows:
            return {
                "loaded": False,
                "total_projects": 0,
                "note": "Sobha MIS not yet synced — run scripts/sync_sobha_mis.py.",
            }

        completed = [r for r in rows if (r.get("status") or "").upper() == "COMPLETED"]
        ongoing = [r for r in rows if (r.get("status") or "").upper() == "ONGOING"]

        total_units = sum(int(r.get("total_units") or 0) for r in rows)
        total_sa = sum(float(r.get("saleable_area_sqft") or 0) for r in rows)
        ongoing_units = sum(int(r.get("total_units") or 0) for r in ongoing)

        # Communities by activity
        communities: dict[str, int] = {}
        for r in rows:
            c = r.get("community") or "unknown"
            communities[c] = communities.get(c, 0) + 1

        # Top 5 ongoing projects by saleable area (= flagship signal)
        top_ongoing = sorted(
            ongoing,
            key=lambda r: float(r.get("saleable_area_sqft") or 0),
            reverse=True,
        )[:5]

        return {
            "loaded": True,
            "total_projects": len(rows),
            "completed_count": len(completed),
            "ongoing_count": len(ongoing),
            "total_units": total_units,
            "total_saleable_sqft": int(total_sa),
            "ongoing_units": ongoing_units,
            "communities": communities,
            "flagship_ongoing": [
                {
                    "name": p.get("project_name"),
                    "community": p.get("community"),
                    "units": p.get("total_units"),
                    "sa_sqft": int(float(p.get("saleable_area_sqft") or 0)),
                    "sold_pct": p.get("sold_pct"),
                    "avg_psf": p.get("avg_sold_psf_aed"),
                }
                for p in top_ongoing
            ],
        }

    return _cached("summary", _build)


def get_projects_in_region(region: str) -> list[dict[str, Any]]:
    """All Sobha projects in a given region. Region values match
    classifier output: dubai | abu_dhabi | usa | australia | other.
    """
    region = (region or "").lower()
    return _cached(
        f"region:{region}",
        lambda: [
            r for r in _fetch_all_projects()
            if (r.get("region") or "dubai").lower() == region
        ],
    )


def get_competitive_overlap(dev_slug: str | None) -> dict[str, Any]:
    """Compute Sobha's exposure to a specific competitor's move.

    Returns a dict suitable for direct injection into LLM prompts:
      {
        "dev_slug": "emaar",
        "regions": ["dubai"],
        "communities_overlap": [...],
        "adjacent_to_sobha": [...],   # Sobha projects/communities adjacent
        "segment_overlap": [...],
        "notes": "...",                # one-line analyst priors
        "sobha_projects_in_region": [  # actual MIS projects in that region
          {"name": ..., "community": ..., "units": ..., "sold_pct": ...},
          ...
        ],
      }

    Empty/missing dev_slug → returns a "no overlap data" dict so the
    LLM can fall back gracefully.
    """
    if not dev_slug:
        return {
            "dev_slug": None,
            "loaded": False,
            "note": "No developer slug provided — assess generically.",
        }

    base = _COMPETITOR_OVERLAP.get(dev_slug.lower(), {
        "regions": [],
        "communities_overlap": [],
        "adjacent_to_sobha": [],
        "segment_overlap": [],
        "notes": f"Developer {dev_slug} is not in Sobha's tracked-competitor map; treat as adjacent or background signal.",
    })

    # Augment with actual Sobha projects in the overlapping region(s)
    sobha_in_region: list[dict[str, Any]] = []
    for r in base.get("regions", []):
        for p in get_projects_in_region(r):
            sobha_in_region.append({
                "name": p.get("project_name"),
                "community": p.get("community"),
                "status": p.get("status"),
                "units": p.get("total_units"),
                "sold_pct": p.get("sold_pct"),
                "avg_psf_aed": p.get("avg_sold_psf_aed"),
            })

    # Adjacency-aware exposure (Tier 1 location intelligence). For each
    # competitor sub-market in communities_overlap, find Sobha projects
    # whose own community is adjacent. Captures exposures the curated
    # adjacent_to_sobha list might miss.
    adjacencies_found: list[dict[str, str]] = []
    for competitor_area in base.get("communities_overlap", []):
        adj_set = set(get_adjacent_communities(competitor_area))
        if not adj_set:
            continue
        for proj in sobha_in_region:
            proj_comm = (proj.get("community") or "").strip().lower()
            if proj_comm and proj_comm in adj_set:
                adjacencies_found.append({
                    "competitor_area": competitor_area,
                    "sobha_project": proj.get("name"),
                    "sobha_community": proj.get("community"),
                    "relation": "adjacent (shared boundary / demand corridor)",
                })

    return {
        "dev_slug": dev_slug,
        "loaded": True,
        "regions": base.get("regions", []),
        "communities_overlap": base.get("communities_overlap", []),
        "adjacent_to_sobha": base.get("adjacent_to_sobha", []),
        "segment_overlap": base.get("segment_overlap", []),
        "notes": base.get("notes", ""),
        "sobha_projects_in_region": sobha_in_region,
        "geographic_adjacencies": adjacencies_found,
    }


# ─── LLM-prompt formatters ──────────────────────────────────────────
def format_for_llm(scope: str = "summary", region: str | None = None) -> str:
    """Compact text block suitable for prepending to a system prompt.

    Scopes:
      - "summary"  (~120 tokens)  — global portfolio digest
      - "regional" (~250 tokens)  — region-specific projects + summary
      - "full"     (~500 tokens)  — every project, name + community + status

    `region` only matters for scope="regional".
    """
    s = get_portfolio_summary()

    if not s.get("loaded"):
        return (
            "Sobha portfolio context: NOT YET WIRED "
            "(scripts/sync_sobha_mis.py hasn't run). "
            "Reason about competitor signals generically; do not invent Sobha projects."
        )

    if scope == "kpi_lens":
        # Standalone KPI framework — pair this with "summary" or
        # "regional" when a stage needs both anchor + framework.
        return KPI_LENS

    if scope == "summary":
        flagship_lines = "\n".join(
            f"  - {p['name']} ({p['community']}, {p['units']} units, "
            f"sold {(p['sold_pct'] or 0)*100:.0f}%, "
            f"avg AED {int(p['avg_psf'] or 0)}/sqft)"
            for p in s["flagship_ongoing"]
        )
        return (
            f"REFERENCE FRAME (Sobha portfolio — anchor only, never the subject):\n"
            f"{s['total_projects']} projects "
            f"({s['completed_count']} completed, {s['ongoing_count']} ongoing), "
            f"{s['total_units']:,} total units across "
            f"{', '.join(list(s['communities'].keys())[:6])}.\n"
            f"Flagship ongoing baselines:\n{flagship_lines}\n\n"
            f"Use this frame to: (a) filter signal relevance, (b) compute "
            f"benchmarks (competitor PSF / sold% vs Sobha comparable), "
            f"(c) flag exposure by community. NEVER make Sobha the subject."
        )

    if scope == "regional":
        region = (region or "dubai").lower()
        projects = get_projects_in_region(region)
        if not projects:
            return (
                f"Sobha portfolio (region={region}): no current projects in MIS. "
                f"Sobha's planned expansion includes this region but no live "
                f"product. Treat competitor moves here as watch-list, not "
                f"head-to-head."
            )
        lines = "\n".join(
            f"  - {p.get('project_name')} ({p.get('community')}, "
            f"{p.get('total_units') or '?'} units, "
            f"{p.get('status', '?')})"
            for p in projects[:15]
        )
        return (
            f"Sobha portfolio in {region}: {len(projects)} projects.\n{lines}"
        )

    if scope == "full":
        all_p = _fetch_all_projects()
        lines = "\n".join(
            f"  - {p.get('project_name')} ({p.get('community')}, "
            f"{p.get('status')}, {p.get('total_units') or '?'} units)"
            for p in all_p[:50]
        )
        return f"Sobha full portfolio ({len(all_p)} projects):\n{lines}"

    return format_for_llm("summary")


def format_overlap_for_llm(dev_slug: str | None) -> str:
    """Format `get_competitive_overlap()` as a system-prompt context block.

    Drop straight into the user message of investigation/deep-dive prompts.
    """
    o = get_competitive_overlap(dev_slug)
    if not o.get("loaded"):
        return f"Sobha competitive overlap for {dev_slug or '?'}: not mapped. Reason generically."

    lines = []
    if o.get("regions"):
        lines.append(f"  Regions where both operate: {', '.join(o['regions'])}")
    if o.get("communities_overlap"):
        lines.append(f"  Sub-market overlap: {'; '.join(o['communities_overlap'])}")
    if o.get("adjacent_to_sobha"):
        lines.append(f"  Sobha projects/communities adjacent: {'; '.join(o['adjacent_to_sobha'])}")
    if o.get("segment_overlap"):
        lines.append(f"  Product segment overlap: {'; '.join(o['segment_overlap'])}")
    if o.get("sobha_projects_in_region"):
        proj_names = [p["name"] for p in o["sobha_projects_in_region"][:8] if p.get("name")]
        if proj_names:
            lines.append(f"  Sobha live projects in region: {', '.join(proj_names)}")
    if o.get("geographic_adjacencies"):
        adj_lines = [
            f"    {a['sobha_project']} ({a['sobha_community']}) ↔ {a['competitor_area']}"
            for a in o["geographic_adjacencies"][:8]
        ]
        if adj_lines:
            lines.append("  Adjacency-based exposure (shared boundary / demand corridor):")
            lines.extend(adj_lines)
    if o.get("notes"):
        lines.append(f"  Analyst notes: {o['notes']}")

    body = "\n".join(lines) if lines else "  (no specific overlap mapped)"
    return f"Sobha competitive overlap with {dev_slug}:\n{body}"


# ─── Sync entrypoint (called by scripts/sync_sobha_mis.py) ──────────
def sync_from_xlsx(path: str) -> int:
    """Parse the MIS xlsx, upsert rows into sobha_projects, return count.

    Joins '💰 Finance MIS Input' + '📊 Dev MIS Input' on Project Name.
    Dev sheet is authoritative for shape (units/area); Finance sheet adds
    sales/pricing/margin. Either sheet alone is sufficient — missing
    fields land as null.
    """
    try:
        import openpyxl
    except ImportError as e:
        raise RuntimeError("openpyxl is required: pip install openpyxl") from e

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)

    finance_sheet = next((s for s in wb.sheetnames if "Finance MIS" in s), None)
    dev_sheet = next((s for s in wb.sheetnames if "Dev MIS" in s), None)
    if not finance_sheet and not dev_sheet:
        raise RuntimeError(f"MIS xlsx missing required sheets. Found: {wb.sheetnames}")

    def _read_sheet(name: str) -> list[dict[str, Any]]:
        if not name:
            return []
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [str(h).strip() if h is not None else "" for h in rows[0]]
        out = []
        for raw in rows[1:]:
            # Skip the "UPDATE YELLOW FIELDS" instruction row + blank rows
            first = raw[0]
            if first is None:
                continue
            first_s = str(first).strip()
            if not first_s or first_s.startswith("⬇") or first_s.startswith("─"):
                continue
            out.append({
                headers[i]: raw[i]
                for i in range(min(len(headers), len(raw)))
                if headers[i]
            })
        return out

    dev_rows = _read_sheet(dev_sheet)
    fin_rows = _read_sheet(finance_sheet)

    # Index by project_name
    dev_by_name = {r.get("Project Name"): r for r in dev_rows if r.get("Project Name")}
    fin_by_name = {r.get("Project Name"): r for r in fin_rows if r.get("Project Name")}

    all_names = set(dev_by_name.keys()) | set(fin_by_name.keys())
    log.info("[sobha_context] sync: dev_rows=%d finance_rows=%d unique=%d",
             len(dev_by_name), len(fin_by_name), len(all_names))

    # Build upsert payload
    payload: list[dict[str, Any]] = []
    for name in sorted(all_names):
        d = dev_by_name.get(name) or {}
        f = fin_by_name.get(name) or {}

        community = (d.get("Community") or f.get("Community") or "").strip() or "Unknown"

        # Default region: Dubai for now (the MIS only contains Dubai
        # projects). When Sobha launches in AD/Texas/Brisbane, the MIS
        # will gain those rows; we'd extend the inference here.
        region = "dubai"

        status = (d.get("Status") or f.get("Status") or "").strip() or "UNKNOWN"

        # Launch date — varies in shape ("01-May-16" string vs datetime).
        launch_raw = f.get("Launch Date") or d.get("Launch Date")
        launch_iso: str | None = None
        if launch_raw is not None:
            try:
                if hasattr(launch_raw, "isoformat"):
                    launch_iso = launch_raw.date().isoformat() if hasattr(launch_raw, "date") else launch_raw.isoformat()
                else:
                    # Try string parsing a few formats
                    from datetime import datetime as _dt
                    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                        try:
                            launch_iso = _dt.strptime(str(launch_raw).strip(), fmt).date().isoformat()
                            break
                        except ValueError:
                            continue
            except Exception:
                launch_iso = None

        def _num(d: dict, key: str) -> float | None:
            v = d.get(key)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _int(d: dict, key: str) -> int | None:
            v = _num(d, key)
            return int(v) if v is not None else None

        row = {
            "project_name": name,
            "community": community,
            "region": region,
            "status": status,
            "launch_date": launch_iso,

            # Dev MIS
            "total_units": _int(d, "Total Units"),
            "resi_units": _int(d, "Resi Units"),
            "retail_other": _int(d, "Retail/Other"),
            "plot_area_sqft": _num(d, "Plot Area (sft)"),
            "gfa_sqft": _num(d, "GFA (sft)"),
            "bua_sqft": _num(d, "BUA (sft)"),
            "saleable_area_sqft": _num(d, "Saleable Area (sft)"),
            "far": _num(d, "FAR"),
            "sa_per_gfa": _num(d, "SA/GFA"),
            "sa_per_bua": _num(d, "SA/BUA"),

            # Finance MIS
            "project_topline_aed": _num(f, "Project Topline (AED)"),
            "sa_launched_sqft": _num(f, "SA Launched (sft)"),
            "total_sales_aed": _num(f, "Total Sales till Date (AED)"),
            "sold_sa_sqft": _num(f, "Sold SA (sft)"),
            "sold_pct": _num(f, "Sold %"),
            "avg_sold_psf_aed": _num(f, "Avg Sold Rate (PSF)"),
            "unsold_sa_sqft": _num(f, "Unsold SA (sft)"),
            "unsold_pct": _num(f, "Unsold %"),
            "unsold_value_aed": _num(f, "Unsold Value (AED)"),
            "construction_pct": _num(f, "POCM / Construction %"),
            "gp_pct_inception": _num(f, "GP% Since Inception"),
            "gp_pct_fy2026_ytd": _num(f, "GP% FY2026 YTD"),
            "ytd_collection_aed": _num(f, "YTD Collection (AED)"),
            "revenue_recognized_aed": _num(f, "Revenue Recognized YTD (AED)"),

            "raw_dev_json": {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                              for k, v in d.items()} if d else None,
            "raw_finance_json": {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                                  for k, v in f.items()} if f else None,
        }
        payload.append(row)

    if not payload:
        log.warning("[sobha_context] sync produced 0 rows — check sheet names")
        return 0

    sb.client().table("sobha_projects").upsert(payload, on_conflict="project_name").execute()
    invalidate_cache()
    log.info("[sobha_context] synced %d projects to Supabase", len(payload))
    return len(payload)
