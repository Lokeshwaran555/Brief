"""Google News early-signal scout.

Same query set as api/intel.js in the dashboard. Fetches 7 narrow RSS
queries tuned for leak-vocabulary (pre-launch, EOI, commission bumps,
hiring, handover delays). Returns deduped raw items ready for scoring.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any
from urllib.parse import quote, urlparse

import feedparser
import httpx

log = logging.getLogger(__name__)


def _gnews(q: str, *, gl: str = "AE", hl: str = "en") -> str:
    """gl is country (drives ranking + locale); hl is language."""
    ceid = f"{gl}:{hl}"
    return f"https://news.google.com/rss/search?q={quote(q)}&hl={hl}&gl={gl}&ceid={ceid}"


# Region-tagged query set. Each query stamps its region + country at
# scout time so promotion can route signals to the right MD Scan tab
# without waiting on the LLM classifier. Dubai queries unchanged
# from the original v2 set; AD / USA / AU / global added 2026-04-25.
QUERIES: list[dict[str, str]] = [
    # ─── DUBAI (existing v2 set) ─────────────────────────────────
    {
        "tag": "pre-launch",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("pre-launch" OR "pre launch" OR "EOI" OR "Expression of Interest") '
            "Dubai (Emaar OR DAMAC OR Aldar OR Nakheel OR Meraas OR Binghatti OR "
            "Azizi OR Omniyat OR Ellington) when:7d", gl="AE",
        ),
    },
    {
        "tag": "private-sale",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("friends and family" OR "private sale" OR "exclusive access" OR '
            '"invite only") Dubai property when:7d', gl="AE",
        ),
    },
    {
        "tag": "commission",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("broker commission" OR "5% commission" OR "6% commission" OR '
            '"commission increase") Dubai real estate when:14d', gl="AE",
        ),
    },
    {
        "tag": "discount",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("payment plan" OR "post-handover" OR "DLD waiver" OR '
            '"service charge waiver" OR "cashback") Dubai (Emaar OR DAMAC OR '
            "Aldar OR Nakheel OR Binghatti OR Azizi) when:14d", gl="AE",
        ),
    },
    {
        "tag": "hiring",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("Head of Sales" OR "VP Sales" OR "Sales Director" OR '
            '"Chief Commercial") Dubai developer hiring when:30d', gl="AE",
        ),
    },
    {
        "tag": "delay",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '(delay OR "handover delay" OR "escrow issue" OR "stalled") Dubai '
            "(tower OR project OR community) when:14d", gl="AE",
        ),
    },
    {
        "tag": "launch-date",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("launching" OR "new tower" OR "master plan" OR "phase 2") Dubai '
            "(Emaar OR DAMAC OR Aldar OR Nakheel OR Meraas OR Binghatti OR Omniyat) when:7d", gl="AE",
        ),
    },
    # ─── UAE VISA / RESIDENCY POLICY (added 2026-04-30) ──────────
    # MD ask: "regulatory policy of changing visa thresholds" needs
    # to surface. Existing scouts had golden_visa as a substring
    # filter on WAM/Zawya only — this query catches mainstream UAE
    # press coverage of any visa-threshold or residency-tier change.
    {
        "tag": "uae-visa-policy",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("Golden Visa" OR "property visa" OR "investor visa" OR '
            '"residency visa" OR "long-term visa" OR "10-year visa" OR '
            '"5-year visa" OR "visa threshold" OR "visa eligibility" OR '
            '"residence permit" OR "remote work visa" OR "digital nomad visa" OR '
            '"retirement visa") (UAE OR Dubai OR "Abu Dhabi") when:30d',
            gl="AE",
        ),
    },
    {
        "tag": "uae-property-residency",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '(UAE OR Dubai) ("property residency" OR "real estate visa" OR '
            '"AED 2 million" OR "AED 750,000" OR "AED 1 million" OR '
            '"residency reform" OR "freehold expansion") when:30d', gl="AE",
        ),
    },
    # ─── CONTECH INNOVATION (added 2026-04-30 round 5) ───────────
    # MD wants the innovation half of construction tech: prefab,
    # modular, 3D-print, robotics, fast-build records, AI-for-design.
    # The materials_scout covers cost; this catches methodology.
    {
        "tag": "contech-innovation",
        "region": None, "country_code": None,
        "url": _gnews(
            '("prefabricated" OR "modular construction" OR "off-site construction" OR '
            '"3D printed building" OR "3D-printed tower" OR "construction robot" OR '
            '"robotic mason" OR "bricklaying robot" OR "concrete printing" OR '
            '"built in 14 days" OR "built in 30 days" OR "fastest tower" OR '
            '"speed build" OR "rapid construction" OR "tower in days") '
            '("real estate" OR construction OR builder OR developer) when:30d',
        ),
    },
    {
        "tag": "contech-uae",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '(UAE OR Dubai OR "Abu Dhabi" OR "Saudi Arabia" OR Riyadh OR NEOM) '
            '("3D printed" OR "modular" OR "prefab" OR "robot" OR "AI design" OR '
            '"digital twin" OR "BIM" OR "off-site construction" OR "speed build") '
            '(building OR tower OR project) when:30d', gl="AE",
        ),
    },
    # ─── AI FRONTIER (added 2026-04-30 round 5) ──────────────────
    # MD wants Claude / GPT / Anthropic / OpenAI moves alongside
    # PropTech / ConTech. This is global tech-trend awareness, not
    # RE-anchored — domain whitelist on the gnews source is the
    # filter (these queries hit Google News which is broad).
    {
        "tag": "ai-frontier",
        "region": None, "country_code": None,
        "url": _gnews(
            '(Anthropic OR Claude OR OpenAI OR "GPT-5" OR "GPT-6" OR '
            '"GPT-4o" OR "model launch" OR "new model release" OR '
            '"frontier AI" OR "AI breakthrough" OR DeepMind OR Mistral OR '
            'Perplexity OR xAI OR Grok OR "Hugging Face" OR Cohere OR '
            '"agentic AI") when:7d',
        ),
    },
    {
        "tag": "ai-funding-2026",
        "region": None, "country_code": None,
        "url": _gnews(
            '(AI OR "artificial intelligence") '
            '("Series B" OR "Series C" OR "Series D" OR "valuation" OR '
            '"raised" OR "fundraise" OR "funding round" OR "$100M" OR '
            '"$500M" OR "$1B" OR "$50B" OR "billion valuation") when:14d',
        ),
    },
    # ─── ABU DHABI ───────────────────────────────────────────────
    # Expanded 2026-05-08 from 3 → 10 queries per stakeholder
    # feedback that AD coverage was structurally thin (only 2 of 200
    # signals in /api/intel were AD-tagged). Mirrors the Dubai
    # vocabulary — pre-launch, partnerships, capital, talent, distress,
    # Golden Visa, infrastructure — pinned to AD developers + areas.
    {
        "tag": "ad-launches",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("Abu Dhabi" AND ("launch" OR "phase" OR "pre-launch" OR "master plan" OR "off-plan" OR "ground-breaking")) '
            "(Aldar OR Mubadala OR IHC OR ADQ OR Reportage OR Modon OR Eagle Hills OR Imkan OR Bloom OR Tamouh) when:14d", gl="AE",
        ),
    },
    {
        "tag": "ad-capital",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("Abu Dhabi" AND (sukuk OR "capital raise" OR "joint venture" OR "M&A" OR acquisition OR "bond issuance" OR REIT OR IPO)) '
            "(Aldar OR Mubadala OR IHC OR ADQ OR ADIA OR ADX) when:30d", gl="AE",
        ),
    },
    {
        "tag": "ad-saadiyat",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("Saadiyat Island" OR "Saadiyat Cultural District" OR "Louvre Abu Dhabi" OR "Guggenheim Abu Dhabi" OR "Saadiyat Beach" OR "Saadiyat Grove") '
            "(launch OR sold OR partnership OR development OR residence OR villa OR apartment) when:30d", gl="AE",
        ),
    },
    {
        "tag": "ad-yas-reem",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("Yas Island" OR "Yas Bay" OR "Yas Acres" OR "Reem Island" OR "Al Maryah Island" OR "Al Reem" OR "Hudayriyat" OR "Al Jurf") '
            "(launch OR development OR residence OR tower OR community) when:30d", gl="AE",
        ),
    },
    {
        "tag": "ad-aldar",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            'Aldar (Properties OR Investments OR Education OR Estates) '
            "(launch OR partnership OR acquisition OR JV OR development OR sukuk OR IPO OR earnings OR dividend) when:30d", gl="AE",
        ),
    },
    {
        "tag": "ad-mubadala-ihc-adq",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            "(Mubadala OR IHC OR \"International Holding Company\" OR ADQ OR ADIA) "
            '(real estate OR property OR developer OR REIT OR "joint venture" OR acquisition OR launch) when:30d', gl="AE",
        ),
    },
    {
        "tag": "ad-modon-reportage",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '(Modon OR "Modon Properties" OR Reportage OR "Reportage Properties" OR "Eagle Hills Abu Dhabi" OR Imkan OR Bloom OR Tamouh) '
            "(launch OR project OR development OR partnership OR sales) when:30d", gl="AE",
        ),
    },
    {
        "tag": "ad-adgm-hub71",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("ADGM" OR "Abu Dhabi Global Market" OR "Hub71" OR "ADX" OR "Abu Dhabi Securities Exchange") '
            "(license OR fund OR REIT OR IPO OR listing OR fintech OR proptech) when:30d", gl="AE",
        ),
    },
    {
        "tag": "ad-policy-visa",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("Abu Dhabi" OR ADGM) ("Golden Visa" OR "investor visa" OR "freehold" OR "property residency" OR "DCT" OR "Department of Culture and Tourism") when:30d', gl="AE",
        ),
    },
    {
        "tag": "ad-infrastructure",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("Abu Dhabi" OR "Etihad Rail" OR "Al Maktoum") ("infrastructure" OR "metro" OR "airport expansion" OR "smart city" OR "Hudayriyat" OR "rail corridor") when:30d', gl="AE",
        ),
    },
    # ─── USA (Texas first, nationwide multifamily / REIT) ────────
    {
        "tag": "us-texas",
        "region": "usa", "country_code": "US",
        "url": _gnews(
            '(Texas OR Austin OR Dallas OR Houston) ("multifamily" OR "build-to-rent" OR "Class A apartments" OR "land deal") when:7d', gl="US",
        ),
    },
    {
        "tag": "us-reit-activity",
        "region": "usa", "country_code": "US",
        "url": _gnews(
            '("8-K filing" OR "REIT" OR "build-to-rent") (Camden OR "Mid-America" OR "Mill Creek" OR "Tricon Residential" OR Greystar) when:14d', gl="US",
        ),
    },
    {
        "tag": "us-proptech",
        "region": "usa", "country_code": "US",
        "url": _gnews(
            '("PropTech" OR "AI walkthrough" OR "modular construction" OR "construction tech") (US OR "United States" OR Texas) when:14d', gl="US",
        ),
    },
    # ─── AUSTRALIA (Brisbane priority + Sydney + Melbourne + national)
    # Expanded 2026-05-07 — stakeholder feedback that the Australia
    # tab was empty. Original 3 queries were too narrow; this set adds
    # Sydney/Melbourne city-level coverage, broader developer set,
    # macro (RBA rates, housing affordability), foreign investment,
    # and a generic gl=AU sweep so editorial coverage rolls up.
    {
        "tag": "au-brisbane",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '(Brisbane OR Queensland) ("property launch" OR development OR "luxury apartments" OR "Olympic infrastructure" OR "off the plan" OR apartments OR "house prices") when:14d', gl="AU",
        ),
    },
    {
        "tag": "au-sydney",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '(Sydney OR "New South Wales" OR NSW) ("property launch" OR development OR "luxury apartments" OR "off the plan" OR "house prices" OR auction OR "median price") when:14d', gl="AU",
        ),
    },
    {
        "tag": "au-melbourne",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '(Melbourne OR Victoria) (property OR "apartment market" OR "house prices" OR "build-to-rent" OR development OR auction) when:14d', gl="AU",
        ),
    },
    {
        "tag": "au-developers",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '(Mirvac OR Stockland OR Lendlease OR "Goodman Group" OR "Charter Hall" OR Meriton OR "Frasers Property" OR "GPT Group" OR "Dexus" OR "Crown Group") '
            "(launch OR partnership OR acquisition OR REIT OR project OR DA OR development) when:14d", gl="AU",
        ),
    },
    {
        "tag": "au-prefab",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '("modular construction" OR "prefab" OR "build-to-rent" OR "BTR sector" OR "social housing") Australia when:30d', gl="AU",
        ),
    },
    {
        "tag": "au-rates-housing",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '("RBA" OR "Reserve Bank of Australia" OR "cash rate" OR "interest rate" OR "mortgage rate" OR "housing affordability" OR "first home buyer") (Australia OR Sydney OR Melbourne OR Brisbane) when:14d', gl="AU",
        ),
    },
    {
        "tag": "au-foreign-investment",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '("FIRB" OR "foreign investment review" OR "Chinese investor" OR "Indian investor" OR "Singapore investor") (Australia OR property OR "real estate") when:30d', gl="AU",
        ),
    },
    {
        "tag": "au-luxury",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '(Australia OR Sydney OR Melbourne OR Brisbane OR Perth OR "Gold Coast") ("luxury home" OR "penthouse" OR "branded residence" OR "ultra-luxury" OR "trophy home" OR "record sale") when:30d', gl="AU",
        ),
    },
    {
        "tag": "au-general-realestate",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '("real estate" OR "property market" OR "housing market" OR "auction clearance" OR "new launch" OR "off the plan") (Australia OR Sydney OR Melbourne OR Brisbane OR Perth) when:7d', gl="AU",
        ),
    },
    # ─── GLOBAL / EMERGING (catch-all; classifier fills country) ──
    {
        "tag": "global-luxury",
        "region": None, "country_code": None,
        "url": _gnews(
            '("luxury residences" OR "branded residences" OR "ultra-luxury") (launch OR partnership OR "sold out") when:14d',
        ),
    },
    {
        "tag": "global-tech",
        "region": None, "country_code": None,
        "url": _gnews(
            '("real estate" OR "real-estate") ("AI" OR "PropTech" OR "blockchain" OR "tokenisation") when:14d',
        ),
    },
    # ─── QUERY GAPS (added 2026-04-26 per source audit) ────────
    # Pre-construction signal — land bank moves usually announce
    # 6-12 months before a project launch.
    {
        "tag": "land-bank-dubai",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("land bank" OR "land plot" OR "land deal" OR "G+P" OR "G+B+P+R") Dubai when:14d', gl="AE",
        ),
    },
    {
        "tag": "land-bank-ad",
        "region": "abu_dhabi", "country_code": "AE",
        "url": _gnews(
            '("land bank" OR "land plot" OR "land deal" OR "land allocation") "Abu Dhabi" when:14d', gl="AE",
        ),
    },
    # EPC / main contractor awards — schedule + scale before
    # marketing campaign starts.
    {
        "tag": "epc-uae",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("EPC contract" OR "main contractor awarded" OR "construction tender" OR "contract awarded") (Dubai OR "Abu Dhabi") when:14d', gl="AE",
        ),
    },
    {
        "tag": "epc-us",
        "region": "usa", "country_code": "US",
        "url": _gnews(
            '("general contractor" OR "construction tender" OR "GMP contract") (multifamily OR "luxury residential") (Texas OR USA) when:14d', gl="US",
        ),
    },
    # Branded residences — partnership window often opens months
    # before announcement.
    {
        "tag": "branded-residence-global",
        "region": None, "country_code": None,
        "url": _gnews(
            '("branded residence" OR "fashion house residence" OR "marque licensing" OR "couture residence" OR "auto-brand residence") (launch OR partnership OR licensing) when:30d',
        ),
    },
    # Capital structure — green sukuk, ESG bond, sustainability
    # bond, REIT IPO are early signals of expansion narratives.
    {
        "tag": "capital-mena",
        "region": None, "country_code": None,
        "url": _gnews(
            '("green sukuk" OR "ESG sukuk" OR "sustainability bond" OR "perpetual sukuk" OR "REIT IPO") (UAE OR "Saudi Arabia" OR Qatar OR Bahrain) when:30d',
        ),
    },
    # Off-market / private placement — capital flow tells often
    # appear in trade press before official announcement.
    {
        "tag": "off-market-dubai",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("off-market" OR "private placement" OR "invite only" OR "off-market transaction" OR "block deal") Dubai property when:14d', gl="AE",
        ),
    },
    # Soft launches / VIP previews — earliest pricing signal.
    {
        "tag": "soft-launch-dubai",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("soft launch" OR "VIP preview" OR "broker preview" OR "exclusive launch event" OR "limited release") Dubai when:7d', gl="AE",
        ),
    },
    # Secondary market / assignment fees — leading indicator of
    # off-plan demand softening.
    {
        "tag": "secondary-market-dubai",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("assignment fee" OR "flip" OR "secondary market" OR "reassignment" OR "title transfer") Dubai property when:14d', gl="AE",
        ),
    },
    # Conferences — Cityscape / MIPIM / IPS attendance reveals who's
    # planning launches in the next quarter.
    {
        "tag": "conferences-mena",
        "region": None, "country_code": None,
        "url": _gnews(
            '("Cityscape Global" OR "Cityscape Dubai" OR "MIPIM" OR "International Property Show" OR "IPS Dubai") (Dubai OR "Abu Dhabi") when:30d',
        ),
    },
    # Buyer-mix / nationality — drives Dubai HNW demand cycles.
    {
        "tag": "buyer-mix-dubai",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("buyer nationality" OR "Indian HNW" OR "Russian HNW" OR "Chinese HNW" OR "British HNW" OR "Pakistani investor") Dubai property when:30d', gl="AE",
        ),
    },
    # ─── ROUND 2 GAP FILL (added 2026-04-26 per source audit §B) ──
    # DLD/registration leakage — earliest off-plan-marketing signal.
    {
        "tag": "dld-leakage",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("DLD waived" OR "registration waived" OR "oqood issued" OR "escrow opened" OR "Trakheesi") Dubai when:14d', gl="AE",
        ),
    },
    # Project rebrand / relaunch — competitor pricing-reset signal.
    {
        "tag": "rebrand",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '(rebranded OR relaunched OR "new identity" OR "renamed") Dubai (tower OR project OR community) when:30d', gl="AE",
        ),
    },
    # Sell-through claims — absorption-velocity proxy.
    {
        "tag": "sold-out",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("sold out" OR "100% sold" OR "fully subscribed" OR "oversubscribed") Dubai (Emaar OR DAMAC OR Aldar OR Sobha OR Nakheel OR Binghatti OR Azizi OR Omniyat) when:14d', gl="AE",
        ),
    },
    # SWF / sovereign capital movements into UAE real estate.
    {
        "tag": "swf-mena",
        "region": None, "country_code": None,
        "url": _gnews(
            '(PIF OR Mubadala OR ADIA OR QIA OR ADQ OR GIC) ("real estate" OR property OR "Dubai" OR "Abu Dhabi") (acquired OR "joint venture" OR "anchor investor" OR "first close") when:30d',
        ),
    },
    # Distress / restructuring tells — capital-pressure read.
    {
        "tag": "distress-uae",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '(distressed OR "loan modification" OR "covenant breach" OR restructuring OR "credit watch") (Emaar OR DAMAC OR Aldar OR Nakheel OR Damac OR "Dubai developer") when:30d', gl="AE",
        ),
    },
    # Construction milestone — handover-window forecasting.
    {
        "tag": "construction-milestone",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("topping out" OR "structural completion" OR "DEWA energization" OR "fire-safety approval" OR "Building Completion Certificate") Dubai when:14d', gl="AE",
        ),
    },
    # Texas multifamily concessions — lease-up-pressure leading indicator.
    {
        "tag": "us-concessions",
        "region": "usa", "country_code": "US",
        "url": _gnews(
            '("rent concessions" OR "month free" OR "two months free" OR "lease-up" OR "merchant builder" OR "cap rate compression") (Austin OR Dallas OR Houston OR "Sun Belt") multifamily when:30d', gl="US",
        ),
    },
    # Brisbane DA / Olympic-corridor signals.
    {
        "tag": "au-da-olympics",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '("DA approved" OR "Development Application" OR "Olympics 2032" OR "off-the-plan" OR "stamp duty Queensland") Brisbane when:30d', gl="AU",
        ),
    },
    # Regulatory frame shifts — opens/closes whole markets.
    {
        "tag": "reg-frame-emerging",
        "region": None, "country_code": None,
        "url": _gnews(
            '("freehold law" OR "expat ownership" OR "REIT framework" OR "REIT regulation" OR "FDI cap" OR "foreign ownership cap") (passed OR raised OR introduced OR amended) when:60d',
        ),
    },
    # Talent churn at competitors — restructuring forerunner.
    {
        "tag": "talent-churn",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("resigned" OR "stepped down" OR "new CEO" OR "appointed CEO" OR "Chief Sales Officer" OR "Chief Marketing Officer") (Emaar OR DAMAC OR Aldar OR Sobha OR Nakheel OR Meraas OR Binghatti) when:30d', gl="AE",
        ),
    },
    # ─── INFRASTRUCTURE / MEGAPROJECT COVERAGE (added 2026-04-26) ──
    # Major infrastructure announcements (metro extensions, rail
    # corridors, smart-city scope, port/airport expansion) shift land
    # values + buyer geography for years. The Gold Line Metro extension
    # was missed by the prior query set — these queries close that
    # vocabulary gap. Wider when:30d window because press cycle for
    # infrastructure announcements is slower than tactical RE moves.
    {
        "tag": "transit-uae",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("Gold Line" OR "Blue Line" OR "Etihad Rail" OR "RTA corridor" OR '
            '"Dubai Metro extension" OR "Dubai Loop" OR "Hyperloop UAE" OR '
            '"tram extension" OR "monorail") (Dubai OR UAE OR Sharjah) when:30d', gl="AE",
        ),
    },
    {
        "tag": "infra-dubai",
        "region": "dubai", "country_code": "AE",
        "url": _gnews(
            '("airport expansion" OR "DXB" OR "Al Maktoum" OR "Jebel Ali" OR '
            '"smart city" OR "Expo City" OR "infrastructure project" OR '
            '"megaproject") Dubai when:30d', gl="AE",
        ),
    },
    {
        "tag": "megaprojects-mena",
        "region": None, "country_code": None,
        "url": _gnews(
            '("NEOM" OR "AlUla" OR "Mukaab" OR "Vision 2030" OR "Diriyah Gate" OR '
            '"megaproject" OR "smart city") (Saudi OR UAE OR Qatar OR MENA) when:30d',
        ),
    },
    {
        "tag": "transit-us",
        "region": "usa", "country_code": "US",
        "url": _gnews(
            '("high-speed rail" OR "Hyperloop" OR "Brightline" OR "transit '
            'megaproject" OR "rail corridor" OR "light rail") (Texas OR Florida OR USA) when:30d', gl="US",
        ),
    },
    {
        "tag": "transit-au",
        "region": "australia", "country_code": "AU",
        "url": _gnews(
            '("Brisbane Metro" OR "Cross River Rail" OR "Melbourne Metro" OR '
            '"Sydney Metro" OR "Olympic transit" OR "rail corridor") Australia when:30d', gl="AU",
        ),
    },
]


def _dedup_key(title: str, url: str) -> str:
    """sha256(title_head || source_domain) — stable across re-runs."""
    domain = urlparse(url).netloc.lower() if url else ""
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


async def _fetch_one(
    client: httpx.AsyncClient, query: dict[str, str]
) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            query["url"],
            headers={"User-Agent": "Mozilla/5.0 (SobhaMDI/1.0 intel-scout)"},
            timeout=8.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[gnews:%s] fetch failed: %s", query["tag"], e)
        return []

    parsed = feedparser.parse(resp.text)
    items: list[dict[str, Any]] = []
    for e in parsed.entries[:30]:
        title = (getattr(e, "title", "") or "").strip()[:300]
        url = (getattr(e, "link", "") or "").strip()
        if not title or not url:
            continue
        summary = (getattr(e, "summary", "") or "")[:600]
        date = (
            getattr(e, "published", None)
            or getattr(e, "updated", None)
            or None
        )
        items.append(
            {
                "source": f"gnews:{query['tag']}",
                "source_url": url,
                "title": title,
                "summary": summary,
                "raw_json": {
                    "query_tag": query["tag"],
                    "date": date,
                    # Region + country fall through to the signals row
                    # via ingest_flow._promote_to_signals. Null for
                    # global queries — classifier infers them.
                    "region": query.get("region"),
                    "country_code": query.get("country_code"),
                },
                "dedup_key": _dedup_key(title, url),
            }
        )
    return items


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        k = it["dedup_key"]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


async def run(limit: int = 40) -> list[dict[str, Any]]:
    """Scrape all queries in parallel, dedupe, return up to `limit` items."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch_one(client, q) for q in QUERIES], return_exceptions=False
        )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info("[gnews] fetched=%d deduped=%d", len(flat), len(deduped))
    return deduped[:limit]
