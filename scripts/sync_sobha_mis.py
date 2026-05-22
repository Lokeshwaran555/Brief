"""One-shot CLI runner: parse Sobha MIS xlsx → upsert sobha_projects.

Usage:
    python scripts/sync_sobha_mis.py \\
        --file ~/Downloads/Sobha_MIS_Consolidated_Mar2026.xlsx

Run after every monthly MIS update. Idempotent (upsert on project_name);
re-running is safe and updates KPIs in place.

Reads Supabase creds from settings (so .env or Railway env both work).
Migration 012_sobha_projects.sql must already be applied.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root: python scripts/sync_sobha_mis.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import sobha_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Sobha MIS xlsx → sobha_projects table.")
    parser.add_argument("--file", required=True, help="Path to Sobha_MIS_Consolidated_*.xlsx")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    print(f"Syncing from {path} ...")
    count = sobha_context.sync_from_xlsx(str(path))
    print(f"OK — {count} projects upserted into sobha_projects.")

    summary = sobha_context.get_portfolio_summary()
    if summary.get("loaded"):
        print(
            f"  total_units={summary['total_units']:,}  "
            f"completed={summary['completed_count']}  "
            f"ongoing={summary['ongoing_count']}  "
            f"saleable={summary['total_saleable_sqft']:,} sqft"
        )
        print("  flagship ongoing:")
        for p in summary["flagship_ongoing"]:
            print(
                f"    - {p['name']} ({p['community']}, {p['units']} units, "
                f"sold {(p['sold_pct'] or 0)*100:.0f}%, AED {int(p['avg_psf'] or 0)}/sqft)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
