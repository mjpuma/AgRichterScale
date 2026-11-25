#!/usr/bin/env python3
"""Validate registry coverage, event identifiers and evidence metadata.

Usage: python scripts/validate_historical_registry.py
Requires only Python 3.8+ standard library.
This validates structural integrity, *not* the accuracy of historical claims.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "legacy": ROOT / "ancillary" / "event_attributes.csv",
    "registry": ROOT / "ancillary" / "historical_event_registry.csv",
    "evidence": ROOT / "ancillary" / "historical_loss_evidence_seed.csv",
}


def load(name):
    with PATHS[name].open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    legacy = load("legacy")
    registry = load("registry")
    evidence = load("evidence")
    errors = []
    legacy_ids = [row["event_name"].strip() for row in legacy]
    registry_ids = [row["event_id"].strip() for row in registry]
    obs_ids = [row["observation_id"].strip() for row in evidence]

    if len(legacy_ids) != len(set(legacy_ids)):
        errors.append("Duplicate legacy event identifiers")
    if len(registry_ids) != len(set(registry_ids)):
        errors.append("Duplicate event identifiers in registry")
    if len(obs_ids) != len(set(obs_ids)):
        errors.append("Duplicate observation identifiers in evidence")
    if set(legacy_ids) - set(registry_ids):
        errors.append("Missing legacy IDs: " + ", ".join(sorted(set(legacy_ids) - set(registry_ids))))
    if any(not row.get("event_id") or not row.get("candidate_status") or not row.get("year_start") for row in registry):
        errors.append("Registry rows missing required identifier/status/start year")
    for row in registry:
        if row.get("catalog_origin") == "event_attributes.csv" and row["event_id"] not in legacy_ids:
            errors.append("Non-legacy event incorrectly claims legacy origin: " + row["event_id"])
        if row.get("quantitative_comparison_ready") == "yes":
            eligible = [
                e for e in evidence
                if e["event_id"] == row["event_id"]
                and e.get("compatible_with_agrichter_grain_loss") == "yes"
                and e.get("verification_status") == "harmonized_verified"
            ]
            if not eligible:
                errors.append("Comparison-ready event lacks harmonized verified evidence: " + row["event_id"])

    for row in evidence:
        if row["event_id"] not in registry_ids:
            errors.append("Evidence references unknown event: " + row["event_id"])
        if row.get("value") and not (row.get("unit") and row.get("metric") and row.get("source_url")):
            errors.append("Numeric evidence lacks unit/metric/source: " + row["observation_id"])
        if row.get("compatible_with_agrichter_grain_loss") not in {"yes", "no", "conditional"}:
            errors.append("Invalid comparability flag: " + row["observation_id"])

    print(f"Legacy events: {len(legacy)} | Registry candidates: {len(registry)} | Evidence observations: {len(evidence)}")
    if errors:
        for msg in errors:
            print("ERROR:", msg, file=sys.stderr)
        return 1
    print("PASS: Registry coverage and evidence foreign keys are structurally consistent.")
    print("NOTE: Historical source claims and crop-region-year comparability remain unvalidated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
