"""Label for the historical-event numbers the figures plot.

The calculator sums SPAM 2020 harvested area and production for the crops
in the run, over the places it received, and treats that sum as a loss.
No event-specific loss fraction is applied, so the implicit fraction is 1.
This module names that quantity. It does not recompute it.
"""
import csv
from pathlib import Path

from agrichter.core.constants import HISTORICAL_EVENTS

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "ancillary" / "historical_event_registry.csv"

QUANTITY = (
    "SPAM 2020 harvested area and production of the selected crops "
    "inside the places the calculator received, counted in full"
)
GEOGRAPHY_YEAR = "2020"
LOSS_FRACTION_APPLIED = "1"
FIGURES_REDRAWN = "no"

FIELDS = [
    "event_id",
    "on_current_figures",
    "quantity",
    "geography_year",
    "loss_fraction_applied",
    "observed_affected_area_ha",
    "observed_crop_loss_fraction",
    "historical_baseline",
    "figures_redrawn",
    "notes",
]


def build_rows():
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        event_ids = [row["event_id"] for row in csv.DictReader(handle)]
    plotted = set(HISTORICAL_EVENTS)
    rows = []
    for event_id in event_ids:
        on_figure = event_id in plotted
        if on_figure:
            note = (
                "The figure files were not redrawn, and this label does not "
                "recompute their values."
            )
        else:
            note = (
                "This registry case has no disruption sheet, so it is not a "
                "point on the current figures."
            )
        rows.append({
            "event_id": event_id,
            "on_current_figures": "yes" if on_figure else "no",
            "quantity": QUANTITY if on_figure else "",
            "geography_year": GEOGRAPHY_YEAR if on_figure else "",
            "loss_fraction_applied": LOSS_FRACTION_APPLIED if on_figure else "",
            "observed_affected_area_ha": "",
            "observed_crop_loss_fraction": "",
            "historical_baseline": "",
            "figures_redrawn": FIGURES_REDRAWN,
            "notes": note,
        })
    return rows
