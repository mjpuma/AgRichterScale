"""Unverified compilation percents and crop-failure claims are not stated as fact."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files that describe events in prose. The source-check and crosswalk tables are left out
# because they quote the compilation sentence on purpose and mark it unverified.
PROSE = [
    "docs/MAJOR_FAMINE_LOSS_SOURCES.md",
    "docs/DOCUMENTARY_SEARCH_LOG.md",
    "docs/HISTORICAL_LOSSES_METHOD.md",
    "ancillary/documentary_search_log_v1.csv",
    "ancillary/event_attributes.csv",
    "ancillary/food_disruptions.csv",
    "README.md",
    "Supplementary_Information.tex",
]

FORBIDDEN = [
    "total harvest **25% below normal**",
    "Tanjore total harvest is reported 25 percent below normal",
    "A Tanjore record puts the total harvest 25 percent below normal",
    "not above one-quarter of average",
    "~18.8% decline",
    "Laki volcanic eruption caused crop failures",
    "Potato blight destroyed primary food crop",
    "Crop failure from excessive rain and cold",
    "Davies and Wheatcroft put the 1932 USSR harvest near 56",
]


class TestSoftenedClaims(unittest.TestCase):
    def test_forbidden_phrases_are_gone(self):
        for name in PROSE:
            path = ROOT / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for phrase in FORBIDDEN:
                self.assertNotIn(phrase, text, f"{phrase!r} in {name}")

    def test_compilation_percents_only_appear_as_unverified(self):
        for name in PROSE:
            path = ROOT / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for claim in ("86 percent", "12-44", "12–44", "63-94", "63–94", "almost 100 percent"):
                self.assertNotIn(claim, text, f"{claim!r} in {name}")


if __name__ == "__main__":
    unittest.main()
