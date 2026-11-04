"""Settings (config.json) and the repository's folders, defined once.

Every path in the project is built from ROOT, so scripts work from any directory.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the repository root
RESULTS = ROOT / "results"                         # result records (JSON)
REPORTS = ROOT / "reports"                         # test reports (markdown)
DEMO = ROOT / "demo"                               # offline pages and screenshots

CFG = json.loads((ROOT / "config.json").read_text())


def snapshot():
    """Settings that affect results, stored inside every result file."""
    return {k: v for k, v in CFG.items() if not k.startswith("_") and k not in ("pilot", "server")}
