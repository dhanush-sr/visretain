"""The single source of settings: config.json next to this file."""

import json
from pathlib import Path

CFG = json.loads(Path(__file__).with_name("config.json").read_text())


def snapshot():
    """Settings that affect results, stored inside every result file."""
    return {k: v for k, v in CFG.items() if not k.startswith("_") and k not in ("pilot", "server")}
