"""Loader for the 2WikiMultihopQA development set."""

from __future__ import annotations

from pathlib import Path

from .hotpotqa import HotpotQA, slug


class TwoWikiMultihopQA(HotpotQA):
    """Load the 2WikiMultihopQA development JSON array."""

    name = "2wiki"
    _expected_filename = "dev.json"
    _download_url = (
        "https://raw.githubusercontent.com/Alab-NII/2wikimultihop/"
        "main/data/dev.json"
    )


__all__ = ["TwoWikiMultihopQA", "slug"]
