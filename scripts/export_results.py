"""Convenience runner: regenerate every artifact the dashboard reads.

Runs the full backtest export and the walk-forward validation in sequence so a
single command refreshes data/backtest_results.json and
data/walk_forward_results.json.

Usage:
    python scripts/export_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_backtest  # noqa: E402  (scripts dir is on sys.path)
import walk_forward  # noqa: E402

if __name__ == "__main__":
    run_backtest.main()
    print()
    walk_forward.main()
