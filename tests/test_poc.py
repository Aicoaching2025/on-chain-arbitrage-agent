"""Test suite for the arbitrage POC.

Covers the success criteria that matter most for an auditable backtest:
  * determinism (same input -> identical output)
  * data validation actually rejects bad rows
  * no look-ahead in the indicator computation
  * cost model and metric correctness
  * risk-gate behavior

Run:  pytest -q      (from the project root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtester import EventDrivenBacktester
from src.data_pipeline import DataPipeline
from src.model import MicrostructureModel, build_signal_frame
from src.risk_agent import RiskMonitoringAgent
from src.utils import (
    CostParams,
    daily_sharpe,
    estimate_slippage_bps,
    load_config,
    max_drawdown,
    round_trip_cost_usd,
)


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    # Use a small in-memory-ish window so tests are fast.
    c["backtest_start_date"] = "2023-01-01"
    c["backtest_end_date"] = "2023-04-01"
    c["db_path"] = "data/_test.db"
    return c


@pytest.fixture(scope="module")
def bars(cfg):
    pipe = DataPipeline(config=cfg)
    raw = pipe.fetch_historical_swaps()
    out = pipe.compute_microstructure(raw)
    pipe.close()
    return out


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_synthetic_data_is_deterministic(cfg):
    p1 = DataPipeline(config=cfg)
    a = p1.fetch_historical_swaps()
    p1.close()
    p2 = DataPipeline(config=cfg)
    b = p2.fetch_historical_swaps()
    p2.close()
    pd.testing.assert_frame_equal(a, b)


def test_backtest_is_deterministic(cfg, bars):
    def run_once():
        model = MicrostructureModel(cfg["ma_window_bars"], cfg["vol_window_bars"],
                                    cfg["z_threshold"], cfg["use_garch"])
        model.fit(bars["midprice"].values[: len(bars) // 2])
        sig = build_signal_frame(bars, model)
        bt = EventDrivenBacktester(cfg["initial_capital_usd"], cfg["max_position_size_usd"],
                                   CostParams.from_config(cfg), bar_frequency=cfg["bar_frequency"])
        bt.run(sig)
        return bt.get_results()

    r1, r2 = run_once(), run_once()
    assert r1["metrics"] == r2["metrics"]
    assert r1["equity_curve"] == r2["equity_curve"]
    assert r1["trade_log"] == r2["trade_log"]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validation_rejects_bad_rows(cfg):
    pipe = DataPipeline(config=cfg)
    df = pipe.fetch_historical_swaps().head(1000).copy()
    n0 = len(df)
    # Inject one null, one non-positive price, one duplicate.
    df.iloc[0, df.columns.get_loc("effective_spot_price")] = np.nan
    df.iloc[1, df.columns.get_loc("effective_spot_price")] = -5.0
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    report = pipe.validate(df)
    pipe.close()
    assert report["rejected"]["null_fields"] >= 1
    assert report["rejected"]["non_positive_price"] >= 1
    assert report["rejected"]["duplicates"] >= 1
    assert report["output_rows"] < n0 + 1


def test_clean_synthetic_data_has_no_rejections(cfg):
    pipe = DataPipeline(config=cfg)
    df = pipe.fetch_historical_swaps()
    report = pipe.validate(df)
    pipe.close()
    assert sum(report["rejected"].values()) == 0


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #

def test_indicators_are_causal(bars):
    """Indicator at bar t must not change when future bars are appended."""
    model = MicrostructureModel(ma_window=48, vol_window=48)
    prices = bars["midprice"].reset_index(drop=True)
    cut = 1000
    ind_full = model.compute_indicators(prices)
    ind_partial = model.compute_indicators(prices.iloc[: cut + 1])
    # The z-score at bar `cut` should be identical whether or not future data exists.
    a = ind_full["zscore"].iloc[cut]
    b = ind_partial["zscore"].iloc[cut]
    assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9


# --------------------------------------------------------------------------- #
# Cost model + metrics
# --------------------------------------------------------------------------- #

def test_slippage_scales_with_size():
    params = CostParams(2.0, 500.0, 5.0, 0.5)
    small = estimate_slippage_bps(1_000_000, 1e7, params)
    big = estimate_slippage_bps(5_000_000, 1e7, params)
    assert big > small
    # Floor respected for a tiny trade in a deep pool.
    assert estimate_slippage_bps(1_000, 1e8, params) == pytest.approx(params.min_slippage_bps, abs=0.01)


def test_round_trip_cost_components():
    params = CostParams(2.0, 500.0, 5.0, 0.5)
    c = round_trip_cost_usd(10_000, 1e8, params)
    assert c["gas_usd"] == 4.0  # 2 txns
    assert c["total_cost_usd"] == pytest.approx(
        c["gas_usd"] + c["slippage_usd"] + c["mev_usd"]
    )


def test_max_drawdown_known_series():
    eq = np.array([100, 120, 90, 110, 80])
    # Peak 120 -> trough 80 = -33.33%
    assert max_drawdown(eq) == pytest.approx(-1 / 3, rel=1e-6)


def test_daily_sharpe_positive_for_uptrend():
    idx = pd.date_range("2023-01-01", periods=30, freq="D")
    eq = np.linspace(100, 130, 30)
    assert daily_sharpe(idx, eq) > 0


# --------------------------------------------------------------------------- #
# Risk agent
# --------------------------------------------------------------------------- #

def test_entry_gate_rejects_on_drawdown():
    agent = RiskMonitoringAgent(max_daily_drawdown_pct=5.0)
    ok = agent.check_entry_gate(
        "t0",
        {"current_delta": 0.0, "daily_drawdown_pct": -6.0},
        {"volatility": 1.0, "historical_vol": 1.0, "signal": "BUY"},
    )
    assert ok is False
    assert agent.decision_log[-1]["decision"] == "REJECT"


def test_exit_gate_stop_loss_long():
    agent = RiskMonitoringAgent(stop_loss_pct=5.0)
    pos = {"entry_price": 100.0, "side": "long"}
    should_exit, reason = agent.check_exit_gate("t1", pos, current_price=94.0, bars_held=1)
    assert should_exit is True
    assert "stop-loss" in reason
