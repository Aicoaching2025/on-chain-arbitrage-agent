"""Walk-forward, out-of-sample validation.

Slide a (train, test) window across the dataset. For each window: fit the model
on the train slice, backtest only on the test slice, and record metrics. Robust
edge shows up as a stable, positive average test Sharpe across windows - not one
lucky in-sample fit.

Usage:
    python scripts/walk_forward.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.backtester import EventDrivenBacktester
from src.data_pipeline import DataPipeline
from src.model import MicrostructureModel, build_signal_frame
from src.risk_agent import RiskMonitoringAgent
from src.utils import CostParams, bars_per_day, load_config, project_path


def main() -> None:
    cfg = load_config()
    pipeline = DataPipeline(config=cfg)
    pipeline.build()
    bars = pipeline.load_bars("uniswap_bars")
    pipeline.close()

    bpd = bars_per_day(cfg["bar_frequency"])
    train_w = int(cfg["train_window_days"] * bpd)
    test_w = int(cfg["test_window_days"] * bpd)
    slide = int(cfg["walk_forward_slide_days"] * bpd)
    cost_params = CostParams.from_config(cfg)

    print("=" * 70)
    print(f"Walk-forward: train={cfg['train_window_days']}d  test={cfg['test_window_days']}d  "
          f"slide={cfg['walk_forward_slide_days']}d  ({len(bars)} bars total)")
    print("=" * 70)

    results = []
    window = 0
    start = 0
    while start + train_w + test_w <= len(bars):
        train_slice = bars.iloc[start:start + train_w]
        test_slice = bars.iloc[start + train_w:start + train_w + test_w].reset_index(drop=True)

        model = MicrostructureModel(
            ma_window=int(cfg["ma_window_bars"]),
            vol_window=int(cfg["vol_window_bars"]),
            z_threshold=float(cfg["z_threshold"]),
            use_garch=bool(cfg["use_garch"]),
        )
        model.fit(train_slice["midprice"].values)
        signals = build_signal_frame(test_slice, model)

        risk_agent = RiskMonitoringAgent(
            max_delta=float(cfg["max_portfolio_delta"]),
            max_daily_drawdown_pct=float(cfg["max_daily_drawdown_pct"]),
            stop_loss_pct=float(cfg["stop_loss_pct"]),
            vol_spike_multiple=float(cfg["vol_spike_multiple"]),
            max_hold_bars=int(cfg["max_hold_bars"]),
        )
        bt = EventDrivenBacktester(
            initial_capital=float(cfg["initial_capital_usd"]),
            max_position_size=float(cfg["max_position_size_usd"]),
            cost_params=cost_params,
            risk_agent=risk_agent,
            bar_frequency=cfg["bar_frequency"],
        )
        m = bt.run(signals)
        window += 1
        test_start = str(test_slice["timestamp"].iloc[0].date())
        test_end = str(test_slice["timestamp"].iloc[-1].date())
        m_row = {
            "window": window,
            "test_start": test_start,
            "test_end": test_end,
            "sharpe": round(m["sharpe"], 3),
            "max_drawdown_pct": round(m["max_drawdown_pct"], 2),
            "win_rate": round(m["win_rate"], 3),
            "total_trades": m["total_trades"],
            "total_return_pct": round(m["total_return_pct"], 2),
        }
        results.append(m_row)
        print(f"  win {window:>2} [{test_start}..{test_end}]  "
              f"Sharpe={m_row['sharpe']:>6}  DD={m_row['max_drawdown_pct']:>6}%  "
              f"win={m_row['win_rate']:>5}  trades={m_row['total_trades']:>3}")
        start += slide

    if not results:
        print("Not enough data for even one walk-forward window. "
              "Reduce train/test window sizes in config.json.")
        return

    sharpes = np.array([r["sharpe"] for r in results])
    win_rates = np.array([r["win_rate"] for r in results])
    summary = {
        "n_windows": len(results),
        "avg_sharpe": round(float(np.mean(sharpes)), 3),
        "std_sharpe": round(float(np.std(sharpes)), 3),
        "avg_win_rate": round(float(np.mean(win_rates)), 3),
        "pct_windows_positive_sharpe": round(float(np.mean(sharpes > 0)), 3),
        "windows": results,
    }
    print("-" * 70)
    print(f"  AVG Sharpe={summary['avg_sharpe']} (std={summary['std_sharpe']})  "
          f"avg win-rate={summary['avg_win_rate']}  "
          f"positive windows={summary['pct_windows_positive_sharpe']:.0%}")

    out = project_path("data", "walk_forward_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote -> {out}")


if __name__ == "__main__":
    main()
