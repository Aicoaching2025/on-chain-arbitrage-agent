"""End-to-end entry point: data -> model -> backtest -> export.

Usage:
    python scripts/run_backtest.py

Steps:
  1. Build the DuckDB dataset (synthetic by default; see config.json).
  2. Fit the microstructure model on the training window.
  3. Generate point-in-time signals over the full period.
  4. Run the event-driven backtest with the risk agent attached.
  5. Export results + decision log to JSON for the dashboard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.backtester import EventDrivenBacktester
from src.data_pipeline import DataPipeline
from src.model import MicrostructureModel, build_signal_frame
from src.risk_agent import RiskMonitoringAgent
from src.utils import CostParams, bars_per_day, load_config, project_path


def main() -> None:
    cfg = load_config()
    print("=" * 70)
    print(f"On-Chain Arbitrage Agent | pair={cfg['pair']} | source={cfg['data_source']}")
    print("=" * 70)

    # 1. Data ----------------------------------------------------------- #
    pipeline = DataPipeline(config=cfg)
    print("[1/5] Building dataset ...")
    report = pipeline.build()
    print(f"      validated {report['output_rows']}/{report['input_rows']} swaps, "
          f"{report['bars']} bars @ {report['bar_frequency']}")
    print(f"      rejected: {report['rejected']} | time gaps>2h: {report['time_gaps_over_2h']}")
    bars = pipeline.load_bars("uniswap_bars")
    pipeline.close()

    # 2. Model fit (train window only, no look-ahead) ------------------- #
    print("[2/5] Fitting microstructure model ...")
    bpd = bars_per_day(cfg["bar_frequency"])
    train_bars = int(cfg["train_window_days"] * bpd)
    train_bars = min(train_bars, len(bars) - 2)
    model = MicrostructureModel(
        ma_window=int(cfg["ma_window_bars"]),
        vol_window=int(cfg["vol_window_bars"]),
        z_threshold=float(cfg["z_threshold"]),
        use_garch=bool(cfg["use_garch"]),
    )
    model.fit(bars["midprice"].values[:train_bars])

    # 3. Signals -------------------------------------------------------- #
    print("[3/5] Generating signals ...")
    signals = build_signal_frame(bars, model)
    n_buy = int((signals["signal"] == "BUY").sum())
    n_sell = int((signals["signal"] == "SELL").sum())
    print(f"      signals: BUY={n_buy}  SELL={n_sell}  HOLD={len(signals) - n_buy - n_sell}")

    # 4. Backtest ------------------------------------------------------- #
    print("[4/5] Running event-driven backtest ...")
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
        cost_params=CostParams.from_config(cfg),
        risk_agent=risk_agent,
        bar_frequency=cfg["bar_frequency"],
    )
    metrics = bt.run(signals)
    results = bt.get_results()
    results["decision_log"] = risk_agent.get_decision_log()
    results["config"] = cfg
    results["signals_preview"] = (
        signals.tail(500)
        .assign(timestamp=lambda d: d["timestamp"].astype(str))
        .to_dict(orient="records")
    )

    print("      ---- metrics ----")
    for k, v in metrics.items():
        print(f"      {k:>18}: {v}")

    # 5. Export --------------------------------------------------------- #
    out_path = project_path(cfg["results_path"])
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"[5/5] Wrote results -> {out_path}")
    print(f"      decision-log entries: {len(results['decision_log'])}")
    print("Done. Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
