"""Event-driven backtester.

Processes one bar at a time: ingest market data -> consult the model signal ->
run risk gates -> simulate fills with realistic costs -> update portfolio. State
is explicit and mutation is sequential, so a given (data, config) pair always
produces an identical equity curve and trade log (determinism is asserted in the
test suite).

Position model (POC): at most one open position at a time, fixed notional.
- BUY signal  -> open a LONG  (expect mean reversion upward)
- SELL signal -> open a SHORT (expect mean reversion downward)
- Opposite signal, model-neutral revert, risk-forced exit, or max-hold closes it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .risk_agent import RiskMonitoringAgent
from .utils import CostParams, round_trip_cost_usd, summarize_performance, bars_per_year, monthly_returns, underwater_curve


class EventDrivenBacktester:
    def __init__(
        self,
        initial_capital: float,
        max_position_size: float,
        cost_params: CostParams,
        risk_agent: RiskMonitoringAgent | None = None,
        bar_frequency: str = "1h",
    ):
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.max_position_size = float(max_position_size)
        self.cost_params = cost_params
        self.risk_agent = risk_agent
        self.bar_frequency = bar_frequency

        self.position: dict[str, Any] | None = None
        self.trade_log: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []
        self.equity_index: list[Any] = []
        self._day_start_equity = self.capital
        self._current_day = None
        self._bar_count = 0

    # ------------------------------------------------------------------ #

    def _mark_to_market(self, price: float) -> float:
        """Current total equity = cash + unrealized P&L of any open position."""
        if self.position is None:
            return self.capital
        pos = self.position
        if pos["side"] == "long":
            unrealized = pos["size_usd"] * (price - pos["entry_price"]) / pos["entry_price"]
        else:
            unrealized = pos["size_usd"] * (pos["entry_price"] - price) / pos["entry_price"]
        return self.capital + unrealized

    def _daily_drawdown_pct(self, equity: float) -> float:
        if self._day_start_equity <= 0:
            return 0.0
        return (equity / self._day_start_equity - 1.0) * 100.0

    # ------------------------------------------------------------------ #

    def process_bar(self, row: pd.Series) -> None:
        """Handle a single market bar."""
        ts = row["timestamp"]
        price = float(row["price"])
        signal = row["signal"]
        liquidity = float(row.get("liquidity", 0.0))
        vol = float(row.get("vol", np.nan))
        hist_vol = float(row.get("hist_vol", np.nan))

        # Reset the daily-drawdown anchor at each new calendar day.
        day = pd.Timestamp(ts).date()
        if self._current_day != day:
            self._current_day = day
            self._day_start_equity = self._mark_to_market(price)

        # ---- Manage an open position first --------------------------- #
        if self.position is not None:
            self._bar_count += 1
            bars_held = self._bar_count - self.position["entry_bar"]
            exit_now, reason = False, ""

            if self.risk_agent is not None:
                exit_now, reason = self.risk_agent.check_exit_gate(ts, self.position, price, bars_held)

            # Model-driven exit: reversion complete (z crossed back) or opposite signal.
            if not exit_now:
                z = float(row.get("zscore", np.nan))
                if self.position["side"] == "long" and (signal == "SELL" or z >= 0):
                    exit_now, reason = True, "reversion complete / opposite signal"
                elif self.position["side"] == "short" and (signal == "BUY" or z <= 0):
                    exit_now, reason = True, "reversion complete / opposite signal"

            if exit_now:
                self._close_position(ts, price, liquidity, reason)
        else:
            self._bar_count += 1

        # ---- Consider a new entry ----------------------------------- #
        if self.position is None and signal in ("BUY", "SELL"):
            equity = self._mark_to_market(price)
            portfolio_state = {
                "current_delta": 0.0,  # flat before entry
                "daily_drawdown_pct": self._daily_drawdown_pct(equity),
            }
            market_state = {"volatility": vol, "historical_vol": hist_vol, "signal": signal}
            approved = True
            if self.risk_agent is not None:
                approved = self.risk_agent.check_entry_gate(ts, portfolio_state, market_state)
            if approved:
                self._open_position(ts, price, signal, liquidity)

        # ---- Record equity ------------------------------------------ #
        self.equity_curve.append(self._mark_to_market(price))
        self.equity_index.append(ts)

    # ------------------------------------------------------------------ #

    def _open_position(self, ts: Any, price: float, signal: str, liquidity: float) -> None:
        size_usd = min(self.max_position_size, self.capital)
        costs = round_trip_cost_usd(size_usd, liquidity, self.cost_params)
        # Charge the entry-leg portion of costs now (half of round trip).
        entry_cost = costs["gas_usd"] / 2 + costs["slippage_usd"] / 2 + costs["mev_usd"] / 2
        self.capital -= entry_cost
        self.position = {
            "entry_date": str(ts),
            "entry_price": price,
            "size_usd": size_usd,
            "side": "long" if signal == "BUY" else "short",
            "entry_bar": self._bar_count,
            "entry_cost": entry_cost,
            "entry_liquidity": liquidity,
            "round_trip_cost": costs,
        }

    def _close_position(self, ts: Any, price: float, liquidity: float, reason: str) -> None:
        pos = self.position
        assert pos is not None
        if pos["side"] == "long":
            gross_pnl = pos["size_usd"] * (price - pos["entry_price"]) / pos["entry_price"]
        else:
            gross_pnl = pos["size_usd"] * (pos["entry_price"] - price) / pos["entry_price"]

        # Exit-leg costs (use current liquidity for the exit slippage).
        exit_costs = round_trip_cost_usd(pos["size_usd"], liquidity, self.cost_params)
        exit_cost = exit_costs["gas_usd"] / 2 + exit_costs["slippage_usd"] / 2 + exit_costs["mev_usd"] / 2
        total_cost = pos["entry_cost"] + exit_cost
        net_pnl = gross_pnl - exit_cost  # entry_cost already deducted from capital

        self.capital += gross_pnl - exit_cost
        self.trade_log.append(
            {
                "entry_date": pos["entry_date"],
                "exit_date": str(ts),
                "side": pos["side"],
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(price, 4),
                "size_usd": round(pos["size_usd"], 2),
                "gross_pnl": round(gross_pnl, 2),
                "total_cost": round(total_cost, 4),
                "pnl": round(net_pnl - pos["entry_cost"], 2),
                "return_pct": round((net_pnl - pos["entry_cost"]) / pos["size_usd"] * 100.0, 4),
                "exit_reason": reason,
            }
        )
        self.position = None

    # ------------------------------------------------------------------ #

    def run(self, signals_df: pd.DataFrame) -> dict[str, Any]:
        """Run the backtest over a frame with columns:
        timestamp, price, signal, zscore, vol, liquidity, hist_vol."""
        self._reset()
        for _, row in signals_df.iterrows():
            self.process_bar(row)
        # Force-close any position at the final bar so metrics are complete.
        if self.position is not None:
            last = signals_df.iloc[-1]
            self._close_position(last["timestamp"], float(last["price"]),
                                 float(last.get("liquidity", 0.0)), "end of backtest")
            self.equity_curve[-1] = self.capital
        return self.get_metrics()

    def _reset(self) -> None:
        self.capital = self.initial_capital
        self.position = None
        self.trade_log = []
        self.equity_curve = []
        self.equity_index = []
        self._day_start_equity = self.capital
        self._current_day = None
        self._bar_count = 0
        if self.risk_agent is not None:
            self.risk_agent.reset()

    # ------------------------------------------------------------------ #

    def get_metrics(self) -> dict[str, Any]:
        idx = pd.DatetimeIndex(self.equity_index)
        equity = np.array(self.equity_curve, dtype=float)
        ppy = bars_per_year(self.bar_frequency)
        metrics = summarize_performance(idx, equity, self.trade_log, ppy)
        return metrics

    def get_results(self) -> dict[str, Any]:
        """Full result bundle for export/dashboard."""
        idx = pd.DatetimeIndex(self.equity_index)
        equity = np.array(self.equity_curve, dtype=float)
        mr = monthly_returns(idx, equity) if len(equity) else pd.Series(dtype=float)
        return {
            "metrics": self.get_metrics(),
            "equity_curve": [float(x) for x in equity],
            "equity_dates": [str(t) for t in idx],
            "underwater": [float(x) for x in underwater_curve(equity)],
            "monthly_returns": {str(k.date()): float(v) for k, v in mr.items()},
            "trade_log": self.trade_log,
        }
