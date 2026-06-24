"""Autonomous risk-monitoring agent.

Sits between the model's signals and the backtester's execution. Every decision
(approve/reject entry, force exit) is logged with structured reasoning so the
whole run is auditable. The agent holds no market state of its own - it is fed
portfolio and market snapshots and returns a decision, which keeps it
deterministic and easy to unit-test.
"""
from __future__ import annotations

from typing import Any


class RiskMonitoringAgent:
    def __init__(
        self,
        max_delta: float = 0.15,
        max_daily_drawdown_pct: float = 5.0,
        stop_loss_pct: float = 5.0,
        vol_spike_multiple: float = 2.5,
        max_hold_bars: int = 168,
    ):
        self.max_delta = max_delta
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.vol_spike_multiple = vol_spike_multiple
        self.max_hold_bars = max_hold_bars
        self.decision_log: list[dict[str, Any]] = []

    def _log(self, timestamp: Any, decision_type: str, decision: bool, reasons: list[str], extra: dict | None = None) -> None:
        entry = {
            "timestamp": str(timestamp),
            "decision_type": decision_type,
            "decision": "APPROVE" if decision else "REJECT",
            "reasons": "; ".join(reasons) if reasons else "all checks passed",
        }
        if extra:
            entry.update(extra)
        self.decision_log.append(entry)

    # ------------------------------------------------------------------ #

    def check_entry_gate(self, timestamp: Any, portfolio_state: dict, market_state: dict) -> bool:
        """Approve or reject a new position. Logs the decision either way."""
        reasons: list[str] = []

        if abs(portfolio_state.get("current_delta", 0.0)) >= self.max_delta:
            reasons.append(
                f"delta {portfolio_state['current_delta']:.3f} >= limit {self.max_delta:.3f}"
            )
        if portfolio_state.get("daily_drawdown_pct", 0.0) <= -self.max_daily_drawdown_pct:
            reasons.append(
                f"daily drawdown {portfolio_state['daily_drawdown_pct']:.2f}% breached "
                f"-{self.max_daily_drawdown_pct:.2f}%"
            )
        hist_vol = market_state.get("historical_vol")
        cur_vol = market_state.get("volatility")
        if hist_vol and cur_vol and cur_vol > hist_vol * self.vol_spike_multiple:
            reasons.append(
                f"vol spike: {cur_vol:.5f} > {self.vol_spike_multiple:.1f}x hist {hist_vol:.5f}"
            )

        approved = len(reasons) == 0
        self._log(timestamp, "entry_gate", approved, reasons,
                  extra={"signal": market_state.get("signal", "")})
        return approved

    def check_exit_gate(self, timestamp: Any, position: dict, current_price: float, bars_held: int) -> tuple[bool, str]:
        """Force-exit checks independent of the model signal.

        Returns (should_exit, reason). Only logs when an exit is forced, to keep
        the decision log focused on actionable events.
        """
        reasons: list[str] = []
        entry_price = position["entry_price"]
        side = position.get("side", "long")

        # Directional P&L for stop-loss.
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100.0

        if pnl_pct <= -self.stop_loss_pct:
            reasons.append(f"stop-loss hit: {pnl_pct:.2f}% <= -{self.stop_loss_pct:.2f}%")
        if bars_held >= self.max_hold_bars:
            reasons.append(f"max hold reached: {bars_held} >= {self.max_hold_bars} bars")

        should_exit = len(reasons) > 0
        reason = "; ".join(reasons)
        if should_exit:
            self._log(timestamp, "exit_gate", True, reasons,
                      extra={"forced_pnl_pct": round(pnl_pct, 3)})
        return should_exit, reason

    # ------------------------------------------------------------------ #

    def get_decision_log(self) -> list[dict[str, Any]]:
        return self.decision_log

    def reset(self) -> None:
        self.decision_log = []
