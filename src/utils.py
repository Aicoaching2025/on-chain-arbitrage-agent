"""Shared helpers: config loading, cost modeling, and performance metrics.

Everything here is deterministic and dependency-light so the backtest can be
reproduced bit-for-bit. No global state, no I/O side effects beyond reading
the config file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_path(*parts: str) -> str:
    """Resolve a path relative to the project root."""
    return os.path.join(PROJECT_ROOT, *parts)


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config.json. Defaults to data/config.json at the project root."""
    path = path or project_path("data", "config.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Map a pandas-style frequency string to number of bars per calendar year.
# Used for annualizing Sharpe and converting day-based windows to bar counts.
_FREQ_PER_YEAR = {
    "1h": 24 * 365,
    "h": 24 * 365,
    "4h": 6 * 365,
    "1d": 365,
    "d": 365,
}


def bars_per_year(freq: str) -> float:
    key = freq.lower().strip()
    if key not in _FREQ_PER_YEAR:
        raise ValueError(f"Unsupported bar_frequency '{freq}'. Use one of {list(_FREQ_PER_YEAR)}")
    return float(_FREQ_PER_YEAR[key])


def bars_per_day(freq: str) -> float:
    return bars_per_year(freq) / 365.0


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostParams:
    """Static cost parameters pulled from config."""

    gas_per_tx_usd: float
    slippage_impact_factor: float
    min_slippage_bps: float
    mev_loss_bps: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CostParams":
        return cls(
            gas_per_tx_usd=float(cfg["gas_per_tx_usd"]),
            slippage_impact_factor=float(cfg["slippage_impact_factor"]),
            min_slippage_bps=float(cfg["min_slippage_bps"]),
            mev_loss_bps=float(cfg["mev_loss_bps"]),
        )


def estimate_slippage_bps(
    position_size_usd: float,
    available_liquidity_usd: float,
    params: CostParams,
) -> float:
    """Per-leg execution cost in bps as a function of trade size vs. pool depth.

        cost_bps = min_slippage_bps + (size / liquidity) * impact_factor

    The floor ``min_slippage_bps`` represents the fixed execution cost a swap
    always pays - for the ETH-USDC 0.05% pool that is the 5 bps LP fee, which
    dominates for the small trades in a deep pool typical of this POC. The
    second term is price impact: it scales linearly with the fraction of pool
    depth consumed (impact_factor is bps per unit of size/liquidity). Real AMMs
    are convex; this linear approximation is intentionally conservative and is
    documented as a limitation in the README.
    """
    if available_liquidity_usd <= 0:
        return params.min_slippage_bps + params.slippage_impact_factor
    impact_bps = (position_size_usd / available_liquidity_usd) * params.slippage_impact_factor
    return float(params.min_slippage_bps + impact_bps)


def round_trip_cost_usd(
    position_size_usd: float,
    available_liquidity_usd: float,
    params: CostParams,
) -> dict[str, float]:
    """Full cost breakdown for an entry+exit round trip.

    Returns a dict so the trade log can store each component for auditability.
    Gas is charged per transaction (2 txns per round trip). Slippage and MEV
    are charged on notional, once per leg.
    """
    slip_bps = estimate_slippage_bps(position_size_usd, available_liquidity_usd, params)
    slippage_usd = position_size_usd * (slip_bps / 1e4) * 2.0
    mev_usd = position_size_usd * (params.mev_loss_bps / 1e4) * 2.0
    gas_usd = params.gas_per_tx_usd * 2.0
    total = slippage_usd + mev_usd + gas_usd
    return {
        "slippage_bps": slip_bps,
        "slippage_usd": slippage_usd,
        "mev_usd": mev_usd,
        "gas_usd": gas_usd,
        "total_cost_usd": total,
    }


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #


def compute_returns(equity: np.ndarray) -> np.ndarray:
    """Simple period-over-period returns from an equity series."""
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return np.array([])
    prev = equity[:-1]
    # Guard against zero/negative equity ruining the division.
    prev = np.where(prev == 0, np.nan, prev)
    return np.diff(equity) / prev


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


# Crypto markets trade 24/7, so a year is 365 trading days.
CRYPTO_TRADING_DAYS = 365


def daily_sharpe(equity_index: pd.DatetimeIndex, equity: np.ndarray) -> float:
    """Sharpe computed on DAILY-resampled equity, annualized by sqrt(365).

    We deliberately resample to daily rather than annualizing the native
    (hourly) bar returns: annualizing high-frequency returns by a large sqrt
    factor is statistically noisy and tends to overstate Sharpe on short
    trending windows. Daily resampling gives a stable, conventional figure.
    """
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return 0.0
    s = pd.Series(equity, index=pd.DatetimeIndex(equity_index))
    daily = s.resample("D").last().dropna()
    rets = daily.pct_change().dropna().to_numpy()
    if rets.size == 0 or np.std(rets) == 0:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * np.sqrt(CRYPTO_TRADING_DAYS))


def max_drawdown(equity: np.ndarray) -> float:
    """Max drawdown as a fraction (negative). Computed on the equity curve
    directly, not on cumulative returns, so it is exact."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def underwater_curve(equity: np.ndarray) -> np.ndarray:
    """Per-period drawdown fraction (<= 0) for the underwater plot."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return np.array([])
    running_max = np.maximum.accumulate(equity)
    return (equity - running_max) / running_max


def win_rate(trade_log: list[dict[str, Any]]) -> float:
    if not trade_log:
        return 0.0
    wins = sum(1 for t in trade_log if t.get("pnl", 0.0) > 0)
    return wins / len(trade_log)


def monthly_returns(equity_index: pd.DatetimeIndex, equity: np.ndarray) -> pd.Series:
    """Resample the equity curve to month-end and compute monthly % returns."""
    s = pd.Series(np.asarray(equity, dtype=float), index=pd.DatetimeIndex(equity_index))
    monthly = s.resample("ME").last()
    return monthly.pct_change().dropna()


def summarize_performance(
    equity_index: pd.DatetimeIndex,
    equity: np.ndarray,
    trade_log: list[dict[str, Any]],
    periods_per_year: float,
) -> dict[str, Any]:
    """One-stop metrics dict used by the backtester and dashboard."""
    equity = np.asarray(equity, dtype=float)
    pnls = [t.get("pnl", 0.0) for t in trade_log]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    return {
        "sharpe": daily_sharpe(equity_index, equity),
        "max_drawdown_pct": max_drawdown(equity) * 100.0,
        "win_rate": win_rate(trade_log),
        "total_trades": len(trade_log),
        "final_capital": float(equity[-1]) if equity.size else 0.0,
        "total_return_pct": (float(equity[-1] / equity[0]) - 1.0) * 100.0 if equity.size else 0.0,
        "avg_trade_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
    }
