"""Statistical microstructure model: mean-reversion via rolling z-score.

Hypothesis: when the spot price diverges far from its moving average (measured
in units of conditional volatility), it tends to revert. We trade that
reversion. Volatility is estimated either with a simple rolling std (default,
fast, robust) or a GARCH(1,1) fit (optional, set use_garch=true in config).

The model is fit on a training window and produces point-in-time signals so the
backtest has no look-ahead: the z-score at bar t uses only the MA and vol
estimated from data up to and including t.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MicrostructureModel:
    def __init__(
        self,
        ma_window: int = 48,
        vol_window: int = 48,
        z_threshold: float = 2.0,
        use_garch: bool = False,
    ):
        self.ma_window = ma_window
        self.vol_window = vol_window
        self.z_threshold = z_threshold
        self.use_garch = use_garch
        self._garch_sigma_ann: float | None = None  # fitted unconditional vol scale
        self._train_log_vol: float | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #

    def fit(self, prices: np.ndarray) -> "MicrostructureModel":
        """Estimate model-level parameters from a training price series.

        For the rolling-std path this mostly validates inputs and records the
        training-set volatility scale. For the GARCH path it fits GARCH(1,1)
        on log returns and stores the conditional-vol scale used to normalize
        the rolling estimate.
        """
        prices = np.asarray(prices, dtype=float)
        prices = prices[prices > 0]
        if prices.size < self.ma_window + 2:
            raise ValueError("Not enough training data for the configured ma_window.")
        log_ret = np.diff(np.log(prices))
        self._train_log_vol = float(np.std(log_ret))

        if self.use_garch:
            self._fit_garch(log_ret)
        self._fitted = True
        return self

    def _fit_garch(self, log_ret: np.ndarray) -> None:
        """Fit GARCH(1,1) on (scaled) log returns via statsmodels.

        Returns are scaled to %-units for numerical stability, as recommended
        by statsmodels/arch conventions. Falls back to rolling std if the fit
        fails to converge.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA  # noqa: F401  (ensures statsmodels present)
            from arch import arch_model  # type: ignore

            scaled = log_ret * 100.0
            res = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Zero").fit(disp="off")
            # Long-run (unconditional) conditional vol in original units.
            params = res.params
            omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
            uncond = np.sqrt(omega / max(1e-9, (1 - alpha - beta))) / 100.0
            self._garch_sigma_ann = float(uncond)
        except Exception:
            # arch not installed or fit failed: silently use rolling-std path.
            self.use_garch = False
            self._garch_sigma_ann = None

    # ------------------------------------------------------------------ #

    def compute_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """Point-in-time MA, volatility, and z-score for a price series.

        All rolling windows are causal (use only past+current data), so the
        resulting frame is safe to backtest on directly.
        """
        prices = pd.Series(np.asarray(prices, dtype=float)).reset_index(drop=True)
        ma = prices.rolling(self.ma_window, min_periods=self.ma_window).mean()
        log_ret = np.log(prices).diff()
        roll_vol = log_ret.rolling(self.vol_window, min_periods=self.vol_window).std()
        # Convert return-vol into a price-level band around the MA.
        price_vol = ma * roll_vol
        if self.use_garch and self._garch_sigma_ann:
            # Blend the GARCH unconditional scale with the rolling estimate.
            price_vol = ma * np.maximum(roll_vol, self._garch_sigma_ann)
        z = (prices - ma) / price_vol.replace(0, np.nan)
        return pd.DataFrame(
            {
                "price": prices,
                "ma": ma,
                "vol": price_vol,
                "zscore": z,
            }
        )

    def generate_signal(self, z_score: float) -> str:
        """Map a z-score to a discrete signal. Reversion logic:
        very low z (price below MA) -> BUY (expect rise);
        very high z (price above MA) -> SELL (expect fall)."""
        if np.isnan(z_score):
            return "HOLD"
        if z_score < -self.z_threshold:
            return "BUY"
        if z_score > self.z_threshold:
            return "SELL"
        return "HOLD"

    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """Vectorized indicators + signal column for a whole series."""
        ind = self.compute_indicators(prices)
        ind["signal"] = ind["zscore"].apply(self.generate_signal)
        return ind

    def get_model_state(self, prices: pd.Series) -> dict[str, Any]:
        """Latest MA / vol / z-score / signal for the dashboard."""
        ind = self.generate_signals(prices)
        last = ind.iloc[-1]
        return {
            "price": float(last["price"]),
            "ma": float(last["ma"]) if not np.isnan(last["ma"]) else None,
            "vol": float(last["vol"]) if not np.isnan(last["vol"]) else None,
            "zscore": float(last["zscore"]) if not np.isnan(last["zscore"]) else None,
            "signal": last["signal"],
            "z_threshold": self.z_threshold,
        }


def build_signal_frame(
    bars: pd.DataFrame,
    model: "MicrostructureModel",
    hist_vol_window: int = 480,
) -> pd.DataFrame:
    """Assemble the column set the backtester expects from bars + model.

    Adds ``hist_vol`` (a slow rolling baseline of the volatility band) so the
    risk agent can detect volatility spikes relative to the recent norm.
    """
    ind = model.generate_signals(bars["midprice"].reset_index(drop=True))
    out = pd.DataFrame(
        {
            "timestamp": bars["timestamp"].values,
            "price": ind["price"].values,
            "ma": ind["ma"].values,
            "vol": ind["vol"].values,
            "zscore": ind["zscore"].values,
            "signal": ind["signal"].values,
            "liquidity": bars["liquidity"].values,
        }
    )
    out["hist_vol"] = out["vol"].rolling(hist_vol_window, min_periods=20).median()
    return out
