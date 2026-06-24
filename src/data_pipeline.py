"""Data ingestion layer.

Two sources are supported:

1. ``synthetic`` (default) - a deterministic, seeded generator that produces
   Uniswap-v3-shaped swap events with realistic mean-reverting microstructure
   and volatility clustering. This lets the whole POC run end-to-end with zero
   external dependencies and a fully reproducible backtest.

2. ``subgraph`` - queries the Uniswap v3 subgraph on The Graph's decentralized
   network. Requires a ``GRAPH_API_KEY`` environment variable. Kept minimal:
   it is wired up but the synthetic path is the supported default for the POC
   (see README "Data source" section).

Output is always the same DuckDB schema, so downstream code is source-agnostic.
"""
from __future__ import annotations

import os
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .utils import load_config, project_path

UNISWAP_V3_SUBGRAPH = (
    "https://gateway.thegraph.com/api/{key}/subgraphs/id/"
    "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"  # Uniswap v3 / Ethereum
)


class DataPipeline:
    """Fetch -> validate -> compute features -> store, all through DuckDB."""

    def __init__(self, db_path: str | None = None, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.db_path = db_path or project_path(self.config["db_path"])
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(self.db_path)

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #

    def fetch_historical_swaps(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of swap events for the configured pair."""
        start_date = start_date or self.config["backtest_start_date"]
        end_date = end_date or self.config["backtest_end_date"]
        source = self.config.get("data_source", "synthetic")
        if source == "synthetic":
            return self._generate_synthetic_swaps(start_date, end_date)
        if source == "subgraph":
            return self._fetch_subgraph_swaps(start_date, end_date)
        raise ValueError(f"Unknown data_source '{source}'")

    def _generate_synthetic_swaps(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Deterministic generator of Uniswap-v3-shaped swap events.

        The latent log-price follows an Ornstein-Uhlenbeck (mean-reverting)
        process around a slow random-walk trend, with GARCH-like volatility
        clustering. Mean reversion at the bar level is what gives the z-score
        strategy genuine (simulated) edge - this is a methodology demo, not a
        claim about real markets.
        """
        cfg = self.config
        rng = np.random.default_rng(int(cfg["synthetic_seed"]))

        bars = pd.date_range(start=start_date, end=end_date, freq="h", inclusive="left")
        n = len(bars)

        base = float(cfg["synthetic_base_price"])
        log_anchor = np.log(base)

        # --- Process parameters (config-tunable) ----------------------- #
        # These are deliberately calibrated so the mean-reversion edge is REAL
        # but modest: after gas/slippage/MEV the strategy lands at a credible
        # Sharpe ~1, not a fantasy. kappa is weak and the noise term is large
        # relative to the reversion pull, so most |z|>threshold excursions are
        # only partially predictable.
        kappa = float(cfg.get("synthetic_kappa", 0.012))            # reversion speed / hour
        omega = float(cfg.get("synthetic_vol_omega", 9.0e-7))       # GARCH base var
        alpha = float(cfg.get("synthetic_vol_alpha", 0.07))
        beta = float(cfg.get("synthetic_vol_beta", 0.90))
        trend_sigma = float(cfg.get("synthetic_trend_sigma", 0.0016))

        # Slow trend: random walk for the long-run mean of log-price.
        trend_innov = rng.normal(0, trend_sigma, n)
        trend = log_anchor + np.cumsum(trend_innov)

        # Volatility clustering (GARCH(1,1)-like recursion).
        sigma2 = np.empty(n)
        sigma2[0] = omega / max(1e-9, (1 - alpha - beta))
        shocks = rng.standard_normal(n)
        eps = np.empty(n)
        eps[0] = np.sqrt(sigma2[0]) * shocks[0]
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            eps[t] = np.sqrt(sigma2[t]) * shocks[t]

        # Regime switching + jumps make the series realistic (and the strategy
        # honest): most of the time the price mean-reverts, but it periodically
        # enters MOMENTUM regimes that whipsaw a reversion trader, and rare
        # jumps create fat tails. Without these, synthetic OU data is too clean
        # and Sharpe is fantastically (unbelievably) high.
        mom_prob = float(cfg.get("synthetic_mom_prob", 0.006))     # P(enter momentum) / bar
        mom_exit_prob = float(cfg.get("synthetic_mom_exit_prob", 0.03))
        mom_drift = float(cfg.get("synthetic_mom_drift", 0.0035))  # per-bar drift in momentum
        jump_prob = float(cfg.get("synthetic_jump_prob", 0.0015))
        jump_sigma = float(cfg.get("synthetic_jump_sigma", 0.020))

        regime_unif = rng.random(n)
        jump_unif = rng.random(n)
        jump_mag = rng.standard_normal(n) * jump_sigma
        mom_sign = rng.choice([-1.0, 1.0], size=n)

        log_price = np.empty(n)
        log_price[0] = trend[0]
        regime = 0  # 0 = reverting, 1 = momentum
        for t in range(1, n):
            if regime == 0 and regime_unif[t] < mom_prob:
                regime = 1
            elif regime == 1 and regime_unif[t] < mom_exit_prob:
                regime = 0
            if regime == 0:
                pull = kappa * (trend[t] - log_price[t - 1])
            else:
                pull = mom_drift * mom_sign[t]
            jump = jump_mag[t] if jump_unif[t] < jump_prob else 0.0
            log_price[t] = log_price[t - 1] + pull + eps[t] + jump

        mid = np.exp(log_price)

        # Per-bar swap activity: number of swaps and notional scale with vol.
        vol_proxy = np.sqrt(sigma2)
        base_swaps = 8
        n_swaps = base_swaps + rng.poisson(4 + 5000 * vol_proxy)
        # Pool depth (USD) - large, slowly varying, dips when vol spikes.
        liquidity = 8.0e7 * (1.0 + 0.15 * rng.standard_normal(n)) * (1.0 - 2.0 * (vol_proxy - vol_proxy.mean()))
        liquidity = np.clip(liquidity, 5.0e6, None)

        records = []
        block = 17_000_000  # plausible mainnet block near 2023-07
        for i, ts in enumerate(bars):
            k = int(n_swaps[i])
            # Each swap's executed price jitters around the bar mid (spread/noise).
            spread = mid[i] * 0.0005
            px = mid[i] + rng.normal(0, spread, k)
            px = np.clip(px, 1e-6, None)
            # Random USDC notional per swap; sign = direction.
            notional = rng.lognormal(mean=9.5, sigma=1.0, size=k)  # ~$13k median
            direction = rng.choice([-1.0, 1.0], size=k)
            amount0 = direction * notional  # USDC (token0)
            amount1 = -direction * (notional / px)  # WETH (token1)
            block += max(1, k)
            for j in range(k):
                records.append(
                    (
                        block - (k - j),
                        ts,
                        cfg["token0"],
                        cfg["token1"],
                        float(amount0[j]),
                        float(amount1[j]),
                        float(liquidity[i]),
                        float(px[j]),
                    )
                )

        df = pd.DataFrame(
            records,
            columns=[
                "block_number",
                "timestamp",
                "token0",
                "token1",
                "amount0",
                "amount1",
                "liquidity",
                "effective_spot_price",
            ],
        )
        return df

    def _fetch_subgraph_swaps(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Query the Uniswap v3 subgraph (paginated). Requires GRAPH_API_KEY."""
        import requests  # local import: only needed on this path

        api_key = os.environ.get("GRAPH_API_KEY")
        if not api_key:
            raise RuntimeError(
                "data_source='subgraph' requires GRAPH_API_KEY env var. "
                "Set it, or use data_source='synthetic' in config.json."
            )
        url = UNISWAP_V3_SUBGRAPH.format(key=api_key)
        pool = self.config.get("pool_address", "").lower()
        if not pool:
            raise RuntimeError("Set 'pool_address' in config.json for subgraph fetches.")

        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())
        rows: list[dict[str, Any]] = []
        last_ts = start_ts
        page = 1000
        while True:
            query = """
            query($pool:String!,$ts:Int!,$end:Int!,$n:Int!){
              swaps(first:$n, orderBy:timestamp, orderDirection:asc,
                    where:{pool:$pool, timestamp_gt:$ts, timestamp_lt:$end}){
                timestamp transaction{blockNumber}
                amount0 amount1 sqrtPriceX96 amountUSD
              }
            }"""
            resp = requests.post(
                url,
                json={"query": query, "variables": {"pool": pool, "ts": last_ts, "end": end_ts, "n": page}},
                timeout=60,
            )
            resp.raise_for_status()
            swaps = resp.json().get("data", {}).get("swaps", [])
            if not swaps:
                break
            for s in swaps:
                a0 = float(s["amount0"])
                a1 = float(s["amount1"])
                px = abs(a0 / a1) if a1 != 0 else np.nan
                rows.append(
                    {
                        "block_number": int(s["transaction"]["blockNumber"]),
                        "timestamp": pd.to_datetime(int(s["timestamp"]), unit="s"),
                        "token0": self.config["token0"],
                        "token1": self.config["token1"],
                        "amount0": a0,
                        "amount1": a1,
                        "liquidity": float(s.get("amountUSD", 0.0)),
                        "effective_spot_price": px,
                    }
                )
            last_ts = int(swaps[-1]["timestamp"])
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run data-quality checks. Returns a report and drops bad rows in place.

        Checks: nulls, duplicate (block, price), non-positive prices, and time
        gaps. Rejected rows are logged with a reason rather than silently
        dropped.
        """
        report: dict[str, Any] = {"input_rows": len(df), "rejected": {}}

        null_mask = df[["effective_spot_price", "amount0", "amount1"]].isnull().any(axis=1)
        report["rejected"]["null_fields"] = int(null_mask.sum())

        bad_price_mask = df["effective_spot_price"] <= 0
        report["rejected"]["non_positive_price"] = int(bad_price_mask.sum())

        zero_amount_mask = (df["amount0"] == 0) | (df["amount1"] == 0)
        report["rejected"]["zero_amount"] = int(zero_amount_mask.sum())

        drop_mask = null_mask | bad_price_mask | zero_amount_mask
        df.drop(index=df[drop_mask].index, inplace=True)

        dup_mask = df.duplicated(subset=["block_number", "effective_spot_price", "amount0"])
        report["rejected"]["duplicates"] = int(dup_mask.sum())
        df.drop(index=df[dup_mask].index, inplace=True)

        # Time-gap check on hourly bars derived from swap timestamps.
        hours = df["timestamp"].dt.floor("h").drop_duplicates().sort_values()
        if len(hours) > 1:
            gaps = hours.diff().dropna()
            big_gaps = gaps[gaps > pd.Timedelta(hours=2)]
            report["time_gaps_over_2h"] = int(len(big_gaps))
        else:
            report["time_gaps_over_2h"] = 0

        # Staleness: most recent swap vs. configured end date.
        most_recent = df["timestamp"].max()
        report["most_recent_swap"] = str(most_recent)
        report["output_rows"] = len(df)
        report["passed"] = report["output_rows"] > 0
        return report

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #

    def compute_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample raw swaps into time bars with microstructure features.

        Produces one row per bar (config bar_frequency) with:
        midprice (VWAP), midprice_ma, spread, volume_usd, liquidity, n_swaps.
        """
        freq = self.config["bar_frequency"]
        ma_window = int(self.config["ma_window_bars"])
        df = df.copy()
        df["bar"] = df["timestamp"].dt.floor(freq.replace("1", "") if freq.startswith("1") else freq)
        df["abs_usd"] = df["amount0"].abs()

        def _agg(group: pd.DataFrame) -> pd.Series:
            w = group["abs_usd"]
            px = group["effective_spot_price"]
            vwap = float((px * w).sum() / w.sum()) if w.sum() > 0 else float(px.mean())
            return pd.Series(
                {
                    "midprice": vwap,
                    "high": float(px.max()),
                    "low": float(px.min()),
                    "spread": float(px.max() - px.min()),
                    "volume_usd": float(w.sum()),
                    "liquidity": float(group["liquidity"].mean()),
                    "n_swaps": int(len(group)),
                }
            )

        bars = df.groupby("bar", group_keys=True).apply(_agg, include_groups=False)
        bars.index.name = "timestamp"
        bars = bars.sort_index()
        bars["midprice_ma"] = bars["midprice"].rolling(ma_window, min_periods=1).mean()
        bars["log_return"] = np.log(bars["midprice"]).diff()
        bars = bars.reset_index()
        return bars

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #

    def store_to_db(self, df: pd.DataFrame, table_name: str) -> None:
        """Write (replace) a DataFrame into DuckDB."""
        self.conn.register("_tmp_df", df)
        self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
        self.conn.unregister("_tmp_df")

    def load_bars(self, table_name: str = "uniswap_bars") -> pd.DataFrame:
        df = self.conn.execute(f"SELECT * FROM {table_name} ORDER BY timestamp").fetchdf()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def build(self) -> dict[str, Any]:
        """Full pipeline: fetch -> validate -> features -> store. Returns report."""
        raw = self.fetch_historical_swaps()
        report = self.validate(raw)
        bars = self.compute_microstructure(raw)
        self.store_to_db(raw, "uniswap_swaps")
        self.store_to_db(bars, "uniswap_bars")
        report["bars"] = len(bars)
        report["bar_frequency"] = self.config["bar_frequency"]
        return report
