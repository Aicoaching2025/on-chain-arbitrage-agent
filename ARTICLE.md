# I Built an On-Chain Arbitrage Agent — and the Most Important Number Was the One That *Wasn't* Impressive

On-chain "arbitrage" sounds like free money. Prices on a decentralized exchange wobble around their average; fade the wobble, collect the reversion, repeat. The pitch is seductive — and it's exactly why most backtests of it are quietly dishonest.

So I built the whole thing end to end: a data pipeline, a statistical model, an event-driven backtester, walk-forward validation, an autonomous risk agent, and a live dashboard. Then I let the costs tell the truth.

**🔗 Live dashboard:** https://onchain-arbitrage-agent.streamlit.app/
**💻 Code:** https://github.com/Aicoaching2025/on-chain-arbitrage-agent

## The idea

On Uniswap v3, the spot price of a pair like ETH-USDC drifts above and below its moving average. The hypothesis: when price diverges far enough from that average — measured in standard deviations (a z-score) — it tends to revert. When `|z| > 2`, fade the move. When it reverts, exit.

That's a classic mean-reversion strategy. The interesting part isn't the signal — it's whether the signal *survives contact with reality*.

## The architecture

I deliberately built this as **unified quant infrastructure**, the same shape a real desk would use, just lightweight:

```
Data pipeline (DuckDB)  →  z-score model  →  event-driven backtester
   → walk-forward validation  →  autonomous risk agent  →  live dashboard
```

- **Data pipeline** ingests Uniswap-v3-shaped swap events, validates them (nulls, duplicates, time gaps), and engineers microstructure features (VWAP midprice, moving average, volatility band, liquidity) into hourly bars stored in DuckDB.
- **Model** computes a *causal* rolling z-score — every indicator at time *t* uses only data up to *t*, so there's no look-ahead. (There's a unit test that asserts this.)
- **Backtester** is a custom event loop: one bar at a time, it consults the signal, runs risk gates, simulates a fill, and updates the portfolio. Same input → byte-identical output, every run.
- **Risk agent** sits between signal and execution. It can reject entries (e.g., during volatility spikes) and force exits (stop-loss, max-hold), logging every decision with machine-readable reasoning.

## The part everyone skips: costs

Here's where most arbitrage backtests fall apart. They model the strategy and forget that *trading isn't free*. I priced in three things per round trip:

- **Gas:** ~$2 per transaction (two per round trip).
- **Execution/slippage:** a 5 bps floor representing the 0.05% LP fee tier — the dominant cost for small trades in a deep pool — plus a price-impact term that scales with how much of the pool's depth you consume.
- **MEV:** ~0.5 bps, an order-of-magnitude estimate from public Flashbots data.

For a $10k trade in an ~$80M pool, that's about **$11 per round trip**. Doesn't sound like much — until you realize it's what turns a gross-positive signal into a razor-thin net edge.

## The honest result

After two years of hourly data and all costs:

| Metric | Value |
|---|---|
| Sharpe ratio | **0.67** |
| Max drawdown | **−4.9%** |
| Win rate | **71.9%** |
| Profit factor | **1.09** |

Notice the profit factor: **1.09**. The strategy wins 72% of the time but barely makes money, because costs eat most of the gross edge. That thin number is the whole point. A backtest that handed me a Sharpe of 6 would mean I'd made a mistake — not a fortune.

I validated it the way you're supposed to: **walk-forward**, 16 out-of-sample windows. Average Sharpe **0.73**, but with real dispersion — some windows hit +5, others went negative, and only **56% were profitable**. The edge is *regime-dependent*: it makes money in choppy, mean-reverting markets and bleeds during trends. Hiding that behind a single inflated number would have been the real red flag.

## What the risk agent caught

Across the backtest the agent logged **866 autonomous decisions**: 719 approved entries, **12 rejected** because volatility spiked above 1.8× its normal level (stand aside when the market is abnormal), and **135 forced exits** via stop-loss or max-hold. Every one is timestamped with its reasoning — the entire run is auditable, which is exactly what you'd need before trusting a system with capital.

## What I'd tell a recruiter (or my future self)

Three things this project actually demonstrates:

1. **End-to-end ownership.** Not a notebook — a pipeline, a tested engine, a deployed app with a shareable link.
2. **Cost-awareness and statistical honesty.** The hard part of quant isn't finding a signal; it's proving the signal survives costs and out-of-sample testing. This project leans *into* that instead of away from it.
3. **Reproducibility.** Deterministic backtest, a pytest suite covering determinism / validation / no-look-ahead / cost modeling, and a self-bootstrapping dashboard.

## Honest limitations

The default dataset is synthetic — realistically shaped (mean reversion + volatility clustering + momentum regimes + jumps), but synthetic, because the live Uniswap subgraph now needs a paid API key. A real-data path is wired in behind an API key; the rest of the pipeline is source-agnostic. The slippage model is a linear approximation of a convex curve, MEV is a flat estimate, and execution is assumed instant. All documented in the README.

The absolute numbers are illustrative. The *methodology* is the deliverable — and the discipline to report a 0.67 instead of dressing it up as something it isn't.

**Built with:** Python, pandas, NumPy, statsmodels, DuckDB, Plotly, Streamlit.

**Try it:** https://onchain-arbitrage-agent.streamlit.app/
