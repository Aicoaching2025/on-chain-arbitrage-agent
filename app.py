"""Streamlit dashboard for the On-Chain Arbitrage Agent.

Reads the JSON artifacts produced by scripts/run_backtest.py and
scripts/walk_forward.py and renders signals, equity/underwater curves, the
trade log, the risk-agent decision log, and walk-forward robustness.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "data", "backtest_results.json")
WALK = os.path.join(HERE, "data", "walk_forward_results.json")

st.set_page_config(layout="wide", page_title="On-Chain Arbitrage Agent", page_icon="⚡")

# Subdued on-chain palette.
GREEN, RED, BLUE, GREY = "#16c784", "#ea3943", "#3861fb", "#8a8d98"


@st.cache_data
def load_results(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def signal_badge(sig: str) -> str:
    color = {"BUY": GREEN, "SELL": RED}.get(sig, GREY)
    return f"<span style='background:{color};color:white;padding:4px 14px;border-radius:6px;font-weight:700'>{sig}</span>"


@st.cache_resource(show_spinner="First load: building dataset and running backtest (~1 min)…")
def bootstrap() -> bool:
    """Generate results on a fresh deploy if the committed artifacts are absent.

    Lets the app stand up anywhere (Streamlit Cloud, HF Spaces, a clean clone)
    without a manual build step. Runs once per container, then cached.
    """
    import sys
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    import run_backtest  # noqa: E402
    import walk_forward  # noqa: E402
    run_backtest.main()
    walk_forward.main()
    return True


st.title("⚡ On-Chain DEX Arbitrage Agent")
st.caption("Statistical mean-reversion on Uniswap v3 microstructure · backtested with gas / slippage / MEV costs · autonomous risk monitoring")

results = load_results(RESULTS)
walk = load_results(WALK)

if results is None:
    bootstrap()
    load_results.clear()
    results = load_results(RESULTS)
    walk = load_results(WALK)

if results is None:
    st.error("Could not generate results. Check the logs / run `python scripts/run_backtest.py`.")
    st.stop()

metrics = results["metrics"]
cfg = results.get("config", {})

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
c2.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
c3.metric("Win Rate", f"{metrics['win_rate']:.1%}")
c4.metric("Total Return", f"{metrics['total_return_pct']:.1f}%")
c5.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
c6.metric("Total Trades", f"{metrics['total_trades']}")

st.divider()

# --------------------------------------------------------------------------- #
# Current signal + portfolio state  |  microstructure
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 2])

signals_preview = pd.DataFrame(results.get("signals_preview", []))
with left:
    st.subheader("Current Signal & State")
    if not signals_preview.empty:
        last = signals_preview.iloc[-1]
        st.markdown(signal_badge(last["signal"]), unsafe_allow_html=True)
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Spot Price", f"${last['price']:,.2f}")
        m2.metric("MA (band center)", f"${last['ma']:,.2f}" if pd.notna(last["ma"]) else "—")
        z = last["zscore"]
        m1.metric("Z-Score", f"{z:.2f}" if pd.notna(z) else "—")
        m2.metric("Z-Threshold", f"±{cfg.get('z_threshold', 2.0)}")
        st.caption(
            f"Risk limits — max delta {cfg.get('max_portfolio_delta')} · "
            f"daily DD {cfg.get('max_daily_drawdown_pct')}% · "
            f"stop-loss {cfg.get('stop_loss_pct')}%"
        )

with right:
    st.subheader("Microstructure — Price vs. MA Band (recent)")
    if not signals_preview.empty:
        df = signals_preview.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        upper = df["ma"] + cfg.get("z_threshold", 2.0) * df["vol"]
        lower = df["ma"] - cfg.get("z_threshold", 2.0) * df["vol"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=lower, fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(56,97,251,0.10)", name="±z band"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ma"], line=dict(color=GREY, dash="dot"), name="MA"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price"], line=dict(color=BLUE), name="Spot"))
        buys = df[df["signal"] == "BUY"]
        sells = df[df["signal"] == "SELL"]
        fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["price"], mode="markers",
                                 marker=dict(color=GREEN, size=7, symbol="triangle-up"), name="BUY"))
        fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["price"], mode="markers",
                                 marker=dict(color=RED, size=7, symbol="triangle-down"), name="SELL"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(fig, width="stretch")

st.divider()

# --------------------------------------------------------------------------- #
# Equity curve + underwater
# --------------------------------------------------------------------------- #
st.subheader("Equity Curve & Drawdown (full backtest)")
eq = results["equity_curve"]
dates = pd.to_datetime(results["equity_dates"])
under = results["underwater"]
fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04,
                     subplot_titles=("Equity (USD)", "Underwater (drawdown %)"))
fig2.add_trace(go.Scatter(x=dates, y=eq, line=dict(color=BLUE), name="Equity"), row=1, col=1)
fig2.add_trace(go.Scatter(x=dates, y=[u * 100 for u in under], fill="tozeroy",
                          line=dict(color=RED), name="Drawdown"), row=2, col=1)
fig2.update_layout(height=430, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
st.plotly_chart(fig2, width="stretch")

mr = results.get("monthly_returns", {})
if mr:
    with st.expander("Monthly returns"):
        mr_df = pd.DataFrame({"month": list(mr.keys()), "return_%": [v * 100 for v in mr.values()]})
        bar = go.Figure(go.Bar(x=mr_df["month"], y=mr_df["return_%"],
                               marker_color=[GREEN if v >= 0 else RED for v in mr_df["return_%"]]))
        bar.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="%")
        st.plotly_chart(bar, width="stretch")

st.divider()

# --------------------------------------------------------------------------- #
# Trade log + decision log
# --------------------------------------------------------------------------- #
tl_col, dl_col = st.columns(2)
with tl_col:
    st.subheader("Trade Log")
    trades = pd.DataFrame(results["trade_log"])
    if not trades.empty:
        st.dataframe(
            trades[["entry_date", "exit_date", "side", "entry_price", "exit_price",
                    "pnl", "return_pct", "exit_reason"]].iloc[::-1],
            width="stretch", height=360, hide_index=True,
        )
        st.caption(f"{len(trades)} trades · avg P&L ${metrics['avg_trade_pnl']:.2f}")

with dl_col:
    st.subheader("Risk Agent Decision Log")
    dl = pd.DataFrame(results.get("decision_log", []))
    if not dl.empty:
        rejections = (dl["decision"] == "REJECT").sum()
        st.dataframe(dl[["timestamp", "decision_type", "decision", "reasons"]].iloc[::-1],
                     width="stretch", height=360, hide_index=True)
        st.caption(f"{len(dl)} logged decisions · {rejections} rejections/forced exits")
    else:
        st.info("No risk decisions logged.")

st.divider()

# --------------------------------------------------------------------------- #
# Walk-forward robustness
# --------------------------------------------------------------------------- #
st.subheader("Walk-Forward Validation (out-of-sample)")
if walk is None:
    st.info("Run `python scripts/walk_forward.py` to populate walk-forward results.")
else:
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Avg Sharpe", f"{walk['avg_sharpe']:.2f}")
    w2.metric("Sharpe Std", f"{walk['std_sharpe']:.2f}")
    w3.metric("Avg Win Rate", f"{walk['avg_win_rate']:.1%}")
    w4.metric("Positive Windows", f"{walk['pct_windows_positive_sharpe']:.0%}")
    wdf = pd.DataFrame(walk["windows"])
    wf = go.Figure(go.Bar(x=wdf["test_start"], y=wdf["sharpe"],
                          marker_color=[GREEN if s >= 0 else RED for s in wdf["sharpe"]]))
    wf.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                     yaxis_title="Sharpe", xaxis_title="test window start")
    st.plotly_chart(wf, width="stretch")
    with st.expander("Per-window detail"):
        st.dataframe(wdf, width="stretch", hide_index=True)

st.caption("⚠️ Backtested POC on synthetic Uniswap-v3-shaped data — not live trading. See README for methodology, costs, and limitations.")
