"""Dealer Positioning Dashboard.

Detects market-maker hedging obligations from an uploaded option chain:
net gamma exposure (GEX), whether dealers are forced to hedge with or
against the market, and the key levels (gamma flip, call/put walls).
"""

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dealer_gex.analytics import Analysis, analyze, fmt_dollars
from dealer_gex.parsing import ChainParseError, normalize_chain, read_chain
from dealer_gex.report import build_markdown, markdown_to_html, regime_text

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_option_chain.csv"
SAMPLE_ASOF = date(2026, 7, 17)  # quote date baked into the sample chain

st.set_page_config(
    page_title="Dealer Positioning Dashboard",
    page_icon="⚖️",
    layout="wide",
)


# --- palette (validated: dataviz reference instance) -------------------------

def _theme_mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if theme is not None and getattr(theme, "type", "") == "dark" else "light"


_MODE = _theme_mode()
C = {
    "light": dict(
        call="#2a78d6", put="#e34948", line="#2a78d6",
        ink="#0b0b0b", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        good="#0ca30c", good_text="#006300", critical="#d03b3b",
        wash_pos="rgba(42,120,214,0.07)", wash_neg="rgba(227,73,72,0.07)",
    ),
    "dark": dict(
        call="#3987e5", put="#e66767", line="#3987e5",
        ink="#ffffff", muted="#898781", grid="#2c2c2a", axis="#383835",
        good="#0ca30c", good_text="#0ca30c", critical="#d03b3b",
        wash_pos="rgba(57,135,229,0.10)", wash_neg="rgba(230,103,103,0.10)",
    ),
}[_MODE]


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=C["muted"], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"])
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"])
    return fig


# --- data loading ------------------------------------------------------------

@st.cache_data
def _parse(file_bytes: bytes) -> tuple[pd.DataFrame, float | None]:
    return read_chain(file_bytes)


def _manual_mapping_ui(name: str, file_bytes: bytes) -> pd.DataFrame | None:
    """Fallback column-mapping UI when auto-detection fails."""
    try:
        raw = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001 - surface any read failure to the user
        st.sidebar.error(f"{name}: not a readable CSV ({exc})")
        return None
    st.sidebar.warning(f"Couldn't auto-detect columns in **{name}** — map them below.")
    cols = ["(none)"] + list(raw.columns)
    fields = {
        "expiry": "Expiry", "strike": "Strike", "type": "Call/Put",
        "open_interest": "Open interest", "iv": "Implied volatility",
        "volume": "Volume (optional)", "gamma": "Gamma (optional)",
    }
    mapping = {}
    with st.sidebar.expander(f"Column mapping — {name}", expanded=True):
        for field, label in fields.items():
            pick = st.selectbox(label, cols, key=f"map_{name}_{field}")
            mapping[field] = None if pick == "(none)" else pick
    try:
        return normalize_chain(raw, mapping)
    except ChainParseError as exc:
        st.sidebar.info(f"Mapping incomplete: {exc}")
        return None


def load_data() -> tuple[pd.DataFrame | None, float | None, str, date]:
    """Sidebar data controls. Returns (chain, inferred_spot, ticker, default_asof)."""
    st.sidebar.header("Data")
    source = st.sidebar.radio(
        "Option-chain source",
        ["Sample data (synthetic SPY-like)", "Upload CSV"],
        help="Upload broker or CBOE option-chain exports. Calls and puts may be "
             "in separate rows (long format) or side-by-side per strike (CBOE).",
    )

    if source.startswith("Sample"):
        chain, spot = _parse(SAMPLE_PATH.read_bytes())
        return chain, spot, "SPY (sample)", SAMPLE_ASOF

    files = st.sidebar.file_uploader(
        "Option chain CSV(s)", type=["csv", "dat", "txt"], accept_multiple_files=True,
        help="Multiple files are combined (e.g. one file per expiry).",
    )
    if not files:
        st.sidebar.info("Upload at least one chain CSV, or switch to sample data.")
        return None, None, "", date.today()

    chains, spot = [], None
    for f in files:
        data = f.getvalue()
        try:
            chain, inferred = _parse(data)
        except ChainParseError:
            chain, inferred = _manual_mapping_ui(f.name, data), None
        if chain is not None:
            chains.append(chain)
            spot = spot or inferred
    if not chains:
        return None, None, "", date.today()
    return pd.concat(chains, ignore_index=True), spot, "", date.today()


# --- UI sections -------------------------------------------------------------

def verdict_banner(a: Analysis) -> None:
    title, body = regime_text(a.regime)
    long_g = a.regime == "long_gamma"
    color = C["good_text"] if long_g else C["critical"]
    icon = "🛡️" if long_g else "⚠️"
    obligation = (
        "Obligation: <b>buy dips, sell rips</b> (stabilizing)" if long_g
        else "Obligation: <b>sell weakness, buy strength</b> (destabilizing)"
    )
    if a.gamma_flip is not None:
        dist = (a.gamma_flip / a.spot - 1) * 100
        flip_note = f"Gamma flip at <b>{a.gamma_flip:,.2f}</b> ({dist:+.1f}% from spot)."
    else:
        flip_note = "No gamma flip inside ±15% of spot."
    st.markdown(
        f"""<div style="border:1px solid {color}; border-left:6px solid {color};
             border-radius:8px; padding:0.9rem 1.2rem; margin-bottom:0.5rem;">
          <div style="font-size:1.15rem; font-weight:700; color:{color};">
            {icon} {title}</div>
          <div style="margin-top:0.4rem;">{body}</div>
          <div style="margin-top:0.4rem;">{obligation} &nbsp;·&nbsp; {flip_note}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def metrics_row(a: Analysis) -> None:
    cols = st.columns(5)
    flip_val = f"{a.gamma_flip:,.2f}" if a.gamma_flip is not None else "—"
    flip_delta = (
        f"{(a.gamma_flip / a.spot - 1) * 100:+.1f}% vs spot" if a.gamma_flip is not None else None
    )
    cols[0].metric("Spot", f"{a.spot:,.2f}")
    cols[1].metric("Gamma flip", flip_val, flip_delta, delta_color="off")
    cols[2].metric("Call wall", f"{a.call_wall:,.2f}",
                   f"strike {a.call_wall_strike:,.0f}", delta_color="off")
    cols[3].metric("Put wall", f"{a.put_wall:,.2f}",
                   f"strike {a.put_wall_strike:,.0f}", delta_color="off")
    cols[4].metric("Net GEX / 1% move", fmt_dollars(a.total_gex))


def _level_lines(fig: go.Figure, a: Analysis, walls: bool = True) -> None:
    fig.add_vline(x=a.spot, line_dash="dot", line_color=C["ink"], line_width=1,
                  annotation_text="spot", annotation_position="top right",
                  annotation_font_color=C["ink"])
    if a.gamma_flip is not None:
        fig.add_vline(x=a.gamma_flip, line_dash="dash", line_color=C["muted"], line_width=1,
                      annotation_text="flip", annotation_position="bottom left",
                      annotation_font_color=C["muted"])
    if walls:
        fig.add_vline(x=a.call_wall, line_dash="dash", line_color=C["call"], line_width=1,
                      annotation_text="call wall", annotation_position="top right",
                      annotation_font_color=C["call"])
        fig.add_vline(x=a.put_wall, line_dash="dash", line_color=C["put"], line_width=1,
                      annotation_text="put wall", annotation_position="top left",
                      annotation_font_color=C["put"])


def gex_by_strike_chart(a: Analysis) -> None:
    lo, hi = a.spot * 0.88, a.spot * 1.12
    df = a.by_strike.query("@lo <= strike <= @hi")
    fig = go.Figure()
    fig.add_bar(x=df["strike"], y=df["call_gex"] / 1e6, name="Calls (dealers long)",
                marker_color=C["call"],
                hovertemplate="strike %{x}<br>call GEX $%{y:,.0f}M<extra></extra>")
    fig.add_bar(x=df["strike"], y=df["put_gex"] / 1e6, name="Puts (dealers short)",
                marker_color=C["put"],
                hovertemplate="strike %{x}<br>put GEX $%{y:,.0f}M<extra></extra>")
    fig.update_layout(barmode="relative", title="Dealer gamma by strike",
                      yaxis_title="GEX ($M per 1% move)")
    _level_lines(fig, a)
    st.plotly_chart(_style(fig), use_container_width=True)


def gex_curve_chart(a: Analysis) -> None:
    curve = a.curve
    fig = go.Figure()
    if a.gamma_flip is not None:
        fig.add_vrect(x0=curve["spot_level"].min(), x1=a.gamma_flip,
                      fillcolor=C["wash_neg"], line_width=0,
                      annotation_text="short-gamma zone", annotation_position="bottom left",
                      annotation_font_color=C["muted"])
        fig.add_vrect(x0=a.gamma_flip, x1=curve["spot_level"].max(),
                      fillcolor=C["wash_pos"], line_width=0,
                      annotation_text="long-gamma zone", annotation_position="bottom right",
                      annotation_font_color=C["muted"])
    fig.add_scatter(x=curve["spot_level"], y=curve["total_gex"] / 1e9, mode="lines",
                    name="Net dealer GEX", line=dict(color=C["line"], width=2),
                    hovertemplate="spot %{x:,.2f}<br>net GEX $%{y:,.2f}B<extra></extra>")
    fig.add_scatter(x=[a.spot], y=[a.total_gex / 1e9], mode="markers",
                    name="Current spot", marker=dict(color=C["ink"], size=9),
                    hovertemplate="spot %{x:,.2f}<br>net GEX $%{y:,.2f}B<extra></extra>")
    fig.add_hline(y=0, line_color=C["axis"], line_width=1)
    fig.update_layout(title="Net dealer gamma vs spot level",
                      yaxis_title="Net GEX ($B per 1% move)")
    _level_lines(fig, a, walls=False)
    st.plotly_chart(_style(fig), use_container_width=True)


def oi_chart(a: Analysis) -> None:
    lo, hi = a.spot * 0.88, a.spot * 1.12
    df = a.by_strike.query("@lo <= strike <= @hi")
    fig = go.Figure()
    fig.add_bar(x=df["strike"], y=df["call_oi"], name="Call OI", marker_color=C["call"],
                hovertemplate="strike %{x}<br>call OI %{y:,.0f}<extra></extra>")
    fig.add_bar(x=df["strike"], y=-df["put_oi"], name="Put OI", marker_color=C["put"],
                hovertemplate="strike %{x}<br>put OI %{customdata:,.0f}<extra></extra>",
                customdata=df["put_oi"])
    fig.update_layout(barmode="relative", title="Open interest by strike (puts shown downward)",
                      yaxis_title="Contracts")
    _level_lines(fig, a)
    st.plotly_chart(_style(fig), use_container_width=True)


def tables(a: Analysis) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Per-expiry positioning")
        exp = a.by_expiry.copy()
        exp["net_gex"] = exp["net_gex"].map(fmt_dollars)
        exp.columns = ["Expiry", "Net GEX", "Call OI", "Put OI"]
        st.dataframe(exp, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Top strikes by |net GEX|")
        top = a.by_strike.reindex(
            a.by_strike["net_gex"].abs().sort_values(ascending=False).index
        ).head(10)[["strike", "net_gex", "call_oi", "put_oi"]].copy()
        top["net_gex"] = top["net_gex"].map(fmt_dollars)
        top.columns = ["Strike", "Net GEX", "Call OI", "Put OI"]
        st.dataframe(top, use_container_width=True, hide_index=True)


def report_section(a: Analysis, ticker: str) -> None:
    st.subheader("Report")
    md = build_markdown(a, ticker=ticker)
    stem = f"dealer-positioning-{a.asof:%Y%m%d}"
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Download report (.md)", md, file_name=f"{stem}.md",
                       mime="text/markdown")
    c2.download_button("Download report (.html)", markdown_to_html(md),
                       file_name=f"{stem}.html", mime="text/html")
    with st.expander("Preview report"):
        st.markdown(md)


# --- main --------------------------------------------------------------------

def main() -> None:
    st.title("⚖️ Dealer Positioning Dashboard")
    st.caption(
        "Market-maker hedging obligations from options open interest: net gamma "
        "exposure (GEX), forced-hedging direction, and the levels where it flips."
    )

    chain, inferred_spot, default_ticker, default_asof = load_data()
    if chain is None:
        st.info("⬅️ Choose a data source in the sidebar to begin.")
        return

    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Ticker (label only)", value=default_ticker)
    spot = st.sidebar.number_input(
        "Spot price", min_value=0.01, value=float(inferred_spot or 100.0),
        format="%.2f",
        help="Auto-filled when the file includes an underlying price.",
    )
    asof = st.sidebar.date_input(
        "Analysis as-of date", value=default_asof,
        help="Time-to-expiry is measured from this date. Use the chain's quote date.",
    )
    rate = st.sidebar.number_input("Risk-free rate (%)", 0.0, 15.0, 4.5, 0.25) / 100

    all_exp = sorted(d for d in chain["expiry"].dt.date.dropna().unique())
    picked = st.sidebar.multiselect(
        "Expiries", all_exp, default=all_exp,
        help="Near-dated expiries dominate dealer hedging obligations.",
    )
    if picked and len(picked) < len(all_exp):
        chain = chain[chain["expiry"].dt.date.isin(picked)]

    try:
        a = analyze(chain, spot, asof, rate)
    except ValueError as exc:
        st.error(f"{exc} — check the as-of date against the chain's expiries.")
        return

    if inferred_spot is None:
        st.warning(
            "No underlying price found in the file — set the spot price in the "
            "sidebar. All levels depend on it.", icon="📍",
        )

    verdict_banner(a)
    metrics_row(a)

    gex_by_strike_chart(a)
    col1, col2 = st.columns(2)
    with col1:
        gex_curve_chart(a)
    with col2:
        oi_chart(a)

    tables(a)
    report_section(a, ticker)

    with st.expander("Methodology & assumptions"):
        st.markdown(
            f"""
- **Convention** — dealers are assumed **long customer-sold calls, short
  customer-bought puts** (the standard GEX convention), so call open interest
  contributes positive dealer gamma and put OI negative. Real dealer books can
  differ; treat every level as an estimate.
- **GEX** = gamma × OI × 100 × spot² × 1% — dollar hedging demand per 1% move.
- **Gamma flip** — net GEX recomputed across a ±15% spot grid (Black-Scholes
  gamma from each contract's IV, held fixed); the zero crossing nearest spot,
  bisected to cent precision.
- **Walls** — pinpoint levels, not strike-rounded: the exact spot level where
  each side's aggregate dollar gamma peaks, searched around the heaviest
  strike (shown as the anchor). Price tends to pin at walls in a long-gamma
  regime and accelerate through them in a short-gamma regime.
- Missing greeks are filled with Black-Scholes gamma (rate {rate:.2%}).
  Expired contracts are excluded. Open interest updates daily — this is
  positioning analysis, **not trading advice**.
- Analyzing **{a.n_contracts:,} contracts** across **{len(a.expiries)}
  expiries**, max pain {a.max_pain:,.2f}.
"""
        )


main()
