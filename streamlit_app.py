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

from dealer_gex.analytics import (
    Analysis, analyze, block_levels, confluence_levels, darkpool_levels,
    BLOCK_TIER_NOTE, block_tier_summary, block_type_breakdown, data_quality,
    flow_books, flow_type_breakdown, fmt_dollars, institutional_mask,
    intraday_flow, level_hit_rate, magnet_levels, oi_levels, oi_walls,
    tuned_layer_weights,
)

_CONF_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}
from dealer_gex.forecast import (
    ExpectedMoveForecast, forecast_expected_move, lightgbm_available,
)
from dealer_gex.parsing import (
    ChainParseError, ParsedFile, aggregate_prints, normalize_chain, parse_file,
)
from dealer_gex.report import (
    build_markdown, build_playbook, executive_summary, key_ladder,
    markdown_to_html, regime_text,
)

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
def _parse(file_bytes: bytes):
    return parse_file(file_bytes)


@st.cache_data(show_spinner=False)
def _analyze_cached(chain, spot, asof, rate, weight, multiplier):
    """Cached analyze — reused by the scenario slider and history loop so
    unchanged (spot, IV) states don't re-run the GEX grid + flip bisection."""
    return analyze(chain, spot, asof, rate, weight=weight, multiplier=multiplier)


@st.cache_data(show_spinner=False)
def _main_bundle(chain, spot, asof, rate, weight, multiplier, prints, dark):
    """All main-view analytics in one cached unit, keyed on the inputs that
    actually change. Keeps the whole level stack stable while the scenario
    slider (which passes a *different* spot) drags — only the scenario's own
    analyze recomputes, so interaction is near-instant on big files."""
    a = analyze(chain, spot, asof, rate, weight=weight, multiplier=multiplier)
    magnets = magnet_levels(a)
    oi_lvls = oi_levels(a)
    blk_books, blk_lvls = {}, pd.DataFrame()
    if prints is not None and institutional_mask(prints).any():
        blk_books = flow_books(prints, a.spot, a.asof, rate, multiplier=multiplier)
        blk_lvls = block_levels(prints, a.spot)
    dark_lvls = darkpool_levels(dark, a.spot) if dark is not None else pd.DataFrame()
    master = confluence_levels(a, magnets, oi_lvls,
                               blk_lvls if not blk_lvls.empty else None,
                               dark_lvls if not dark_lvls.empty else None)
    dq = data_quality(a, prints)
    return a, magnets, oi_lvls, blk_books, blk_lvls, dark_lvls, master, dq


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


def load_data() -> tuple[list, str]:
    """Sidebar data controls. Returns (parsed_files, ticker_label)."""
    st.sidebar.header("Data")
    source = st.sidebar.radio(
        "Option-chain source",
        ["Sample data (synthetic SPY-like)", "Upload CSV"],
        help="Chain snapshots (broker long format, CBOE side-by-side) and "
             "trade-level order-flow exports (QuantData) are auto-detected.",
    )

    if source.startswith("Sample"):
        pf = _parse(SAMPLE_PATH.read_bytes())
        pf = ParsedFile(chain=pf.chain, spots=pf.spots, asof=SAMPLE_ASOF,
                        tickers=pf.tickers, prints=pf.prints)
        return [pf], "SPY (sample)"

    files = st.sidebar.file_uploader(
        "Option chain CSV(s)", type=["csv", "dat", "txt"], accept_multiple_files=True,
        help="Multiple files combine into one book — or, when they cover "
             "different days, can be compared as level history.",
    )
    if not files:
        st.sidebar.info("Upload at least one chain CSV, or switch to sample data.")
        return [], ""

    pfs = []
    for f in files:
        data = f.getvalue()
        try:
            pfs.append(_parse(data))
        except ChainParseError:
            mapped = _manual_mapping_ui(f.name, data)
            if mapped is not None:
                pfs.append(ParsedFile(chain=mapped))
    return pfs, ""


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


TRADING_HOURS = 6.5


def scenario_section(a: Analysis, chain: pd.DataFrame, asof, rate: float,
                     weight: str, multiplier: float) -> None:
    st.subheader("🎛️ Scenario simulator")
    st.caption(
        "Re-price the whole book at a hypothetical spot and IV — see the "
        "regime, flip, and forced hedge flow *before* the market goes there. "
        "Sticky-strike IV, open interest held fixed."
    )
    c1, c2 = st.columns(2)
    spot_h = c1.slider(
        "Hypothetical spot", min_value=round(a.spot * 0.90, 2),
        max_value=round(a.spot * 1.10, 2), value=round(a.spot, 2),
        step=0.01, format="%.2f",
    )
    iv_shift = c2.slider(
        "IV shift (vol points)", -10.0, 10.0, 0.0, 0.5,
        help="Applied to every contract's IV. Vol crush = negative.",
    )

    ch = chain
    if iv_shift:
        ch = chain.assign(iv=(chain["iv"] + iv_shift / 100.0).clip(lower=0.005))
    try:
        s = _analyze_cached(ch, spot_h, asof, rate, weight, multiplier)
    except ValueError as exc:
        st.error(str(exc))
        return

    hedge = -(s.dex - a.dex)  # dealers trade against their delta change
    flips = s.regime != a.regime
    cols = st.columns(4)
    cols[0].metric(
        "Regime at that spot",
        "LONG gamma" if s.regime == "long_gamma" else "SHORT gamma",
        "REGIME FLIPS" if flips else "unchanged", delta_color="off",
    )
    cols[1].metric("Net GEX there", fmt_dollars(s.total_gex),
                   f"{fmt_dollars(s.total_gex - a.total_gex)} vs now", delta_color="off")
    cols[2].metric("Gamma flip there",
                   f"{s.gamma_flip:,.2f}" if s.gamma_flip is not None else "—",
                   help="The flip itself moves when IV shifts.")
    cols[3].metric(
        "Hedge flow on the way",
        f"{'+' if hedge >= 0 else '-'}{fmt_dollars(abs(hedge))}",
        "dealers buy" if hedge >= 0 else "dealers sell", delta_color="off",
        help="Dealer delta change from current spot/IV to the scenario, "
             "hedged by trading the opposite — the mechanical flow released "
             "if price/IV actually go there.",
    )
    if flips:
        st.warning(
            f"At {spot_h:,.2f}{f' with IV {iv_shift:+.1f}pt' if iv_shift else ''} "
            f"the regime flips to **{s.regime.replace('_', ' ')}** — dealer "
            "hedging changes character there "
            f"({'amplifying moves' if s.regime == 'short_gamma' else 'dampening moves'}).",
            icon="🔄",
        )


def metrics_row(a: Analysis, zero_dte: bool = False) -> None:
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

    def flow(x: float) -> str:
        return f"{'+' if x >= 0 else '-'}{fmt_dollars(abs(x))}"

    cols = st.columns(5)
    em_val = f"±{a.expected_move:,.2f}" if a.expected_move is not None else "—"
    if zero_dte:
        em_label, em_note = "Expected move (to close)", "by today's close"
        charm_label, charm_val = "Charm flow / hour", flow(a.charm_flow / TRADING_HOURS)
        charm_help = ("Forced dealer re-hedging per trading hour from delta "
                      "decay — 0DTE decay is realized within the session.")
    else:
        em_label = "Expected move (1σ)"
        em_note = f"into {a.nearest_expiry}" if a.expected_move is not None else None
        charm_label, charm_val = "Charm flow / day", flow(a.charm_flow)
        charm_help = "Forced dealer re-hedging per calendar day from delta decay."
    cols[0].metric(em_label, em_val, em_note, delta_color="off",
                   help="Straddle approximation from near-the-money IV at the nearest expiry.")
    cols[1].metric("Net DEX", fmt_dollars(a.dex),
                   help="Net dealer delta inventory under the standard convention.")
    cols[2].metric("Vanna flow / -1 IV pt", flow(a.vanna_flow),
                   "buying" if a.vanna_flow >= 0 else "selling", delta_color="off",
                   help="Forced dealer re-hedging if implied vol drops one point.")
    cols[3].metric(charm_label, charm_val,
                   "buying" if a.charm_flow >= 0 else "selling", delta_color="off",
                   help=charm_help)
    mode_val = {"volume": "Volume", "flow": "Flow"}.get(a.weight_mode, "OI")
    mode_note = {"volume": "intraday flow", "flow": "signed order flow"}.get(
        a.weight_mode, "positioning")
    cols[4].metric("Weighting", mode_val, mode_note, delta_color="off",
                   help="Volume mode reads today's traded flow (intraday/0DTE); "
                        "Flow mode signs positions from ask/bid side codes; "
                        "OI mode reads standing positioning.")


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


def gex_by_strike_chart(a: Analysis, magnets: pd.DataFrame | None = None) -> None:
    lo, hi = a.spot * 0.88, a.spot * 1.12
    df = a.by_strike.query("@lo <= strike <= @hi")
    fig = go.Figure()
    if a.expected_move is not None:
        fig.add_vrect(x0=max(lo, a.spot - a.expected_move),
                      x1=min(hi, a.spot + a.expected_move),
                      fillcolor=C["wash_pos"], line_width=0,
                      annotation_text="±1σ expected move",
                      annotation_position="bottom right",
                      annotation_font_color=C["muted"])
    flow = a.weight_mode == "flow"
    fig.add_bar(x=df["strike"], y=df["call_gex"] / 1e6,
                name="Calls (net dealer)" if flow else "Calls (dealers long)",
                marker_color=C["call"],
                hovertemplate="strike %{x}<br>call GEX $%{y:,.0f}M<extra></extra>")
    fig.add_bar(x=df["strike"], y=df["put_gex"] / 1e6,
                name="Puts (net dealer)" if flow else "Puts (dealers short)",
                marker_color=C["put"],
                hovertemplate="strike %{x}<br>put GEX $%{y:,.0f}M<extra></extra>")
    if magnets is not None and not magnets.empty:
        for kind, color, symbol in (("magnet", C["call"], "diamond"),
                                    ("accelerator", C["put"], "diamond-open")):
            sub = magnets[magnets["kind"] == kind]
            if sub.empty:
                continue
            fig.add_scatter(
                x=sub["level"], y=[0] * len(sub), mode="markers",
                name=f"{'🧲 Magnet' if kind == 'magnet' else '⚡ Accelerator'}",
                marker=dict(color=color, symbol=symbol, size=11,
                            line=dict(width=1, color=C["ink"])),
                customdata=sub["strength"],
                hovertemplate="%{x:,.2f} · strength %{customdata:.0f}<extra></extra>",
            )
    fig.update_layout(barmode="relative", title="Dealer gamma by strike",
                      yaxis_title="GEX ($M per 1% move)")
    _level_lines(fig, a)
    st.plotly_chart(_style(fig), use_container_width=True)


def magnet_section(a: Analysis, magnets: pd.DataFrame) -> None:
    st.subheader("🧲 Magnet levels")
    if magnets.empty:
        st.info("No gamma concentration peaks found within ±10% of spot.")
        return
    left, right = st.columns([3, 2])
    with left:
        view = magnets.copy()
        view["Kind"] = view["kind"].map({"magnet": "🧲 Magnet", "accelerator": "⚡ Accelerator"})
        view["Level"] = view["level"].map(lambda x: f"{x:,.2f}")
        view["Distance"] = view["distance_pct"].map(lambda x: f"{x:+.1f}%")
        view["Anchor"] = view["anchor_strike"].map(lambda x: f"{x:,.0f}")
        st.dataframe(
            view[["Level", "Kind", "strength", "Distance", "Anchor"]].rename(
                columns={"strength": "Pull"}),
            use_container_width=True, hide_index=True,
            column_config={"Pull": st.column_config.ProgressColumn(
                "Pull", min_value=0, max_value=100, format="%.0f")},
        )
    with right:
        st.markdown(
            "- **🧲 Magnets** — positive dealer-gamma peaks: hedging fades "
            "moves around them, pulling price in. Pin power is strongest "
            "into expiry and while dealers stay net long gamma there.\n"
            "- **⚡ Accelerators** — negative-gamma peaks: hedging pushes "
            "price away, so moves through them tend to extend.\n"
            "- Diamonds on the strike chart mark these levels; hover for "
            "pull strength."
        )


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


def oi_chart(a: Analysis, levels: pd.DataFrame | None = None) -> None:
    lo, hi = a.spot * 0.88, a.spot * 1.12
    df = a.by_strike.query("@lo <= strike <= @hi")
    fig = go.Figure()
    fig.add_bar(x=df["strike"], y=df["call_oi"], name="Call OI", marker_color=C["call"],
                hovertemplate="strike %{x}<br>call OI %{y:,.0f}<extra></extra>")
    fig.add_bar(x=df["strike"], y=-df["put_oi"], name="Put OI", marker_color=C["put"],
                hovertemplate="strike %{x}<br>put OI %{customdata:,.0f}<extra></extra>",
                customdata=df["put_oi"])
    if levels is not None and not levels.empty:
        fig.add_scatter(
            x=levels["level"], y=[0] * len(levels), mode="markers",
            name="OI level",
            marker=dict(color=C["ink"], symbol="diamond", size=11,
                        line=dict(width=1, color=C["muted"])),
            customdata=levels["strength"],
            hovertemplate="%{x:,.2f} · OI weight %{customdata:.0f}<extra></extra>",
        )
    ow = oi_walls(a)
    if ow.call is not None and lo <= ow.call <= hi:
        fig.add_vline(x=ow.call, line_dash="dashdot", line_color=C["call"], line_width=1,
                      annotation_text="call OI wall", annotation_position="top right",
                      annotation_font_color=C["call"])
    if ow.put is not None and lo <= ow.put <= hi:
        fig.add_vline(x=ow.put, line_dash="dashdot", line_color=C["put"], line_width=1,
                      annotation_text="put OI wall", annotation_position="bottom left",
                      annotation_font_color=C["put"])
    fig.update_layout(barmode="relative", title="Open interest by strike (puts shown downward)",
                      yaxis_title="Contracts")
    _level_lines(fig, a)
    st.plotly_chart(_style(fig), use_container_width=True)


def oi_levels_section(a: Analysis, levels: pd.DataFrame) -> None:
    st.subheader("📊 OI levels (raw open interest)")
    ow = oi_walls(a)
    wcols = st.columns(4)
    bs = a.by_strike
    if ow.call is not None:
        n = bs.loc[bs["strike"] == ow.call_strike, "call_oi"].iloc[0]
        wcols[0].metric("Call OI wall", f"{ow.call:,.2f}",
                        f"strike {ow.call_strike:,.0f} · {n:,.0f} contracts",
                        delta_color="off",
                        help="Peak of smoothed call open interest — the classic "
                             "rally cap / pin level, pinpointed off the grid; "
                             "the anchor strike holds the largest raw call OI.")
    if ow.put is not None:
        n = bs.loc[bs["strike"] == ow.put_strike, "put_oi"].iloc[0]
        wcols[1].metric("Put OI wall", f"{ow.put:,.2f}",
                        f"strike {ow.put_strike:,.0f} · {n:,.0f} contracts",
                        delta_color="off",
                        help="Peak of smoothed put open interest — the classic "
                             "support / capitulation level, pinpointed off the "
                             "grid; the anchor strike holds the largest raw put OI.")
    if levels.empty:
        st.info("No open-interest concentration peaks found within ±10% of spot.")
        return
    left, right = st.columns([3, 2])
    with left:
        view = levels.copy()
        view["Side"] = view["side"].map(
            {"call": "📈 Call-heavy", "put": "📉 Put-heavy", "mixed": "⚖️ Mixed"})
        view["Level"] = view["level"].map(lambda x: f"{x:,.2f}")
        view["Distance"] = view["distance_pct"].map(lambda x: f"{x:+.1f}%")
        view["Call OI"] = view["call_oi"].map(lambda x: f"{x:,.0f}")
        view["Put OI"] = view["put_oi"].map(lambda x: f"{x:,.0f}")
        st.dataframe(
            view[["Level", "Side", "strength", "Distance", "Call OI", "Put OI"]]
            .rename(columns={"strength": "Weight"}),
            use_container_width=True, hide_index=True,
            column_config={"Weight": st.column_config.ProgressColumn(
                "Weight", min_value=0, max_value=100, format="%.0f")},
        )
    with right:
        st.markdown(
            "- **Raw OI marks where positions sit** — independent of today's "
            "gamma. These are the classic support/resistance and expiry-pin "
            "levels.\n"
            "- **📈 Call-heavy above spot** tends to cap rallies; **📉 "
            "put-heavy below** tends to catch selloffs; big clusters attract "
            "price into expiry.\n"
            "- The 🧲 magnet table weights this same OI by its hedging force "
            "*today*; this table is the raw standing size. Levels on both "
            "lists are the highest-conviction ones."
        )


def confluence_section(a: Analysis, master: pd.DataFrame, dq: dict) -> None:
    st.subheader("🎯 Master levels (confluence)")
    if dq["level"] != "high":
        icon = "⚠️" if dq["level"] == "low" else "📊"
        st.warning(
            f"{_CONF_ICON[dq['level']]} Data quality: **{dq['level']}**. "
            + " ".join(dq["notes"]) + " Treat levels with extra caution.",
            icon=icon,
        )
    if master.empty:
        st.info("Not enough level structure to score confluence.")
        return
    view = pd.DataFrame({
        "Level": master["level"].map(lambda x: f"{x:,.2f}"),
        "Role": master["role"],
        "score": master["score"],
        "Confidence": master["confidence"].map(lambda c: f"{_CONF_ICON[c]} {c}"),
        "Layers": master["n_layers"],
        "Confirmed by": master["layers"].map(lambda ls: ", ".join(ls)),
        "Distance": master["distance_pct"].map(lambda x: f"{x:+.1f}%"),
    })
    st.dataframe(
        view, use_container_width=True, hide_index=True,
        column_config={"score": st.column_config.ProgressColumn(
            "Confluence", min_value=0, max_value=100, format="%.0f")},
    )
    st.caption(
        "Every level system fused into one ranking — the more independent "
        "layers agree on a price, the higher the score and confidence. "
        "🟢 = 3+ layers confirm, 🟡 = 2, 🔴 = 1 (thin). Trade the top rows; "
        "treat lone-layer levels as tentative."
    )


def history_section(hist: pd.DataFrame) -> None:
    st.subheader("🗓️ Level migration")
    series = [
        ("spot", "Spot", C["ink"], "solid"),
        ("flip", "Gamma flip", C["muted"], "dash"),
        ("call_wall", "Call wall (gamma)", C["call"], "solid"),
        ("put_wall", "Put wall (gamma)", C["put"], "solid"),
        ("call_oi_wall", "Call OI wall", C["call"], "dot"),
        ("put_oi_wall", "Put OI wall", C["put"], "dot"),
    ]
    fig = go.Figure()
    for col, name, color, dash in series:
        if hist[col].notna().any():
            fig.add_scatter(
                x=hist["date"], y=hist[col], mode="lines+markers", name=name,
                line=dict(color=color, width=2, dash=dash), marker=dict(size=8),
                hovertemplate=name + " %{y:,.2f}<extra></extra>",
            )
    fig.update_layout(title="Key levels by day", yaxis_title="Price")
    fig.update_xaxes(type="category")
    st.plotly_chart(_style(fig), use_container_width=True)

    if len(hist) >= 2:
        prev, last = hist.iloc[-2], hist.iloc[-1]
        st.markdown(f"**Change {prev['date']} → {last['date']}:**")
        cols = st.columns(6)
        for i, (col, name, *_rest) in enumerate(series):
            if pd.notna(last[col]) and pd.notna(prev[col]):
                cols[i].metric(name, f"{last[col]:,.2f}",
                               f"{last[col] - prev[col]:+,.2f}", delta_color="off")
        if prev["regime"] != last["regime"]:
            st.warning(
                f"Regime flipped between days: {prev['regime'].replace('_', ' ')} "
                f"→ {last['regime'].replace('_', ' ')}.", icon="🔄",
            )


_VENUE_ICON = {"floor": "🏛️", "auto": "⚡", "cross": "🔁", "cob": "📚",
               "cob auction": "📣", "auction": "📣", "iso": "🏃"}
_SHAPE_ICON = {"block": "🧱", "sweep": "🌊", "split": "✂️", "multi": "🧬"}


def _flow_label(row) -> str:
    """Icons + the precise consolidated type of one print ('' if untyped)."""
    label = row.get("flow_type", "")
    if not label or label == "single":
        return ""
    venue, shape = row.get("flow_venue", ""), row.get("flow_shape", "single")
    icons = f"{_VENUE_ICON.get(venue, '')}{_SHAPE_ICON.get(shape, '')}"
    return f"{icons} {label}".strip()


_TIER_ICON = {"negotiated": "🤝", "facilitated": "📣", "electronic": "⚡",
              "fragment": "🧩"}


def block_type_section(prints: pd.DataFrame, spot: float) -> None:
    """Blocks only, split by *how they printed* — the distinction that
    decides whether a 'block' is someone finding a counterparty for size or
    one leg of an auto-executed spread."""
    br = block_type_breakdown(prints, spot)
    if br.empty:
        return
    tiers = block_tier_summary(br)
    st.markdown(f"**Block types — what the block book is actually made of** "
                f"(spot **{spot:,.2f}**):")

    cols = st.columns(max(len(tiers), 1))
    for col, (_, t) in zip(cols, tiers.iterrows()):
        col.metric(f"{_TIER_ICON.get(t['tier'], '')} {t['tier'].title()}",
                   fmt_dollars(t["premium"]),
                   f"{t['premium_share']:.0%} of block premium · "
                   f"{t['prints']:,.0f} prints",
                   delta_color="off", help=BLOCK_TIER_NOTE.get(t["tier"], ""))

    view = br.copy()
    view["Type"] = [
        f"{_TIER_ICON.get(r['tier'], '')} {r['block_type']}"
        + ("  🔗" if r["tied"] else "")
        for _, r in br.iterrows()
    ]
    view["Premium"] = view["premium"].map(fmt_dollars)
    view["Contracts"] = view["contracts"].map(lambda x: f"{x:,.0f}")
    view["Prints"] = view["prints"].map(lambda x: f"{x:,.0f}")
    view["Median print"] = view["median_premium"].map(fmt_dollars)
    view["Level"] = view["level"].map(
        lambda x: f"{x:,.2f}" if pd.notna(x) else "—")
    view["vs spot"] = view["distance_pct"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    view["Net"] = [
        f"{'🟢' if r['direction'] == 'bought' else '🔴' if r['direction'] == 'sold' else '⚪'} "
        f"{r['direction']} ({r['net_contracts']:+,.0f})"
        for _, r in br.iterrows()
    ]
    view["Share"] = view["premium_share"] * 100
    st.dataframe(
        view[["Type", "Premium", "Contracts", "Prints", "Median print",
              "Level", "vs spot", "Net", "Share"]],
        use_container_width=True, hide_index=True,
        column_config={"Share": st.column_config.ProgressColumn(
            "% block premium", min_value=0, max_value=100, format="%.1f%%")},
    )
    st.caption(
        "🤝 **negotiated** (floor, cross) is the block flow that means the "
        "most — size someone had to find a counterparty for, off the public "
        "book. 📣 **facilitated** (complex-order book, auction) is real size "
        "worked publicly for price improvement. ⚡ **electronic** is the "
        "default route. 🧩 **fragment** is one leg of a spread package — it "
        "can dominate the *print count* while carrying a few percent of "
        "premium, so read premium, not prints. 🔗 marks **stock-tied** "
        "(delta-hedged) prints: a volatility trade, not a directional one — "
        "its delta is neutralized by the accompanying stock. Levels and the "
        "block book exclude fragments by default. **Level** is the "
        "premium-weighted strike that type traded at — where the money "
        "actually sat — and **vs spot** places it against the current "
        "underlying price."
    )


def flow_type_table(prints: pd.DataFrame) -> None:
    """Consolidated premium and quantity per precise execution type —
    floor vs auto vs block vs sweep, never averaged together."""
    br = flow_type_breakdown(prints)
    if br.empty:
        return
    st.markdown("**Consolidated flow by type** — premium and size per "
                "execution type, as the file tagged it:")
    view = br.copy()
    view["Type"] = [
        f"{_VENUE_ICON.get(r['venue'], '')}{_SHAPE_ICON.get(r['shape'], '')} "
        f"{r['flow_type']}" + ("  ⭐" if r["institutional"] else "")
        for _, r in br.iterrows()
    ]  # venue+shape icons, then the file's own label
    view["Premium"] = view["premium"].map(fmt_dollars)
    view["Contracts"] = view["contracts"].map(lambda x: f"{x:,.0f}")
    view["Prints"] = view["prints"].map(lambda x: f"{x:,.0f}")
    view["Avg / print"] = view["avg_premium"].map(fmt_dollars)
    view["Net"] = [
        f"{'🟢' if r['direction'] == 'bought' else '🔴' if r['direction'] == 'sold' else '⚪'} "
        f"{r['direction']} ({r['net_contracts']:+,.0f})"
        for _, r in br.iterrows()
    ]
    view["Share"] = view["premium_share"] * 100
    st.dataframe(
        view[["Type", "Premium", "Contracts", "Prints", "Avg / print", "Net", "Share"]],
        use_container_width=True, hide_index=True,
        column_config={"Share": st.column_config.ProgressColumn(
            "% premium", min_value=0, max_value=100, format="%.1f%%")},
    )
    st.caption(
        "⭐ counts as institutional size — block-shaped, floor (🏛️), or cross "
        "(🔁) prints. ⚡ auto is the electronic default route, not a size "
        "signal, so it is excluded unless the print is also block-tagged. "
        "**Net** is signed by side code: 🟢 customers net bought (dealers "
        "pushed short), 🔴 net sold. A type your file uses that isn't mapped "
        "here shows with its raw code rather than being silently bucketed."
    )


def block_section(a: Analysis, prints: pd.DataFrame, books: dict,
                  lvls: pd.DataFrame) -> None:
    st.subheader("🧱 Block intelligence (smart vs fast money)")

    block_type_section(prints, a.spot)
    with st.expander("All flow types (sweeps, splits, everything else)"):
        flow_type_table(prints)

    if books:
        rows = []
        for name, label, mask in (
            ("blocks", "🧱 Blocks (institutional)", institutional_mask(prints)),
            ("sweeps", "🌊 Sweeps (urgent)", prints["is_sweep"]),
            ("floor", "🏛️ Floor (negotiated)",
             prints["is_floor"] if "is_floor" in prints else pd.Series(False, index=prints.index)),
        ):
            if name not in books:
                continue
            bk = books[name]
            rows.append({
                "Book": label,
                "Prints": f"{int(mask.sum()):,}",
                "Premium": fmt_dollars(prints.loc[mask, "premium"].sum()),
                "Customer delta": fmt_dollars(-bk.dex),
                "Net GEX": fmt_dollars(bk.total_gex),
                "Regime": bk.regime.replace("_", " "),
                "Flip": f"{bk.gamma_flip:,.2f}" if bk.gamma_flip is not None else "—",
                "Put wall": f"{bk.put_wall:,.2f}",
                "Call wall": f"{bk.call_wall:,.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if "blocks" in books and "sweeps" in books:
            b, s = books["blocks"], books["sweeps"]
            b_dir, s_dir = -b.dex, -s.dex
            aligned = (b.regime == s.regime) and (b_dir >= 0) == (s_dir >= 0)
            if aligned:
                st.success(
                    f"**Aligned** — institutional blocks and the urgent tape lean the "
                    f"same way ({'long' if b_dir >= 0 else 'short'}, both "
                    f"{b.regime.replace('_', ' ')}). Higher-conviction read.",
                    icon="✅",
                )
            else:
                msg = (
                    f"**Divergent** — blocks lean **{'long' if b_dir >= 0 else 'short'}** "
                    f"({fmt_dollars(abs(b_dir))} stock-equiv., {b.regime.replace('_', ' ')}) "
                    f"while sweeps lean **{'long' if s_dir >= 0 else 'short'}** "
                    f"({fmt_dollars(abs(s_dir))}, {s.regime.replace('_', ' ')}). "
                    "Patient money vs urgent money disagree: favor blocks on swing "
                    "horizon, sweeps intraday."
                )
                # escape $ so paired amounts aren't parsed as LaTeX math
                st.warning(msg.replace("$", "\\$"), icon="⚔️")

    if not lvls.empty:
        st.markdown("**Block commitment levels** — where negotiated size actually traded:")
        view = lvls.copy()
        view["Level"] = view["level"].map(lambda x: f"{x:,.2f}")
        view["Side"] = view["side"].map({"call": "📈 Calls", "put": "📉 Puts", "mixed": "⚖️ Mixed"})
        view["Direction"] = view["direction"].map(
            {"buy": "🟢 Net bought", "sell": "🔴 Net sold", "mixed": "⚪ Two-way"})
        view["Premium"] = view["premium"].map(fmt_dollars)
        view["Distance"] = view["distance_pct"].map(lambda x: f"{x:+.1f}%")
        st.dataframe(
            view[["Level", "Side", "Direction", "strength", "Premium", "Distance"]]
            .rename(columns={"strength": "Weight"}),
            use_container_width=True, hide_index=True,
            column_config={"Weight": st.column_config.ProgressColumn(
                "Weight", min_value=0, max_value=100, format="%.0f")},
        )
        st.caption(
            "Read the pair: side says *what* traded, direction says *which way*. "
            "Puts net **sold** at a level = institutions comfortable owning risk "
            "there (bullish commitment); puts net **bought** = paid-for protection. "
            "Cross-check against the walls: block-confirmed levels are the "
            "committed ones."
        )
    elif not books:
        st.info("No block prints in this file.")


def hit_rate_section(detail, s, tuned: dict | None = None) -> None:
    if detail is None or s is None or s["tested"] == 0:
        return
    st.subheader("✅ Level hit-rate (did they hold?)")
    if s["pairs"] < 3:
        st.caption(f"Only {s['pairs']} day-pair(s) so far — upload 4-5 daily "
                   "exports for a meaningful rate.")
    cols = st.columns(4)
    rate = s["rate"]
    cols[0].metric("Overall hit-rate", f"{rate:.0%}" if rate is not None else "—",
                   f"{s['held']}/{s['tested']} tested", delta_color="off")
    hc = s["by_confidence"]["high"]
    cols[1].metric("🟢 High-confidence", f"{hc['held']}/{hc['tested']}" if hc["tested"] else "—",
                   "levels held", delta_color="off")
    sup, res = s["by_type"]["support"], s["by_type"]["resistance"]
    cols[2].metric("Support held", f"{sup['held']}/{sup['tested']}" if sup["tested"] else "—",
                   delta_color="off")
    cols[3].metric("Resistance held", f"{res['held']}/{res['tested']}" if res["tested"] else "—",
                   delta_color="off")

    view = detail[detail["tested"]].copy()
    view["Level"] = view["level"].map(lambda x: f"{x:,.2f}")
    view["Held?"] = view["outcome"].map({"held": "✅ held", "broke": "❌ broke"})
    view["Confidence"] = view["confidence"].map(lambda c: f"{_CONF_ICON[c]} {c}")
    view["Set"] = view["from_date"].astype(str) + " → " + view["to_date"].astype(str)
    st.dataframe(
        view[["Set", "Level", "role", "Confidence", "Held?"]].rename(columns={"role": "Role"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Each prior-day confluence level tested against the next day's range "
        "(reconstructed from print reference prices — a proxy for OHLC, small "
        "sample). Validation, not a promise: it shows whether these levels "
        "have actually been respected on this ticker recently. Watch whether "
        "🟢 high-confidence levels hold better than 🔴 lone-layer ones."
    )
    if tuned:
        from dealer_gex.analytics import _FAMILY_LABEL
        up = [f"{_FAMILY_LABEL.get(k, k)} ×{v}" for k, v in sorted(tuned.items())
              if v > 1.0]
        down = [f"{_FAMILY_LABEL.get(k, k)} ×{v}" for k, v in sorted(tuned.items())
                if v < 1.0]
        note = "🎚️ **Confluence tuned from this history** — "
        parts = []
        if up:
            parts.append("up-weighted " + ", ".join(up))
        if down:
            parts.append("down-weighted " + ", ".join(down))
        note += "; ".join(parts) if parts else "all families near neutral"
        note += ". The master table above reflects what actually held here."
        st.info(note.replace("$", "\\$"), icon="🎚️")


def forecast_section(f: ExpectedMoveForecast) -> None:
    """Learned expected move: what positioning says the next session holds,
    against what the option market is charging for it."""
    if f is None or f.baseline_pct is None:
        return
    st.subheader("🤖 Expected-move model (LightGBM)")

    cols = st.columns(4)
    cols[0].metric("Implied (1σ, per session)", f"±{f.implied_sigma:,.2f}",
                   "what options price", delta_color="off",
                   help="The straddle expected move rescaled to one trading "
                        "session — the baseline the model has to beat.")
    if f.used_model:
        rich = f.richness
        cols[1].metric("Model (1σ, per session)", f"±{f.predicted_sigma:,.2f}",
                       f"{rich:.0%} of implied", delta_color="off",
                       help="Implied and model blended by measured skill.")
        cols[2].metric("Out-of-sample skill", f"{f.skill:+.0%}",
                       f"{f.weight:.0%} model weight", delta_color="off",
                       help="1 − model MAE / implied MAE on walk-forward "
                            "predictions the model never trained on.")
    else:
        cols[1].metric("Model (1σ, per session)", "—", "no edge yet",
                       delta_color="off")
        skill = f"{f.skill:+.0%}" if f.skill is not None else "—"
        cols[2].metric("Out-of-sample skill", skill, "0% model weight",
                       delta_color="off")
    cols[3].metric("Scored sessions", f"{f.n_oos}",
                   f"{f.n_train} labelled day-pairs", delta_color="off",
                   help="Every scored session was predicted by a model fit "
                        "only on the days before it.")

    icon = "🤖" if f.used_model else "ℹ️"
    (st.success if f.used_model else st.info)(f.message, icon=icon)

    if not f.backtest.empty:
        bt = f.backtest
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt["target_date"], y=bt["actual"] * 100, name="realized",
            mode="lines+markers", line=dict(color=C["ink"], width=2)))
        fig.add_trace(go.Scatter(
            x=bt["target_date"], y=bt["baseline"] * 100, name="implied",
            mode="lines", line=dict(color=C["muted"], width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(
            x=bt["target_date"], y=bt["predicted"] * 100, name="model",
            mode="lines", line=dict(color=C["call"], width=1.5, dash="dash")))
        fig.update_layout(title="Walk-forward: realized move vs implied vs model")
        fig.update_yaxes(title_text="|move| (% of spot)", ticksuffix="%")
        st.plotly_chart(_style(fig), use_container_width=True)

    if not f.importance.empty and f.used_model:
        top = f.importance.head(8).copy()
        top["Weight"] = top["share"].map(lambda x: f"{x:.0%}")
        st.dataframe(
            top[["feature", "Weight"]].rename(columns={"feature": "Feature"}),
            use_container_width=True, hide_index=True,
        )
        st.caption("What the model leaned on. Importance is descriptive, not "
                   "causal — with this few sessions, read it as a hint.")

    st.caption(
        f"Engine: **{f.engine}**. Target is the next session's absolute move "
        "(close to close from reconstructed session closes, √t-scaled when "
        "uploads skip days); features are the positioning state at each day's "
        "close, so no fit ever sees its own answer. The blend weight *is* the "
        "measured skill — beat implied by 20% out of sample and the model gets "
        "20% of the number. Small samples, reconstructed closes, one ticker: "
        "this is a calibration aid, **not trading advice**."
    )
    if not lightgbm_available():
        st.caption("⚠️ `lightgbm` is not installed — running the ridge "
                   "fallback. `pip install lightgbm` for the gradient-boosted "
                   "model.")


def campaign_section(day_prints: list) -> None:
    from dealer_gex.analytics import block_campaigns

    camps = block_campaigns(day_prints)
    if camps.empty:
        return
    st.subheader("🎯 Block campaigns (multi-day institutional builds)")
    view = pd.DataFrame({
        "Contract": camps.apply(
            lambda r: f"{r['strike']:,.0f}{r['type']} {r['expiry']:%m-%d}"
            if pd.notna(r["expiry"]) else f"{r['strike']:,.0f}{r['type']}", axis=1),
        "Days": camps["days"],
        "Span": camps.apply(lambda r: f"{r['first_day']} → {r['last_day']}", axis=1),
        "Direction": camps["direction"].map(
            {"buy": "🟢 Building (bought)", "sell": "🔴 Building (sold)", "mixed": "⚪ Two-way"}),
        "Net size": camps["net_size"].map(lambda x: f"{x:+,.0f}"),
        "Premium": camps["premium"].map(fmt_dollars),
        "By day": camps["daily"],
    })
    st.dataframe(
        view, use_container_width=True, hide_index=True,
        column_config={"By day": st.column_config.BarChartColumn(
            "Signed size by day", help="Per-day net block flow — the build")},
    )
    st.caption(
        "Contracts hit by blocks on multiple days — the same institution "
        "adding over time, the strongest footprint in the data. A strike "
        "**built up** day after day (consistent direction) is a conviction "
        "position; check whether its strike lines up with your walls."
    )


def intraday_timeline_section(prints: pd.DataFrame) -> None:
    flow = intraday_flow(prints)
    if flow.empty or len(flow) < 3:
        return
    st.subheader("⏱️ Intraday flow timeline")
    fig = go.Figure()
    for col, name, color, dash in (
        ("cum_all", "All flow", C["ink"], "solid"),
        ("cum_block", "Blocks", C["call"], "dot"),
        ("cum_sweep", "Sweeps", C["put"], "dash"),
    ):
        if flow[col].abs().sum() > 0:
            fig.add_scatter(x=flow["time"], y=flow[col], mode="lines", name=name,
                            line=dict(color=color, width=2, dash=dash),
                            hovertemplate=name + " %{y:,.0f}<extra></extra>")
    fig.add_hline(y=0, line_color=C["axis"], line_width=1)
    fig.update_layout(title="Cumulative signed flow through the session",
                      yaxis_title="Net contracts (buy + / sell −)")
    st.plotly_chart(_style(fig), use_container_width=True)
    st.caption(
        "When — and who — the flow landed. Rising = net customer buying "
        "(dealers pushed short); falling = net selling. Compare the block and "
        "sweep lines: patient institutional size vs urgent aggressive flow, "
        "and whether the close was one-sided."
    )


def darkpool_section(dark_lvls: pd.DataFrame, spot: float) -> None:
    st.subheader("🌑 Dark-pool levels (institutional block prints)")
    if dark_lvls.empty:
        st.info("No dark-pool concentration within ±10% of spot.")
        return
    view = dark_lvls.copy()
    view["Level"] = view["level"].map(lambda x: f"{x:,.2f}")
    view["Premium"] = view["premium"].map(fmt_dollars)
    view["Shares"] = view["size"].map(lambda x: f"{x:,.0f}")
    view["Distance"] = view["distance_pct"].map(lambda x: f"{x:+.1f}%")
    st.dataframe(
        view[["Level", "strength", "Premium", "Shares", "Distance"]]
        .rename(columns={"strength": "Weight"}),
        use_container_width=True, hide_index=True,
        column_config={"Weight": st.column_config.ProgressColumn(
            "Weight", min_value=0, max_value=100, format="%.0f")},
    )
    st.caption(
        "Prices where large off-exchange equity blocks concentrated — "
        "institutional interest zones that often act as support/resistance. "
        "Dark-pool prints carry **no direction** (mid-executed) and no "
        "hedging obligation, so this is a confluence layer, not a signal: a "
        "level that also appears in the master table is stronger for it."
    )


def notable_flow_section(prints: pd.DataFrame, spot: float) -> None:
    st.subheader("🐋 Notable flow (biggest premium prints)")
    top = prints.reindex(prints["premium"].sort_values(ascending=False).index).head(10)
    if top["premium"].max() <= 0:
        st.info("No premium data in this file.")
        return
    view = pd.DataFrame({
        "Time": top["trade_time"].dt.strftime("%m-%d %H:%M").fillna("—"),
        "Expiry": pd.to_datetime(top["expiry"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d"),
        "Strike": top["strike"].map(lambda x: f"{x:,.0f}"),
        "C/P": top["type"].astype(str).str.upper().str[0],
        "Side": top["side"].map(lambda s: "Buy (ask)" if s in ("A", "AA")
                                else "Sell (bid)" if s in ("B", "BB") else "Mid"),
        "Size": top["size"].map(lambda x: f"{x:,.0f}"),
        "Premium": top["premium"].map(fmt_dollars),
        "Flags": top.apply(lambda r: " ".join(filter(None, [
            "🟡 golden" if r["is_golden"] else "",
            _flow_label(r),
            "🚨 unusual" if r["is_unusual"] else "",
            "🆕 opening" if r["is_opening"] else "",
        ])) or "—", axis=1),
    })
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(
        "Largest single prints by dollar premium — conviction flow worth "
        "cross-checking against the levels above. Ask-side call buying near a "
        "wall is an attack on it; put buying near the put wall reinforces it."
    )


def playbook_section(a: Analysis, master: pd.DataFrame | None = None,
                     ) -> None:
    st.subheader("Trading interpretation")
    left, right = st.columns([3, 2])
    with left:
        # escape $ so st.markdown doesn't read paired dollars as LaTeX math
        st.markdown("\n".join(
            f"- {b}" for b in build_playbook(a, master=master)
        ).replace("$", "\\$"))
    with right:
        ladder = key_ladder(a).copy()
        ladder["Price"] = ladder["Price"].map(lambda x: f"{x:,.2f}")
        st.dataframe(ladder, use_container_width=True, hide_index=True)
        st.caption("Level ladder — all actionable levels, price-sorted.")


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


def report_section(a: Analysis, ticker: str, hist: pd.DataFrame | None = None,
                   blk_books: dict | None = None,
                   blk_lvls: pd.DataFrame | None = None,
                   master: pd.DataFrame | None = None,
                   dq: dict | None = None,
                   dark_lvls: pd.DataFrame | None = None,
                   forecast: ExpectedMoveForecast | None = None,
                   flow_types: pd.DataFrame | None = None,
                   block_types: pd.DataFrame | None = None) -> None:
    st.subheader("Report")
    md = build_markdown(a, ticker=ticker, history=hist,
                        block_books=blk_books, block_lvls=blk_lvls,
                        master=master, dq=dq, dark_lvls=dark_lvls,
                        forecast=forecast, flow_types=flow_types,
                        block_types=block_types)
    stem = f"dealer-positioning-{a.asof:%Y%m%d}"
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Download report (.md)", md, file_name=f"{stem}.md",
                       mime="text/markdown")
    c2.download_button("Download report (.html)", markdown_to_html(md),
                       file_name=f"{stem}.html", mime="text/html")
    with st.expander("Preview report"):
        st.markdown(md.replace("$", "\\$"))


# --- main --------------------------------------------------------------------

def main() -> None:
    st.title("⚖️ Dealer Positioning Dashboard")
    st.caption(
        "Market-maker hedging obligations from options open interest: net gamma "
        "exposure (GEX), forced-hedging direction, and the levels where it flips."
    )

    pfs_all, default_ticker = load_data()
    if not pfs_all:
        st.info("⬅️ Choose a data source in the sidebar to begin.")
        return

    # dark-pool (equity block) files are a confluence overlay, not chains
    dark_pfs = [pf for pf in pfs_all if pf.dark is not None]
    pfs = [pf for pf in pfs_all if pf.dark is None]
    dark_all = pd.concat([pf.dark for pf in dark_pfs], ignore_index=True) if dark_pfs else None
    if not pfs:
        st.warning("Only a dark-pool file was uploaded — it's an overlay. "
                   "Add an options-chain / order-flow export to analyze.", icon="📊")
        return
    if dark_pfs:
        st.sidebar.caption(f"🌑 Dark-pool overlay: {sum(len(pf.dark) for pf in dark_pfs):,} prints.")

    # multiple files covering different days can be compared as history
    dated = sorted({pf.asof for pf in pfs if pf.asof is not None})
    history = False
    if len(pfs) > 1 and len(dated) > 1:
        mode = st.sidebar.radio(
            f"Multiple days detected ({len(dated)})",
            ["Compare days (level history)", "Combine into one book"],
            help="History mode analyzes each file on its own date and shows "
                 "how the levels migrated. Combining files from different "
                 "days would double-count open interest.",
        )
        history = mode.startswith("Compare")

    chain = pd.concat([pf.chain for pf in pfs], ignore_index=True)
    spots: dict = {}
    for pf in pfs:
        spots.update(pf.spots)
    default_asof = max(dated) if dated else date.today()

    st.sidebar.header("Parameters")
    file_tickers = []
    if "ticker" in chain.columns:
        file_tickers = [t for t in chain["ticker"].value_counts().index if str(t)]
    if len(file_tickers) > 1:
        ticker = st.sidebar.selectbox(
            "Ticker (from file)", file_tickers,
            help="This file contains several tickers; each is analyzed separately.",
        )
        chain = chain[chain["ticker"] == ticker]
    elif len(file_tickers) == 1:
        ticker = st.sidebar.text_input("Ticker", value=str(file_tickers[0]))
    else:
        ticker = st.sidebar.text_input("Ticker (label only)", value=default_ticker)

    # conviction filter: rebuild the book from flagged prints only
    all_prints = [pf.prints for pf in pfs if pf.prints is not None]
    conv_picked: list[str] = []
    if all_prints:
        conv_picked = st.sidebar.multiselect(
            "Conviction filter (flow files)",
            ["Sweeps", "Blocks", "Floor", "Cross", "Auto", "Splits",
             "Golden sweeps", "Unusual", "Opening positions"],
            help="Rebuild every level from flagged prints only. Sweeps = "
                 "urgent aggressive flow; blocks = large negotiated "
                 "institutional trades; floor/cross = negotiated off the "
                 "electronic book; auto = electronic default route; splits = "
                 "one order worked across executions. Empty = all prints.",
        )

    _conv_flags = {
        "Sweeps": "is_sweep", "Blocks": "is_block", "Splits": "is_split",
        "Floor": "is_floor", "Cross": "is_cross", "Auto": "is_auto",
        "Golden sweeps": "is_golden", "Unusual": "is_unusual",
        "Opening positions": "is_opening",
    }

    def _conv_mask(p: pd.DataFrame) -> pd.Series:
        m = pd.Series(False, index=p.index)
        for label, col in _conv_flags.items():
            if label in conv_picked and col in p.columns:
                m |= p[col]
        return m

    def _build_chain(pf: ParsedFile) -> pd.DataFrame | None:
        ch = pf.chain
        if conv_picked and pf.prints is not None:
            picked_prints = pf.prints[_conv_mask(pf.prints)]
            if picked_prints.empty:
                return None
            ch = aggregate_prints(picked_prints)
        if file_tickers and "ticker" in ch.columns:
            ch = ch[ch["ticker"] == ticker] if len(file_tickers) > 1 else ch
        return None if ch.empty else ch

    if conv_picked and all_prints:
        conv_chains = [c for c in (_build_chain(pf) for pf in pfs) if c is not None]
        if conv_chains:
            chain = pd.concat(conv_chains, ignore_index=True)
            n_conv = sum(int(_conv_mask(p).sum()) for p in all_prints)
            st.sidebar.caption(f"Conviction book: {n_conv:,} prints → {len(chain):,} contracts.")
        else:
            st.sidebar.warning("No prints match the conviction filter — using all prints.")
            conv_picked = []

    inferred_spot = spots.get(ticker, spots.get("", None))
    if history:
        spot, asof = float(inferred_spot or 100.0), default_asof
        st.sidebar.caption("History mode: each day uses its own file's spot "
                           "and trade date; expiry filter is disabled.")
    else:
        spot = st.sidebar.number_input(
            "Spot price", min_value=0.01, value=float(inferred_spot or 100.0),
            format="%.2f", key=f"spot_{ticker}_{inferred_spot}",
            help="Auto-filled from the file (underlying/reference price) when present.",
        )
        asof = st.sidebar.date_input(
            "Analysis as-of date", value=default_asof, key=f"asof_{default_asof}",
            help="Time-to-expiry is measured from this date. For flow exports this "
                 "defaults to the file's last trade date.",
        )
    rate = st.sidebar.number_input("Risk-free rate (%)", 0.0, 15.0, 4.5, 0.25) / 100
    multiplier = st.sidebar.number_input(
        "Contract multiplier", min_value=1.0, value=100.0, step=1.0,
        help="Units of underlying per contract: stocks/ETFs/index options = 100, "
             "ES = 50, NQ = 20, CL = 1000. Only dollar figures scale with this; "
             "levels are unaffected.",
    )

    has_flow = ("net_customer_size" in chain.columns
                and chain["net_customer_size"].abs().sum() > 0)
    weight_options = ["Open interest (positioning)", "Volume (intraday / 0DTE flow)"]
    if has_flow:
        weight_options.append("Signed order flow (side codes)")
    weight = st.sidebar.radio(
        "Weighting", weight_options,
        help="Open interest reads the standing dealer book (updates overnight). "
             "Volume weights by today's traded contracts (intraday/0DTE). "
             "Signed order flow infers dealer positioning from actual trade "
             "direction — ask-side prints = customer bought (dealer short), "
             "bid-side = customer sold (dealer long); mid prints carry no "
             "direction.",
    )
    weight = ("volume" if weight.startswith("Volume")
              else "flow" if weight.startswith("Signed") else "open_interest")

    zero_dte = False
    if not history:
        all_exp = sorted(d for d in chain["expiry"].dt.date.dropna().unique())
        if asof in all_exp:
            zero_dte = st.sidebar.toggle(
                f"0DTE mode — {asof} expiry only",
                help="Analyze only contracts expiring on the as-of date: the "
                     "gamma that actually binds today. Charm switches to "
                     "per-hour and the expected move is to the close.",
            )
        if zero_dte:
            chain = chain[chain["expiry"].dt.date == asof]
        else:
            picked = st.sidebar.multiselect(
                "Expiries", all_exp, default=all_exp,
                help="Near-dated expiries dominate dealer hedging obligations.",
            )
            if picked and len(picked) < len(all_exp):
                chain = chain[chain["expiry"].dt.date.isin(picked)]

    # prints for this ticker (flow-file extras) — needed by both paths
    merged_prints = None
    if all_prints:
        merged_prints = pd.concat(all_prints, ignore_index=True)
        if file_tickers and len(file_tickers) > 1:
            merged_prints = merged_prints[merged_prints["ticker"] == ticker]
        if merged_prints.empty:
            merged_prints = None

    # dark-pool overlay for this ticker
    dark_t = dark_all
    if dark_all is not None and "ticker" in dark_all.columns:
        by_tk = dark_all[dark_all["ticker"] == ticker]
        dark_t = by_tk if not by_tk.empty else (dark_all if not file_tickers else None)

    def _derive(a, prints, dark=None):
        """Level stack from an Analysis + prints (shared by both paths)."""
        magnets = magnet_levels(a)
        oi_lvls = oi_levels(a)
        blk_books, blk_lvls = {}, pd.DataFrame()
        if prints is not None and institutional_mask(prints).any():
            blk_books = flow_books(prints, a.spot, a.asof, rate, multiplier=multiplier)
            blk_lvls = block_levels(prints, a.spot)
        dark_lvls = darkpool_levels(dark, a.spot) if dark is not None else pd.DataFrame()
        master = confluence_levels(a, magnets, oi_lvls,
                                   blk_lvls if not blk_lvls.empty else None,
                                   dark_lvls if not dark_lvls.empty else None)
        return magnets, oi_lvls, blk_books, blk_lvls, dark_lvls, master

    hist = None
    campaign_days: list = []
    hr_days: list = []
    fc_days: list = []
    forecast = None
    if history:
        from dealer_gex.analytics import oi_walls as _oiw, session_range

        rows, analyses = [], []
        for pf in sorted([p for p in pfs if p.asof], key=lambda p: p.asof):
            ch = _build_chain(pf)
            sp = pf.spots.get(ticker, pf.spots.get("", None))
            if ch is None or sp is None:
                continue
            try:
                ai = _analyze_cached(ch, sp, pf.asof, rate, weight, multiplier)
            except ValueError:
                continue
            walls_oi = _oiw(ai)
            rows.append({
                "date": pf.asof, "spot": sp, "flip": ai.gamma_flip,
                "call_wall": ai.call_wall, "put_wall": ai.put_wall,
                "call_oi_wall": walls_oi.call, "put_oi_wall": walls_oi.put,
                "max_pain": ai.max_pain, "net_gex": ai.total_gex,
                "regime": ai.regime,
            })
            analyses.append(ai)
            dp = None
            if pf.prints is not None:
                dp = pf.prints
                if file_tickers and len(file_tickers) > 1:
                    dp = dp[dp["ticker"] == ticker]
                campaign_days.append((pf.asof, dp))
            # per-day confluence + reconstructed range for hit-rate validation
            *_, day_master = _derive(ai, dp)
            rng = session_range(dp) if dp is not None else None
            if rng is not None:
                hr_days.append({"date": pf.asof, "master": day_master,
                                "low": rng[0], "high": rng[1], "close": rng[2]})
            # expected-move model: positioning state + realized session move
            fc_days.append({
                "date": pf.asof, "analysis": ai,
                "close": rng[2] if rng is not None else sp,
                "low": rng[0] if rng is not None else None,
                "high": rng[1] if rng is not None else None,
            })
        if not analyses:
            st.error("No day could be analyzed — check that each file carries "
                     "a spot price and unexpired contracts.")
            return
        a = analyses[-1]
        hist = pd.DataFrame(rows)
        st.info(f"History mode: levels below are for the latest day "
                f"(**{a.asof}**); the migration view covers {len(hist)} days.",
                icon="🗓️")
        magnets, oi_lvls, blk_books, blk_lvls, dark_lvls, master = _derive(
            a, merged_prints, dark_t)
        dq = data_quality(a, merged_prints)
        # validate prior levels, then tune the latest master by what held
        hr_detail = hr_summary = None
        tuned_w: dict = {}
        if len(hr_days) >= 2:
            hr_detail, hr_summary = level_hit_rate(hr_days)
            tuned_w = tuned_layer_weights(hr_detail)
            if tuned_w:
                master = confluence_levels(
                    a, magnets, oi_lvls,
                    blk_lvls if not blk_lvls.empty else None,
                    dark_lvls if not dark_lvls.empty else None, weights=tuned_w)
        if len(fc_days) >= 2:
            forecast = forecast_expected_move(fc_days)
    else:
        try:
            a, magnets, oi_lvls, blk_books, blk_lvls, dark_lvls, master, dq = _main_bundle(
                chain, spot, asof, rate, weight, multiplier, merged_prints, dark_t)
        except ValueError as exc:
            st.error(f"{exc} — check the as-of date against the chain's expiries.")
            return

    if weight == "volume" and chain["volume"].sum() == 0:
        st.warning("This file has no volume data — volume weighting shows nothing. "
                   "Switch back to open interest.", icon="⚠️")
    if inferred_spot is None:
        st.warning(
            "No underlying price found in the file — set the spot price in the "
            "sidebar. All levels depend on it.", icon="📍",
        )

    # --- render ---
    verdict_banner(a)
    st.markdown("**TL;DR** — "
                + executive_summary(a, master, forecast).replace("$", "\\$"))
    if zero_dte:
        st.caption(f"⏱️ 0DTE mode: {a.n_contracts:,} contracts expiring {a.asof} — "
                   "this is the gamma that binds into today's close.")
    metrics_row(a, zero_dte)

    confluence_section(a, master, dq)

    if hist is not None and len(hist) >= 2:
        history_section(hist)
        hit_rate_section(hr_detail, hr_summary, tuned_w)
        forecast_section(forecast)
        if len(campaign_days) >= 2:
            campaign_section(campaign_days)

    gex_by_strike_chart(a, magnets)
    col1, col2 = st.columns(2)
    with col1:
        gex_curve_chart(a)
    with col2:
        oi_chart(a, oi_lvls)

    magnet_section(a, magnets)
    oi_levels_section(a, oi_lvls)

    if dark_t is not None:
        darkpool_section(dark_lvls, a.spot)

    if blk_books or not blk_lvls.empty:
        block_section(a, merged_prints, blk_books, blk_lvls)

    if not history:
        scenario_section(a, chain, asof, rate, weight, multiplier)

    if merged_prints is not None:
        intraday_timeline_section(merged_prints)
        notable_flow_section(merged_prints, a.spot)

    playbook_section(a, master)
    tables(a)
    report_section(a, ticker, hist, blk_books, blk_lvls, master, dq,
                   dark_lvls, forecast,
                   flow_type_breakdown(merged_prints) if merged_prints is not None else None,
                   block_type_breakdown(merged_prints) if merged_prints is not None else None)

    with st.expander("Methodology & assumptions"):
        st.markdown(
            f"""
- **Convention** — dealers are assumed **long customer-sold calls, short
  customer-bought puts** (the standard GEX convention), so call open interest
  contributes positive dealer gamma and put OI negative. Real dealer books can
  differ; treat every level as an estimate.
- **GEX** = gamma × OI × multiplier × spot² × 1% — dollar hedging demand per
  1% move (contract multiplier {a.multiplier:g}; set it in the sidebar for
  futures options like ES=50 or NQ=20).
- **Gamma flip** — net GEX recomputed across a ±15% spot grid (Black-Scholes
  gamma from each contract's IV, held fixed); the zero crossing nearest spot,
  bisected to cent precision.
- **Walls** — pinpoint levels, not strike-rounded: the exact spot level where
  each side's aggregate dollar gamma peaks, searched around the heaviest
  strike (shown as the anchor). Price tends to pin at walls in a long-gamma
  regime and accelerate through them in a short-gamma regime.
- **Vanna / charm flows** — Black-Scholes estimates of dealer re-hedging
  forced by a 1-point IV drop and by one day of delta decay. **Expected
  move** is the 1σ straddle approximation from near-the-money IV at the
  nearest expiry.
- **Expected-move model** (multi-day uploads) — a LightGBM regression from
  the day's positioning state to the *next session's* realized absolute
  move. Features are known at each day's close, the score is
  expanding-window walk-forward, and the model is blended into the implied
  move at exactly its measured out-of-sample skill: no skill, no weight.
- **Weighting** — open interest reads the standing book (updates
  overnight); volume mode weights by today's traded contracts for
  intraday/0DTE reads.
- Missing greeks are filled with Black-Scholes gamma (rate {rate:.2%}).
  Expired contracts are excluded. Open interest updates daily — this is
  positioning analysis, **not trading advice**.
- Analyzing **{a.n_contracts:,} contracts** across **{len(a.expiries)}
  expiries**, max pain {a.max_pain:,.2f}.
"""
        )


main()
