"""Markdown / HTML report generation for a dealer-positioning analysis."""

from __future__ import annotations

import html

import pandas as pd

from dealer_gex.analytics import Analysis, fmt_dollars, magnet_levels, oi_levels, oi_walls

_REGIME_TEXT = {
    "long_gamma": (
        "LONG GAMMA — dealers are obliged to hedge against the move",
        "Net dealer gamma is positive: as price rises dealers must sell the "
        "underlying, and as it falls they must buy. This hedging obligation "
        "dampens moves — expect mean-reversion, pinning near large-OI strikes, "
        "and fading of intraday extremes while spot holds above the gamma flip.",
    ),
    "short_gamma": (
        "SHORT GAMMA — dealers are forced to hedge with the move",
        "Net dealer gamma is negative: as price falls dealers must sell more, "
        "and as it rises they must buy more. This hedging obligation amplifies "
        "moves — expect wider ranges, momentum extension, and accelerated "
        "selloffs while spot holds below the gamma flip.",
    ),
}


def regime_text(regime: str) -> tuple[str, str]:
    return _REGIME_TEXT[regime]


def key_ladder(a: Analysis) -> pd.DataFrame:
    """All actionable levels in one price-sorted ladder (highest first)."""
    rows = [
        ("Call wall", a.call_wall, "Rallies stall / pin; premium-selling zone, not a breakout-buy zone"),
        ("Spot", a.spot, "Current price"),
        ("Max pain", a.max_pain, "Expiry gravitation level"),
        ("Put wall", a.put_wall, "Support in long-gamma regime; acceleration marker below the flip"),
    ]
    cw, pw = oi_walls(a)
    if cw is not None:
        rows.append(("Call OI wall", cw,
                     "Largest raw call open interest — classic cap / pin strike"))
    if pw is not None:
        rows.append(("Put OI wall", pw,
                     "Largest raw put open interest — classic support marker"))
    if a.gamma_flip is not None:
        rows.append(("Gamma flip", a.gamma_flip,
                     "Regime switch: stabilizing above, destabilizing below"))
    if a.expected_move is not None:
        rows.append(("+1σ expected move", a.spot + a.expected_move,
                     f"1-sigma range top into {a.nearest_expiry}"))
        rows.append(("-1σ expected move", a.spot - a.expected_move,
                     f"1-sigma range bottom into {a.nearest_expiry}"))
    df = pd.DataFrame(rows, columns=["Level", "Price", "Reading"])
    return df.sort_values("Price", ascending=False).reset_index(drop=True)


def _fmt_flow(x: float) -> str:
    verb = "buying" if x >= 0 else "selling"
    return f"~{fmt_dollars(abs(x))} of {verb}"


def build_playbook(a: Analysis) -> list[str]:
    """Turn the day's numbers into a trading interpretation (markdown bullets)."""
    long_g = a.regime == "long_gamma"
    lines = []

    if long_g:
        lines.append(
            "**Regime — long gamma (stabilizing).** Dealer hedging leans against "
            "price: expect compressed ranges, bought dips, faded breakouts, and "
            "pinning near heavy strikes. Mean-reversion setups are favored over "
            "momentum chasing."
        )
    else:
        lines.append(
            "**Regime — short gamma (destabilizing).** Dealer hedging pushes with "
            "price: expect extended moves, accelerating selloffs, and squeezes. "
            "Momentum is favored; fading moves means fighting forced flow."
        )

    if a.gamma_flip is not None:
        dist = (a.gamma_flip / a.spot - 1) * 100
        side = "below" if a.gamma_flip < a.spot else "above"
        lines.append(
            f"**Watch {a.gamma_flip:,.2f} (gamma flip, {dist:+.1f}%).** The regime "
            f"switch sits {abs(dist):.1f}% {side} spot — a sustained break "
            f"{'below turns the shock absorbers into amplifiers: widen stops and expect range expansion' if long_g else 'above restores dampening: expect moves to lose steam'}."
        )
    else:
        lines.append("**No gamma flip within ±15% of spot** — the current regime is entrenched.")

    lines.append(
        f"**Range frame: {a.put_wall:,.2f} — {a.call_wall:,.2f}** (put wall to call "
        f"wall). Into the call wall, dealer selling intensifies; "
        f"{'toward the put wall, mechanical buying tends to catch price' if long_g else 'through the put wall, hedge selling can accelerate'}."
    )

    if a.expected_move is not None:
        lines.append(
            f"**Options price a ±{a.expected_move:,.2f} (1σ) move** into "
            f"{a.nearest_expiry} ({a.spot - a.expected_move:,.2f} – "
            f"{a.spot + a.expected_move:,.2f}). Compare with the walls: a wall "
            "inside the expected move is likely to be tested; outside, likely to hold."
        )

    magnets = magnet_levels(a)
    pins = magnets[magnets["kind"] == "magnet"]
    below = pins[pins["level"] < a.spot]
    above = pins[pins["level"] >= a.spot]
    if not below.empty or not above.empty:
        parts = []
        if not below.empty:
            r = below.loc[below["level"].idxmax()]
            parts.append(f"{r['level']:,.2f} (pull {r['strength']:.0f}) below")
        if not above.empty:
            r = above.loc[above["level"].idxmin()]
            parts.append(f"{r['level']:,.2f} (pull {r['strength']:.0f}) above")
        lines.append(
            f"**Nearest magnets: {' / '.join(parts)}.** In a quiet "
            f"{'long-gamma tape price drifts toward the stronger pull' if long_g else 'tape magnets bind less while dealers are net short gamma — respect accelerators instead'}."
        )

    oi = oi_levels(a)
    caps = oi[(oi["side"] == "call") & (oi["level"] > a.spot)]
    floors = oi[(oi["side"] == "put") & (oi["level"] < a.spot)]
    if not caps.empty or not floors.empty:
        parts = []
        if not caps.empty:
            r = caps.loc[caps["strength"].idxmax()]
            parts.append(f"call-OI cap at {r['level']:,.2f} (weight {r['strength']:.0f})")
        if not floors.empty:
            r = floors.loc[floors["strength"].idxmax()]
            parts.append(f"put-OI floor at {r['level']:,.2f} (weight {r['strength']:.0f})")
        lines.append(
            f"**Raw OI structure: {' and '.join(parts)}.** These are where "
            "positions actually sit — classic resistance/support and the pin "
            "candidates into expiry, regardless of today's gamma."
        )

    lines.append(
        f"**Passive flows:** IV down 1pt forces {_fmt_flow(a.vanna_flow)} (vanna); "
        f"each day of decay forces {_fmt_flow(a.charm_flow)} (charm). In a quiet "
        "tape these flows lean on price in that direction, typically strongest "
        "into the close and ahead of expiry."
    )

    if a.weight_mode == "volume":
        lines.append(
            "**Volume-weighted (intraday) view** — levels reflect today's traded "
            "flow rather than standing open interest; best for 0DTE reads, "
            "noisier for multi-day positioning."
        )
    elif a.weight_mode == "flow":
        lines.append(
            "**Signed order-flow view** — dealer positioning is inferred from "
            "actual trade direction (ask-side = customer bought, dealer short; "
            "bid-side = customer sold, dealer long), not the standard OI "
            "convention. Walls here are the peaks of positive (pin) and "
            "negative (acceleration) net dealer gamma from the captured flow; "
            "mid-market prints carry no direction and are excluded."
        )
    return lines


def build_markdown(a: Analysis, ticker: str = "",
                   history: pd.DataFrame | None = None) -> str:
    title, body = regime_text(a.regime)
    label = f"{ticker.upper()} " if ticker else ""
    flip = f"{a.gamma_flip:,.2f}" if a.gamma_flip is not None else "no crossing in ±15% range"

    lines = [
        f"# {label}Dealer Positioning Report — {a.asof:%Y-%m-%d}",
        "",
        f"## Verdict: {title}",
        "",
        body,
        "",
        "## Key levels",
        "",
        "| Level | Value | Meaning |",
        "|---|---|---|",
        f"| Spot (as of analysis) | {a.spot:,.2f} | Reference price for all calculations |",
        f"| Gamma flip (zero-gamma) | {flip} | Below: dealers short gamma (destabilizing); above: long gamma (stabilizing) |",
        f"| Call wall (gamma) | {a.call_wall:,.2f} (strike {a.call_wall_strike:,.0f}) | Peak aggregate dealer call gamma — rallies tend to stall/pin here |",
        f"| Put wall (gamma) | {a.put_wall:,.2f} (strike {a.put_wall_strike:,.0f}) | Peak aggregate dealer put gamma — selloffs tend to accelerate below, or find support at, this level |",
        f"| Max pain | {a.max_pain:,.2f} | Level minimizing option-holder payout at expiry |",
    ]
    cw, pw = oi_walls(a)
    if cw is not None:
        lines.append(
            f"| Call OI wall | {cw:,.2f} | Largest raw call open interest — classic cap / pin strike |")
    if pw is not None:
        lines.append(
            f"| Put OI wall | {pw:,.2f} | Largest raw put open interest — classic support marker |")
    lines += [
        f"| Net GEX | {fmt_dollars(a.total_gex)} / 1% move | Total dealer hedging demand per 1% move in spot |",
        f"| Net DEX | {fmt_dollars(a.dex)} | Net dealer delta inventory (convention-based) |",
        f"| Vanna flow | {_fmt_flow(a.vanna_flow)} per -1 IV pt | Forced re-hedging if implied vol drops one point |",
        f"| Charm flow | {_fmt_flow(a.charm_flow)} per day | Forced re-hedging from delta decay |",
        "",
        "## Trading interpretation",
        "",
    ]
    for bullet in build_playbook(a):
        lines.append(f"- {bullet}")

    ladder = key_ladder(a)
    lines += [
        "",
        "## Level ladder",
        "",
        "| Level | Price | Reading |",
        "|---|---|---|",
    ]
    for _, r in ladder.iterrows():
        lines.append(f"| {r['Level']} | {r['Price']:,.2f} | {r['Reading']} |")

    magnets = magnet_levels(a)
    if not magnets.empty:
        lines += [
            "",
            "## Magnet levels",
            "",
            "| Level | Kind | Pull (0-100) | Distance | Anchor strike |",
            "|---|---|---|---|---|",
        ]
        for _, r in magnets.iterrows():
            kind = "Magnet (pull/pin)" if r["kind"] == "magnet" else "Accelerator (repel)"
            lines.append(
                f"| {r['level']:,.2f} | {kind} | {r['strength']:.0f} "
                f"| {r['distance_pct']:+.1f}% | {r['anchor_strike']:,.0f} |"
            )

    if history is not None and len(history) >= 2:
        lines += [
            "",
            "## Level migration",
            "",
            "| Date | Spot | Flip | Call wall | Put wall | Call OI wall | Put OI wall | Max pain | Net GEX | Regime |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in history.iterrows():
            def _n(v):
                return f"{v:,.2f}" if pd.notna(v) else "—"
            lines.append(
                f"| {r['date']} | {_n(r['spot'])} | {_n(r['flip'])} | {_n(r['call_wall'])} "
                f"| {_n(r['put_wall'])} | {_n(r['call_oi_wall'])} | {_n(r['put_oi_wall'])} "
                f"| {_n(r['max_pain'])} | {fmt_dollars(r['net_gex'])} "
                f"| {str(r['regime']).replace('_', ' ')} |"
            )

    oi = oi_levels(a)
    if not oi.empty:
        lines += [
            "",
            "## OI levels (raw open interest)",
            "",
            "| Level | Side | Weight (0-100) | Distance | Call OI | Put OI |",
            "|---|---|---|---|---|---|",
        ]
        side_label = {"call": "Call-heavy", "put": "Put-heavy", "mixed": "Mixed"}
        for _, r in oi.iterrows():
            lines.append(
                f"| {r['level']:,.2f} | {side_label[r['side']]} | {r['strength']:.0f} "
                f"| {r['distance_pct']:+.1f}% | {r['call_oi']:,.0f} | {r['put_oi']:,.0f} |"
            )

    lines += [
        "",
        "## Per-expiry breakdown",
        "",
        "| Expiry | Net GEX | Call OI | Put OI |",
        "|---|---|---|---|",
    ]
    for _, row in a.by_expiry.iterrows():
        lines.append(
            f"| {row['expiry']} | {fmt_dollars(row['net_gex'])} "
            f"| {row['call_oi']:,.0f} | {row['put_oi']:,.0f} |"
        )

    top = a.by_strike.reindex(
        a.by_strike["net_gex"].abs().sort_values(ascending=False).index
    ).head(10)
    lines += [
        "",
        "## Top 10 strikes by |net GEX|",
        "",
        "| Strike | Net GEX | Call GEX | Put GEX | Call OI | Put OI |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['strike']:,.2f} | {fmt_dollars(r['net_gex'])} | {fmt_dollars(r['call_gex'])} "
            f"| {fmt_dollars(r['put_gex'])} | {r['call_oi']:,.0f} | {r['put_oi']:,.0f} |"
        )

    lines += [
        "",
        "## Methodology & assumptions",
        "",
        f"- {a.n_contracts:,} contracts across {len(a.expiries)} expiries; "
        f"risk-free rate {a.rate:.2%}; contract multiplier {a.multiplier:g}; "
        "weighting: "
        + ("today's traded volume (intraday/0DTE view)." if a.weight_mode == "volume"
           else "signed order flow — dealer positions inferred from actual trade "
                "direction via ask/bid side codes, not the OI convention."
           if a.weight_mode == "flow"
           else "open interest (standing positioning)."),
        "- Vanna/charm flows are Black-Scholes estimates of dealer re-hedging "
        "from IV and time changes; the expected move is the 1-sigma straddle "
        "approximation from near-the-money IV at the nearest expiry.",
        "- Dealer positioning uses the standard GEX convention: dealers assumed "
        "long customer-sold calls and short customer-bought puts, so call OI "
        "contributes positive dealer gamma and put OI negative. Actual dealer "
        "books can differ — treat levels as estimates, not guarantees.",
        "- GEX = gamma x OI x 100 x spot^2 x 1%, i.e. dollar hedging demand per 1% move.",
        "- Missing greeks are filled with Black-Scholes gamma from each "
        "contract's implied volatility; the gamma-flip curve re-prices gamma "
        "across hypothetical spot levels holding IV fixed (sticky-strike).",
        "- Levels are pinpoint, not strike-rounded: the flip is bisected to "
        "cent precision, walls are the exact spot level where each side's "
        "aggregate dollar gamma peaks (anchor strike shown alongside), and "
        "max pain is interpolated between strikes.",
        "- Open interest updates once daily; intraday flows are not captured. "
        "This report is analysis of positioning, not trading advice.",
        "",
    ]
    return "\n".join(lines)


def markdown_to_html(md: str, title: str = "Dealer Positioning Report") -> str:
    """Minimal markdown→HTML for the report's own structure (headings,
    tables, bullets, paragraphs) — avoids an extra dependency."""
    out = []
    table: list[str] = []

    def flush_table():
        if not table:
            return
        rows = [r.strip().strip("|").split("|") for r in table]
        out.append('<table><thead><tr>')
        out.extend(f"<th>{html.escape(c.strip())}</th>" for c in rows[0])
        out.append("</tr></thead><tbody>")
        for r in rows[2:]:  # skip separator row
            out.append("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in r) + "</tr>")
        out.append("</tbody></table>")
        table.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            out.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    flush_table()

    body = "\n".join(out)
    return f"""<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 860px; margin: 2rem auto; padding: 0 1rem;
         color: #0b0b0b; background: #fcfcfb; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 1.6em; }}
  table {{ border-collapse: collapse; margin: .75em 0; font-variant-numeric: tabular-nums; }}
  th, td {{ border: 1px solid #e1e0d9; padding: .35em .7em; text-align: left; }}
  th {{ background: #f9f9f7; }}
  li {{ margin: .3em 0; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #ffffff; background: #1a1a19; }}
    th {{ background: #0d0d0d; }}
    th, td {{ border-color: #2c2c2a; }}
  }}
</style>
<body>{body}</body>"""
