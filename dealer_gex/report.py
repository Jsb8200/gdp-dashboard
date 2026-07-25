"""Markdown / HTML report generation for a dealer-positioning analysis."""

from __future__ import annotations

import html

import pandas as pd

from dealer_gex.analytics import (
    Analysis, block_tier_summary, fmt_dollars, magnet_levels, oi_levels,
    oi_walls,
)

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
    ow = oi_walls(a)
    if ow.call is not None:
        rows.append(("Call OI wall", ow.call,
                     "Largest raw call open interest — classic cap / pin strike"))
    if ow.put is not None:
        rows.append(("Put OI wall", ow.put,
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


def executive_summary(a: Analysis, master: pd.DataFrame | None = None,
                      forecast=None) -> str:
    """One-paragraph TL;DR fusing regime, the top confluence level, and the
    expected move — the single-glance read the rest of the page expands."""
    regime = "long gamma (moves dampened, mean-reverting)" if a.regime == "long_gamma" \
        else "short gamma (moves amplified, trending)"
    bits = [f"Dealers are **{regime}**"]
    if a.gamma_flip is not None:
        bits[0] += f"; the regime flips at **{a.gamma_flip:,.2f}**"
    if master is not None and not master.empty:
        top = master.iloc[0]
        bits.append(
            f"the strongest level is **{top['level']:,.2f}** "
            f"({top['role'].lower()}, {int(top['n_layers'])} systems agree, "
            f"{top['confidence']} confidence)"
        )
    if a.expected_move is not None:
        bits.append(f"options price a ±{a.expected_move:,.2f} move to "
                    f"{a.nearest_expiry}")
    if forecast is not None and forecast.used_model:
        rich = forecast.richness or 1.0
        verdict = ("under-pricing" if rich > 1.05 else
                   "over-pricing" if rich < 0.95 else "fairly pricing")
        bits.append(
            f"the model expects **±{forecast.predicted_sigma:,.2f}** next "
            f"session ({rich:.0%} of implied — the market is {verdict} it)"
        )
    return ". ".join(s[0].upper() + s[1:] for s in bits) + "."


def build_playbook(a: Analysis, master: pd.DataFrame | None = None) -> list[str]:
    """Turn the day's numbers into a trading interpretation (markdown bullets)."""
    long_g = a.regime == "long_gamma"
    lines = []

    if master is not None and not master.empty:
        top = master.head(3)
        named = "; ".join(
            f"**{r['level']:,.2f}** ({r['role'].lower()}, score {r['score']:.0f})"
            for _, r in top.iterrows()
        )
        lines.append(
            f"**Key levels by confluence: {named}.** These are where the most "
            "independent systems agree — trade their reactions first; the "
            "individual walls and magnets below are the components."
        )

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
                   history: pd.DataFrame | None = None,
                   block_books: dict | None = None,
                   block_lvls: pd.DataFrame | None = None,
                   master: pd.DataFrame | None = None,
                   dq: dict | None = None,
                   dark_lvls: pd.DataFrame | None = None,
                   forecast=None,
                   flow_types: pd.DataFrame | None = None,
                   block_types: pd.DataFrame | None = None) -> str:
    title, body = regime_text(a.regime)
    label = f"{ticker.upper()} " if ticker else ""
    flip = f"{a.gamma_flip:,.2f}" if a.gamma_flip is not None else "no crossing in ±15% range"

    lines = [
        f"# {label}Dealer Positioning Report — {a.asof:%Y-%m-%d}",
        "",
        f"**TL;DR** — {executive_summary(a, master, forecast)}",
        "",
        f"## Verdict: {title}",
        "",
        body,
        "",
    ]

    if dq is not None and dq["level"] != "high":
        lines += [f"> **Data quality: {dq['level']}.** {' '.join(dq['notes'])}", ""]

    if forecast is not None and forecast.baseline_pct is not None:
        f = forecast
        lines += [
            "## Expected move — implied vs model",
            "",
            "| Estimate | 1σ move | Range |",
            "|---|---|---|",
            f"| Implied (options, per session) | ±{f.implied_sigma:,.2f} "
            f"| {a.spot - f.implied_sigma:,.2f} – {a.spot + f.implied_sigma:,.2f} |",
        ]
        if f.used_model:
            lines.append(
                f"| Model ({f.engine}, blended) | ±{f.predicted_sigma:,.2f} "
                f"| {a.spot - f.predicted_sigma:,.2f} – "
                f"{a.spot + f.predicted_sigma:,.2f} |"
            )
        lines += ["", f.message, ""]
        if f.used_model:
            lines += [
                f"Skill is measured out of sample: {f.n_oos} walk-forward "
                f"session(s), model MAE {f.mae_model * 100:.3f}% of spot vs "
                f"implied {f.mae_baseline * 100:.3f}%. The model is fit only "
                "on days before each prediction, on positioning features known "
                "at that day's close.",
                "",
            ]
            if not f.importance.empty:
                top = ", ".join(
                    f"{r['feature']} ({r['share']:.0%})"
                    for _, r in f.importance.head(5).iterrows()
                )
                lines += [f"Leading features: {top}.", ""]

    if master is not None and not master.empty:
        lines += [
            "## Master levels (confluence)",
            "",
            "The levels confirmed by the most independent systems — trade the "
            "top rows, treat lone-layer levels as tentative.",
            "",
            "| Level | Role | Confluence | Confidence | Confirmed by |",
            "|---|---|---|---|---|",
        ]
        for _, r in master.iterrows():
            lines.append(
                f"| {r['level']:,.2f} | {r['role']} | {r['score']:.0f} "
                f"| {r['confidence']} ({r['n_layers']} layers) "
                f"| {', '.join(r['layers'])} |"
            )
        lines.append("")

    lines += [
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
    ow = oi_walls(a)
    if ow.call is not None:
        lines.append(
            f"| Call OI wall | {ow.call:,.2f} (strike {ow.call_strike:,.0f}) "
            "| Peak raw call open interest — classic cap / pin level |")
    if ow.put is not None:
        lines.append(
            f"| Put OI wall | {ow.put:,.2f} (strike {ow.put_strike:,.0f}) "
            "| Peak raw put open interest — classic support marker |")
    lines += [
        f"| Net GEX | {fmt_dollars(a.total_gex)} / 1% move | Total dealer hedging demand per 1% move in spot |",
        f"| Net DEX | {fmt_dollars(a.dex)} | Net dealer delta inventory (convention-based) |",
        f"| Vanna flow | {_fmt_flow(a.vanna_flow)} per -1 IV pt | Forced re-hedging if implied vol drops one point |",
        f"| Charm flow | {_fmt_flow(a.charm_flow)} per day | Forced re-hedging from delta decay |",
        "",
        "## Trading interpretation",
        "",
    ]
    for bullet in build_playbook(a, master=master):
        lines.append(f"- {bullet}")

    # The confluence "Master levels" table supersedes the flat price-sorted
    # ladder; only fall back to the ladder when confluence wasn't computed.
    if master is None or master.empty:
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

    if block_books:
        lines += [
            "",
            "## Block intelligence (smart vs fast money)",
            "",
            "| Book | Customer delta | Net GEX | Regime | Flip | Put wall | Call wall |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, label in (("blocks", "Blocks (institutional)"),
                            ("sweeps", "Sweeps (urgent)"),
                            ("floor", "Floor (negotiated)")):
            if name in block_books:
                bk = block_books[name]
                flip_v = f"{bk.gamma_flip:,.2f}" if bk.gamma_flip is not None else "—"
                lines.append(
                    f"| {label} | {fmt_dollars(-bk.dex)} | {fmt_dollars(bk.total_gex)} "
                    f"| {bk.regime.replace('_', ' ')} | {flip_v} "
                    f"| {bk.put_wall:,.2f} | {bk.call_wall:,.2f} |"
                )
        if "blocks" in block_books and "sweeps" in block_books:
            b, s = block_books["blocks"], block_books["sweeps"]
            aligned = (b.regime == s.regime) and ((-b.dex >= 0) == (-s.dex >= 0))
            lines.append("")
            lines.append(
                "Books are **aligned** — higher-conviction read." if aligned else
                "Books are **divergent** — patient block money and the urgent tape "
                "disagree; favor blocks on swing horizon, sweeps intraday."
            )

    if block_types is not None and not block_types.empty:
        tiers = block_tier_summary(block_types)
        lines += [
            "",
            "## Block types (how the size printed)",
            "",
            "| Tier | Premium | % | Prints | Reading |",
            "|---|---|---|---|---|",
        ]
        for _, t in tiers.iterrows():
            lines.append(
                f"| **{t['tier']}** | {fmt_dollars(t['premium'])} "
                f"| {t['premium_share'] * 100:.0f}% | {t['prints']:,.0f} "
                f"| {t['note']} |"
            )
        lines += [
            "",
            "| Block type | Premium | Contracts | Prints | Median print | Net | % |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in block_types.iterrows():
            tied = " (stock-tied)" if r["tied"] else ""
            lines.append(
                f"| {r['block_type']}{tied} | {fmt_dollars(r['premium'])} "
                f"| {r['contracts']:,.0f} | {r['prints']:,.0f} "
                f"| {fmt_dollars(r['median_premium'])} "
                f"| {r['direction']} ({r['net_contracts']:+,.0f}) "
                f"| {r['premium_share'] * 100:.1f}% |"
            )
        lines += [
            "",
            "Read premium, not print count: spread *legs* are fragments of a "
            "package and can be most of the prints while carrying a few "
            "percent of the premium. Stock-tied prints are delta-hedged on "
            "the trade — a volatility position, not a directional one.",
        ]

    if flow_types is not None and not flow_types.empty:
        lines += [
            "",
            "## Consolidated flow by type",
            "",
            "Premium and size per execution type as the file tagged it — floor "
            "and cross prints are negotiated size, auto is the electronic "
            "default route. ★ marks the types counted as institutional.",
            "",
            "| Type | Premium | Contracts | Prints | Avg / print | Net | % premium |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in flow_types.iterrows():
            star = " ★" if r["institutional"] else ""
            lines.append(
                f"| {r['flow_type']}{star} | {fmt_dollars(r['premium'])} "
                f"| {r['contracts']:,.0f} | {r['prints']:,.0f} "
                f"| {fmt_dollars(r['avg_premium'])} "
                f"| {r['direction']} ({r['net_contracts']:+,.0f}) "
                f"| {r['premium_share'] * 100:.1f}% |"
            )

    if block_lvls is not None and not block_lvls.empty:
        lines += [
            "",
            "## Block commitment levels",
            "",
            "| Level | Side | Direction | Weight | Premium | Distance |",
            "|---|---|---|---|---|---|",
        ]
        dir_label = {"buy": "Net bought", "sell": "Net sold", "mixed": "Two-way"}
        side_label = {"call": "Calls", "put": "Puts", "mixed": "Mixed"}
        for _, r in block_lvls.iterrows():
            lines.append(
                f"| {r['level']:,.2f} | {side_label[r['side']]} | {dir_label[r['direction']]} "
                f"| {r['strength']:.0f} | {fmt_dollars(r['premium'])} | {r['distance_pct']:+.1f}% |"
            )

    if dark_lvls is not None and not dark_lvls.empty:
        lines += [
            "",
            "## Dark-pool levels (institutional block prints)",
            "",
            "| Level | Weight | Premium | Distance |",
            "|---|---|---|---|",
        ]
        for _, r in dark_lvls.iterrows():
            lines.append(
                f"| {r['level']:,.2f} | {r['strength']:.0f} "
                f"| {fmt_dollars(r['premium'])} | {r['distance_pct']:+.1f}% |"
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
        *(["- The expected-move model is a gradient-boosted-tree regression "
           "(LightGBM) from the day's positioning state to the next session's "
           "realized absolute move, scored by expanding-window walk-forward "
           "against the implied move and blended at exactly its measured "
           "out-of-sample skill — zero skill, zero weight. Realized moves come "
           "from session closes reconstructed from print reference prices, on "
           "a sample of days, on one ticker."]
          if forecast is not None and forecast.baseline_pct is not None else []),
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
