"""Markdown / HTML report generation for a dealer-positioning analysis."""

from __future__ import annotations

import html

import pandas as pd

from dealer_gex.analytics import Analysis, fmt_dollars

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


def build_markdown(a: Analysis, ticker: str = "") -> str:
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
        f"| Call wall | {a.call_wall:,.2f} (strike {a.call_wall_strike:,.0f}) | Peak aggregate dealer call gamma — rallies tend to stall/pin here |",
        f"| Put wall | {a.put_wall:,.2f} (strike {a.put_wall_strike:,.0f}) | Peak aggregate dealer put gamma — selloffs tend to accelerate below, or find support at, this level |",
        f"| Max pain | {a.max_pain:,.2f} | Level minimizing option-holder payout at expiry |",
        f"| Net GEX | {fmt_dollars(a.total_gex)} / 1% move | Total dealer hedging demand per 1% move in spot |",
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
        f"risk-free rate {a.rate:.2%}.",
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
