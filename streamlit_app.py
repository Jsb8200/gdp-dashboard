import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path

import detectors as det
import schema

st.set_page_config(
    page_title='Magnitude detector',
    page_icon=':chart_with_upwards_trend:',
    layout='wide',
)

DATA_DIR = Path(__file__).parent / 'data'

# Categorical slots 1-2 + status red, validated per the dataviz method.
CALL_COLOR = '#2a78d6'
PUT_COLOR = '#008300'
FLAG_COLOR = '#e34948'
TYPE_SCALE = alt.Scale(domain=['call', 'put'], range=[CALL_COLOR, PUT_COLOR])

# -----------------------------------------------------------------------------
# Data loading

@st.cache_data
def read_csv_bytes(payload):
    from io import BytesIO
    return pd.read_csv(BytesIO(payload))


@st.cache_data
def load_chain(payload, overrides_items):
    raw = read_csv_bytes(payload)
    return schema.load_chain(raw, dict(overrides_items))


def normalize_price_history(raw):
    """Map a price-history CSV's date/close/iv columns by alias."""
    targets = {
        'date': ['date', 'timestamp', 'day', 'quotedate', 'tradedate'],
        'close': ['close', 'price', 'last', 'adjclose', 'closeprice'],
        'iv': ['iv', 'impliedvolatility', 'impliedvol', 'impvol'],
    }
    out = {}
    for raw_col in raw.columns:
        norm = schema.normalize_header(raw_col)
        for canonical, aliases in targets.items():
            if canonical not in out and norm in aliases:
                out[canonical] = raw[raw_col]
    if 'date' not in out or 'close' not in out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ('close', 'iv'):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['date', 'close'])


def warn_missing(missing):
    lines = '\n'.join(f'- **{c}** (accepted names: {schema.alias_help(c)})'
                      for c in missing)
    st.warning('This detector needs columns your CSV is missing:\n' + lines)


# -----------------------------------------------------------------------------
# Sidebar: uploads + mapping

with st.sidebar:
    st.header('Data')
    chain_file = st.file_uploader('Options chain CSV', type='csv')
    price_file = st.file_uploader('Price history CSV (optional)', type='csv',
                                  help='Columns: date, close, and optionally iv')

    if chain_file is not None:
        chain_bytes = chain_file.getvalue()
        chain_source = chain_file.name
    else:
        chain_bytes = (DATA_DIR / 'sample_options_chain.csv').read_bytes()
        chain_source = 'bundled sample (upload your own above)'
    st.caption(f'Chain: {chain_source}')

overrides = st.session_state.get('mapping_overrides', {})
chain, info = load_chain(chain_bytes, tuple(sorted(overrides.items())))

with st.sidebar:
    with st.expander('Column mapping', expanded=bool(info['unmapped'])):
        st.dataframe(
            pd.DataFrame(
                [{'column': c, 'mapped from': info['mapping'].get(c, '—')}
                 for c in schema.ALIASES]),
            hide_index=True, height=250)
        if info['unmapped'] and info['ignored']:
            st.caption('Assign unmapped columns manually:')
            new_overrides = dict(overrides)
            for canonical in info['unmapped']:
                choice = st.selectbox(
                    canonical, ['(none)'] + info['ignored'],
                    key=f'override_{canonical}')
                if choice != '(none)':
                    new_overrides[canonical] = choice
                else:
                    new_overrides.pop(canonical, None)
            if new_overrides != overrides:
                st.session_state['mapping_overrides'] = new_overrides
                st.rerun()
    for note in info['notes']:
        st.info(note)

    if 'expiration' in chain and chain['expiration'].notna().any():
        expiries = sorted(chain['expiration'].dropna().unique())
        labels = [pd.Timestamp(e).date().isoformat() for e in expiries]
        picked = st.multiselect('Expirations', labels, default=labels)
        keep = [e for e, lab in zip(expiries, labels) if lab in picked]
        attrs = chain.attrs
        chain = chain[chain['expiration'].isin(keep)]
        chain.attrs.update(attrs)

# Price-history source: uploaded file, else derived from a multi-date chain,
# else the bundled sample (only when the chain is the sample too).
if price_file is not None:
    prices = normalize_price_history(read_csv_bytes(price_file.getvalue()))
    price_source = price_file.name
elif ('timestamp' in chain and 'underlying_price' in chain
      and chain['timestamp'].nunique() >= 25):
    prices = (chain.groupby('timestamp')['underlying_price'].median()
              .rename('close').reset_index().rename(columns={'timestamp': 'date'}))
    price_source = 'derived from the options CSV timestamps'
elif chain_file is None:
    prices = normalize_price_history(
        read_csv_bytes((DATA_DIR / 'sample_price_history.csv').read_bytes()))
    price_source = 'bundled sample (upload your own above)'
else:
    prices = pd.DataFrame()
    price_source = None

with st.sidebar:
    if price_source:
        st.caption(f'Price history: {price_source}')

# -----------------------------------------------------------------------------
# Header + metric row

'''
# :chart_with_upwards_trend: Magnitude detector

How big are the moves — priced in, already happening, or hiding in the flow?
Upload an options-chain CSV (any common column naming) and, optionally, a
price history. Four detectors below; each tells you what it needs if a
column is missing.
'''

spot, spot_method = det.infer_spot(chain)
moves = det.implied_move(chain, spot)
ua = det.unusual_activity(chain)

nearest = moves.iloc[0] if not moves.empty else None

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric('Spot', f'{spot:,.2f}' if spot else 'n/a')
    if spot_method:
        st.caption(f'via {spot_method}')
with m2:
    atm_iv = nearest['atm_iv'] if nearest is not None else None
    st.metric('ATM IV (nearest expiry)',
              f'{atm_iv:.1%}' if pd.notna(atm_iv) else 'n/a')
with m3:
    st.metric('Implied move (nearest expiry)',
              f'±{nearest["move_pct"]:.1%}' if nearest is not None else 'n/a')
with m4:
    st.metric('Unusual contracts flagged',
              int(ua['flagged'].sum()) if not ua.empty else 'n/a')

''

tab_move, tab_flow, tab_price, tab_iv = st.tabs(
    ['Implied move', 'Unusual activity', 'Price moves', 'IV spikes'])

# -----------------------------------------------------------------------------
# Tab 1: implied move

with tab_move:
    st.subheader('Expected move per expiry', divider='gray')
    missing = schema.missing_columns(chain, det.IMPLIED_MOVE_REQUIRED)
    if missing:
        warn_missing(missing)
    elif moves.empty:
        st.info('Could not compute implied moves — no strikes with both a '
                'call and a put quote, or no usable spot price.')
    else:
        long = moves.melt(
            ['expiration'], ['straddle_move', 'iv_move'], 'method', 'move_$')
        long['method'] = long['method'].map(
            {'straddle_move': '85% of ATM straddle', 'iv_move': 'IV × √t (1σ)'})
        long['move_%'] = long['move_$'] / spot
        long = long.dropna(subset=['move_$'])

        bar = alt.Chart(long).mark_bar(size=18, cornerRadiusEnd=4).encode(
            x=alt.X('yearmonthdate(expiration):O', title='Expiration'),
            xOffset='method:N',
            y=alt.Y('move_%:Q', title='Expected move', axis=alt.Axis(format='%')),
            color=alt.Color('method:N', title='Method', scale=alt.Scale(
                range=[CALL_COLOR, PUT_COLOR])),
            tooltip=[alt.Tooltip('yearmonthdate(expiration)', title='Expiration'),
                     'method:N',
                     alt.Tooltip('move_%:Q', format='.1%'),
                     alt.Tooltip('move_$:Q', format='$.2f')],
        ).properties(height=280)
        st.altair_chart(bar, width='stretch')

        range_bars = alt.Chart(moves).mark_bar(size=10, color=CALL_COLOR).encode(
            y=alt.Y('yearmonthdate(expiration):O', title='Expiration'),
            x=alt.X('range_low:Q', title='Implied price range',
                    scale=alt.Scale(zero=False)),
            x2='range_high:Q',
            tooltip=[alt.Tooltip('yearmonthdate(expiration)', title='Expiration'),
                     alt.Tooltip('range_low:Q', format='.2f'),
                     alt.Tooltip('range_high:Q', format='.2f')],
        )
        spot_rule = alt.Chart(pd.DataFrame({'spot': [spot]})).mark_rule(
            strokeDash=[4, 3], color='gray').encode(x='spot:Q')
        st.altair_chart((range_bars + spot_rule).properties(height=200),
                        width='stretch')
        st.caption('Bars: implied range per expiry (spot ± expected move). '
                   'Dashed line: spot.')

        st.dataframe(
            moves[['expiration', 'dte', 'atm_strike', 'straddle', 'atm_iv',
                   'straddle_move', 'iv_move', 'move_pct', 'range_low',
                   'range_high']],
            hide_index=True, width='stretch',
            column_config={
                'expiration': st.column_config.DateColumn('Expiration'),
                'dte': 'DTE',
                'atm_strike': st.column_config.NumberColumn('ATM strike', format='%.1f'),
                'straddle': st.column_config.NumberColumn('Straddle $', format='%.2f'),
                'atm_iv': st.column_config.NumberColumn('ATM IV', format='%.1%'),
                'straddle_move': st.column_config.NumberColumn('Move $ (straddle)', format='%.2f'),
                'iv_move': st.column_config.NumberColumn('Move $ (IV)', format='%.2f'),
                'move_pct': st.column_config.NumberColumn('Move %', format='%.1%'),
                'range_low': st.column_config.NumberColumn('Range low', format='%.2f'),
                'range_high': st.column_config.NumberColumn('Range high', format='%.2f'),
            })

# -----------------------------------------------------------------------------
# Tab 2: unusual activity

with tab_flow:
    st.subheader('Unusual options activity', divider='gray')
    missing = schema.missing_columns(chain, det.UNUSUAL_REQUIRED)
    if missing:
        warn_missing(missing)
    else:
        c1, c2, c3 = st.columns(3)
        min_volume = c1.slider('Minimum volume', 0, 1000, 100, step=25)
        ratio_thr = c2.slider('Volume / OI threshold', 0.5, 10.0, 2.0, step=0.25)
        premium_floor = c3.slider('Premium floor ($)', 0, 2_000_000, 100_000,
                                  step=25_000)
        ua = det.unusual_activity(chain, min_volume, ratio_thr, premium_floor)
        if ua.empty:
            st.info('No contracts above the volume floor. Lower the minimum '
                    'volume, or check that the volume column parsed as numbers.')
        else:
            has_oi = 'open_interest' in ua and ua['open_interest'].notna().any()
            if not has_oi:
                st.info('No open-interest column — ranking by premium only.')
            if 'premium' not in ua or ua['premium'].isna().all():
                st.info('No usable prices — ranking by volume/OI only.')

            if has_oi:
                base = ua.dropna(subset=['open_interest', 'volume'])
                pts = alt.Chart(base).mark_circle(size=60).encode(
                    x=alt.X('open_interest:Q', title='Open interest'),
                    y=alt.Y('volume:Q', title='Volume'),
                    color=alt.Color('type:N', title='Type', scale=TYPE_SCALE),
                    size=alt.Size('premium:Q', title='Premium $', legend=None),
                    tooltip=['type:N', 'strike:Q',
                             alt.Tooltip('yearmonthdate(expiration)', title='Expiration'),
                             'volume:Q', 'open_interest:Q',
                             alt.Tooltip('vol_oi:Q', format='.2f'),
                             alt.Tooltip('premium:Q', format='$,.0f'),
                             alt.Tooltip('score:Q', format='.0f')],
                )
                rings = alt.Chart(base[base['flagged']]).mark_point(
                    size=220, shape='circle', color=FLAG_COLOR,
                    strokeWidth=2).encode(
                    x='open_interest:Q', y='volume:Q')
                st.altair_chart((pts + rings).properties(height=320),
                                width='stretch')
                st.caption('Each dot is a contract, sized by premium. '
                           'Red rings: flagged as unusual.')

            st.dataframe(
                ua.head(50), hide_index=True, width='stretch',
                column_config={
                    'type': 'Type',
                    'strike': st.column_config.NumberColumn('Strike', format='%.1f'),
                    'expiration': st.column_config.DateColumn('Expiration'),
                    'dte': 'DTE',
                    'mid': st.column_config.NumberColumn('Mid', format='%.2f'),
                    'volume': st.column_config.NumberColumn('Volume', format='%d'),
                    'open_interest': st.column_config.NumberColumn('OI', format='%d'),
                    'vol_oi': st.column_config.NumberColumn('Vol/OI', format='%.2f'),
                    'premium': st.column_config.NumberColumn('Premium $', format='$%.0f'),
                    'score': st.column_config.ProgressColumn(
                        'Score', min_value=0, max_value=100, format='%.0f'),
                    'flagged': 'Flagged',
                })

# -----------------------------------------------------------------------------
# Tab 3: big price moves

with tab_price:
    st.subheader('Big moves in the underlying', divider='gray')
    if prices.empty:
        st.info('No price history available. Upload a CSV with `date` and '
                '`close` columns in the sidebar, or provide an options CSV '
                'with many timestamps and an underlying-price column.')
    else:
        c1, c2 = st.columns(2)
        window = c1.slider('Volatility window (days)', 5, 60, 20)
        z_thr = c2.slider('Z-score threshold', 1.5, 4.0, 2.5, step=0.25)
        pm = det.big_price_moves(prices, window, z_thr)
        if pm.empty:
            st.info(f'Need at least {window + 5} rows of price history for a '
                    f'{window}-day window — this source has {len(prices)}.')
        else:
            flagged = pm[pm['big_move']]
            line = alt.Chart(pm).mark_line(color=CALL_COLOR, strokeWidth=2).encode(
                x=alt.X('date:T', title=None),
                y=alt.Y('close:Q', title='Close', scale=alt.Scale(zero=False)),
                tooltip=['date:T', alt.Tooltip('close:Q', format='.2f')],
            )
            marks = alt.Chart(flagged).mark_point(
                size=90, color=FLAG_COLOR, filled=True).encode(
                x='date:T', y='close:Q',
                tooltip=['date:T', alt.Tooltip('close:Q', format='.2f'),
                         alt.Tooltip('log_ret:Q', title='Log return', format='.2%'),
                         alt.Tooltip('z:Q', format='.2f')],
            )
            st.altair_chart((line + marks).properties(height=300),
                            width='stretch')
            st.caption(f'Red dots: days where |z| ≥ {z_thr:g} vs the trailing '
                       f'{window}-day volatility.')

            zline = alt.Chart(pm.dropna(subset=['z'])).mark_line(
                color=CALL_COLOR, strokeWidth=2).encode(
                x=alt.X('date:T', title=None),
                y=alt.Y('z:Q', title='Move z-score'),
                tooltip=['date:T', alt.Tooltip('z:Q', format='.2f')],
            )
            rules = alt.Chart(pd.DataFrame({'y': [z_thr, -z_thr]})).mark_rule(
                strokeDash=[4, 3], color='gray').encode(y='y:Q')
            st.altair_chart((zline + rules).properties(height=200),
                            width='stretch')

            burst_days = int(pm['vol_burst'].sum())
            st.caption(f'Volatility bursts (5-day vol ≥ 1.5× {window}-day vol): '
                       f'{burst_days} day(s).')
            st.dataframe(
                flagged[['date', 'close', 'log_ret', 'z']],
                hide_index=True, width='stretch',
                column_config={
                    'date': st.column_config.DateColumn('Date'),
                    'close': st.column_config.NumberColumn('Close', format='%.2f'),
                    'log_ret': st.column_config.NumberColumn('Log return', format='%.2%'),
                    'z': st.column_config.NumberColumn('Z-score', format='%.2f'),
                })

# -----------------------------------------------------------------------------
# Tab 4: IV spikes

with tab_iv:
    st.subheader('IV outliers across the chain', divider='gray')
    missing = schema.missing_columns(chain, det.IV_SNAPSHOT_REQUIRED)
    if missing:
        warn_missing(missing)
    elif spot is None:
        st.info('No usable spot price, so moneyness cannot be computed.')
    else:
        c1, c2, c3 = st.columns(3)
        moneyness = c1.slider('Moneyness band (|K/S − 1|)', 0.05, 0.50, 0.20,
                              step=0.05)
        snap_z = c2.slider('Robust z threshold', 2.0, 5.0, 3.0, step=0.25)
        include_crushes = c3.checkbox('Include IV crushes (low outliers)')
        ivo = det.iv_snapshot_outliers(chain, spot, moneyness, snap_z,
                                       include_crushes)
        if ivo.empty:
            st.info('No IV values inside the selected moneyness band.')
        else:
            pts = alt.Chart(ivo).mark_circle(size=45).encode(
                x=alt.X('strike:Q', title='Strike', scale=alt.Scale(zero=False)),
                y=alt.Y('implied_volatility:Q', title='Implied volatility',
                        axis=alt.Axis(format='%'), scale=alt.Scale(zero=False)),
                color=alt.Color('type:N', title='Type', scale=TYPE_SCALE),
                tooltip=['type:N', 'strike:Q',
                         alt.Tooltip('yearmonthdate(expiration)', title='Expiration'),
                         alt.Tooltip('implied_volatility:Q', format='.1%'),
                         alt.Tooltip('robust_z:Q', format='.2f')],
            )
            rings = alt.Chart(ivo[ivo['flagged']]).mark_point(
                size=200, color=FLAG_COLOR, strokeWidth=2).encode(
                x='strike:Q', y='implied_volatility:Q')
            st.altair_chart((pts + rings).properties(height=300),
                            width='stretch')
            st.caption('IV by strike, all selected expiries. Red rings: '
                       "outliers vs the expiry's fitted smile.")

            outliers = ivo[ivo['flagged']]
            if outliers.empty:
                st.info('No IV outliers at the current threshold.')
            else:
                st.dataframe(
                    outliers, hide_index=True, width='stretch',
                    column_config={
                        'type': 'Type',
                        'strike': st.column_config.NumberColumn('Strike', format='%.1f'),
                        'expiration': st.column_config.DateColumn('Expiration'),
                        'dte': 'DTE',
                        'mid': st.column_config.NumberColumn('Mid', format='%.2f'),
                        'volume': st.column_config.NumberColumn('Volume', format='%d'),
                        'implied_volatility': st.column_config.NumberColumn('IV', format='%.1%'),
                        'iv_fit': st.column_config.NumberColumn('Smile-fit IV', format='%.1%'),
                        'robust_z': st.column_config.NumberColumn('Robust z', format='%.2f'),
                        'flagged': 'Flagged',
                    })

        term = det.iv_term_structure(chain, spot)
        if not term.empty:
            st.subheader('ATM IV term structure', divider='gray')
            tline = alt.Chart(term).mark_line(
                color=CALL_COLOR, strokeWidth=2, point=alt.OverlayMarkDef(
                    size=70, filled=True, color=CALL_COLOR)).encode(
                x=alt.X('dte:Q', title='Days to expiry'),
                y=alt.Y('atm_iv:Q', title='ATM IV', axis=alt.Axis(format='%'),
                        scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip('yearmonthdate(expiration)', title='Expiration'),
                         'dte:Q', alt.Tooltip('atm_iv:Q', format='.1%')],
            )
            st.altair_chart(tline.properties(height=220),
                            width='stretch')

    st.subheader('IV spikes over time', divider='gray')
    if not prices.empty and 'iv' in prices and prices['iv'].notna().any():
        iv_series = prices[['date', 'iv']]
        iv_source = price_source
    else:
        iv_series = det.iv_series_from_chain(chain)
        iv_source = 'near-ATM mean IV per timestamp in the options CSV'
    ts = det.iv_time_series_spikes(iv_series)
    if ts.empty:
        st.info('No IV history available — a single-snapshot chain has no IV '
                'time series. Include an `iv` column in the price-history CSV, '
                'or upload a chain covering many dates.')
    else:
        st.caption(f'Source: {iv_source}')
        spikes = ts[ts['flagged']]
        line = alt.Chart(ts).mark_line(color=CALL_COLOR, strokeWidth=2).encode(
            x=alt.X('date:T', title=None),
            y=alt.Y('iv:Q', title='Implied volatility', axis=alt.Axis(format='%'),
                    scale=alt.Scale(zero=False)),
            tooltip=['date:T', alt.Tooltip('iv:Q', format='.1%')],
        )
        marks = alt.Chart(spikes).mark_point(
            size=90, color=FLAG_COLOR, filled=True).encode(
            x='date:T', y='iv:Q',
            tooltip=['date:T', alt.Tooltip('iv:Q', format='.1%'),
                     alt.Tooltip('iv_change:Q', title='1-day change', format='+.1%'),
                     alt.Tooltip('z:Q', format='.2f')],
        )
        st.altair_chart((line + marks).properties(height=250),
                        width='stretch')
        if not spikes.empty:
            st.dataframe(
                spikes[['date', 'iv', 'iv_change', 'z']],
                hide_index=True, width='stretch',
                column_config={
                    'date': st.column_config.DateColumn('Date'),
                    'iv': st.column_config.NumberColumn('IV', format='%.1%'),
                    'iv_change': st.column_config.NumberColumn('1-day change', format='%.1%'),
                    'z': st.column_config.NumberColumn('Z-score', format='%.2f'),
                })
