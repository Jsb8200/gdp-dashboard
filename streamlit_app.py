import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import lightgbm as lgb

st.set_page_config(
    page_title='Market forecast dashboard',
    page_icon=':chart_with_upwards_trend:',
    layout='wide',
)

# -----------------------------------------------------------------------------
# Palette (validated reference palette from the dataviz method)

C = {
    'price': '#2a78d6',       # categorical slot 1 (blue)
    'forecast': '#eb6834',    # categorical slot 2 (orange)
    'band': 'rgba(42, 120, 214, 0.15)',
    'up': '#0ca30c',          # status: good
    'down': '#d03b3b',        # status: critical
    'ink': '#0b0b0b',
    'ink2': '#52514e',
    'muted': '#898781',
    'grid': '#e1e0d9',
    'axis': '#c3c2b7',
    'surface': '#fcfcfb',
    'seq': ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95'],
}

PLOTLY_LAYOUT = dict(
    template='none',
    paper_bgcolor=C['surface'],
    plot_bgcolor=C['surface'],
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
              color=C['ink2'], size=13),
    margin=dict(l=50, r=80, t=30, b=10),
    hovermode='x unified',
    xaxis=dict(gridcolor=C['grid'], linecolor=C['axis'], zeroline=False,
               automargin=True, showspikes=True, spikemode='across',
               spikethickness=1, spikecolor=C['axis'], spikedash='dot'),
    yaxis=dict(gridcolor=C['grid'], linecolor=C['axis'], zeroline=False,
               automargin=True, tickformat=',.0f'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
)

# -----------------------------------------------------------------------------
# Data

@st.cache_data
def make_sample_market(n_bars: int = 1500, seed: int = 7, start_price: float = 4000.0):
    """Synthetic index-like price series: geometric Brownian motion with
    regime-switching drift/volatility, so the model has structure to learn."""
    rng = np.random.default_rng(seed)

    # Regimes: (annualized drift, annualized vol), avg regime length ~120 bars
    regimes = [(0.12, 0.13), (-0.18, 0.28), (0.04, 0.18), (0.30, 0.16)]
    probs = [0.40, 0.15, 0.30, 0.15]
    dt = 1 / 252

    rets = np.empty(n_bars)
    i = 0
    while i < n_bars:
        mu, sigma = regimes[rng.choice(len(regimes), p=probs)]
        length = min(int(rng.exponential(120)) + 20, n_bars - i)
        z = rng.standard_normal(length)
        # A touch of autocorrelation so momentum features carry signal
        for k in range(1, length):
            z[k] = 0.25 * z[k - 1] + np.sqrt(1 - 0.25 ** 2) * z[k]
        rets[i:i + length] = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        i += length

    close = start_price * np.exp(np.cumsum(rets))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_bars)
    return pd.DataFrame({'Date': dates, 'Close': close})


def load_uploaded_csv(file) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.sidebar.error(f'Could not read CSV: {e}')
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    if 'close' not in cols:
        st.sidebar.error('CSV needs a "Close" column (and ideally a "Date" column).')
        return None
    out = pd.DataFrame()
    if 'date' in cols:
        out['Date'] = pd.to_datetime(df[cols['date']], errors='coerce')
    else:
        out['Date'] = pd.RangeIndex(len(df))
    out['Close'] = pd.to_numeric(df[cols['close']], errors='coerce')
    out = out.dropna().sort_values('Date').reset_index(drop=True)
    if len(out) < 300:
        st.sidebar.error(f'Need at least 300 rows to train; got {len(out)}.')
        return None
    return out

# -----------------------------------------------------------------------------
# Features & models

def build_features(close: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index)
    ret = close.pct_change()
    for lag in (1, 2, 3, 5, 10):
        f[f'ret_{lag}'] = close.pct_change(lag)
    f['vol_10'] = ret.rolling(10).std()
    f['vol_20'] = ret.rolling(20).std()
    f['sma20_dist'] = close / close.rolling(20).mean() - 1
    f['sma50_dist'] = close / close.rolling(50).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    f['mom_20'] = close.pct_change(20)
    return f


LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=15, max_depth=4,
    min_child_samples=30, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, verbosity=-1, random_state=42,
)


@st.cache_resource(show_spinner=False)
def train_horizon(close_values: tuple, horizon: int):
    """Train quantile regressors (q10/q50/q90, target = forward move in points)
    and a direction classifier for one horizon. Returns models + honest
    validation metrics from a chronological 80/20 split."""
    close = pd.Series(close_values)
    X = build_features(close)
    y = close.shift(-horizon) - close  # forward move in points

    valid = X.notna().all(axis=1)
    fit_rows = valid & y.notna()
    X_fit, y_fit = X[fit_rows], y[fit_rows]

    split = int(len(X_fit) * 0.8)
    X_tr, y_tr = X_fit.iloc[:split], y_fit.iloc[:split]
    X_va, y_va = X_fit.iloc[split:], y_fit.iloc[split:]

    quantiles = {}
    for alpha in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(objective='quantile', alpha=alpha, **LGB_PARAMS)
        m.fit(X_tr, y_tr)
        quantiles[alpha] = m

    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    clf.fit(X_tr, (y_tr > 0).astype(int))

    pred_va = quantiles[0.5].predict(X_va)
    metrics = {
        'dir_accuracy': float(np.mean(np.sign(pred_va) == np.sign(y_va))),
        'mae_points': float(np.mean(np.abs(pred_va - y_va))),
        'n_train': len(X_tr),
        'n_valid': len(X_va),
    }
    backtest = pd.DataFrame({
        'idx': X_va.index, 'actual': y_va.to_numpy(), 'predicted': pred_va,
    })

    # Forecast from the most recent fully-featured bar
    X_now = X[valid].iloc[[-1]]
    q10, q50, q90 = (float(quantiles[a].predict(X_now)[0]) for a in (0.1, 0.5, 0.9))
    q10, q50, q90 = sorted((q10, q50, q90))  # guard against quantile crossing
    forecast = {
        'move_q10': q10, 'move_q50': q50, 'move_q90': q90,
        'prob_up': float(clf.predict_proba(X_now)[0, 1]),
    }
    importance = pd.Series(quantiles[0.5].feature_importances_,
                           index=X.columns).sort_values()
    return forecast, metrics, backtest, importance

# -----------------------------------------------------------------------------
# Sidebar

st.sidebar.header('Data')
source = st.sidebar.radio('Source', ['Sample market (synthetic)', 'Upload CSV'],
                          help='The sample series is generated with geometric Brownian '
                               'motion + regime shifts. Upload any CSV with Date and Close '
                               'columns to forecast real data.')
df = None
if source == 'Upload CSV':
    up = st.sidebar.file_uploader('CSV with Date + Close columns', type='csv')
    if up is not None:
        df = load_uploaded_csv(up)
    if df is None:
        st.sidebar.info('Using the sample market until a valid CSV is uploaded.')
if df is None:
    seed = st.sidebar.number_input('Sample seed', 0, 999, 7,
                                   help='Regenerates a different synthetic market.')
    df = make_sample_market(seed=int(seed))

st.sidebar.header('Forecast')
horizons = st.sidebar.multiselect('Horizons (bars ahead)', [1, 3, 5, 10, 15, 20, 30],
                                  default=[5, 10, 15])
horizons = sorted(horizons) or [5, 10, 15]
lookback = st.sidebar.slider('Chart lookback (bars)', 60, 500, 180, step=20)

# -----------------------------------------------------------------------------
# Header

st.title(':chart_with_upwards_trend: Where is the market going?')
st.caption('LightGBM quantile models forecast the move over the next '
           f'{", ".join(str(h) for h in horizons)} bars — direction, size in points, '
           'and the from → to price range. Educational demo, not financial advice.')

close = df['Close'].reset_index(drop=True)
dates = df['Date'].reset_index(drop=True)
last_close = float(close.iloc[-1])
last_date = dates.iloc[-1]

results = {}
with st.spinner('Training LightGBM models…'):
    for h in horizons:
        results[h] = train_horizon(tuple(close.to_numpy()), h)

# -----------------------------------------------------------------------------
# KPI row: one tile per horizon

cols = st.columns(len(horizons) + 1)
with cols[0]:
    st.metric('Last close', f'{last_close:,.1f}',
              f'{close.iloc[-1] - close.iloc[-2]:+,.1f} pts vs prev bar')

for col, h in zip(cols[1:], horizons):
    fc, _, _, _ = results[h]
    move = fc['move_q50']
    target = last_close + move
    with col:
        st.metric(
            f'Next {h} bars',
            f'{target:,.1f}',
            f'{move:+,.1f} pts  ·  P(up) {fc["prob_up"]:.0%}',
        )

# -----------------------------------------------------------------------------
# Main chart: recent price + forecast fan

st.subheader('Price and forecast fan', divider='gray')

hist_n = min(lookback, len(close))
hist_x = dates.iloc[-hist_n:]
hist_y = close.iloc[-hist_n:]

# Future x positions: extend at the median historical bar spacing
if pd.api.types.is_datetime64_any_dtype(dates):
    step = (dates.iloc[-1] - dates.iloc[-min(60, len(dates) - 1)]) / min(60, len(dates) - 1)
    fut_x = {h: last_date + step * h for h in horizons}
else:
    fut_x = {h: last_date + h for h in horizons}

fan_x = [last_date] + [fut_x[h] for h in horizons]
fan_mid = [last_close] + [last_close + results[h][0]['move_q50'] for h in horizons]
fan_lo = [last_close] + [last_close + results[h][0]['move_q10'] for h in horizons]
fan_hi = [last_close] + [last_close + results[h][0]['move_q90'] for h in horizons]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_x, y=hist_y, name='Close', mode='lines',
    line=dict(color=C['price'], width=2),
    hovertemplate='%{y:,.1f}<extra>Close</extra>'))
fig.add_trace(go.Scatter(
    x=fan_x + fan_x[::-1], y=fan_hi + fan_lo[::-1], fill='toself',
    fillcolor=C['band'], mode='lines', line=dict(width=0), name='80% range',
    hoverinfo='skip'))
fig.add_trace(go.Scatter(
    x=fan_x, y=fan_mid, name='Forecast (median)', mode='lines+markers',
    line=dict(color=C['forecast'], width=2, dash='dot'),
    marker=dict(size=8, color=C['forecast']),
    hovertemplate='%{y:,.1f}<extra>Forecast</extra>'))
# Direct-label only the furthest horizon; the KPI row and table carry the rest
h_last = horizons[-1]
y_last = last_close + results[h_last][0]['move_q50']
fig.add_annotation(x=fut_x[h_last], y=y_last,
                   text=f'+{h_last} bars: {y_last:,.0f}',
                   showarrow=False, xanchor='left', xshift=10,
                   font=dict(color=C['ink'], size=12))
fig.update_layout(height=420, **PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Forecast table

st.subheader('Forecast detail', divider='gray')
rows = []
for h in horizons:
    fc = results[h][0]
    move = fc['move_q50']
    rows.append({
        'Horizon': f'next {h} bars',
        'Direction': ('▲ up' if move >= 0 else '▼ down'),
        'P(up)': f'{fc["prob_up"]:.0%}',
        'Point size': f'{move:+,.1f}',
        'From → to': f'{last_close:,.1f} → {last_close + move:,.1f}',
        '80% range': f'{last_close + fc["move_q10"]:,.1f} – {last_close + fc["move_q90"]:,.1f}',
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Backtest: how good are these forecasts, honestly?

st.subheader('Validation backtest (out-of-sample)', divider='gray')
bt_h = st.selectbox('Horizon to inspect', horizons,
                    index=len(horizons) - 1,
                    format_func=lambda h: f'next {h} bars')
fc, metrics, backtest, importance = results[bt_h]

m1, m2, m3 = st.columns(3)
m1.metric('Directional accuracy', f'{metrics["dir_accuracy"]:.0%}',
          help='Share of validation bars where the predicted sign matched the '
               'actual move. Markets are mostly noise — modestly above 50% is realistic.')
m2.metric('Mean abs. error', f'{metrics["mae_points"]:,.1f} pts',
          help=f'For scale: the median |actual move| at this horizon is '
               f'{backtest["actual"].abs().median():,.1f} pts.')
m3.metric('Train / validation bars', f'{metrics["n_train"]:,} / {metrics["n_valid"]:,}',
          help='Strict chronological 80/20 split — the model never sees validation bars.')

left, right = st.columns([3, 2])
with left:
    st.markdown(f'**Predicted vs actual {bt_h}-bar move (points), validation slice**')
    bt_x = dates.iloc[backtest['idx']].reset_index(drop=True)
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt_x, y=backtest['actual'], name=f'Actual {bt_h}-bar move',
        mode='lines', line=dict(color=C['price'], width=2),
        hovertemplate='%{y:+,.1f} pts<extra>Actual</extra>'))
    fig_bt.add_trace(go.Scatter(
        x=bt_x, y=backtest['predicted'], name='Predicted',
        mode='lines', line=dict(color=C['forecast'], width=2),
        hovertemplate='%{y:+,.1f} pts<extra>Predicted</extra>'))
    fig_bt.add_hline(y=0, line=dict(color=C['axis'], width=1))
    fig_bt.update_layout(height=340, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_bt, use_container_width=True)
with right:
    st.markdown('**What the model looks at (split importance)**')
    imp = importance / importance.sum()
    n = len(imp)
    bar_colors = [C['seq'][min(int(i / n * len(C['seq'])), len(C['seq']) - 1)]
                  for i in range(n)]
    fig_imp = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation='h',
        marker=dict(color=bar_colors, cornerradius=4),
        hovertemplate='%{x:.1%}<extra>%{y}</extra>'))
    imp_layout = {**PLOTLY_LAYOUT, 'hovermode': 'closest',
                  'xaxis': {**PLOTLY_LAYOUT['xaxis'],
                            'tickformat': '.0%', 'showspikes': False}}
    fig_imp.update_layout(height=340, bargap=0.35, **imp_layout)
    st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------------

with st.expander('How this works'):
    st.markdown('''
- **Features** are built from closing prices only: lagged returns, rolling
  volatility, distance from 20/50-bar moving averages, RSI-14, MACD histogram,
  and 20-bar momentum.
- **Per horizon** (e.g. next 5, 10, 15 bars) four small LightGBM models are
  trained: quantile regressors at the 10th/50th/90th percentile predicting the
  *forward move in points*, plus a classifier for the probability the move is up.
- The **median quantile** gives the headline "point size" and the from → to
  target; the 10th–90th band is the shaded fan (an ~80% range).
- **Backtest** numbers come from a strict chronological 80/20 split — the model
  never sees validation bars during training. Overlapping forward windows mean
  neighbouring validation bars are correlated, so treat accuracy as indicative.
- Markets are mostly noise: directional accuracy modestly above 50% is
  realistic; anything far higher usually means leakage. **This is an educational
  tool, not financial advice.**
''')
