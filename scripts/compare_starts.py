"""Compare features at pullback-start vs downtrend-start."""
import json, os, sys, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators

db = Database()
with open('data/pullbacks_scan.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pb_starts = []
dt_starts = []
codes = list(data['results'].keys())
random.shuffle(codes)

for code in codes[:80]:
    rows = db.fetchall(
        'SELECT trade_date,open,high,low,close,volume,turn FROM daily_kline WHERE code=? ORDER BY trade_date',
        (code,))
    if len(rows) < 300:
        continue
    dates = [r['trade_date'] for r in rows]
    raw_c = np.array([r['close'] for r in rows], dtype=float)
    raw_h = np.array([r['high'] for r in rows], dtype=float)
    raw_l = np.array([r['low'] for r in rows], dtype=float)
    raw_o = np.array([r['open'] for r in rows], dtype=float)
    n = len(raw_c)

    afs = db.fetchall(
        'SELECT trade_date,adj_factor,fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date',
        (code,))
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
    close, high, low = adj['close'], adj['high'], adj['low']

    rsi14 = indicators.rsi(close, 14)
    macd = indicators.macd(close)
    kdj = indicators.kdj(high, low, close)
    boll = indicators.bollinger(close, 20, 2.0)
    adx_d = indicators.adx(high, low, close, 14)
    ma20 = indicators.sma(close, 20)
    ma60 = indicators.sma(close, 60)
    ma120 = indicators.sma(close, 120)
    atr14 = indicators.atr(high, low, close, 14)

    def g(arr, i, d=0):
        if i >= len(arr):
            return None
        v = arr[i]
        return float(v) if not np.isnan(v) else None

    # Category 1: Pullback starts (peaks within uptrends)
    stock = data['results'].get(code)
    if stock:
        for t in stock['trend_segments']:
            for p in t['pullbacks']:
                try:
                    idx = dates.index(p['peak_date'])
                except ValueError:
                    continue
                if idx < 60 or idx >= n - 60:
                    continue
                price = close[idx]
                pb_starts.append({
                    'rsi': g(rsi14, idx),
                    'kdj_k': g(kdj['k'], idx),
                    'macd_dif': g(macd['dif'], idx),
                    'macd_above_zero': 1 if g(macd['dif'], idx, 0) > 0 else 0,
                    'bb_pos': round((price - boll['lower'][idx]) / (boll['upper'][idx] - boll['lower'][idx]) * 100, 1)
                    if g(boll['lower'], idx) and g(boll['upper'], idx) and boll['upper'][idx] != boll['lower'][idx] else None,
                    'ma20_dist': round((price - ma20[idx]) / ma20[idx] * 100, 1) if g(ma20, idx) else None,
                    'ma60_dist': round((price - ma60[idx]) / ma60[idx] * 100, 1) if g(ma60, idx) else None,
                    'ma120_dist': round((price - ma120[idx]) / ma120[idx] * 100, 1) if g(ma120, idx) else None,
                    'adx': g(adx_d['adx'], idx),
                    'atr_pct': round(g(atr14, idx, 0) / price * 100, 2) if price > 0 else None,
                    'gain_20d': round((close[idx] - close[max(0, idx - 20)]) / close[max(0, idx - 20)] * 100, 1),
                    'gain_60d': round((close[idx] - close[max(0, idx - 60)]) / close[max(0, idx - 60)] * 100, 1),
                })

    # Category 2: Downtrend starts
    for i in range(120, n - 60):
        if high[i] < max(high[i - 60:i]):
            continue
        if high[i] <= max(high[i + 1:i + 6]):
            continue

        lookahead = min(i + 120, n - 1)
        min_after = min(low[i:lookahead])
        drop = (high[i] - min_after) / high[i] * 100
        if drop < 20:
            continue

        max_after = max(high[i:lookahead])
        if max_after > high[i] * 1.02:
            continue

        price = close[i]
        dt_starts.append({
            'rsi': g(rsi14, i),
            'kdj_k': g(kdj['k'], i),
            'macd_dif': g(macd['dif'], i),
            'macd_above_zero': 1 if g(macd['dif'], i, 0) > 0 else 0,
            'bb_pos': round((price - boll['lower'][i]) / (boll['upper'][i] - boll['lower'][i]) * 100, 1)
            if g(boll['lower'], i) and g(boll['upper'], i) and boll['upper'][i] != boll['lower'][i] else None,
            'ma20_dist': round((price - ma20[i]) / ma20[i] * 100, 1) if g(ma20, i) else None,
            'ma60_dist': round((price - ma60[i]) / ma60[i] * 100, 1) if g(ma60, i) else None,
            'ma120_dist': round((price - ma120[i]) / ma120[i] * 100, 1) if g(ma120, i) else None,
            'adx': g(adx_d['adx'], i),
            'atr_pct': round(g(atr14, i, 0) / price * 100, 2) if price > 0 else None,
            'gain_20d': round((close[i] - close[max(0, i - 20)]) / close[max(0, i - 20)] * 100, 1),
            'gain_60d': round((close[i] - close[max(0, i - 60)]) / close[max(0, i - 60)] * 100, 1),
        })

print(f'Pullback starts: {len(pb_starts)}')
print(f'Downtrend starts: {len(dt_starts)}')
print()

features = [
    ('rsi', ''),
    ('kdj_k', ''),
    ('macd_dif', ''),
    ('macd_above_zero', 'rate'),
    ('bb_pos', '%'),
    ('ma20_dist', '%'),
    ('ma60_dist', '%'),
    ('ma120_dist', '%'),
    ('adx', ''),
    ('atr_pct', '%'),
    ('gain_20d', '%'),
    ('gain_60d', '%'),
]

print(f'{"Feature":>16}  {"Pullback":>10}  {"Downtrend":>10}  {"Diff":>8}')
print('-' * 52)
for name, unit in features:
    if 'rate' in unit:
        sv = sum(r[name] for r in pb_starts if r[name] is not None) / max(
            len([r for r in pb_starts if r[name] is not None]), 1) * 100
        fv = sum(r[name] for r in dt_starts if r[name] is not None) / max(
            len([r for r in dt_starts if r[name] is not None]), 1) * 100
        print(f'{name:>16}  {sv:>9.0f}%  {fv:>9.0f}%  {sv - fv:>+8.0f}')
    else:
        sv = [r[name] for r in pb_starts if r[name] is not None]
        fv = [r[name] for r in dt_starts if r[name] is not None]
        if sv and fv:
            print(f'{name:>16}  {np.mean(sv):>10.1f}{unit}  {np.mean(fv):>10.1f}{unit}  {np.mean(sv) - np.mean(fv):>+8.1f}')

for name, buckets in [
    ('gain_60d', [(-50, 0), (0, 20), (20, 50), (50, 100), (100, 300)]),
    ('gain_20d', [(-30, 0), (0, 5), (5, 15), (15, 30), (30, 100)]),
    ('adx', [(0, 20), (20, 30), (30, 40), (40, 60)]),
]:
    print(f'\n=== {name} ===')
    for lo, hi in buckets:
        pb_n = sum(1 for r in pb_starts if r[name] is not None and lo <= r[name] < hi)
        dt_n = sum(1 for r in dt_starts if r[name] is not None and lo <= r[name] < hi)
        if pb_n + dt_n > 5:
            pb_pct = pb_n / (pb_n + dt_n) * 100
            bar = '#' * int(pb_pct / 3)
            print(f'  {lo:>4}~{hi:<4}: {pb_n:>4}pb  {dt_n:>4}dt  pb占比={pb_pct:.0f}%  {bar}')
