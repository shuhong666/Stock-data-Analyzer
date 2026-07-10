"""Analyze all technical indicators for pullback quality."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators

db = Database()

with open('data/pullbacks_scan.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []
count = 0

for code, stock in data['results'].items():
    if count >= 6000:
        break

    rows = db.fetchall(
        'SELECT trade_date, open, high, low, close, volume, turn '
        'FROM daily_kline WHERE code=? ORDER BY trade_date',
        (code,)
    )
    if len(rows) < 120:
        continue
    dates = [r['trade_date'] for r in rows]
    raw_close = np.array([r['close'] for r in rows], dtype=float)
    raw_high = np.array([r['high'] for r in rows], dtype=float)
    raw_low = np.array([r['low'] for r in rows], dtype=float)
    raw_open = np.array([r['open'] for r in rows], dtype=float)
    vol = np.array([r['volume'] or 0 for r in rows], dtype=float)
    turn = np.array([r['turn'] or 0 for r in rows], dtype=float)
    n = len(raw_close)

    # Load adjust factors & apply forward-adjustment for price-based indicators
    afs = db.fetchall('SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date', (code,))
    adj = indicators.forward_adjust(raw_close, raw_high, raw_low, raw_open, dates, afs)
    close = adj['close']
    high = adj['high']
    low = adj['low']

    # All indicators (now on adjusted prices)
    rsi14 = indicators.rsi(close, 14)
    macd = indicators.macd(close)
    kdj = indicators.kdj(high, low, close)
    atr14 = indicators.atr(high, low, close, 14)
    boll = indicators.bollinger(close, 20, 2.0)
    adx_data = indicators.adx(high, low, close, 14)
    obv_arr = indicators.obv(close, vol)
    vol_ratio = indicators.vol_ratio(vol, 5)
    ma5 = indicators.sma(close, 5)
    ma10 = indicators.sma(close, 10)
    ma20 = indicators.sma(close, 20)
    ma30 = indicators.sma(close, 30)
    ma60 = indicators.sma(close, 60)
    ma120 = indicators.sma(close, 120)

    for t in stock['trend_segments']:
        for p in t['pullbacks']:
            if p.get('recovery_days') is None:
                continue
            try:
                idx = dates.index(p['trough_date'])
            except ValueError:
                continue
            if idx < 60 or idx >= n - 20:
                continue

            price = close[idx]
            success = close[min(idx + 20, n - 1)] > price

            def g(arr, default=None):
                if arr is None or idx >= len(arr):
                    return default
                v = arr[idx]
                return float(v) if not np.isnan(v) else default

            results.append({
                'retrace_ratio': p.get('retracement_ratio'),
                'success': success,
                'code': code,
                'rsi': g(rsi14),
                'macd_dif': g(macd.get('dif')),
                'macd_dea': g(macd.get('dea')),
                'macd_hist': g(macd.get('histogram')),
                'macd_dif_above_zero': 1 if g(macd.get('dif'), 0) > 0 else 0,
                'macd_dif_above_dea': 1 if g(macd.get('dif'), 0) > g(macd.get('dea'), 0) else 0,
                'kdj_k': g(kdj.get('k')),
                'kdj_d': g(kdj.get('d')),
                'kdj_j': g(kdj.get('j')),
                'atr_pct': round(g(atr14, 0) / price * 100, 2) if price > 0 else None,
                'bb_position': round(
                    (price - g(boll.get('lower'), price)) /
                    (g(boll.get('upper'), price) - g(boll.get('lower'), price)) * 100, 1
                ) if g(boll.get('upper')) and g(boll.get('lower')) and
                     g(boll.get('upper')) != g(boll.get('lower')) else None,
                'bb_width': g(boll.get('width')),
                'adx': g(adx_data.get('adx')),
                'adx_plus_di': g(adx_data.get('plus_di')),
                'adx_minus_di': g(adx_data.get('minus_di')),
                'vol_ratio': g(vol_ratio),
                'ma5_dist': round((price - g(ma5, price)) / g(ma5, price) * 100, 1) if g(ma5) and g(ma5) > 0 else None,
                'ma10_dist': round((price - g(ma10, price)) / g(ma10, price) * 100, 1) if g(ma10) and g(ma10) > 0 else None,
                'ma20_dist': round((price - g(ma20, price)) / g(ma20, price) * 100, 1) if g(ma20) and g(ma20) > 0 else None,
                'ma30_dist': round((price - g(ma30, price)) / g(ma30, price) * 100, 1) if g(ma30) and g(ma30) > 0 else None,
                'ma60_dist': round((price - g(ma60, price)) / g(ma60, price) * 100, 1) if g(ma60) and g(ma60) > 0 else None,
                'ma120_dist': round((price - g(ma120, price)) / g(ma120, price) * 100, 1) if g(ma120) and g(ma120) > 0 else None,
                'ma_below_count': sum(1 for m in [ma5, ma10, ma20, ma30, ma60, ma120] if g(m, 999) < price),
            })
            count += 1

print(f'Samples: {len(results)}')
success = [r for r in results if r['success']]
fail = [r for r in results if not r['success']]
print(f'Success: {len(success)} ({len(success)/len(results)*100:.0f}%)  Fail: {len(fail)} ({len(fail)/len(results)*100:.0f}%)')
print()

# Feature comparison
feature_list = [
    ('retrace_ratio', '%'),
    ('rsi', ''),
    ('macd_dif', ''),
    ('macd_dea', ''),
    ('macd_hist', ''),
    ('macd_dif_above_zero', 'rate'),
    ('macd_dif_above_dea', 'rate'),
    ('kdj_k', ''),
    ('kdj_d', ''),
    ('kdj_j', ''),
    ('atr_pct', '%'),
    ('bb_position', '%'),
    ('bb_width', '%'),
    ('adx', ''),
    ('adx_plus_di', ''),
    ('adx_minus_di', ''),
    ('vol_ratio', ''),
    ('ma5_dist', '%'),
    ('ma10_dist', '%'),
    ('ma20_dist', '%'),
    ('ma30_dist', '%'),
    ('ma60_dist', '%'),
    ('ma120_dist', '%'),
    ('ma_below_count', ''),
]

valid_list = []
print(f'{"Indicator":>24}  {"Success":>10}  {"Fail":>10}  {"Diff":>8}  {"Verdict":>10}')
print('-' * 75)

for name, unit in feature_list:
    if 'rate' in unit:
        sv_vals = [r[name] for r in success if r[name] is not None]
        fv_vals = [r[name] for r in fail if r[name] is not None]
        if not sv_vals or not fv_vals:
            continue
        sv = sum(sv_vals) / len(sv_vals) * 100
        fv = sum(fv_vals) / len(fv_vals) * 100
        diff = sv - fv
        verdict = 'VALID' if abs(diff) > 5 else 'WEAK'
        print(f'{name:>24}  {sv:>9.0f}%  {fv:>9.0f}%  {diff:>+8.1f}  {verdict:>10}')
        if abs(diff) > 5:
            valid_list.append(name)
        continue

    sv = [r[name] for r in success if r[name] is not None]
    fv = [r[name] for r in fail if r[name] is not None]
    if sv and fv:
        sm = np.mean(sv)
        fm = np.mean(fv)
        diff = sm - fm
        all_v = sv + fv
        p10 = np.percentile(all_v, 10)
        p90 = np.percentile(all_v, 90)
        spread = p90 - p10
        verdict = 'VALID' if spread > 0 and abs(diff) > spread * 0.08 else 'WEAK'
        print(f'{name:>24}  {sm:>10.2f}{unit}  {fm:>10.2f}{unit}  {diff:>+8.2f}  {verdict:>10}')
        if spread > 0 and abs(diff) > spread * 0.08:
            valid_list.append(name)

print()
print(f'VALID indicators: {valid_list}')
print()

# Bucket analysis for top VALID indicators
bucket_configs = [
    ('rsi', 'RSI(14)', [(0,30), (30,40), (40,50), (50,60), (60,70), (70,100)]),
    ('ma5_dist', 'MA5', [(-50,-8), (-8,-4), (-4,-1), (-1,1), (1,4), (4,50)]),
    ('ma10_dist', 'MA10', [(-50,-8), (-8,-4), (-4,-1), (-1,1), (1,4), (4,50)]),
    ('ma20_dist', 'MA20', [(-50,-8), (-8,-4), (-4,-1), (-1,1), (1,4), (4,50)]),
    ('ma60_dist', 'MA60', [(-50,-10), (-10,-5), (-5,0), (0,5), (5,10), (10,50)]),
    ('ma120_dist', 'MA120', [(-50,-10), (-10,-5), (-5,0), (0,5), (5,10), (10,50)]),
    ('atr_pct', 'ATR/Price', [(0,2), (2,3), (3,4), (4,5), (5,8)]),
    ('bb_position', 'BB Position', [(0,20), (20,40), (40,60), (60,80), (80,100)]),
    ('adx', 'ADX', [(0,20), (20,30), (30,40), (40,60), (60,100)]),
    ('kdj_k', 'KDJ-K', [(0,20), (20,40), (40,60), (60,80), (80,100)]),
    ('kdj_j', 'KDJ-J', [(-80,0), (0,30), (30,50), (50,80), (80,150)]),
    ('vol_ratio', 'Vol Ratio', [(0,0.5), (0.5,0.7), (0.7,0.9), (0.9,1.1), (1.1,1.4), (1.4,5)]),
    ('macd_dif_above_zero', 'MACD DIF>0', [(1,), (0,)]),
]

for name, label, buckets in bucket_configs:
    print(f'=== {label} ===')
    for tup in buckets:
        lo, hi = tup[0], tup[-1]
        if name in ('macd_dif_above_zero', 'macd_dif_above_dea'):
            bucket = [r for r in results if r[name] == lo]
            if not bucket:
                continue
            sr = sum(1 for r in bucket if r['success']) / len(bucket) * 100
            tag = 'Yes' if lo else 'No'
            print(f'  {tag:>4}: {len(bucket):>5} samples  success={sr:.0f}%')
        else:
            bucket = [r for r in results if r[name] is not None and lo <= r[name] < hi]
            if len(bucket) < 30:
                continue
            sr = sum(1 for r in bucket if r['success']) / len(bucket) * 100
            bar = '#' * int(sr / 2)
            print(f'  {lo:>5}~{hi:<5}: {len(bucket):>5} samples  success={sr:.0f}%  {bar}')
    print()
