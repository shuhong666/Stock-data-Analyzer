"""Pullback scanner — core logic (config-driven scoring)."""
import json, os, numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.storage.database import Database
from src.server.plugin_mgr import indicators

# Load scoring config
_cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scoring.json")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    CFG = json.load(_f)


def _bucket(val, buckets, scores):
    """Map val to score: buckets=[10,20], scores=[A,B,C] => val<10→A, val<20→B, else→C."""
    for i, th in enumerate(buckets):
        if val < th:
            return scores[i]
    return scores[-1]


def _label(val, buckets, labels):
    for i, th in enumerate(buckets):
        if val < th:
            return labels[i]
    return labels[-1]


def assess_pullback(features):
    """Score from scoring.json config."""
    score = 0
    for key in ["decline_pct", "gain_60d", "kdj_k", "ma20_dist", "ma60_dist", "adx"]:
        cfg = CFG.get(key)
        if not cfg or not cfg.get("scores"):
            continue
        val = features.get(key)
        if val is not None:
            score += _bucket(val, cfg["buckets"], cfg["scores"])

    bb = features.get("bb_pos")
    if bb is not None:
        score += _bucket(bb, CFG["bb_position"]["buckets"], CFG["bb_position"]["scores"])

    if features.get("macd_below_zero"):
        score += CFG["macd_below_zero"]["score"]

    return max(0, min(100, score))


def rsi_display_label(rsi_val):
    """RSI label from config (display only)."""
    if rsi_val is None:
        return None
    cfg = CFG["rsi"]
    return f'{rsi_val:.0f} ' + _label(rsi_val, cfg["buckets"], cfg["labels"])


def scan_stock(code, db):
    """Scan a single stock for current pullback. Returns dict or None."""
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn, pe_ttm, pb_mrq, total_mv "
        "FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),
    )
    if len(rows) < 120:
        return None

    dates = [r['trade_date'] for r in rows]
    raw_c = np.array([r['close'] for r in rows], dtype=float)
    raw_h = np.array([r['high'] for r in rows], dtype=float)
    raw_l = np.array([r['low'] for r in rows], dtype=float)
    raw_o = np.array([r['open'] for r in rows], dtype=float)
    n = len(raw_c)
    latest_date = dates[-1]

    # Forward-adjust
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
    close, high, low = adj['close'], adj['high'], adj['low']

    # Find nearest peak (look back up to 40 days from latest)
    lookback = min(40, n - 1)
    peak_idx = n - 1 - lookback + np.argmax(high[n - 1 - lookback:n])
    peak_price = high[peak_idx]

    # Find trough between peak and latest
    if peak_idx >= n - 2:
        return None
    trough_idx = peak_idx + np.argmin(low[peak_idx:n])
    trough_price = low[trough_idx]

    # Must be currently IN the pullback: trough recent, today not a bounce
    if trough_idx < n - 3:
        return None
    if close[-1] > close[-2]:
        return None
    if (close[-1] - trough_price) / max(trough_price, 0.01) * 100 > 3:
        return None

    decline = (peak_price - trough_price) / peak_price * 100
    if decline < 3:
        return None

    # Compute indicators at trough
    rsi14 = indicators.rsi(close, 14)
    macd = indicators.macd(close)
    kdj = indicators.kdj(high, low, close)
    boll = indicators.bollinger(close, 20, 2.0)
    ma20 = indicators.sma(close, 20)
    ma60 = indicators.sma(close, 60)
    adx_data = indicators.adx(high, low, close, 14)

    def g(arr, d=0):
        if trough_idx >= len(arr):
            return None
        v = arr[trough_idx]
        return float(v) if not np.isnan(v) else None

    price = close[trough_idx]
    gain_60d = round((close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)

    bb_lower = g(boll['lower'], price)
    bb_upper = g(boll['upper'], price)
    bb_pos = None
    if bb_lower and bb_upper and bb_upper != bb_lower:
        bb_pos = round((price - bb_lower) / (bb_upper - bb_lower) * 100, 1)

    features = {
        'decline_pct': round(decline, 1),
        'rsi': g(rsi14),
        'kdj_k': g(kdj['k']),
        'macd_dif': g(macd['dif']),
        'macd_below_zero': g(macd['dif']) < 0 if g(macd['dif']) is not None else None,
        'bb_pos': bb_pos,
        'ma20_dist': round((price - ma20[trough_idx]) / ma20[trough_idx] * 100, 1) if g(ma20) else None,
        'ma60_dist': round((price - ma60[trough_idx]) / ma60[trough_idx] * 100, 1) if g(ma60) else None,
        'gain_20d': round((close[peak_idx] - close[max(0, peak_idx - 20)]) / close[max(0, peak_idx - 20)] * 100, 1),
        'gain_60d': gain_60d,
        'adx': g(adx_data['adx']),
    }

    prob = assess_pullback(features)
    rsi_label = rsi_display_label(features.get('rsi'))

    # Fundamental data
    fund_row = db.fetchone(
        "SELECT pe_ttm, pb_mrq, total_mv, turn FROM daily_kline "
        "WHERE code=? AND total_mv IS NOT NULL ORDER BY trade_date DESC LIMIT 1",
        (code,)
    )
    if not fund_row:
        fund_row = db.fetchone(
            "SELECT pe_ttm, pb_mrq, total_mv, turn FROM daily_kline "
            "WHERE code=? ORDER BY trade_date DESC LIMIT 1",
            (code,)
        )
    fund = {
        'pe_ttm': round(fund_row['pe_ttm'], 1) if fund_row and fund_row['pe_ttm'] else None,
        'pb_mrq': round(fund_row['pb_mrq'], 2) if fund_row and fund_row['pb_mrq'] else None,
        'total_mv': round(fund_row['total_mv'] / 1e8, 1) if fund_row and fund_row['total_mv'] else None,
        'turn': round(fund_row['turn'], 1) if fund_row and fund_row['turn'] else None,
    }

    warnings = []
    if fund['pe_ttm'] is not None and fund['pe_ttm'] < 0:
        warnings.append('PE为负(亏损)')
    if fund['pe_ttm'] is not None and fund['pe_ttm'] > 80:
        warnings.append(f'PE过高({fund["pe_ttm"]:.0f}倍)')
    if fund['turn'] is not None and fund['turn'] > 15:
        warnings.append(f'换手率异常({fund["turn"]:.0f}%)')
    if features['gain_60d'] > 50:
        warnings.append(f'近60日涨幅过大({features["gain_60d"]:.0f}%)')

    name_row = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
    name = name_row['name'] if name_row else ''

    return {
        'code': code, 'name': name, 'latest_date': latest_date,
        'peak_date': dates[peak_idx], 'peak_price': round(float(peak_price), 2),
        'trough_date': dates[trough_idx], 'trough_price': round(float(trough_price), 2),
        'features': features, 'probability': prob, 'rsi_label': rsi_label,
        'fundamentals': fund, 'warnings': warnings,
    }


def scan_all(max_workers=8, min_probability=50, progress_cb=None):
    """Full-market concurrent scan."""
    db = Database()
    codes = db.get_active_stock_codes()
    total = len(codes)
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(scan_stock, code, db): code for code in codes}
        for future in as_completed(futures):
            done += 1
            try:
                r = future.result()
                if r and r['probability'] >= min_probability:
                    results.append(r)
            except Exception:
                pass
            if progress_cb:
                progress_cb(done, total)

    results.sort(key=lambda r: (len(r['warnings']), -r['probability']))
    return results


# ====================================================================
# Historical retrieval: "前期没涨就跌" (60d gain < 0) + high probability
# ====================================================================

def _find_decline_events(close, high, low, dates, lookback=40, min_decline=3, max_decline=50):
    """Find all historical peak→trough decline events in a price series.

    For each bar (going back at most `lookback` bars), find the local high
    and the subsequent low, recording each unique (peak, trough) pair.

    Returns list of dicts sorted by trough_date.
    """
    n = len(close)
    events = []
    for idx in range(lookback + 5, n):
        peak_idx = idx - lookback + np.argmax(high[idx - lookback:idx + 1])
        peak_price = high[peak_idx]
        trough_idx = peak_idx + np.argmin(low[peak_idx:idx + 1])
        trough_price = low[trough_idx]
        decline = (peak_price - trough_price) / peak_price * 100
        if decline < min_decline or decline > max_decline:
            continue
        if peak_idx >= trough_idx:
            continue
        events.append({
            'peak_date': dates[peak_idx],
            'peak_price': round(float(peak_price), 2),
            'trough_date': dates[trough_idx],
            'trough_price': round(float(trough_price), 2),
            'decline_pct': round(decline, 1),
            'peak_idx': peak_idx,
            'trough_idx': trough_idx,
        })

    # Deduplicate by (peak_idx, trough_idx)
    seen = set()
    unique = []
    for e in events:
        key = (e['peak_idx'], e['trough_idx'])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    unique.sort(key=lambda e: e['trough_date'])
    return unique


def _compute_trough_features(close, high, low, peak_idx, trough_idx):
    """Compute technical indicators at a trough point."""
    price = close[trough_idx]
    rsi14 = indicators.rsi(close, 14)
    macd = indicators.macd(close)
    kdj = indicators.kdj(high, low, close)
    boll = indicators.bollinger(close, 20, 2.0)
    ma20 = indicators.sma(close, 20)
    ma60 = indicators.sma(close, 60)
    adx_data = indicators.adx(high, low, close, 14)

    def g(arr):
        if trough_idx >= len(arr):
            return None
        v = arr[trough_idx]
        return float(v) if not np.isnan(v) else None

    gain_60d = round((close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)
    gain_20d = round((close[peak_idx] - close[max(0, peak_idx - 20)]) / close[max(0, peak_idx - 20)] * 100, 1)

    bb_lower = g(boll['lower'])
    bb_upper = g(boll['upper'])
    bb_pos = None
    if bb_lower and bb_upper and bb_upper != bb_lower:
        bb_pos = round((price - bb_lower) / (bb_upper - bb_lower) * 100, 1)

    return {
        'decline_pct': None,  # filled by caller
        'rsi': g(rsi14),
        'kdj_k': g(kdj['k']),
        'macd_dif': g(macd['dif']),
        'macd_below_zero': g(macd['dif']) < 0 if g(macd['dif']) is not None else None,
        'bb_pos': bb_pos,
        'ma20_dist': round((price - ma20[trough_idx]) / ma20[trough_idx] * 100, 1) if g(ma20) else None,
        'ma60_dist': round((price - ma60[trough_idx]) / ma60[trough_idx] * 100, 1) if g(ma60) else None,
        'gain_20d': gain_20d,
        'gain_60d': gain_60d,
        'adx': g(adx_data['adx']),
    }


def _compute_recovery(dates, close, trough_idx, peak_price, max_lookahead=504):
    """Days until close >= peak_price after trough_idx. Returns (recovery_date, days) or None."""
    n = len(close)
    for j in range(trough_idx + 1, min(trough_idx + 1 + max_lookahead, n)):
        if close[j] >= peak_price:
            return {
                'recovery_date': dates[j],
                'recovery_days': j - trough_idx,
                'recovery_price': round(float(close[j]), 2),
            }
    return None


def retrieve_no_prior_gain(code, db, min_probability=80, min_decline=3, max_decline=50):
    """Retrieve historical pullback events for a single stock matching:
    - Prior 60-day gain < 0 (didn't run up before the drop)
    - Probability >= min_probability

    Returns list of dicts, each containing event info + features + probability + recovery.
    """
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn "
        "FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),
    )
    if len(rows) < 120:
        return []

    dates = [r['trade_date'] for r in rows]
    raw_c = np.array([r['close'] for r in rows], dtype=float)
    raw_h = np.array([r['high'] for r in rows], dtype=float)
    raw_l = np.array([r['low'] for r in rows], dtype=float)
    raw_o = np.array([r['open'] for r in rows], dtype=float)

    # Forward-adjust
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
    close, high, low = adj['close'], adj['high'], adj['low']

    events = _find_decline_events(close, high, low, dates, min_decline=min_decline, max_decline=max_decline)

    name_row = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
    name = name_row['name'] if name_row else ''

    results = []
    for e in events:
        # Filter: only events where 60-day gain at peak < 0 (no prior run-up)
        pk_idx = e['peak_idx']
        gain_60d = round((close[pk_idx] - close[max(0, pk_idx - 60)]) / close[max(0, pk_idx - 60)] * 100, 1)
        if gain_60d >= 0:
            continue

        features = _compute_trough_features(close, high, low, e['peak_idx'], e['trough_idx'])
        features['decline_pct'] = e['decline_pct']
        prob = assess_pullback(features)
        if prob < min_probability:
            continue

        recovery = _compute_recovery(dates, close, e['trough_idx'], e['peak_price'])
        rsi_label = rsi_display_label(features.get('rsi'))

        # 5/10/20/60 day forward returns from trough
        n = len(close)
        ti = e['trough_idx']
        fwd_returns = {}
        for horizon, lbl in [(5, 'r5d'), (10, 'r10d'), (20, 'r20d'), (60, 'r60d')]:
            if ti + horizon < n:
                fwd_returns[lbl] = round((close[ti + horizon] - close[ti]) / close[ti] * 100, 1)
            else:
                fwd_returns[lbl] = None

        results.append({
            'code': code,
            'name': name,
            'peak_date': e['peak_date'],
            'peak_price': e['peak_price'],
            'trough_date': e['trough_date'],
            'trough_price': e['trough_price'],
            'features': features,
            'probability': prob,
            'rsi_label': rsi_label,
            'recovery': recovery,
            'forward_returns': fwd_returns,
        })

    return results


def retrieve_all_no_prior_gain(max_workers=8, min_probability=80, min_decline=3, max_decline=50,
                                progress_cb=None, codes_filter=None):
    """Full-market concurrent retrieval of "前期没涨就跌" pullback events.

    Args:
        max_workers: thread pool size.
        min_probability: minimum probability score (0-100).
        min_decline: minimum decline % to consider.
        max_decline: maximum decline % to consider.
        progress_cb: optional callback(done, total).
        codes_filter: optional list of codes to restrict scan (None = all active).

    Returns:
        dict with keys: params, summary, results (list of event dicts).
    """
    db = Database()
    codes = codes_filter if codes_filter else db.get_active_stock_codes()
    total = len(codes)
    all_results = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(retrieve_no_prior_gain, code, db, min_probability, min_decline, max_decline): code
            for code in codes
        }
        for future in as_completed(futures):
            done += 1
            try:
                stock_results = future.result()
                if stock_results:
                    all_results.extend(stock_results)
            except Exception:
                pass
            if progress_cb:
                progress_cb(done, total)

    # Sort: fewer warnings first, then higher probability
    def _warn_count(r):
        w = []
        f = r['features']
        if f.get('gain_60d', 0) > 50:
            w.append(1)
        return len(w)

    all_results.sort(key=lambda r: (_warn_count(r), -r['probability']))

    return {
        'params': {
            'min_probability': min_probability,
            'min_decline_pct': min_decline,
            'max_decline_pct': max_decline,
            'condition': 'gain_60d < 0',
        },
        'summary': {
            'total_scanned': total,
            'total_events': len(all_results),
            'stocks_with_events': len(set(r['code'] for r in all_results)),
        },
        'results': all_results,
    }


# ====================================================================
# Current-day scanner: "前期没涨就跌" (gain_60d < 0) + probability threshold
# ====================================================================

def scan_no_prior_gain_current(code, db, min_probability=80):
    """Scan a single stock for CURRENT pullback with gain_60d < 0.

    Same as scan_stock() but additionally requires:
    - gain_60d < 0 (no prior run-up before the drop)
    - probability >= min_probability
    """
    result = scan_stock(code, db)
    if result is None:
        return None
    if result['features']['gain_60d'] >= 0:
        return None
    if result['probability'] < min_probability:
        return None
    return result


def scan_all_no_prior_gain_current(max_workers=8, min_probability=80, progress_cb=None):
    """Full-market concurrent scan for CURRENT pullbacks with no prior gain.

    Returns list of stocks currently in a pullback where:
    - The stock didn't run up before the drop (60d gain at peak < 0)
    - The pullback probability >= min_probability

    Results sorted by: fewer warnings first, then higher probability.
    """
    db = Database()
    codes = db.get_active_stock_codes()
    total = len(codes)
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(scan_no_prior_gain_current, code, db, min_probability): code
            for code in codes
        }
        for future in as_completed(futures):
            done += 1
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception:
                pass
            if progress_cb:
                progress_cb(done, total)

    results.sort(key=lambda r: (len(r['warnings']), -r['probability']))
    return results


def retrieve_no_prior_gain_summary(results, group_by=('decline_pct', [(3, 8), (8, 12), (12, 18), (18, 50)])):
    """Generate summary statistics for a result set from retrieve_all_no_prior_gain.

    Args:
        results: list of event dicts from retrieve_no_prior_gain / retrieve_all_no_prior_gain.
        group_by: tuple of (feature_key, [(lo, hi), ...]) for breakdown tables.

    Returns:
        dict with recovery_rate, recovery_days stats, grouped breakdowns.
    """
    if not results:
        return {'total': 0}

    recovered = [r for r in results if r['recovery'] is not None]
    not_recovered = [r for r in results if r['recovery'] is None]
    rec_days = [r['recovery']['recovery_days'] for r in recovered]

    key, bins = group_by
    breakdowns = []
    for lo, hi in bins:
        group = [r for r in results if lo <= r['features'][key] < hi]
        if not group:
            continue
        rec_g = [r for r in group if r['recovery']]
        entry = {
            'range': f'{lo}-{hi}%',
            'count': len(group),
            'recovery_rate': round(len(rec_g) / len(group) * 100, 1) if group else 0,
        }
        if rec_g:
            entry['recovery_days_median'] = round(float(np.median([r['recovery']['recovery_days'] for r in rec_g])), 1)
            entry['recovery_days_mean'] = round(float(np.mean([r['recovery']['recovery_days'] for r in rec_g])), 1)
        breakdowns.append(entry)

    # Forward return averages (for events with data)
    fwd_summary = {}
    for horizon in ['r5d', 'r10d', 'r20d', 'r60d']:
        vals = [r['forward_returns'][horizon] for r in results if r['forward_returns'].get(horizon) is not None]
        if vals:
            fwd_summary[horizon] = {
                'mean': round(float(np.mean(vals)), 1),
                'median': round(float(np.median(vals)), 1),
                'positive_rate': round(len([v for v in vals if v > 0]) / len(vals) * 100, 1),
                'count': len(vals),
            }

    return {
        'total': len(results),
        'recovered': len(recovered),
        'not_recovered': len(not_recovered),
        'recovery_rate': round(len(recovered) / len(results) * 100, 1),
        'recovery_days_mean': round(float(np.mean(rec_days)), 1) if rec_days else None,
        'recovery_days_median': round(float(np.median(rec_days)), 1) if rec_days else None,
        'recovery_days_p25': round(float(np.percentile(rec_days, 25)), 1) if rec_days else None,
        'recovery_days_p75': round(float(np.percentile(rec_days, 75)), 1) if rec_days else None,
        'recovery_days_p90': round(float(np.percentile(rec_days, 90)), 1) if rec_days else None,
        'breakdowns': breakdowns,
        'forward_returns': fwd_summary,
    }
