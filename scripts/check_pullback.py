"""
check_pullback.py — 判断一个下跌是回调还是破位

用法:
  python scripts/check_pullback.py --code sh.601138 --date 2026-04-15

输出: 回调概率 + 关键特征值
"""

import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def find_nearest_peak_trough(close, high, low, dates, target_date, lookback=40):
    """找到 target_date 之前最近的高点，以及高点到 target_date 之间的最低点"""
    # Find nearest trading day (target or earlier)
    idx = None
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] <= target_date:
            idx = i
            break
    if idx is None:
        return None

    if idx < lookback:
        return None

    # 前 40 天内找最高点
    peak_idx = idx - lookback + np.argmax(high[idx - lookback:idx])
    peak_price = high[peak_idx]

    # 峰到当前日期之间找最低点
    trough_idx = peak_idx + np.argmin(low[peak_idx:idx + 1])
    trough_price = low[trough_idx]

    decline = (peak_price - trough_price) / peak_price * 100
    if decline < 3:
        return None  # 跌幅太小，不构成回调

    return {
        'peak_date': dates[peak_idx],
        'peak_price': round(float(peak_price), 2),
        'trough_date': dates[trough_idx],
        'trough_price': round(float(trough_price), 2),
        'decline_pct': round(decline, 1),
        'peak_idx': peak_idx,
        'trough_idx': trough_idx,
    }


def assess_pullback(features):
    """
    基于统计特征判断回调概率。

    两层判断:
      1. 前置涨幅 (gain_60d): 涨幅越大，越可能冲顶衰竭 -> 破位
      2. 当前跌幅 + 技术指标: 辅助微调
    """
    d = features['decline_pct']
    rsi = features.get('rsi', 50)
    ma60_dist = features.get('ma60_dist', 0)
    bb_pos = features.get('bb_pos', 50)
    gain_60d = features.get('gain_60d', 0)

    # Layer 1: Prior gain context (from pullback-start vs downtrend-start comparison)
    if gain_60d is not None:
        if gain_60d < 0:
            prior_adj = +10       # no prior gain, very likely pullback
        elif gain_60d < 20:
            prior_adj = +5
        elif gain_60d < 50:
            prior_adj = 0         # moderate, no adjustment
        elif gain_60d < 100:
            prior_adj = -20       # big prior run-up, exhaustion risk
        else:
            prior_adj = -35       # parabolic, high breakdown risk
    else:
        prior_adj = 0

    # Layer 2: Current decline features
    if d < 10:
        base = 100
    elif d < 15:
        base = 98
    elif d < 20:
        base = 95
    elif d < 25:
        base = 80
    elif d < 30:
        base = 60
    else:
        base = 40

    if rsi is not None:
        if rsi < 30:
            base -= 20
        elif rsi < 40:
            base -= 5
        elif 40 <= rsi <= 55:
            base += 5

    if ma60_dist is not None and ma60_dist < -12:
        base -= 15
    elif ma60_dist is not None and ma60_dist < -6:
        base -= 5

    if bb_pos is not None and bb_pos < 10:
        base -= 10

    return max(0, min(100, base + prior_adj))


def main():
    parser = argparse.ArgumentParser(description="判断下跌是回调还是破位")
    parser.add_argument("--code", required=True, help="股票代码 (sh.601138)")
    parser.add_argument("--date", required=True, help="日期 (2026-04-15)")
    args = parser.parse_args()

    db = Database()
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn "
        "FROM daily_kline WHERE code=? ORDER BY trade_date",
        (args.code,),
    )
    if len(rows) < 120:
        print(f"数据不足: {len(rows)} 条")
        return

    dates = [r['trade_date'] for r in rows]
    raw_c = np.array([r['close'] for r in rows], dtype=float)
    raw_h = np.array([r['high'] for r in rows], dtype=float)
    raw_l = np.array([r['low'] for r in rows], dtype=float)
    raw_o = np.array([r['open'] for r in rows], dtype=float)

    # Forward-adjust
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (args.code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
    close, high, low = adj['close'], adj['high'], adj['low']

    # Find peak->trough
    pt = find_nearest_peak_trough(close, high, low, dates, args.date)
    if pt is None:
        print(f"在 {args.date} 附近未找到有效的下跌结构")
        return

    idx = pt['trough_idx']
    price = close[idx]

    # Compute features
    rsi14 = indicators.rsi(close, 14)
    macd = indicators.macd(close)
    kdj = indicators.kdj(high, low, close)
    boll = indicators.bollinger(close, 20, 2.0)
    ma20 = indicators.sma(close, 20)
    ma60 = indicators.sma(close, 60)

    def g(arr, d=0):
        v = arr[idx] if idx < len(arr) else np.nan
        return float(v) if not np.isnan(v) else None

    peak_idx = pt['peak_idx']
    gain_20d = round((close[peak_idx] - close[max(0, peak_idx - 20)]) / close[max(0, peak_idx - 20)] * 100, 1)
    gain_60d = round((close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)

    features = {
        'decline_pct': pt['decline_pct'],
        'rsi': g(rsi14),
        'kdj_k': g(kdj['k']),
        'macd_dif': g(macd['dif']),
        'macd_below_zero': g(macd['dif']) < 0 if g(macd['dif']) is not None else None,
        'bb_pos': round((price - boll['lower'][idx]) / (boll['upper'][idx] - boll['lower'][idx]) * 100, 1) if idx < len(boll['lower']) else None,
        'ma20_dist': round((price - ma20[idx]) / ma20[idx] * 100, 1) if idx < len(ma20) and not np.isnan(ma20[idx]) else None,
        'ma60_dist': round((price - ma60[idx]) / ma60[idx] * 100, 1) if idx < len(ma60) and not np.isnan(ma60[idx]) else None,
        'gain_20d': gain_20d,
        'gain_60d': gain_60d,
    }

    prob = assess_pullback(features)

    # Output
    print(f"\n{'='*50}")
    print(f"  {args.code}  @ {args.date}")
    print(f"{'='*50}")
    print(f"  前高: {pt['peak_date']}  {pt['peak_price']}")
    print(f"  低点: {pt['trough_date']}  {pt['trough_price']}")
    print(f"  跌幅: {pt['decline_pct']}%")
    print()
    print(f"  --- 前置涨幅 (前高处) ---")
    print(f"  近20日涨幅:  {features['gain_20d']}%")
    print(f"  近60日涨幅:  {features['gain_60d']}%")
    print(f"  --- 低点特征值 ---")
    print(f"  RSI(14):     {features['rsi']:.1f}" if features['rsi'] else "  RSI: N/A")
    print(f"  KDJ-K:       {features['kdj_k']:.1f}" if features['kdj_k'] else "")
    print(f"  MACD DIF:    {features['macd_dif']:.2f}  {'(零轴下)' if features['macd_below_zero'] else '(零轴上)'}" if features['macd_dif'] else "")
    print(f"  BB Position: {features['bb_pos']}% (0=下轨, 100=上轨)" if features['bb_pos'] else "")
    print(f"  MA20 偏离:   {features['ma20_dist']}%" if features['ma20_dist'] else "")
    print(f"  MA60 偏离:   {features['ma60_dist']}%" if features['ma60_dist'] else "")
    print()
    print(f"  =======================")
    print(f"  回调概率: {prob}%")
    print(f"  =======================")

    if prob >= 80:
        print(f"\n  -> 大概率是回调")
    elif prob >= 50:
        print(f"\n  -> 中性偏回调，需警惕")
    else:
        print(f"\n  -> 破位风险较高")
    if features['gain_60d'] > 50:
        print(f"  !! 注意: 近60日涨幅 {features['gain_60d']}%，冲顶衰竭风险")


if __name__ == "__main__":
    main()
