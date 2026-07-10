"""
sample_high_prob.py — 专门抽查概率≥90%的历史回调，深挖恢复规律

用法:
  python scripts/sample_high_prob.py --samples 50 --min-prob 90
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def find_all_decline_events(dates, close, high, low, lookback=40, min_decline=3, max_decline=50):
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
            'peak_date': dates[peak_idx], 'peak_price': round(float(peak_price), 2),
            'trough_date': dates[trough_idx], 'trough_price': round(float(trough_price), 2),
            'decline_pct': round(decline, 1), 'peak_idx': peak_idx, 'trough_idx': trough_idx,
        })
    seen = set()
    unique = []
    for e in events:
        key = (e['peak_idx'], e['trough_idx'])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    unique.sort(key=lambda e: e['trough_date'])
    return unique


def compute_recovery(dates, close, trough_idx, peak_price, max_lookahead=504):
    n = len(close)
    for j in range(trough_idx + 1, min(trough_idx + 1 + max_lookahead, n)):
        if close[j] >= peak_price:
            return {'recovery_date': dates[j], 'recovery_days': j - trough_idx, 'recovery_price': round(float(close[j]), 2)}
    return None


def compute_features_at_trough(close, high, low, peak_idx, trough_idx):
    price = close[trough_idx]
    rsi14 = indicators.rsi(close, 14)
    macd = indicators.macd(close)
    kdj = indicators.kdj(high, low, close)
    boll = indicators.bollinger(close, 20, 2.0)
    ma20 = indicators.sma(close, 20)
    ma60 = indicators.sma(close, 60)
    adx_data = indicators.adx(high, low, close, 14)

    def g(arr):
        if trough_idx >= len(arr): return None
        v = arr[trough_idx]
        return float(v) if not np.isnan(v) else None

    gain_60d = round((close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)
    gain_20d = round((close[peak_idx] - close[max(0, peak_idx - 20)]) / close[max(0, peak_idx - 20)] * 100, 1)

    bb_lower = g(boll['lower']); bb_upper = g(boll['upper'])
    bb_pos = round((price - bb_lower) / (bb_upper - bb_lower) * 100, 1) if bb_lower and bb_upper and bb_upper != bb_lower else None

    return {
        'rsi': g(rsi14), 'kdj_k': g(kdj['k']), 'kdj_j': g(kdj['j']),
        'macd_dif': g(macd['dif']), 'macd_below_zero': g(macd['dif']) < 0 if g(macd['dif']) is not None else None,
        'bb_pos': bb_pos,
        'ma20_dist': round((price - ma20[trough_idx]) / ma20[trough_idx] * 100, 1) if g(ma20) else None,
        'ma60_dist': round((price - ma60[trough_idx]) / ma60[trough_idx] * 100, 1) if g(ma60) else None,
        'gain_20d': gain_20d, 'gain_60d': gain_60d, 'adx': g(adx_data['adx']),
    }


def assess_pullback_probability(features):
    d = features['decline_pct']
    rsi = features.get('rsi', 50); ma60_dist = features.get('ma60_dist', 0)
    bb_pos = features.get('bb_pos', 50); gain_60d = features.get('gain_60d', 0)

    if gain_60d is not None:
        if gain_60d < 0: prior_adj = +10
        elif gain_60d < 20: prior_adj = +5
        elif gain_60d < 50: prior_adj = 0
        elif gain_60d < 100: prior_adj = -20
        else: prior_adj = -35
    else:
        prior_adj = 0

    if d < 10: base = 100
    elif d < 15: base = 98
    elif d < 20: base = 95
    elif d < 25: base = 80
    elif d < 30: base = 60
    elif d < 40: base = 40
    else: base = 25

    if rsi is not None:
        if rsi < 30: base -= 20
        elif rsi < 40: base -= 5
        elif 40 <= rsi <= 55: base += 5
    if ma60_dist is not None and ma60_dist < -12: base -= 15
    elif ma60_dist is not None and ma60_dist < -6: base -= 5
    if bb_pos is not None and bb_pos < 10: base -= 10
    return max(0, min(100, base + prior_adj))


def collect_high_prob_events(db, sample_count=50, min_prob=90, min_decline=3, max_decline=50, seed=42,
                              min_trough_date=None, max_trough_date=None):
    """收集概率≥min_prob的高质量回调事件"""
    random.seed(seed); np.random.seed(seed)
    codes = db.get_active_stock_codes()
    sampled = random.sample(codes, min(sample_count * 3, len(codes)))  # oversample

    results = []
    for code in sampled:
        rows = db.fetchall(
            "SELECT trade_date, open, high, low, close, volume, turn "
            "FROM daily_kline WHERE code=? ORDER BY trade_date", (code,))
        if len(rows) < 120: continue

        dates = [r['trade_date'] for r in rows]
        raw_c = np.array([r['close'] for r in rows], dtype=float)
        raw_h = np.array([r['high'] for r in rows], dtype=float)
        raw_l = np.array([r['low'] for r in rows], dtype=float)
        raw_o = np.array([r['open'] for r in rows], dtype=float)

        afs = db.fetchall(
            "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date", (code,))
        adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
        close, high, low = adj['close'], adj['high'], adj['low']

        events = find_all_decline_events(dates, close, high, low, min_decline=min_decline, max_decline=max_decline)

        name_row = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
        stock_name = name_row['name'] if name_row else ''

        for e in events:
            features = compute_features_at_trough(close, high, low, e['peak_idx'], e['trough_idx'])
            features['decline_pct'] = e['decline_pct']
            prob = assess_pullback_probability(features)
            if prob < min_prob: continue
            if min_trough_date and e['trough_date'] < min_trough_date: continue
            if max_trough_date and e['trough_date'] > max_trough_date: continue

            recovery = compute_recovery(dates, close, e['trough_idx'], e['peak_price'])
            # Also track 5d/10d/20d/60d returns from trough
            n = len(close)
            ti = e['trough_idx']
            returns = {}
            for horizon, label in [(5, 'r5d'), (10, 'r10d'), (20, 'r20d'), (60, 'r60d')]:
                if ti + horizon < n:
                    ret = (close[ti + horizon] - close[ti]) / close[ti] * 100
                    returns[label] = round(ret, 1)
                else:
                    returns[label] = None

            results.append({
                'code': code, 'name': stock_name,
                'peak_date': e['peak_date'], 'peak_price': e['peak_price'],
                'trough_date': e['trough_date'], 'trough_price': e['trough_price'],
                'decline_pct': e['decline_pct'],
                'gain_60d': features['gain_60d'], 'gain_20d': features['gain_20d'],
                'rsi': features['rsi'], 'kdj_k': features['kdj_k'],
                'bb_pos': features['bb_pos'],
                'ma20_dist': features['ma20_dist'], 'ma60_dist': features['ma60_dist'],
                'adx': features['adx'],
                'probability': prob,
                'recovery_days': recovery['recovery_days'] if recovery else None,
                'recovery_date': recovery['recovery_date'] if recovery else None,
                **returns,
            })

        if len(results) >= sample_count * 5:
            break

    return results


def print_detailed_report(results, min_prob):
    """高清报告"""
    print()
    print("=" * 130)
    print(f"  高概率回调深度分析 — 概率 ≥ {min_prob}%")
    print("=" * 130)

    # === 明细表 ===
    header = (
        f"{'代码':<12} {'名称':<8} {'低点日期':<12} {'跌幅%':>6} {'60日涨%':>7} "
        f"{'RSI':>5} {'KDJ-K':>6} {'BB%':>5} {'MA20%':>6} {'MA60%':>6} "
        f"{'概率%':>6} {'恢复天':>6} {'5日':>6} {'10日':>6} {'20日':>6} {'60日':>6}"
    )
    print()
    print(header)
    print("-" * 130)

    results.sort(key=lambda r: (r['probability'], -(r['recovery_days'] or 9999)), reverse=True)

    for r in results:
        rec = f"{r['recovery_days']}d" if r['recovery_days'] else "未"
        bb = f"{r['bb_pos']:.0f}" if r['bb_pos'] is not None else '--'
        ma20 = f"{r['ma20_dist']:+.1f}" if r['ma20_dist'] is not None else '--'
        ma60 = f"{r['ma60_dist']:+.1f}" if r['ma60_dist'] is not None else '--'
        rsi_s = f"{r['rsi']:.0f}" if r['rsi'] is not None else '--'
        kdj_s = f"{r['kdj_k']:.1f}" if r['kdj_k'] is not None else '--'
        r5 = f"{r['r5d']:+.1f}%" if r['r5d'] is not None else '--'
        r10 = f"{r['r10d']:+.1f}%" if r['r10d'] is not None else '--'
        r20 = f"{r['r20d']:+.1f}%" if r['r20d'] is not None else '--'
        r60 = f"{r['r60d']:+.1f}%" if r['r60d'] is not None else '--'

        print(
            f"{r['code']:<12} {r['name']:<8} {r['trough_date']:<12} {r['decline_pct']:>5.1f}% {r['gain_60d']:>+6.1f}% "
            f"{rsi_s:>5} {kdj_s:>6} {bb:>5} {ma20:>6} {ma60:>6} "
            f"{r['probability']:>5.0f}% {rec:>6} {r5:>6} {r10:>6} {r20:>6} {r60:>6}"
        )

    # === 统计摘要 ===
    recovered = [r for r in results if r['recovery_days'] is not None]
    not_recovered = [r for r in results if r['recovery_days'] is None]

    print()
    print("=" * 130)
    print("  统计摘要")
    print("=" * 130)
    print(f"  高概率事件总数: {len(results)}")
    print(f"  已恢复到前高: {len(recovered)} ({len(recovered)/max(len(results),1)*100:.1f}%)")
    print(f"  未恢复: {len(not_recovered)} ({len(not_recovered)/max(len(results),1)*100:.1f}%)")

    if recovered:
        days = [r['recovery_days'] for r in recovered]
        print(f"\n  恢复天数分布:")
        print(f"    均值: {np.mean(days):.1f}d  中位数: {np.median(days):.0f}d  标准差: {np.std(days):.1f}d")
        print(f"    最短: {min(days)}d  最长: {max(days)}d")
        pcts = [25, 50, 75, 90]
        for p in pcts:
            print(f"    P{p}: {np.percentile(days, p):.0f}d")

    # === 按跌幅细分恢复 ===
    print(f"\n  按跌幅分组:")
    bins = [(3, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 50)]
    for lo, hi in bins:
        group = [r for r in results if lo <= r['decline_pct'] < hi]
        if not group: continue
        rec_g = [r for r in group if r['recovery_days']]
        rec_rate = len(rec_g) / len(group) * 100
        if rec_g:
            avg = np.mean([r['recovery_days'] for r in rec_g])
            med = np.median([r['recovery_days'] for r in rec_g])
            print(f"    {lo}-{hi}%: {len(group)}次, 恢复率{rec_rate:.0f}%, 均值{avg:.0f}d, 中位数{med:.0f}d")
        else:
            print(f"    {lo}-{hi}%: {len(group)}次, 恢复率{rec_rate:.0f}%")

    # === 按 RSI 细分 ===
    print(f"\n  按 RSI 分组 (低点处):")
    for lo, hi, label in [(0, 30, "超卖<30"), (30, 40, "弱势30-40"), (40, 55, "中性40-55"), (55, 100, "偏强>55")]:
        group = [r for r in results if r['rsi'] is not None and lo <= r['rsi'] < hi]
        if not group: continue
        rec_g = [r for r in group if r['recovery_days']]
        rec_rate = len(rec_g) / len(group) * 100
        if rec_g:
            avg = np.mean([r['recovery_days'] for r in rec_g])
            print(f"    {label}: {len(group)}次, 恢复率{rec_rate:.0f}%, 均值{avg:.0f}d")
        else:
            print(f"    {label}: {len(group)}次, 恢复率{rec_rate:.0f}%")

    # === 按前置涨幅细分 ===
    print(f"\n  按前置60日涨幅分组:")
    for lo, hi, label in [(-999, 0, "前期没涨<0"), (0, 20, "温和上涨0-20"), (20, 50, "较强20-50"), (50, 999, "暴涨>50")]:
        group = [r for r in results if lo <= r['gain_60d'] < hi]
        if not group: continue
        rec_g = [r for r in group if r['recovery_days']]
        rec_rate = len(rec_g) / len(group) * 100
        if rec_g:
            avg = np.mean([r['recovery_days'] for r in rec_g])
            print(f"    {label}: {len(group)}次, 恢复率{rec_rate:.0f}%, 均值{avg:.0f}d")
        else:
            print(f"    {label}: {len(group)}次, 恢复率{rec_rate:.0f}%")

    # === 反弹力度分析 (从低点起算) ===
    print(f"\n  从低点起的反弹幅度 (不含未恢复的):")
    for horizon, label in [('r5d', '5日'), ('r10d', '10日'), ('r20d', '20日'), ('r60d', '60日')]:
        vals = [r[horizon] for r in results if r[horizon] is not None]
        if vals:
            print(f"    {label}: 均值 {np.mean(vals):+.1f}%, 中位数 {np.median(vals):+.1f}%, "
                  f"正收益比例 {len([v for v in vals if v > 0])/len(vals)*100:.0f}%")

    # === 最强反弹案例 ===
    print(f"\n  --- 最强反弹 TOP 10 (60日) ---")
    with_r60 = sorted([r for r in results if r['r60d'] is not None], key=lambda r: -r['r60d'])[:10]
    for r in with_r60:
        rec = f"{r['recovery_days']}d" if r['recovery_days'] else "未恢复"
        print(f"    {r['code']} {r['name']:<6} {r['trough_date']} 跌{r['decline_pct']:.1f}% -> "
              f"5日{r['r5d']:+.1f}% 10日{r['r10d']:+.1f}% 20日{r['r20d']:+.1f}% 60日{r['r60d']:+.1f}%  恢复:{rec}")

    # === 最弱反弹案例 ===
    print(f"\n  --- 最弱反弹 BOTTOM 10 (60日) ---")
    with_r60_sorted = sorted([r for r in results if r['r60d'] is not None], key=lambda r: r['r60d'])[:10]
    for r in with_r60_sorted:
        rec = f"{r['recovery_days']}d" if r['recovery_days'] else "未恢复"
        print(f"    {r['code']} {r['name']:<6} {r['trough_date']} 跌{r['decline_pct']:.1f}% -> "
              f"5日{r['r5d']:+.1f}% 10日{r['r10d']:+.1f}% 20日{r['r20d']:+.1f}% 60日{r['r60d']:+.1f}%  恢复:{rec}")

    # === 综合结论 ===
    print()
    print(f"  --- 高概率回调(>= {min_prob}%)的核心特征 ---")
    # 统计高概率事件中常见的指标值
    avg_decline = np.mean([r['decline_pct'] for r in results])
    avg_rsi = np.mean([r['rsi'] for r in results if r['rsi'] is not None])
    avg_ma20 = np.mean([r['ma20_dist'] for r in results if r['ma20_dist'] is not None])
    avg_gain60 = np.mean([r['gain_60d'] for r in results])
    print(f"  平均跌幅: {avg_decline:.1f}%")
    print(f"  平均 RSI: {avg_rsi:.0f}")
    print(f"  平均 MA20偏离: {avg_ma20:+.1f}%")
    print(f"  平均 60日前置涨幅: {avg_gain60:+.1f}%")
    if recovered:
        print(f"  恢复把握: {len(recovered)/max(len(results),1)*100:.0f}% 概率恢复, "
              f"平均 {np.mean([r['recovery_days'] for r in recovered]):.0f} 天回到前高")
    print()
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="高概率回调深度抽查")
    parser.add_argument("--samples", type=int, default=50, help="抽样股票基数 (default: 50)")
    parser.add_argument("--min-prob", type=int, default=90, help="最低概率阈值 (default: 90)")
    parser.add_argument("--min-decline", type=float, default=3, help="最小跌幅%")
    parser.add_argument("--max-decline", type=float, default=50, help="最大跌幅%")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-date", type=str, default=None, help="低点起始日期 YYYY-MM-DD")
    parser.add_argument("--to-date", type=str, default=None, help="低点截止日期 YYYY-MM-DD")
    args = parser.parse_args()

    db = Database()
    print(f"\n扫描中... (抽样{args.samples}只, 概率>={args.min_prob}%)")
    results = collect_high_prob_events(
        db, sample_count=args.samples, min_prob=args.min_prob,
        min_decline=args.min_decline, max_decline=args.max_decline,
        seed=args.seed,
        min_trough_date=args.from_date, max_trough_date=args.to_date,
    )
    print(f"找到 {len(results)} 个高概率回调事件")

    if not results:
        print("未找到符合条件的事件")
        return

    print_detailed_report(results, args.min_prob)


if __name__ == "__main__":
    main()
