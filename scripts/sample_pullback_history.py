"""
sample_pullback_history.py — 从历史数据中抽查下跌事件，输出回调概率 + 恢复到前高所需天数

用法:
  python scripts/sample_pullback_history.py --samples 20
  python scripts/sample_pullback_history.py --samples 30 --min-decline 5 --max-decline 30
  python scripts/sample_pullback_history.py --code sh.600000  # 单只股票全部回调
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def find_all_decline_events(dates, close, high, low, lookback=40, min_decline=3, max_decline=50):
    """
    遍历全历史，找出所有"局部高点 → 低点"的下跌事件。
    每个事件记录: peak_date, peak_price, trough_date, trough_price, decline_pct, peak_idx, trough_idx
    """
    n = len(close)
    events = []

    # 对每一天，往前找局部高点
    for idx in range(lookback + 5, n):
        # 前 lookback 天内找最高点
        peak_idx = idx - lookback + np.argmax(high[idx - lookback:idx + 1])
        peak_price = high[peak_idx]

        # 峰到今天之间找最低点
        trough_idx = peak_idx + np.argmin(low[peak_idx:idx + 1])
        trough_price = low[trough_idx]

        decline = (peak_price - trough_price) / peak_price * 100
        if decline < min_decline or decline > max_decline:
            continue
        if peak_idx >= trough_idx:
            continue
        if peak_idx == trough_idx:
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

    # 去重: 按 (peak_idx, trough_idx) 合并
    seen = set()
    unique = []
    for e in events:
        key = (e['peak_idx'], e['trough_idx'])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    # 按 trough_date 排序
    unique.sort(key=lambda e: e['trough_date'])
    return unique


def compute_recovery(dates, close, trough_idx, peak_price, max_lookahead=252):
    """
    计算从 trough_idx 开始，多少天后收盘价 >= peak_price。
    如果 max_lookahead 天内未恢复，返回 None。
    """
    n = len(close)
    for j in range(trough_idx + 1, min(trough_idx + 1 + max_lookahead, n)):
        if close[j] >= peak_price:
            return {
                'recovery_date': dates[j],
                'recovery_days': j - trough_idx,
                'recovery_price': round(float(close[j]), 2),
            }
    return None


def compute_features_at_trough(close, high, low, peak_idx, trough_idx):
    """在低谷位置计算技术指标特征值"""
    price = close[trough_idx]

    rsi14 = indicators.rsi(close, 14)
    macd = indicators.macd(close)
    kdj = indicators.kdj(high, low, close)
    boll = indicators.bollinger(close, 20, 2.0)
    ma20 = indicators.sma(close, 20)
    ma60 = indicators.sma(close, 60)
    adx_data = indicators.adx(high, low, close, 14)

    def g(arr):
        """Safe value getter for arrays that may be shorter than price array"""
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
        'rsi': g(rsi14),
        'kdj_k': g(kdj['k']),
        'kdj_j': g(kdj['j']),
        'macd_dif': g(macd['dif']),
        'macd_below_zero': g(macd['dif']) < 0 if g(macd['dif']) is not None else None,
        'bb_pos': bb_pos,
        'ma20_dist': round((price - ma20[trough_idx]) / ma20[trough_idx] * 100, 1) if g(ma20) else None,
        'ma60_dist': round((price - ma60[trough_idx]) / ma60[trough_idx] * 100, 1) if g(ma60) else None,
        'gain_20d': gain_20d,
        'gain_60d': gain_60d,
        'adx': g(adx_data['adx']),
    }


def assess_pullback_probability(features):
    """
    与 check_pullback.py 一致的评估逻辑。
    返回 0-100 的概率分数。
    """
    d = features['decline_pct']
    rsi = features.get('rsi', 50)
    ma60_dist = features.get('ma60_dist', 0)
    bb_pos = features.get('bb_pos', 50)
    gain_60d = features.get('gain_60d', 0)

    # Layer 1: 前置涨幅
    if gain_60d is not None:
        if gain_60d < 0:
            prior_adj = +10
        elif gain_60d < 20:
            prior_adj = +5
        elif gain_60d < 50:
            prior_adj = 0
        elif gain_60d < 100:
            prior_adj = -20
        else:
            prior_adj = -35
    else:
        prior_adj = 0

    # Layer 2: 当前跌幅
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
    elif d < 40:
        base = 40
    else:
        base = 25

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


def sample_and_analyze(db, sample_count=20, min_decline=3, max_decline=50, seed=42):
    """随机抽查股票，分析历史回调事件"""
    random.seed(seed)
    np.random.seed(seed)

    codes = db.get_active_stock_codes()
    sampled_codes = random.sample(codes, min(sample_count, len(codes)))

    all_results = []

    for code in sampled_codes:
        # 获取数据
        rows = db.fetchall(
            "SELECT trade_date, open, high, low, close, volume, turn "
            "FROM daily_kline WHERE code=? ORDER BY trade_date",
            (code,),
        )
        if len(rows) < 120:
            continue

        dates = [r['trade_date'] for r in rows]
        raw_c = np.array([r['close'] for r in rows], dtype=float)
        raw_h = np.array([r['high'] for r in rows], dtype=float)
        raw_l = np.array([r['low'] for r in rows], dtype=float)
        raw_o = np.array([r['open'] for r in rows], dtype=float)

        # 前复权
        afs = db.fetchall(
            "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
            (code,),
        )
        adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
        close, high, low = adj['close'], adj['high'], adj['low']

        # 找所有下跌事件
        events = find_all_decline_events(dates, close, high, low, min_decline=min_decline, max_decline=max_decline)

        # 随机选最多 3 个事件 per stock
        if len(events) > 3:
            events = random.sample(events, 3)

        name_row = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
        stock_name = name_row['name'] if name_row else ''

        for e in events:
            # 计算特征
            features = compute_features_at_trough(close, high, low, e['peak_idx'], e['trough_idx'])
            features['decline_pct'] = e['decline_pct']

            # 评估概率
            prob = assess_pullback_probability(features)

            # 计算恢复天数
            recovery = compute_recovery(dates, close, e['trough_idx'], e['peak_price'])

            all_results.append({
                'code': code,
                'name': stock_name,
                'peak_date': e['peak_date'],
                'peak_price': e['peak_price'],
                'trough_date': e['trough_date'],
                'trough_price': e['trough_price'],
                'decline_pct': e['decline_pct'],
                'gain_60d': features['gain_60d'],
                'rsi': features['rsi'],
                'kdj_k': features['kdj_k'],
                'bb_pos': features['bb_pos'],
                'ma20_dist': features['ma20_dist'],
                'ma60_dist': features['ma60_dist'],
                'adx': features['adx'],
                'probability': prob,
                'recovery_days': recovery['recovery_days'] if recovery else None,
                'recovery_date': recovery['recovery_date'] if recovery else None,
            })

    return all_results


def print_report(results):
    """美化输出报告"""
    print()
    print("=" * 120)
    print("  历史下跌事件抽查报告 — 回调概率 & 恢复分析")
    print("=" * 120)

    # 分类统计
    recovered = [r for r in results if r['recovery_days'] is not None]
    not_recovered = [r for r in results if r['recovery_days'] is None]
    high_prob = [r for r in results if r['probability'] >= 80]
    mid_prob = [r for r in results if 50 <= r['probability'] < 80]
    low_prob = [r for r in results if r['probability'] < 50]

    # === 明细表 ===
    header = (
        f"{'代码':<12} {'名称':<8} {'前高日期':<12} {'前高价':>8} "
        f"{'低点日期':<12} {'低点价':>8} {'跌幅%':>6} "
        f"{'60日涨%':>7} {'RSI':>5} {'BB%':>5} {'MA20%':>6} "
        f"{'概率%':>6} {'恢复天数':>8} {'恢复日期':<12}"
    )
    print()
    print(header)
    print("-" * 120)

    # 按概率排序: 低概率(可能破位)的排前面, 高概率(健康回调)排后面
    results.sort(key=lambda r: (r['probability'], -(r['recovery_days'] or 9999)))

    for r in results:
        rec_days = f"{r['recovery_days']}d" if r['recovery_days'] else "未恢复"
        rec_date = r['recovery_date'] or '—'
        bb = f"{r['bb_pos']:.0f}" if r['bb_pos'] is not None else 'N/A'
        ma20 = f"{r['ma20_dist']:+.1f}" if r['ma20_dist'] is not None else 'N/A'
        ma60 = f"{r['ma60_dist']:+.1f}" if r['ma60_dist'] is not None else 'N/A'
        rsi_str = f"{r['rsi']:.0f}" if r['rsi'] is not None else 'N/A'
        adx_str = f"{r['adx']:.1f}" if r['adx'] is not None else 'N/A'

        # 颜色标记 (用符号代替)
        flag = ""
        if r['probability'] >= 80:
            flag = "OK"
        elif r['probability'] < 50:
            flag = "XX"

        print(
            f"{r['code']:<12} {r['name']:<8} {r['peak_date']:<12} {r['peak_price']:>8.2f} "
            f"{r['trough_date']:<12} {r['trough_price']:>8.2f} {r['decline_pct']:>5.1f}% "
            f"{r['gain_60d']:>+6.1f}% {rsi_str:>5} {bb:>5} {ma20:>6} "
            f"{r['probability']:>5.0f}% {flag} {rec_days:>8} {rec_date:<12}"
        )

    # === 统计摘要 ===
    print()
    print("=" * 120)
    print("  统计摘要")
    print("=" * 120)
    print(f"  总抽查事件: {len(results)}")
    print(f"  已恢复 (>前高): {len(recovered)} / {len(results)}  ({len(recovered)/max(len(results),1)*100:.1f}%)")
    print(f"  未恢复 (含未触及): {len(not_recovered)} / {len(results)}  ({len(not_recovered)/max(len(results),1)*100:.1f}%)")

    if recovered:
        rec_days_list = [r['recovery_days'] for r in recovered]
        print(f"  恢复天数 — 均值: {np.mean(rec_days_list):.1f}d  中位数: {np.median(rec_days_list):.0f}d  "
              f"最短: {min(rec_days_list)}d  最长: {max(rec_days_list)}d")

    print()
    print(f"  按概率分组:")
    print(f"    高概率 (≥80%): {len(high_prob)} 个, 恢复率 {len([r for r in high_prob if r['recovery_days']])}/{len(high_prob)} = {len([r for r in high_prob if r['recovery_days']])/max(len(high_prob),1)*100:.1f}%")
    if high_prob and [r for r in high_prob if r['recovery_days']]:
        days = [r['recovery_days'] for r in high_prob if r['recovery_days']]
        print(f"      恢复天数均值: {np.mean(days):.1f}d, 中位数: {np.median(days):.0f}d")

    print(f"    中等概率 (50-79%): {len(mid_prob)} 个, 恢复率 {len([r for r in mid_prob if r['recovery_days']])}/{len(mid_prob)} = {len([r for r in mid_prob if r['recovery_days']])/max(len(mid_prob),1)*100:.1f}%")
    if mid_prob and [r for r in mid_prob if r['recovery_days']]:
        days = [r['recovery_days'] for r in mid_prob if r['recovery_days']]
        print(f"      恢复天数均值: {np.mean(days):.1f}d, 中位数: {np.median(days):.0f}d")

    print(f"    低概率 (<50%): {len(low_prob)} 个, 恢复率 {len([r for r in low_prob if r['recovery_days']])}/{len(low_prob)} = {len([r for r in low_prob if r['recovery_days']])/max(len(low_prob),1)*100:.1f}%")
    if low_prob and [r for r in low_prob if r['recovery_days']]:
        days = [r['recovery_days'] for r in low_prob if r['recovery_days']]
        print(f"      恢复天数均值: {np.mean(days):.1f}d, 中位数: {np.median(days):.0f}d")

    # === 按跌幅分组 ===
    print()
    print(f"  按跌幅分组:")
    for lo, hi, label in [(3, 10, "小跌 3-10%"), (10, 20, "中跌 10-20%"),
                           (20, 30, "大跌 20-30%"), (30, 50, "暴跌 30-50%")]:
        group = [r for r in results if lo <= r['decline_pct'] < hi]
        if group:
            rec = [r for r in group if r['recovery_days']]
            rec_rate = len(rec) / len(group) * 100
            avg_days = np.mean([r['recovery_days'] for r in rec]) if rec else 0
            print(f"    {label}: {len(group)} 个, 恢复率 {rec_rate:.0f}%, 平均恢复 {avg_days:.1f}d")

    # === 关键发现 ===
    print()
    print(f"  --- 关键发现 ---")
    # 找规律: 概率和实际恢复的关系
    if recovered:
        probs = [r['probability'] for r in recovered]
        recs = [r['recovery_days'] for r in recovered]
        corr = np.corrcoef(probs, recs)[0, 1] if len(probs) > 2 else 0
        print(f"  概率 vs 恢复天数的相关系数: {corr:.2f}  "
              f"({'负相关: 概率越高恢复越快' if corr < -0.2 else '弱相关' if abs(corr) < 0.2 else '正相关'})")

    # 跌幅 vs 恢复天数
    if recovered:
        declines = [r['decline_pct'] for r in recovered]
        corr2 = np.corrcoef(declines, recs)[0, 1] if len(declines) > 2 else 0
        print(f"  跌幅 vs 恢复天数的相关系数: {corr2:.2f}  "
              f"({'跌幅越大恢复越慢' if corr2 > 0.3 else '弱相关' if abs(corr2) < 0.3 else '跌幅越大恢复越快'})")

    print()
    print("=" * 120)


def analyze_single_stock(db, code):
    """分析单只股票的所有历史回调"""
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn "
        "FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),
    )
    if len(rows) < 120:
        print(f"数据不足: {len(rows)} 条")
        return []

    dates = [r['trade_date'] for r in rows]
    raw_c = np.array([r['close'] for r in rows], dtype=float)
    raw_h = np.array([r['high'] for r in rows], dtype=float)
    raw_l = np.array([r['low'] for r in rows], dtype=float)
    raw_o = np.array([r['open'] for r in rows], dtype=float)

    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
    close, high, low = adj['close'], adj['high'], adj['low']

    name_row = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
    stock_name = name_row['name'] if name_row else ''

    events = find_all_decline_events(dates, close, high, low, min_decline=5, max_decline=40)

    results = []
    for e in events:
        features = compute_features_at_trough(close, high, low, e['peak_idx'], e['trough_idx'])
        features['decline_pct'] = e['decline_pct']
        prob = assess_pullback_probability(features)
        recovery = compute_recovery(dates, close, e['trough_idx'], e['peak_price'])

        results.append({
            'code': code,
            'name': stock_name,
            'peak_date': e['peak_date'],
            'peak_price': e['peak_price'],
            'trough_date': e['trough_date'],
            'trough_price': e['trough_price'],
            'decline_pct': e['decline_pct'],
            'gain_60d': features['gain_60d'],
            'rsi': features['rsi'],
            'kdj_k': features['kdj_k'],
            'bb_pos': features['bb_pos'],
            'ma20_dist': features['ma20_dist'],
            'ma60_dist': features['ma60_dist'],
            'adx': features['adx'],
            'probability': prob,
            'recovery_days': recovery['recovery_days'] if recovery else None,
            'recovery_date': recovery['recovery_date'] if recovery else None,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="抽查历史下跌事件，评估回调概率和恢复时间")
    parser.add_argument("--samples", type=int, default=20, help="抽查股票数量 (default: 20)")
    parser.add_argument("--min-decline", type=float, default=3, help="最小跌幅% (default: 3)")
    parser.add_argument("--max-decline", type=float, default=50, help="最大跌幅% (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--code", type=str, default=None, help="分析单只股票的全部回调")
    args = parser.parse_args()

    db = Database()

    if args.code:
        print(f"\n分析 {args.code} 的全部历史回调...")
        results = analyze_single_stock(db, args.code)
        if not results:
            print("未找到符合条件的回调事件")
            return
        print_report(results)
    else:
        print(f"\n随机抽查 {args.samples} 只股票 (跌幅 {args.min_decline}-{args.max_decline}%)...")
        results = sample_and_analyze(db, args.samples, args.min_decline, args.max_decline, args.seed)
        print_report(results)


if __name__ == "__main__":
    main()
