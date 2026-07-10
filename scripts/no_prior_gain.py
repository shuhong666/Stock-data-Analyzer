"""
no_prior_gain.py — 抽查"前期没涨就跌"(60日涨幅<0)的高概率回调事件
"""
import os, sys, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def find_events(dates, close, high, low, lookback=40, min_d=3, max_d=50):
    n = len(close)
    events = []
    for idx in range(lookback + 5, n):
        pk = idx - lookback + np.argmax(high[idx - lookback:idx + 1])
        tr = pk + np.argmin(low[pk:idx + 1])
        dec = (high[pk] - low[tr]) / high[pk] * 100
        if dec < min_d or dec > max_d or pk >= tr:
            continue
        events.append({
            'pd': dates[pk], 'pp': round(float(high[pk]), 2),
            'td': dates[tr], 'tp': round(float(low[tr]), 2),
            'd': round(dec, 1), 'pk': pk, 'tr': tr,
        })
    seen = set()
    uni = []
    for e in events:
        k = (e['pk'], e['tr'])
        if k not in seen:
            seen.add(k)
            uni.append(e)
    return sorted(uni, key=lambda e: e['td'])


def recovery(dates, close, ti, pp, max_l=504):
    for j in range(ti + 1, min(ti + 1 + max_l, len(close))):
        if close[j] >= pp:
            return {'rd': dates[j], 'days': j - ti}
    return None


def assess(d):
    rsi = d.get('rsi', 50)
    ma60 = d.get('ma60_dist', 0)
    bb = d.get('bb_pos', 50)
    g60 = d.get('gain_60d', 0)
    pa = +10 if g60 < 0 else (+5 if g60 < 20 else (0 if g60 < 50 else (-20 if g60 < 100 else -35)))
    if d['decline_pct'] < 10:
        b = 100
    elif d['decline_pct'] < 15:
        b = 98
    elif d['decline_pct'] < 20:
        b = 95
    elif d['decline_pct'] < 25:
        b = 80
    elif d['decline_pct'] < 30:
        b = 60
    else:
        b = 40
    if rsi is not None:
        if rsi < 30: b -= 20
        elif rsi < 40: b -= 5
        elif 40 <= rsi <= 55: b += 5
    if ma60 is not None and ma60 < -12: b -= 15
    elif ma60 is not None and ma60 < -6: b -= 5
    if bb is not None and bb < 10: b -= 10
    return max(0, min(100, b + pa))


def main():
    db = Database()
    random.seed(999)
    np.random.seed(999)
    codes = random.sample(db.get_active_stock_codes(), 200)
    results = []

    for code in codes:
        rows = db.fetchall(
            "SELECT trade_date, open, high, low, close, volume, turn "
            "FROM daily_kline WHERE code=? ORDER BY trade_date", (code,))
        if len(rows) < 120:
            continue

        dates = [r['trade_date'] for r in rows]
        rc = np.array([r['close'] for r in rows], float)
        rh = np.array([r['high'] for r in rows], float)
        rl = np.array([r['low'] for r in rows], float)
        ro = np.array([r['open'] for r in rows], float)

        afs = db.fetchall(
            "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date", (code,))
        adj = indicators.forward_adjust(rc, rh, rl, ro, dates, afs)
        close, high, low = adj['close'], adj['high'], adj['low']

        events = find_events(dates, close, high, low)
        nm = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
        name = nm['name'] if nm else ''

        for e in events:
            pk = e['pk']
            tr = e['tr']
            price = close[tr]
            rsi14 = indicators.rsi(close, 14)
            kdj = indicators.kdj(high, low, close)
            boll = indicators.bollinger(close, 20, 2.0)
            ma20 = indicators.sma(close, 20)
            ma60 = indicators.sma(close, 60)

            def g(arr):
                if tr >= len(arr): return None
                v = arr[tr]
                return float(v) if not np.isnan(v) else None

            g60 = round((close[pk] - close[max(0, pk - 60)]) / close[max(0, pk - 60)] * 100, 1)
            bb_l = g(boll['lower'])
            bb_u = g(boll['upper'])
            bb_pos = round((price - bb_l) / (bb_u - bb_l) * 100, 1) if bb_l and bb_u and bb_u != bb_l else None

            feats = {
                'decline_pct': e['d'], 'rsi': g(rsi14), 'kdj_k': g(kdj['k']),
                'bb_pos': bb_pos,
                'ma20_dist': round((price - ma20[tr]) / ma20[tr] * 100, 1) if g(ma20) else None,
                'ma60_dist': round((price - ma60[tr]) / ma60[tr] * 100, 1) if g(ma60) else None,
                'gain_60d': g60,
            }
            prob = assess(feats)
            if g60 >= 0: continue      # only prior no-gain
            if prob < 80: continue     # only high prob

            rec = recovery(dates, close, tr, e['pp'])
            n = len(close)
            ti = tr
            rets = {}
            for h, lb in [(5, 'r5'), (10, 'r10'), (20, 'r20'), (60, 'r60')]:
                rets[lb] = round((close[ti + h] - close[ti]) / close[ti] * 100, 1) if ti + h < n else None

            results.append({
                'code': code, 'name': name,
                'td': e['td'], 'tp': e['tp'], 'pp': e['pp'],
                'd': e['d'], 'g60': g60,
                'rsi': feats['rsi'], 'kdj_k': feats['kdj_k'],
                'bb_pos': bb_pos, 'ma20': feats['ma20_dist'], 'ma60': feats['ma60_dist'],
                'prob': prob,
                'rec_d': rec['days'] if rec else None,
                **rets,
            })

        if len(results) >= 120:
            break

    # ============== Print Report ==============
    print()
    print("=" * 130)
    print("  \"前期没涨就跌\" 深度抽查 — 60日涨幅<0 且 回调概率>=80%")
    print("=" * 130)

    rec_all = [r for r in results if r['rec_d'] is not None]
    not_rec = [r for r in results if r['rec_d'] is None]

    hdr = (
        f"{'代码':<12} {'名称':<8} {'低点日期':<12} {'跌幅%':>6} {'前高%':>6} "
        f"{'RSI':>5} {'BB%':>5} {'MA20%':>6} {'MA60%':>6} "
        f"{'概率%':>6} {'恢复天':>7} {'5日':>6} {'10日':>6} {'20日':>6} {'60日':>6}"
    )
    print()
    print(hdr)
    print("-" * 130)

    for r in sorted(results, key=lambda r: r['rec_d'] or 9999):
        rec = f"{r['rec_d']}d" if r['rec_d'] else "未"
        bb = f"{r['bb_pos']:.0f}" if r['bb_pos'] is not None else '--'
        ma20 = f"{r['ma20']:+.1f}" if r['ma20'] is not None else '--'
        ma60 = f"{r['ma60']:+.1f}" if r['ma60'] is not None else '--'
        rsi_s = f"{r['rsi']:.0f}" if r['rsi'] is not None else '--'
        r5 = f"{r['r5']:+.1f}%" if r['r5'] is not None else '--'
        r10 = f"{r['r10']:+.1f}%" if r['r10'] is not None else '--'
        r20 = f"{r['r20']:+.1f}%" if r['r20'] is not None else '--'
        r60 = f"{r['r60']:+.1f}%" if r['r60'] is not None else '--'

        print(
            f"{r['code']:<12} {r['name']:<8} {r['td']:<12} {r['d']:>5.1f}% {r['g60']:>+5.1f}% "
            f"{rsi_s:>5} {bb:>5} {ma20:>6} {ma60:>6} "
            f"{r['prob']:>5.0f}% {rec:>7} {r5:>6} {r10:>6} {r20:>6} {r60:>6}"
        )

    print()
    print("=" * 130)
    print("  统计摘要")
    print("=" * 130)
    print(f"  总数: {len(results)}")
    print(f"  恢复: {len(rec_all)} ({len(rec_all)/max(len(results),1)*100:.1f}%)")
    print(f"  未恢复: {len(not_rec)} ({len(not_rec)/max(len(results),1)*100:.1f}%)")

    if rec_all:
        days = [r['rec_d'] for r in rec_all]
        print(f"\n  恢复天数: 均值 {np.mean(days):.1f}d  中位数 {np.median(days):.0f}d  最短 {min(days)}d  最长 {max(days)}d")
        print(f"  P25={np.percentile(days, 25):.0f}d  P50={np.percentile(days, 50):.0f}d  P75={np.percentile(days, 75):.0f}d  P90={np.percentile(days, 90):.0f}d")

    # 按跌幅分组
    print(f"\n  按跌幅分组:")
    for lo, hi, lb in [(3, 8, '3-8%'), (8, 12, '8-12%'), (12, 18, '12-18%'), (18, 50, '18%+')]:
        g = [r for r in results if lo <= r['d'] < hi]
        if not g: continue
        rg = [r for r in g if r['rec_d']]
        if rg:
            print(f"    {lb}: {len(g)}次, 恢复率{len(rg)/len(g)*100:.0f}%, 中位恢复{np.median([r['rec_d'] for r in rg]):.0f}d")
        else:
            print(f"    {lb}: {len(g)}次, 恢复率0%")

    # 按 RSI 分组
    print(f"\n  按 RSI 分组:")
    for lo, hi, lb in [(0, 30, '超卖<30'), (30, 40, '弱势30-40'), (40, 55, '中性40-55'), (55, 100, '偏强>55')]:
        g = [r for r in results if r['rsi'] is not None and lo <= r['rsi'] < hi]
        if not g: continue
        rg = [r for r in g if r['rec_d']]
        if rg:
            print(f"    {lb}: {len(g)}次, 恢复率{len(rg)/len(g)*100:.0f}%, 中位恢复{np.median([r['rec_d'] for r in rg]):.0f}d")

    # 反弹幅度
    print(f"\n  从低点起的反弹幅度:")
    for h, lb in [('r5', '5日'), ('r10', '10日'), ('r20', '20日'), ('r60', '60日')]:
        vals = [r[h] for r in results if r[h] is not None]
        if vals:
            pos = len([v for v in vals if v > 0])
            print(f"    {lb}: 均值 {np.mean(vals):+.1f}%  中位 {np.median(vals):+.1f}%  正收益 {pos}/{len(vals)} ({pos/len(vals)*100:.0f}%)")

    # 最慢恢复 TOP 10
    print(f"\n  --- 恢复最慢 TOP 10 ---")
    slow = sorted(rec_all, key=lambda r: -r['rec_d'])[:10]
    for r in slow:
        print(f"    {r['code']} {r['name']:<6} {r['td']} 跌{r['d']:.1f}% -> 恢复需{r['rec_d']}d, 60日反弹{r['r60']:+.1f}%")

    # 未恢复的
    print(f"\n  --- 未恢复的 ({len(not_rec)}个) ---")
    for r in sorted(not_rec, key=lambda r: r['td']):
        print(f"    {r['code']} {r['name']:<6} {r['td']} 跌{r['d']:.1f}% 前高{r['g60']:+.1f}% RSI{r['rsi']:.0f} 60日反弹{r['r60']:+.1f}%" if r['r60'] else f"    {r['code']} {r['name']:<6} {r['td']} 跌{r['d']:.1f}% 前高{r['g60']:+.1f}% RSI{r['rsi']:.0f} (数据不足60日)")

    print()
    print("=" * 130)


if __name__ == "__main__":
    main()
