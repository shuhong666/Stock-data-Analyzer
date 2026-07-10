"""
test_pos30.py — 前期没涨 + 60日价格位置>30%, 输出恢复天数
"""
import sys, numpy as np, random
sys.path.insert(0, '.')
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators
from plugins.pullback_scanner.backend.scanner import _find_decline_events, _compute_recovery

db = Database()
random.seed(888); np.random.seed(888)
codes = random.sample(db.get_active_stock_codes(), 100)

all_events = []
print(f"扫描 100 只...")
for code in codes:
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close FROM daily_kline WHERE code=? ORDER BY trade_date", (code,))
    if len(rows) < 120: continue
    dates = [r["trade_date"] for r in rows]
    rc = np.array([r["close"] for r in rows], float)
    rh = np.array([r["high"] for r in rows], float)
    rl = np.array([r["low"] for r in rows], float)
    ro = np.array([r["open"] for r in rows], float)
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date", (code,))
    adj = indicators.forward_adjust(rc, rh, rl, ro, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]
    events = _find_decline_events(close, high, low, dates, min_decline=3, max_decline=50)
    nm = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
    name = nm["name"] if nm else ""

    for e in events:
        pk = e["peak_idx"]; tr = e["trough_idx"]
        g60 = round((close[pk] - close[max(0, pk - 60)]) / close[max(0, pk - 60)] * 100, 1)
        if g60 >= 0:
            continue

        rng_hi = float(np.max(high[max(0, tr - 60):tr + 1]))
        rng_lo = float(np.min(low[max(0, tr - 60):tr + 1]))
        rng_s = rng_hi - rng_lo
        price_pos = round((close[tr] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None

        if price_pos is None or price_pos <= 30:
            continue

        recovery = _compute_recovery(dates, close, tr, e["peak_price"])
        n = len(close); ti = tr
        f60 = round((close[ti + 60] - close[ti]) / close[ti] * 100, 1) if ti + 60 < n else None
        f20 = round((close[ti + 20] - close[ti]) / close[ti] * 100, 1) if ti + 20 < n else None

        all_events.append(dict(
            code=code, name=name,
            peak_date=e["peak_date"], trough_date=e["trough_date"],
            peak_price=e["peak_price"], trough_price=e["trough_price"],
            decline_pct=e["decline_pct"], gain_60d=g60, price_pos=price_pos,
            recovery_days=recovery["recovery_days"] if recovery else None,
            recovery_date=recovery["recovery_date"] if recovery else None,
            r20d=f20, r60d=f60,
        ))

total = len(all_events)
rec = [r for r in all_events if r["recovery_days"] is not None]
not_rec = [r for r in all_events if r["recovery_days"] is None]
rec_rate = len(rec) / max(total, 1) * 100

print(f"\n{'=' * 100}")
print(f"  前期没涨 + 位置 > 30% — 恢复天数分析")
print(f"{'=' * 100}")
print(f"  总事件: {total}")
print(f"  恢复率: {len(rec)}/{total} = {rec_rate:.1f}%")
print(f"  未恢复: {len(not_rec)}")

if rec:
    days = [r["recovery_days"] for r in rec]
    print(f"\n  恢复天数分布:")
    print(f"    均值: {np.mean(days):.0f}d  中位数: {np.median(days):.0f}d  标准差: {np.std(days):.0f}d")
    for p in [25, 50, 75, 90]:
        print(f"    P{p}: {np.percentile(days, p):.0f}d")
    print(f"    最短: {min(days)}d  最长: {max(days)}d")
    print(f"\n  恢复天数直方图:")
    for lo, hi, label in [(1, 5, "  1-5d"), (5, 10, " 5-10d"), (10, 20, "10-20d"),
                           (20, 40, "20-40d"), (40, 80, "40-80d"), (80, 999, " 80d+")]:
        cnt = len([d for d in days if lo <= d < hi])
        pct = cnt / len(days) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label}: {cnt:>5}次 ({pct:>4.0f}%) {bar}")

print(f"\n  按跌幅细分:")
for dl, dh, dlbl in [(3, 8, "  3-8%"), (8, 12, " 8-12%"), (12, 18, "12-18%"), (18, 50, " >18%")]:
    g = [r for r in all_events if dl <= r["decline_pct"] < dh]
    if not g: continue
    rg = [r for r in g if r["recovery_days"] is not None]
    rate = len(rg) / len(g) * 100
    med = np.median([r["recovery_days"] for r in rg]) if rg else 0
    avg = np.mean([r["recovery_days"] for r in rg]) if rg else 0
    print(f"    {dlbl}: {len(g)}次  恢复率{rate:.0f}%  中位{med:.0f}d  均值{avg:.0f}d")

print(f"\n  按位置细分:")
for lo, hi, label in [(30, 50, "  中位 30-50%"), (50, 75, "  高位 50-75%"), (75, 100, "  极高 75-100%")]:
    g = [r for r in all_events if lo <= r["price_pos"] < hi]
    if not g: continue
    rg = [r for r in g if r["recovery_days"] is not None]
    rate = len(rg) / len(g) * 100
    med = np.median([r["recovery_days"] for r in rg]) if rg else 0
    print(f"    {label}: {len(g)}次  恢复率{rate:.0f}%  中位{med:.0f}d")

# 前向收益
print(f"\n  前向收益:")
for h, lb in [("r20d", "20日"), ("r60d", "60日")]:
    vals = [r[h] for r in all_events if r[h] is not None]
    if vals:
        pos = len([v for v in vals if v > 0])
        print(f"    {lb}: 均值{np.mean(vals):+.1f}%  中位{np.median(vals):+.1f}%  胜率{pos}/{len(vals)}({pos/len(vals)*100:.0f}%)")

# 明细
print(f"\n{'=' * 100}")
print(f"  全部事件明细 (按恢复天数排序)")
print(f"{'=' * 100}")
hdr = f"{'代码':<12} {'名称':<8} {'前高日':<12} {'前高价':>7} {'低点日':<12} {'低点价':>7} {'跌幅':>6} {'位%':>5} {'恢复天':>7} {'恢复日':<12}"
print(hdr)
print("-" * 100)

for r in sorted(all_events, key=lambda x: x["recovery_days"] or 9999):
    rd = r["recovery_days"]
    rec_str = f"{rd}d" if rd is not None else "未恢复"
    rdate = r["recovery_date"] or "-"
    print(f"{r['code']:<12} {r['name']:<8} {r['peak_date']:<12} {r['peak_price']:>7.2f} "
          f"{r['trough_date']:<12} {r['trough_price']:>7.2f} {r['decline_pct']:>5.1f}% {r['price_pos']:>4.0f}% {rec_str:>7} {rdate:<12}")
