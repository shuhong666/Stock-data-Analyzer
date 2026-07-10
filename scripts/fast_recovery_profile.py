"""
fast_recovery_profile.py — 前期没涨+位置>30%, 对比快速恢复 vs 慢速恢复的特征
"""
import sys, numpy as np, random
sys.path.insert(0, '.')
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators
from plugins.pullback_scanner.backend.scanner import _find_decline_events, _compute_recovery, _compute_trough_features

db = Database()
random.seed(888); np.random.seed(888)
codes = random.sample(db.get_active_stock_codes(), 100)

fast = []   # <=10d
mid = []    # 20-60d
slow = []   # >120d
no_rec = [] # not recovered

for code in codes:
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date", (code,))
    if len(rows) < 120: continue
    dates = [r["trade_date"] for r in rows]
    rc = np.array([r["close"] for r in rows], float)
    rh = np.array([r["high"] for r in rows], float)
    rl = np.array([r["low"] for r in rows], float)
    ro = np.array([r["open"] for r in rows], float)
    rv = np.array([r["volume"] or 0 for r in rows], float)
    rturn = np.array([r["turn"] or 0 for r in rows], float)
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
        if g60 >= 0: continue

        rng_hi = float(np.max(high[max(0, tr - 60):tr + 1]))
        rng_lo = float(np.min(low[max(0, tr - 60):tr + 1]))
        rng_s = rng_hi - rng_lo
        price_pos = round((close[tr] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None
        if price_pos is None or price_pos <= 30: continue

        features = _compute_trough_features(close, high, low, pk, tr)
        recovery = _compute_recovery(dates, close, tr, e["peak_price"])

        # 缩量比
        pb_len = max(tr - pk, 1)
        pre_vol = float(np.mean(rv[max(0, pk - 10):pk])) if pk >= 10 else 0
        pb_vol = float(np.mean(rv[pk:tr + 1]))
        vol_ratio = round(pb_vol / pre_vol, 2) if pre_vol > 0 else None

        # 累计换手
        cum_turn = round(float(np.sum(rturn[pk:tr + 1])), 1)

        # 峰值到低点的天数
        peak_to_trough_days = tr - pk

        item = dict(
            code=code, name=name,
            decline_pct=e["decline_pct"], gain_60d=g60,
            price_pos=price_pos,
            rsi=features["rsi"], kdj_k=features["kdj_k"],
            ma20_dist=features["ma20_dist"], ma60_dist=features["ma60_dist"],
            bb_pos=features["bb_pos"], adx=features["adx"],
            vol_ratio=vol_ratio, cum_turn=cum_turn,
            peak_to_trough_days=peak_to_trough_days,
            recovery_days=recovery["recovery_days"] if recovery else None,
        )

        rd = item["recovery_days"]
        if rd is None:
            no_rec.append(item)
        elif rd <= 10:
            fast.append(item)
        elif 20 <= rd <= 60:
            mid.append(item)
        elif rd > 120:
            slow.append(item)

print(f"快速(<=10d): {len(fast)}  中等(20-60d): {len(mid)}  慢速(>120d): {len(slow)}  未恢复: {len(no_rec)}")
print()

# ===== Feature comparison =====
def stats(group, label):
    if not group: return
    def m(key, fmt=".1f"):
        vals = [r[key] for r in group if r[key] is not None]
        return f"{np.mean(vals):{fmt}}" if vals else "-"
    def med(key, fmt=".0f"):
        vals = [r[key] for r in group if r[key] is not None]
        return f"{np.median(vals):{fmt}}" if vals else "-"
    print(f"  {label:<16} | {len(group):>5} | {m('decline_pct'):>6}% | {m('price_pos','.0f'):>5}% | {m('rsi','.0f'):>4} | {m('ma20_dist','.1f'):>6}% | {m('ma60_dist','.1f'):>6}% | {m('vol_ratio','.2f'):>6} | {m('cum_turn','.1f'):>6}% | {m('peak_to_trough_days','.0f'):>5}d | {m('gain_60d','.1f'):>6}%")

print(f"  {'':16} | {'数量':>5} | {'跌幅':>6} | {'位置':>5} | {'RSI':>4} | {'MA20%':>6} | {'MA60%':>6} | {'缩量比':>6} | {'累换手':>6} | {'峰谷天':>5} | {'60涨':>6}")
print(f"  {'':->16}-+-{'':->5}-+-{'':->6}-+-{'':->5}-+-{'':->4}-+-{'':->6}-+-{'':->6}-+-{'':->6}-+-{'':->6}-+-{'':->5}-+-{'':->6}")
stats(fast, "快速 <=10d")
stats(mid, "中等 20-60d")
stats(slow, "慢速 >120d")
stats(no_rec, "未恢复")

# ===== Distribution comparisons =====
print(f"\n--- 跌幅分布 ---")
for label, group in [("快速", fast), ("中等", mid), ("慢速", slow), ("未恢复", no_rec)]:
    if not group: continue
    bins = [(3,8),(8,12),(12,18),(18,50)]
    dist = []
    for lo, hi in bins:
        cnt = len([r for r in group if lo <= r["decline_pct"] < hi])
        dist.append(f"{cnt}")
    print(f"  {label}: 3-8%={dist[0]}  8-12%={dist[1]}  12-18%={dist[2]}  >18%={dist[3]}")

print(f"\n--- 位置分布 ---")
for label, group in [("快速", fast), ("中等", mid), ("慢速", slow), ("未恢复", no_rec)]:
    if not group: continue
    bins = [(30,50),(50,75),(75,100)]
    dist = []
    for lo, hi in bins:
        cnt = len([r for r in group if r["price_pos"] is not None and lo <= r["price_pos"] < hi])
        dist.append(f"{cnt}")
    print(f"  {label}: 30-50%={dist[0]}  50-75%={dist[1]}  75-100%={dist[2]}")

print(f"\n--- RSI分布 ---")
for label, group in [("快速", fast), ("中等", mid), ("慢速", slow), ("未恢复", no_rec)]:
    if not group: continue
    bins = [(0,30),(30,40),(40,55),(55,100)]
    dist = []
    for lo, hi in bins:
        cnt = len([r for r in group if r["rsi"] is not None and lo <= r["rsi"] < hi])
        dist.append(f"{cnt}")
    print(f"  {label}: <30={dist[0]}  30-40={dist[1]}  40-55={dist[2]}  >55={dist[3]}")

# ===== Specific examples =====
print(f"\n--- 快速恢复典型样本 ---")
for r in sorted(fast, key=lambda x: x["recovery_days"])[:12]:
    rs = r["rsi"] or 0
    pp = r["price_pos"] or 0
    vr = r["vol_ratio"] or 0
    print(f"  {r['code']} {r['name']:<8} 跌{r['decline_pct']:.1f}% 位{pp:.0f}% RSI{rs:.0f} 缩量{vr:.1f} 峰谷{r['peak_to_trough_days']}d 恢复{r['recovery_days']}d")
