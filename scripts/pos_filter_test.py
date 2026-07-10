"""
pos_filter_test.py — 聚焦60日价格区间位置筛选效果
"""
import sys, numpy as np, random
sys.path.insert(0, '.')
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators
from plugins.pullback_scanner.backend.scanner import (
    _find_decline_events, _compute_trough_features, _compute_recovery, assess_pullback,
)

db = Database()
random.seed(777); np.random.seed(777)
codes = random.sample(db.get_active_stock_codes(), 300)

all_events = []
print("扫描中...")
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
        features = _compute_trough_features(close, high, low, pk, tr)
        features["decline_pct"] = e["decline_pct"]
        prob = assess_pullback(features)
        if prob < 80:
            continue

        # 60日价格区间位置
        rng_hi = float(np.max(high[max(0, tr - 60):tr + 1]))
        rng_lo = float(np.min(low[max(0, tr - 60):tr + 1]))
        rng_s = rng_hi - rng_lo
        price_pos = round((close[tr] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None

        recovery = _compute_recovery(dates, close, tr, e["peak_price"])
        n = len(close); ti = tr
        fwd_60 = round((close[ti + 60] - close[ti]) / close[ti] * 100, 1) if ti + 60 < n else None

        all_events.append(dict(
            code=code, name=name, trough_date=e["trough_date"],
            decline_pct=e["decline_pct"], gain_60d=g60, rsi=features["rsi"],
            probability=prob, price_pos=price_pos,
            recovery_days=recovery["recovery_days"] if recovery else None,
            r60d=fwd_60,
        ))

print(f"总事件: {len(all_events)}")

# ===== Analysis =====
bins = [
    (0, 5, "极低位 0-5%"),
    (5, 15, "低位 5-15%"),
    (15, 30, "中低位 15-30%"),
    (30, 50, "中位 30-50%"),
    (50, 100, "高位 >50%"),
]

total_rec = len([r for r in all_events if r["recovery_days"] is not None])
base_rate = total_rec / max(len(all_events), 1) * 100
print(f"\n基线 (无过滤): {len(all_events)}次  恢复率 {base_rate:.0f}%\n")

print("=" * 90)
print("  60日价格区间位置 — 筛选效果")
print("=" * 90)

for lo, hi, label in bins:
    g = [r for r in all_events if r["price_pos"] is not None and lo <= r["price_pos"] < hi]
    if not g: continue
    rec = [r for r in g if r["recovery_days"] is not None]
    rate = len(rec) / len(g) * 100
    med = np.median([r["recovery_days"] for r in rec]) if rec else 0
    r60s = [r["r60d"] for r in g if r["r60d"] is not None]
    r60a = np.mean(r60s) if r60s else 0
    pos60 = len([v for v in r60s if v > 0]) / max(len(r60s), 1) * 100
    bar = "#" * int(rate / 5)
    print(f"  {label:<15} {len(g):>5}次  恢复率 {rate:.0f}%  中位 {med:.0f}d  60日 {r60a:+.1f}%  胜率 {pos60:.0f}%  {bar}")

# 中高位 vs 极低位
hi_pos = [r for r in all_events if r["price_pos"] is not None and r["price_pos"] >= 15]
lo_pos = [r for r in all_events if r["price_pos"] is not None and r["price_pos"] < 5]
rec_hi = len([r for r in hi_pos if r["recovery_days"] is not None]) / max(len(hi_pos), 1) * 100
rec_lo = len([r for r in lo_pos if r["recovery_days"] is not None]) / max(len(lo_pos), 1) * 100

print(f"\n  -- 对比 --")
print(f"  极低位 (<5%):  {len(lo_pos)}次  恢复率 {rec_lo:.0f}%  <- 过滤掉")
print(f"  中高位 (>=15%): {len(hi_pos)}次  恢复率 {rec_hi:.0f}%  <- 保留")
print(f"  提升: +{rec_hi - base_rate:.0f}pp, 过滤掉 {len(lo_pos)} 个危险信号")

# 跌幅 + 位置交叉
print(f"\n--- 跌幅 + 位置交叉 ---")
for dl, dh, dlbl in [(3, 8, "跌 3-8%"), (8, 12, "跌 8-12%"), (12, 18, "跌 12-18%"), (18, 50, "跌 >18%")]:
    base_g = [r for r in all_events if dl <= r["decline_pct"] < dh]
    hi_g = [r for r in base_g if r["price_pos"] is not None and r["price_pos"] >= 15]
    if not base_g: continue
    r_base = len([r for r in base_g if r["recovery_days"] is not None]) / len(base_g) * 100
    r_hi = len([r for r in hi_g if r["recovery_days"] is not None]) / max(len(hi_g), 1) * 100
    print(f"  {dlbl}: 全部 {r_base:.0f}% ({len(base_g)}次) -> 中高位 {r_hi:.0f}% ({len(hi_g)}次)  提升 {r_hi - r_base:+.0f}pp")

# 样本
print(f"\n--- 中高位样本 (位置 >=15%) ---")
for r in sorted(hi_pos, key=lambda r: r["decline_pct"])[:15]:
    rd = r["recovery_days"]
    rec = f"{rd}d" if rd is not None else "未"
    print(f"  {r['code']} {r['name']:<8} {r['trough_date']} 跌 {r['decline_pct']:.1f}%  位 {r['price_pos']:.0f}%  RSI {r['rsi']:.0f}  恢复 {rec}")
