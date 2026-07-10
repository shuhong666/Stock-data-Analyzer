"""
test_profit_ratio.py — 测试筹码获利比例对假摔反转策略的辅助效果

算法: 基于换手率估算筹码分布
  - 过去N日中, 每日换手 = 当日交易的筹码比例
  - 后续交易会替换掉早期的筹码
  - chip_left[i] = turnover[i] * Π(1 - turnover[j]) for j in (i+1..today)
  - profit_ratio = Σ chip_left[i] where cost[i] < current_price, as % of total chips left
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
codes = random.sample(db.get_active_stock_codes(), 200)


def calc_profit_ratio(turn_rates, close_prices, current_price, window=120):
    """Estimate % of chips in profit at current_price.

    turn_rates: daily turnover ratio (0-100 or 0-1)
    close_prices: corresponding daily close
    current_price: price to evaluate against
    window: lookback days

    Returns (profit_ratio, avg_cost)
    """
    n = len(close_prices)
    start = max(0, n - window)
    turns = np.array(turn_rates[start:n], dtype=float)
    closes = np.array(close_prices[start:n], dtype=float)

    if len(turns) < 10:
        return None, None

    # Normalize turnover to [0, 1] if in percentage (0-100)
    if np.max(turns) > 10:
        turns = turns / 100.0

    # Chips remaining from each day
    # chip_left[i] = turn[i] * product of (1 - turn[j]) for j = i+1 to end
    # Add a "base" that represents chips held from before the window
    m = len(turns)
    chip_left = np.zeros(m)

    # Forward decay: chips from day i survive proportionally
    cumulative_left = 1.0
    # Start from most recent day backwards
    for i in range(m - 1, -1, -1):
        # Chips from day i: turnover[i] * cumulative survival from later days
        chip_left[i] = turns[i] * cumulative_left
        cumulative_left *= (1.0 - turns[i])

    # Remaining = chips from before the window (never traded in window)
    # We model these as having cost = avg price in first 20% of window
    base_chips = cumulative_left  # chips not traded in window
    total_chips = np.sum(chip_left) + base_chips

    if total_chips < 0.001:
        return None, None

    # Chips in profit: cost < current_price
    profit_chips = np.sum(chip_left[closes < current_price])
    # Base chips: assume cost = median price of window, approximate
    if base_chips > 0:
        base_cost = np.median(closes[:max(1, m // 5)])
        if base_cost < current_price:
            profit_chips += base_chips

    # Average cost (weighted by chips remaining from each day + base chips)
    avg_cost = None
    if total_chips > 0.001:
        weighted_sum = np.sum(chip_left * closes) + base_chips * np.median(closes[:max(1, m // 5)])
        avg_cost = round(float(weighted_sum / total_chips), 2)

    profit_ratio = round(profit_chips / total_chips * 100, 1)
    return profit_ratio, round(float(avg_cost), 2) if avg_cost else None


all_events = []
print(f"扫描 200 只股票, 计算筹码获利比例...")

for ci, code in enumerate(codes):
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),
    )
    if len(rows) < 180:
        continue

    dates = [r["trade_date"] for r in rows]
    rc = np.array([r["close"] for r in rows], float)
    rh = np.array([r["high"] for r in rows], float)
    rl = np.array([r["low"] for r in rows], float)
    ro = np.array([r["open"] for r in rows], float)
    rturn = np.array([r["turn"] or 0 for r in rows], float)

    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(rc, rh, rl, ro, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]

    events = _find_decline_events(close, high, low, dates, min_decline=3, max_decline=50)
    nm = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
    name = nm["name"] if nm else ""

    for e in events:
        pk = e["peak_idx"]
        tr = e["trough_idx"]
        g60 = round((close[pk] - close[max(0, pk - 60)]) / close[max(0, pk - 60)] * 100, 1)
        if g60 >= 0:
            continue

        features = _compute_trough_features(close, high, low, pk, tr)
        decl = e["decline_pct"]
        features["decline_pct"] = decl

        # 60日位置
        rng_hi = float(np.max(high[max(0, tr - 60):tr + 1]))
        rng_lo = float(np.min(low[max(0, tr - 60):tr + 1]))
        rng_s = rng_hi - rng_lo
        price_pos = round((close[tr] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None
        if price_pos is None or price_pos < 15:
            continue

        prob = assess_pullback(features)
        if prob < 50:
            continue

        # 计算筹码获利比例
        profit_ratio, avg_cost = calc_profit_ratio(rturn, close, close[tr], window=120)

        recovery = _compute_recovery(dates, close, tr, e["peak_price"])
        n = len(close)
        ti = tr
        f60 = round((close[ti + 60] - close[ti]) / close[ti] * 100, 1) if ti + 60 < n else None

        all_events.append(dict(
            code=code, name=name,
            trough_date=e["trough_date"],
            decline_pct=decl, gain_60d=g60, price_pos=price_pos,
            rsi=features["rsi"], ma20_dist=features["ma20_dist"],
            profit_ratio=profit_ratio, avg_cost=avg_cost,
            recovery_days=recovery["recovery_days"] if recovery else None,
            r60d=f60,
        ))

    if (ci + 1) % 50 == 0:
        print(f"  进度: {ci + 1}/200, 已找到 {len(all_events)} 个")

print(f"\n找到 {len(all_events)} 个事件\n")

# ===== Analysis =====
def analyze(label, group):
    if not group: return
    rec = [r for r in group if r["recovery_days"] is not None]
    rate = len(rec) / len(group) * 100
    med = np.median([r["recovery_days"] for r in rec]) if rec else 0
    r60s = [r["r60d"] for r in group if r["r60d"] is not None]
    r60a = np.mean(r60s) if r60s else 0
    pos60 = len([v for v in r60s if v > 0]) / max(len(r60s), 1) * 100
    print(f"  {label:<18} {len(group):>5}次  恢复率{rate:.0f}%  中位{med:.0f}d  60日{r60a:+.1f}%  胜率{pos60:.0f}%")

# Baseline
rec_all = len([r for r in all_events if r["recovery_days"] is not None])
print(f"基线 (位置>=15%, 前期没涨): {len(all_events)}次  恢复率{rec_all/max(len(all_events),1)*100:.0f}%\n")

# Profit ratio buckets
print("--- 筹码获利比例分组 ---")
for lo, hi, label in [(0, 10, "极低获利 <10%"), (10, 30, "低获利 10-30%"),
                       (30, 50, "中获利 30-50%"), (50, 70, "较高 50-70%"),
                       (70, 100, "高获利 >70%")]:
    g = [r for r in all_events if r["profit_ratio"] is not None and lo <= r["profit_ratio"] < hi]
    analyze(label, g)

# Cross: profit ratio + position
print("\n--- 筹码获利 + 位置交叉 ---")
for pr_lo, pr_hi, pr_label in [(0, 30, "获利盘<30%"), (30, 70, "获利盘30-70%"), (70, 100, "获利盘>70%")]:
    for pp_lo, pp_hi, pp_label in [(15, 30, "位15-30%"), (30, 50, "位30-50%"), (50, 100, "位>50%")]:
        g = [r for r in all_events
             if r["profit_ratio"] is not None and pr_lo <= r["profit_ratio"] < pr_hi
             and r["price_pos"] is not None and pp_lo <= r["price_pos"] < pp_hi]
        if len(g) < 10: continue
        rec = len([r for r in g if r["recovery_days"] is not None]) / len(g) * 100
        print(f"  {pr_label} + {pp_label}: {len(g)}次  恢复率{rec:.0f}%")

# Profit ratio vs recovery correlation
valid = [r for r in all_events if r["profit_ratio"] is not None and r["recovery_days"] is not None]
if valid:
    prs = [r["profit_ratio"] for r in valid]
    rds = [r["recovery_days"] for r in valid]
    corr = np.corrcoef(prs, rds)[0, 1]
    print(f"\n  获利比例 vs 恢复天数 相关性: {corr:.2f}")
    print(f"  (负值 = 获利比例越高恢复越快)")

# Best combo hint
print("\n--- 最佳组合提示 ---")
combo = [r for r in all_events
         if r["profit_ratio"] is not None and r["profit_ratio"] > 30
         and r["price_pos"] is not None and r["price_pos"] > 30]
if combo:
    rec = len([r for r in combo if r["recovery_days"] is not None]) / len(combo) * 100
    print(f"  获利盘>30% + 位置>30%: {len(combo)}次  恢复率{rec:.0f}%")
