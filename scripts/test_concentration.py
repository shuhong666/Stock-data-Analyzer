"""
test_concentration.py — 测试筹码集中度对假摔反转策略的辅助效果
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
    """估算筹码获利比例"""
    n = len(close_prices)
    start = max(0, n - window)
    turns = np.array(turn_rates[start:n], dtype=float)
    closes = np.array(close_prices[start:n], dtype=float)
    if len(turns) < 10: return None
    if np.max(turns) > 10: turns = turns / 100.0
    m = len(turns)
    chip_left = np.zeros(m)
    cumulative_left = 1.0
    for i in range(m - 1, -1, -1):
        chip_left[i] = turns[i] * cumulative_left
        cumulative_left *= (1.0 - turns[i])
    base_chips = cumulative_left
    total_chips = np.sum(chip_left) + base_chips
    if total_chips < 0.001: return None
    profit_chips = np.sum(chip_left[closes < current_price])
    if base_chips > 0:
        base_cost = np.median(closes[:max(1, m // 5)])
        if base_cost < current_price: profit_chips += base_chips
    return round(profit_chips / total_chips * 100, 1)


def calc_concentration(turn_rates, close_prices, window=120):
    """计算多个筹码集中度指标。Returns dict or None."""
    n = len(close_prices)
    start = max(0, n - window)
    turns = np.array(turn_rates[start:n], dtype=float)
    closes = np.array(close_prices[start:n], dtype=float)
    if len(turns) < 10: return None
    if np.max(turns) > 10: turns = turns / 100.0

    m = len(turns)
    chip_left = np.zeros(m)
    cumulative_left = 1.0
    for i in range(m - 1, -1, -1):
        chip_left[i] = turns[i] * cumulative_left
        cumulative_left *= (1.0 - turns[i])
    base_chips = cumulative_left
    total_chips = np.sum(chip_left) + base_chips
    if total_chips < 0.001: return None

    # Build cost distribution
    costs = np.concatenate([closes, [np.median(closes[:max(1, m // 5)])]])
    weights = np.concatenate([chip_left, [base_chips]])
    total_w = np.sum(weights)
    if total_w < 0.001: return None
    weights = weights / total_w

    # Sort by cost
    order = np.argsort(costs)
    sorted_costs = costs[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)

    def cost_at_percentile(p):
        """Cost at the p-th percentile of the chip distribution"""
        idx = np.searchsorted(cum_weights, p / 100.0)
        return float(sorted_costs[min(idx, len(sorted_costs) - 1)])

    # 1. 90% chip concentration: (P95 - P5) / median cost
    p5 = cost_at_percentile(5)
    p95 = cost_at_percentile(95)
    p50 = cost_at_percentile(50)
    conc_90 = round((p95 - p5) / p50 * 100, 1) if p50 > 0 else None

    # 2. 70% chip concentration: (P85 - P15) / median
    p15 = cost_at_percentile(15)
    p85 = cost_at_percentile(85)
    conc_70 = round((p85 - p15) / p50 * 100, 1) if p50 > 0 else None

    # 3. Coefficient of variation of turnover (换手变异系数)
    cv_turn = round(float(np.std(turns) / max(np.mean(turns), 0.001)), 1)

    # 4. Max single-day turnover / avg turnover (单日异常换手)
    max_turn_ratio = round(float(np.max(turns) / max(np.mean(turns), 0.001)), 1)

    # 5. Turnover Gini-like: what % of chips from top 20% of turnover days
    top20_pct = round(float(np.sum(np.sort(chip_left)[-max(1, m // 5):]) / max(np.sum(chip_left), 0.001) * 100), 1)

    return dict(
        conc_90=conc_90, conc_70=conc_70,
        cv_turn=cv_turn, max_turn_ratio=max_turn_ratio,
        top20_pct=top20_pct, avg_turn=round(float(np.mean(turns)) * 100, 2),
    )


all_events = []
print(f"扫描 200 只股票...")

for ci, code in enumerate(codes):
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date", (code,))
    if len(rows) < 180: continue
    dates = [r["trade_date"] for r in rows]
    rc = np.array([r["close"] for r in rows], float)
    rh = np.array([r["high"] for r in rows], float)
    rl = np.array([r["low"] for r in rows], float)
    ro = np.array([r["open"] for r in rows], float)
    rturn = np.array([r["turn"] or 0 for r in rows], float)
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date", (code,))
    adj = indicators.forward_adjust(rc, rh, rl, ro, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]
    events = _find_decline_events(close, high, low, dates, min_decline=3, max_decline=50)
    nm = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,)); name = nm["name"] if nm else ""

    for e in events:
        pk = e["peak_idx"]; tr = e["trough_idx"]
        g60 = round((close[pk] - close[max(0, pk - 60)]) / close[max(0, pk - 60)] * 100, 1)
        if g60 >= 0: continue
        features = _compute_trough_features(close, high, low, pk, tr)
        decl = e["decline_pct"]; features["decline_pct"] = decl
        rng_hi = float(np.max(high[max(0, tr - 60):tr + 1]))
        rng_lo = float(np.min(low[max(0, tr - 60):tr + 1]))
        rng_s = rng_hi - rng_lo
        price_pos = round((close[tr] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None
        if price_pos is None or price_pos < 15: continue

        prob = assess_pullback(features)
        if prob < 50: continue

        profit_ratio = calc_profit_ratio(rturn[:tr + 1], close[:tr + 1], close[tr])
        conc = calc_concentration(rturn[:tr + 1], close[:tr + 1])
        if conc is None: continue

        recovery = _compute_recovery(dates, close, tr, e["peak_price"])
        n = len(close); ti = tr
        f60 = round((close[ti + 60] - close[ti]) / close[ti] * 100, 1) if ti + 60 < n else None

        all_events.append(dict(
            code=code, name=name, trough_date=e["trough_date"],
            decline_pct=decl, gain_60d=g60, price_pos=price_pos,
            rsi=features["rsi"], profit_ratio=profit_ratio,
            recovery_days=recovery["recovery_days"] if recovery else None, r60d=f60,
            **conc,
        ))

    if (ci + 1) % 50 == 0: print(f"  进度: {ci + 1}/200, 已找到 {len(all_events)} 个")

print(f"\n找到 {len(all_events)} 个事件\n")


def analyze_group(label, group):
    if len(group) < 20: return
    rec = [r for r in group if r["recovery_days"] is not None]
    rate = len(rec) / len(group) * 100
    med = np.median([r["recovery_days"] for r in rec]) if rec else 0
    r60s = [r["r60d"] for r in group if r["r60d"] is not None]
    r60a = np.mean(r60s) if r60s else 0
    pos60 = len([v for v in r60s if v > 0]) / max(len(r60s), 1) * 100
    print(f"  {label:<22} {len(group):>5}次  恢复率{rate:.0f}%  中位{med:.0f}d  60日{r60a:+.1f}%  胜率{pos60:.0f}%")


# Test each concentration metric
metrics = [
    ("conc_90", "90%集中度 (P95-P5)/P50"),
    ("conc_70", "70%集中度 (P85-P15)/P50"),
    ("cv_turn", "换手变异系数"),
    ("max_turn_ratio", "单日异常换手比"),
    ("top20_pct", "前20%天数筹码占比"),
    ("avg_turn", "平均换手率"),
]

for key, desc in metrics:
    print(f"\n--- {desc} ({key}) ---")
    vals = [r[key] for r in all_events if r[key] is not None]
    if not vals: continue
    p25 = np.percentile(vals, 25)
    p50 = np.percentile(vals, 50)
    p75 = np.percentile(vals, 75)

    for lo, hi, label in [
        (float('-inf'), p25, f"低 (<P25={p25:.1f})"),
        (p25, p50, f"中低 (P25-P50)"),
        (p50, p75, f"中高 (P50-P75)"),
        (p75, float('inf'), f"高 (>P75={p75:.1f})"),
    ]:
        g = [r for r in all_events if r[key] is not None and lo <= r[key] < hi]
        analyze_group(label, g)

# Best combination hint
print(f"\n{'=' * 70}")
print("  最佳组合探索")
print(f"{'=' * 70}")

# conc_90 + profit_ratio cross
print("\n--- 集中度(conc_90) + 获利比例 交叉 ---")
conc_vals = [r["conc_90"] for r in all_events if r["conc_90"] is not None]
if conc_vals:
    p50c = np.median(conc_vals)
    for pr_label, pr_lo, pr_hi in [("获利<30%", 0, 30), ("获利30-50%", 30, 50), ("获利>50%", 50, 100)]:
        for conc_label, conc_lo, conc_hi in [
            ("集中度低(分散)", float('-inf'), p50c),
            ("集中度高(集中)", p50c, float('inf')),
        ]:
            g = [r for r in all_events
                 if r["conc_90"] is not None and conc_lo <= r["conc_90"] < conc_hi
                 and r["profit_ratio"] is not None and pr_lo <= r["profit_ratio"] < pr_hi]
            if len(g) < 15: continue
            rec = len([r for r in g if r["recovery_days"] is not None]) / len(g) * 100
            print(f"  {pr_label} + {conc_label}: {len(g)}次  恢复率{rec:.0f}%")
