"""
backtest_tp_comparison.py -- 止盈策略对比回测

在同一批入场信号上，对比 4 种止盈方案:
  A. Baseline: 100% 恢复 = 回到前高 (当前策略)
  B. Simple 110%: 统一 110% 恢复
  C. Simple 120%: 统一 120% 恢复
  D. Dynamic: 跌幅<12%→120%, 12-18%→110%, >18%→90%

用法:
  python scripts/backtest_tp_comparison.py --samples 500 --seed 42
  python scripts/backtest_tp_comparison.py --samples 500 --seeds 42,123,456
"""

import argparse, sys, os, random
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def calc_profit_ratio(turn_rates, close_prices, current_price, window=120):
    n = len(close_prices)
    start = max(0, n - window)
    turns = np.array(turn_rates[start:n], dtype=float)
    closes = np.array(close_prices[start:n], dtype=float)
    if len(turns) < 10:
        return None
    if np.max(turns) > 10:
        turns = turns / 100.0
    m = len(turns)
    chip_left = np.zeros(m)
    cumulative_left = 1.0
    for i in range(m - 1, -1, -1):
        chip_left[i] = turns[i] * cumulative_left
        cumulative_left *= (1.0 - turns[i])
    base_chips = cumulative_left
    total_chips = np.sum(chip_left) + base_chips
    if total_chips < 0.001:
        return None
    profit_chips = np.sum(chip_left[closes < current_price])
    if base_chips > 0:
        base_cost = np.median(closes[:max(1, m // 5)])
        if base_cost < current_price:
            profit_chips += base_chips
    return round(profit_chips / total_chips * 100, 1)


def classify_tier(decline_pct, price_pos, rsi, ma20_dist, profit_ratio):
    if price_pos is None or price_pos < 30:
        return None
    if rsi is None or rsi <= 40:
        return None
    if ma20_dist is not None and ma20_dist < -4:
        return None
    if profit_ratio is None or profit_ratio >= 50:
        return None
    if (12 <= decline_pct <= 18 and 40 <= rsi <= 55
            and 30 <= price_pos <= 60 and profit_ratio < 30):
        return ("A", 90)
    if 8 <= decline_pct <= 25:
        return ("B", 70)
    return None


def find_entries(code, db):
    """Find all valid entry signals. Same logic as backtest_strategy.py."""
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),
    )
    if len(rows) < 180:
        return []

    dates = [r["trade_date"] for r in rows]
    raw_c = np.array([r["close"] for r in rows], dtype=float)
    raw_h = np.array([r["high"] for r in rows], dtype=float)
    raw_l = np.array([r["low"] for r in rows], dtype=float)

    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_c, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]
    n = len(close)
    rturn = np.array([r["turn"] or 0 for r in rows], dtype=float)

    entries = []
    for today in range(180, n - 65):
        lookback = 40
        seg_high = high[today - lookback:today + 1]
        peak_offset = np.argmax(seg_high)
        peak_idx = today - lookback + peak_offset
        peak_price = high[peak_idx]

        if peak_idx >= today - 1:
            continue

        trough_idx = peak_idx + np.argmin(low[peak_idx:today + 1])
        trough_price = low[trough_idx]

        if trough_idx < today - 5:
            continue
        if close[today] > close[today - 1] and close[today] > close[today - 2]:
            continue
        if (close[today] - trough_price) / max(trough_price, 0.01) * 100 > 3:
            continue

        decline_pct = (peak_price - trough_price) / peak_price * 100
        if decline_pct < 8 or decline_pct > 50:
            continue

        gain_60d = round((close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)
        if gain_60d >= 0:
            continue

        rng_hi = float(np.max(high[max(0, today - 60):today + 1]))
        rng_lo = float(np.min(low[max(0, today - 60):today + 1]))
        rng_s = rng_hi - rng_lo
        price_pos = round((close[today] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None

        rsi_arr = indicators.rsi(close, 14)
        rsi_val = float(rsi_arr[today]) if today < len(rsi_arr) and not np.isnan(rsi_arr[today]) else None
        ma20_arr = indicators.sma(close, 20)
        ma20_dist = round((close[today] - ma20_arr[today]) / ma20_arr[today] * 100, 1) if today < len(ma20_arr) and not np.isnan(ma20_arr[today]) else None

        profit_ratio = calc_profit_ratio(rturn[:today + 1], close[:today + 1], close[today])

        tier_result = classify_tier(decline_pct, price_pos, rsi_val, ma20_dist, profit_ratio)
        if tier_result is None:
            continue

        tier, score = tier_result
        ma60_arr = indicators.sma(close, 60)
        ma60_dist = round((close[today] - ma60_arr[today]) / ma60_arr[today] * 100, 1) if today < len(ma60_arr) and not np.isnan(ma60_arr[today]) else None

        entries.append({
            "today": today,
            "entry_date": dates[today],
            "entry_price": float(close[today]),
            "peak_price": float(peak_price),
            "peak_idx": peak_idx,
            "trough_idx": trough_idx,
            "trough_price": float(trough_price),
            "decline_pct": decline_pct,
            "gain_60d": gain_60d,
            "price_pos": price_pos,
            "rsi": rsi_val,
            "ma20_dist": ma20_dist,
            "ma60_dist": ma60_dist,
            "profit_ratio": profit_ratio,
            "tier": tier,
            "score": score,
            "close": close,
            "high": high,
            "low": low,
            "dates": dates,
            "n": n,
        })
    return entries


# ══════════════════════════════════════════════════════════════════
# Exit strategies
# ══════════════════════════════════════════════════════════════════

def get_target_price(entry, recovery_ratio):
    """target = entry_price + (peak_price - entry_price) * recovery_ratio"""
    return entry["entry_price"] + (entry["peak_price"] - entry["entry_price"]) * recovery_ratio


def simulate_exit(entry, recovery_ratio):
    """Simulate exit with a fixed recovery ratio target.
    Returns (exit_idx, exit_reason, exit_price) or None.
    """
    close = entry["close"]
    n = entry["n"]
    today = entry["today"]
    peak_price = entry["peak_price"]
    target_price = get_target_price(entry, recovery_ratio)

    for j in range(today + 1, n):
        if close[j] >= target_price:
            return (j, "TP", float(close[j]))
        if j - today >= 60:
            return (j, "Time", float(close[j]))
        if (peak_price - close[j]) / peak_price >= 0.30:
            return (j, "Cut", float(close[j]))
    return None


def simulate_exit_dynamic(entry):
    """Dynamic exit: target depends on decline_pct.
    decline < 12%: target 120% recovery
    decline 12-18%: target 110% recovery
    decline > 18%: target 90% recovery
    """
    d = entry["decline_pct"]
    if d < 12:
        ratio = 1.20
    elif d < 18:
        ratio = 1.10
    else:
        ratio = 0.90
    return simulate_exit(entry, ratio)


# ══════════════════════════════════════════════════════════════════
# Strategy definitions
# ══════════════════════════════════════════════════════════════════

STRATEGIES = {
    "A_Baseline_100%": {
        "desc": "Current: always 100% recovery (return to peak)",
        "fn": lambda e: simulate_exit(e, 1.00),
    },
    "B_Simple_110%": {
        "desc": "Always 110% recovery",
        "fn": lambda e: simulate_exit(e, 1.10),
    },
    "C_Simple_120%": {
        "desc": "Always 120% recovery",
        "fn": lambda e: simulate_exit(e, 1.20),
    },
    "D_Dynamic": {
        "desc": "Dynamic: decline<12%->120%, 12-18%->110%, >18%->90%",
        "fn": simulate_exit_dynamic,
    },
}


# ══════════════════════════════════════════════════════════════════
# Backtest runner
# ══════════════════════════════════════════════════════════════════

def compute_stats(trades):
    """Compute summary stats from trade records."""
    if not trades:
        return None
    rets = [t["return_pct"] for t in trades]
    days = [t["hold_days"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    tp_trades = [t for t in trades if t["exit_reason"] == "TP"]
    time_trades = [t for t in trades if t["exit_reason"] == "Time"]
    cut_trades = [t for t in trades if t["exit_reason"] == "Cut"]
    total = len(trades)

    sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')

    return {
        "total": total,
        "win_rate": len(wins) / total * 100,
        "avg_return": np.mean(rets),
        "med_return": np.median(rets),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,        # 盈亏比: avg_win / |avg_loss|
        "profit_factor": profit_factor,          # 盈利因子: total_profit / |total_loss|
        "avg_days": np.mean(days),
        "med_days": np.median(days),
        "sharpe": sr,
        "max_gain": max(rets),
        "max_loss": min(rets),
        "std_return": np.std(rets),
        "p25_ret": np.percentile(rets, 25),
        "p75_ret": np.percentile(rets, 75),
        "p25_days": np.percentile(days, 25),
        "p75_days": np.percentile(days, 75),
        # Counts
        "tp_count": len(tp_trades),
        "tp_rate": len(tp_trades) / total * 100,
        "time_count": len(time_trades),
        "time_rate": len(time_trades) / total * 100,
        "cut_count": len(cut_trades),
        "cut_rate": len(cut_trades) / total * 100,
        # Per-exit-reason returns
        "tp_avg_ret": np.mean([t["return_pct"] for t in tp_trades]) if tp_trades else 0,
        "tp_med_ret": np.median([t["return_pct"] for t in tp_trades]) if tp_trades else 0,
        "time_avg_ret": np.mean([t["return_pct"] for t in time_trades]) if time_trades else 0,
        "time_med_ret": np.median([t["return_pct"] for t in time_trades]) if time_trades else 0,
        "cut_avg_ret": np.mean([t["return_pct"] for t in cut_trades]) if cut_trades else 0,
        "cut_med_ret": np.median([t["return_pct"] for t in cut_trades]) if cut_trades else 0,
        # Return distribution
        "pct_gt_20": len([r for r in rets if r > 20]) / total * 100,
        "pct_gt_10": len([r for r in rets if r > 10]) / total * 100,
        "pct_neg_5": len([r for r in rets if r < -5]) / total * 100,
        "pct_neg_10": len([r for r in rets if r < -10]) / total * 100,
    }


def run_one_seed(all_entries, tier_filter=None):
    """Run all strategies on the same entry signals."""
    if tier_filter:
        entries = [e for e in all_entries if e["tier"] == tier_filter]
    else:
        entries = all_entries

    a_count = len([e for e in entries if e["tier"] == "A"])
    b_count = len([e for e in entries if e["tier"] == "B"])

    results = {"n_signals": len(entries), "a_count": a_count, "b_count": b_count}

    for name, strat in STRATEGIES.items():
        trades = []
        for entry in entries:
            result = strat["fn"](entry)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "code": "",
                "tier": entry["tier"],
                "entry_date": entry["entry_date"],
                "entry_price": entry["entry_price"],
                "decline_pct": entry["decline_pct"],
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
            })

        results[name] = compute_stats(trades)

    return results


# ══════════════════════════════════════════════════════════════════
# Report printing
# ══════════════════════════════════════════════════════════════════

def print_comparison(all_results):
    """Print a clean comparison table across seeds and strategies."""
    strat_names = list(STRATEGIES.keys())
    strat_descs = {k: v["desc"] for k, v in STRATEGIES.items()}

    # Aggregate across seeds
    metrics = [
        "total", "win_rate", "med_return", "avg_return", "sharpe",
        "avg_win", "avg_loss", "win_loss_ratio", "profit_factor",
        "max_gain", "max_loss", "std_return",
        "avg_days", "med_days", "p25_ret", "p75_ret",
        "tp_rate", "time_rate", "cut_rate",
        "tp_avg_ret", "time_avg_ret", "cut_avg_ret",
        "pct_gt_20", "pct_gt_10", "pct_neg_5", "pct_neg_10",
    ]

    agg = {s: defaultdict(list) for s in strat_names}
    n_signals_list = []

    for seed_results in all_results:
        n_signals_list.append(seed_results.get("n_signals", 0))
        for s in strat_names:
            if s in seed_results and seed_results[s] is not None:
                for m in metrics:
                    val = seed_results[s].get(m, 0)
                    agg[s][m].append(val)

    avg_signals = np.mean(n_signals_list)
    total_trades_base = np.mean(agg["A_Baseline_100%"]["total"])

    # ═══ Header ═══
    print()
    print("=" * 130)
    print("  Take-Profit Strategy Comparison Backtest")
    print("=" * 130)
    print(f"  Seeds: {len(all_results)}  |  Avg signals: {avg_signals:.0f}  |  Avg trades: {total_trades_base:.0f}")
    print()

    # ═══ Table 1: Core Performance ═══
    print(f"  --- Core Performance ---")
    print(f"  {'Strategy':<42} {'Win%':>6} {'MedRet':>8} {'AvgRet':>8} {'Sharpe':>7} {'MedDay':>7} {'TP%':>6} {'Time%':>6} {'Cut%':>6}")
    print(f"  {'-'*110}")

    baseline = {m: np.mean(agg["A_Baseline_100%"][m]) for m in metrics}

    for s in strat_names:
        desc = strat_descs[s]
        r = {m: np.mean(agg[s][m]) for m in metrics}
        sr_d = r["sharpe"] - baseline["sharpe"]
        marker = ""
        if s != "A_Baseline_100%" and sr_d > 0.01:
            marker = " <<"

        print(f"  {desc:<42} {r['win_rate']:>5.1f}% {r['med_return']:>+7.1f}% {r['avg_return']:>+7.1f}% "
              f"{r['sharpe']:>+6.2f} {r['med_days']:>6.0f}d {r['tp_rate']:>5.0f}% {r['time_rate']:>5.0f}% {r['cut_rate']:>5.0f}%{marker}")

    # ═══ Table 2: Risk & Reward ═══
    print(f"\n  --- Risk & Reward ---")
    print(f"  {'Strategy':<42} {'AvgWin':>8} {'AvgLoss':>8} {'盈亏比':>8} {'盈利因子':>9} {'MaxGain':>8} {'MaxLoss':>8} {'StdDev':>7}")
    print(f"  {'-'*110}")

    for s in strat_names:
        desc = strat_descs[s]
        r = {m: np.mean(agg[s][m]) for m in metrics}
        print(f"  {desc:<42} {r['avg_win']:>+7.1f}% {r['avg_loss']:>+7.1f}% {r['win_loss_ratio']:>7.2f} {r['profit_factor']:>8.2f} "
              f"{r['max_gain']:>+7.1f}% {r['max_loss']:>+7.1f}% {r['std_return']:>6.1f}%")

    # ═══ Table 3: Return Distribution ═══
    print(f"\n  --- Return Distribution ---")
    print(f"  {'Strategy':<42} {'P25':>7} {'P50':>7} {'P75':>7} {'>20%':>6} {'>10%':>6} {'<-5%':>6} {'<-10%':>7}")
    print(f"  {'-'*105}")

    for s in strat_names:
        desc = strat_descs[s]
        r = {m: np.mean(agg[s][m]) for m in metrics}
        print(f"  {desc:<42} {r['p25_ret']:>+6.1f}% {r['med_return']:>+6.1f}% {r['p75_ret']:>+6.1f}% "
              f"{r['pct_gt_20']:>5.0f}% {r['pct_gt_10']:>5.0f}% {r['pct_neg_5']:>5.0f}% {r['pct_neg_10']:>5.0f}%")

    # ═══ Table 4: Per-Exit-Reason Returns ═══
    print(f"\n  --- Returns by Exit Reason ---")
    print(f"  {'Strategy':<42} {'TP_Avg':>8} {'Time_Avg':>9} {'Cut_Avg':>9}")
    print(f"  {'-'*80}")

    for s in strat_names:
        desc = strat_descs[s]
        r = {m: np.mean(agg[s][m]) for m in metrics}
        print(f"  {desc:<42} {r['tp_avg_ret']:>+7.1f}% {r['time_avg_ret']:>+8.1f}% {r['cut_avg_ret']:>+8.1f}%")

    # ═══ Table 5: Per-seed detail ═══
    print(f"\n  --- Per-Seed Detail ---")
    for i, seed_results in enumerate(all_results):
        n_sig = seed_results.get("n_signals", 0)
        a_c = seed_results.get("a_count", 0)
        b_c = seed_results.get("b_count", 0)
        print(f"\n  Seed {i+1} (signals: {n_sig}, A:{a_c} B:{b_c})")
        print(f"  {'':<20} {'Win%':>6} {'Med':>7} {'Sharpe':>7} {'盈亏比':>7} {'AvgW/AvgL':>12} {'TP%':>5} {'Time%':>6} {'Cut%':>5}")
        for s in strat_names:
            if s in seed_results and seed_results[s] is not None:
                r = seed_results[s]
                print(f"  {s:<20} {r['win_rate']:>5.1f}% {r['med_return']:>+6.1f}% {r['sharpe']:>+6.2f} "
                      f"{r['win_loss_ratio']:>6.2f} {r['avg_win']:>+5.1f}%/{r['avg_loss']:>+5.1f}% "
                      f"{r['tp_rate']:>4.0f}% {r['time_rate']:>5.0f}% {r['cut_rate']:>4.0f}%")

    # ═══ Recommendation ═══
    print(f"\n  {'='*110}")
    print(f"  Summary")
    print(f"  {'='*110}")

    best_name = None
    best_sr = baseline["sharpe"]
    for s in strat_names:
        sr = np.mean(agg[s]["sharpe"])
        if sr > best_sr:
            best_sr = sr
            best_name = s

    br = {m: np.mean(agg[best_name][m]) for m in metrics}
    bl = baseline
    print(f"\n  Recommended: {strat_descs[best_name]}")
    print(f"  vs Baseline:")
    print(f"    Sharpe:        {bl['sharpe']:.2f} -> {br['sharpe']:.2f}  ({br['sharpe']-bl['sharpe']:+.2f})")
    print(f"    Med Return:    {bl['med_return']:+.1f}% -> {br['med_return']:+.1f}%  ({br['med_return']-bl['med_return']:+.1f}pp)")
    print(f"    Win Rate:      {bl['win_rate']:.1f}% -> {br['win_rate']:.1f}%  ({br['win_rate']-bl['win_rate']:+.1f}pp)")
    print(f"    盈亏比:         {bl['win_loss_ratio']:.2f} -> {br['win_loss_ratio']:.2f}  ({br['win_loss_ratio']-bl['win_loss_ratio']:+.2f})")
    print(f"    Profit Factor: {bl['profit_factor']:.2f} -> {br['profit_factor']:.2f}  ({br['profit_factor']-bl['profit_factor']:+.2f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare take-profit strategies")
    parser.add_argument("--samples", type=int, default=300, help="Stocks per seed")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds")
    parser.add_argument("--tier", type=str, default=None, choices=["A", "B"], help="Only test one tier")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    db = Database()

    all_results = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        all_codes = db.get_active_stock_codes()
        codes = random.sample(all_codes, min(args.samples, len(all_codes)))
        print(f"\n[Seed {seed}] Scanning {len(codes)} stocks...")

        all_entries = []
        for i, code in enumerate(codes):
            entries = find_entries(code, db)
            all_entries.extend(entries)
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(codes)}, {len(all_entries)} signals")

        print(f"  Total signals: {len(all_entries)} (A: {len([e for e in all_entries if e['tier']=='A'])}, B: {len([e for e in all_entries if e['tier']=='B'])})")

        seed_results = run_one_seed(all_entries, args.tier)
        all_results.append(seed_results)

    print_comparison(all_results)


if __name__ == "__main__":
    main()
