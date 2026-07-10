"""
backtest_v4_final.py -- V3 vs V4 最终对比回测

V3 (当前): 止盈100%前高 + 止损30%峰顶跌幅 + 60天时间
V4 (优化): 动态止盈(跌幅三段式) + 动态止损(跌幅三段式) + 60天时间

用法:
  python scripts/backtest_v4_final.py --samples 300 --seeds 42,123,456
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


def classify_tier(decline_pct, price_pos, rsi, ma20_dist, profit_ratio):
    if price_pos is None or price_pos < 30: return None
    if rsi is None or rsi <= 40: return None
    if ma20_dist is not None and ma20_dist < -4: return None
    if profit_ratio is None or profit_ratio >= 50: return None
    if (12 <= decline_pct <= 18 and 40 <= rsi <= 55
            and 30 <= price_pos <= 60 and profit_ratio < 30):
        return ("A", 90)
    if 8 <= decline_pct <= 25: return ("B", 70)
    return None


def find_entries(code, db):
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),)
    if len(rows) < 180: return []
    dates = [r["trade_date"] for r in rows]
    raw_c = np.array([r["close"] for r in rows], dtype=float)
    raw_h = np.array([r["high"] for r in rows], dtype=float)
    raw_l = np.array([r["low"] for r in rows], dtype=float)
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),)
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
        if peak_idx >= today - 1: continue
        trough_idx = peak_idx + np.argmin(low[peak_idx:today + 1])
        trough_price = low[trough_idx]
        if trough_idx < today - 5: continue
        if close[today] > close[today - 1] and close[today] > close[today - 2]: continue
        if (close[today] - trough_price) / max(trough_price, 0.01) * 100 > 3: continue
        decline_pct = (peak_price - trough_price) / peak_price * 100
        if decline_pct < 8 or decline_pct > 50: continue
        gain_60d = round((close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)
        if gain_60d >= 0: continue
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
        if tier_result is None: continue
        tier, score = tier_result
        ma60_arr = indicators.sma(close, 60)
        ma60_dist = round((close[today] - ma60_arr[today]) / ma60_arr[today] * 100, 1) if today < len(ma60_arr) and not np.isnan(ma60_arr[today]) else None
        entries.append({
            "today": today, "entry_date": dates[today],
            "entry_price": float(close[today]),
            "peak_price": float(peak_price), "decline_pct": decline_pct,
            "gain_60d": gain_60d, "price_pos": price_pos,
            "rsi": rsi_val, "ma20_dist": ma20_dist, "ma60_dist": ma60_dist,
            "profit_ratio": profit_ratio, "tier": tier, "score": score,
            "close": close, "high": high, "low": low, "dates": dates, "n": n,
        })
    return entries


# ══════════════════════════════════════════════════════
# Exit simulation
# ══════════════════════════════════════════════════════

def get_tp_target(entry):
    """V4: Dynamic TP based on decline"""
    d = entry["decline_pct"]
    if d < 12: return 1.20
    elif d < 18: return 1.10
    else: return 0.90

def get_sl_pct(entry):
    """V4: Dynamic SL based on decline"""
    d = entry["decline_pct"]
    if d < 12: return 0.35
    elif d < 18: return 0.30
    else: return 0.20


def simulate_v3(entry):
    """V3: TP=100% recovery, SL=30% from peak, max 60d"""
    close = entry["close"]; n = entry["n"]; today = entry["today"]
    peak = entry["peak_price"]; ep = entry["entry_price"]
    tp_price = ep + (peak - ep) * 1.00
    for j in range(today + 1, n):
        if close[j] >= tp_price: return (j, "TP", float(close[j]))
        if j - today >= 60: return (j, "Time", float(close[j]))
        if (peak - close[j]) / peak >= 0.30: return (j, "Cut", float(close[j]))
    return None


def simulate_v4(entry):
    """V4: Dynamic TP + Dynamic SL based on decline"""
    close = entry["close"]; n = entry["n"]; today = entry["today"]
    peak = entry["peak_price"]; ep = entry["entry_price"]
    tp_ratio = get_tp_target(entry)
    sl_pct = get_sl_pct(entry)
    tp_price = ep + (peak - ep) * tp_ratio
    for j in range(today + 1, n):
        if close[j] >= tp_price: return (j, "TP", float(close[j]))
        if j - today >= 60: return (j, "Time", float(close[j]))
        if (peak - close[j]) / peak >= sl_pct: return (j, "Cut", float(close[j]))
    return None


# ══════════════════════════════════════════════════════
# Stats & Report
# ══════════════════════════════════════════════════════

def compute_stats(trades):
    if not trades: return None
    rets = [t["return_pct"] for t in trades]
    days = [t["hold_days"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    tp_t = [t for t in trades if t["exit_reason"] == "TP"]
    time_t = [t for t in trades if t["exit_reason"] == "Time"]
    cut_t = [t for t in trades if t["exit_reason"] == "Cut"]
    total = len(trades)
    sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Per-tier stats
    a_trades = [t for t in trades if t["tier"] == "A"]
    b_trades = [t for t in trades if t["tier"] == "B"]
    a_rets = [t["return_pct"] for t in a_trades]
    b_rets = [t["return_pct"] for t in b_trades]

    # Per-decline-range stats
    d_lo = [t for t in trades if t["decline_pct"] < 12]
    d_mid = [t for t in trades if 12 <= t["decline_pct"] < 18]
    d_hi = [t for t in trades if t["decline_pct"] >= 18]

    return {
        "total": total, "win_rate": len(wins)/total*100,
        "med_return": np.median(rets), "avg_return": np.mean(rets),
        "sharpe": sr, "avg_win": avg_win, "avg_loss": avg_loss,
        "win_loss_ratio": abs(avg_win/avg_loss) if avg_loss else 0,
        "profit_factor": sum(wins)/abs(sum(losses)) if sum(losses) else 0,
        "avg_days": np.mean(days), "med_days": np.median(days),
        "std_return": np.std(rets),
        "max_gain": max(rets), "max_loss": min(rets),
        "p25_ret": np.percentile(rets, 25), "p75_ret": np.percentile(rets, 75),
        "pct_gt_20": len([r for r in rets if r > 20])/total*100,
        "pct_gt_10": len([r for r in rets if r > 10])/total*100,
        "pct_neg_5": len([r for r in rets if r < -5])/total*100,
        "pct_neg_10": len([r for r in rets if r < -10])/total*100,
        # Exit reasons
        "tp_count": len(tp_t), "tp_rate": len(tp_t)/total*100,
        "time_count": len(time_t), "time_rate": len(time_t)/total*100,
        "cut_count": len(cut_t), "cut_rate": len(cut_t)/total*100,
        "tp_avg_ret": np.mean([t["return_pct"] for t in tp_t]) if tp_t else 0,
        "tp_med_ret": np.median([t["return_pct"] for t in tp_t]) if tp_t else 0,
        "time_avg_ret": np.mean([t["return_pct"] for t in time_t]) if time_t else 0,
        "cut_avg_ret": np.mean([t["return_pct"] for t in cut_t]) if cut_t else 0,
        # By tier
        "a_count": len(a_trades), "a_avg_ret": np.mean(a_rets) if a_rets else 0,
        "a_med_ret": np.median(a_rets) if a_rets else 0,
        "b_count": len(b_trades), "b_avg_ret": np.mean(b_rets) if b_rets else 0,
        "b_med_ret": np.median(b_rets) if b_rets else 0,
        # By decline range
        "d_lo_count": len(d_lo), "d_lo_avg": np.mean([t["return_pct"] for t in d_lo]) if d_lo else 0,
        "d_mid_count": len(d_mid), "d_mid_avg": np.mean([t["return_pct"] for t in d_mid]) if d_mid else 0,
        "d_hi_count": len(d_hi), "d_hi_avg": np.mean([t["return_pct"] for t in d_hi]) if d_hi else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,999")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    db = Database()

    all_v3 = []; all_v4 = []

    for seed in seeds:
        random.seed(seed); np.random.seed(seed)
        codes = random.sample(db.get_active_stock_codes(), min(args.samples, len(db.get_active_stock_codes())))
        print(f"\n[Seed {seed}] Scanning {len(codes)} stocks...")
        all_entries = []
        for i, code in enumerate(codes):
            all_entries.extend(find_entries(code, db))
            if (i+1) % 100 == 0: print(f"  {i+1}/{len(codes)}, {len(all_entries)} signals")
        a_c = len([e for e in all_entries if e["tier"]=="A"])
        print(f"  Total: {len(all_entries)} (A:{a_c}, B:{len(all_entries)-a_c})")

        v3_trades = []
        for e in all_entries:
            r = simulate_v3(e)
            if r:
                idx, reason, px = r
                v3_trades.append({
                    "return_pct": (px - e["entry_price"])/e["entry_price"]*100,
                    "hold_days": idx - e["today"], "exit_reason": reason,
                    "tier": e["tier"], "decline_pct": e["decline_pct"],
                })
        all_v3.append(v3_trades)
        print(f"  V3: {len(v3_trades)} trades, win={len([t for t in v3_trades if t['return_pct']>0])/max(len(v3_trades),1)*100:.1f}%")

        v4_trades = []
        for e in all_entries:
            r = simulate_v4(e)
            if r:
                idx, reason, px = r
                v4_trades.append({
                    "return_pct": (px - e["entry_price"])/e["entry_price"]*100,
                    "hold_days": idx - e["today"], "exit_reason": reason,
                    "tier": e["tier"], "decline_pct": e["decline_pct"],
                })
        all_v4.append(v4_trades)
        print(f"  V4: {len(v4_trades)} trades, win={len([t for t in v4_trades if t['return_pct']>0])/max(len(v4_trades),1)*100:.1f}%")

    # ═══ Aggregate & Print ═══
    print("\n" + "=" * 135)
    print("  V3 vs V4 FINAL BACKTEST")
    print("=" * 135)
    print(f"  Seeds: {len(seeds)} x {args.samples} stocks each")
    print()

    v3_stats = compute_stats([t for trades in all_v3 for t in trades])
    v4_stats = compute_stats([t for trades in all_v4 for t in trades])

    # ── Table 1: Core Metrics ──
    print(f"  {'─'*120}")
    print(f"  [Core Performance]")
    print(f"  {'Metric':<28} {'V3 (Current)':>20} {'V4 (Optimized)':>20} {'Change':>15}")
    print(f"  {'─'*90}")

    metrics_display = [
        ("Total Trades", "total", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("Median Return", "med_return", "{:+.1f}%"),
        ("Average Return", "avg_return", "{:+.1f}%"),
        ("Sharpe Ratio", "sharpe", "{:.2f}"),
        ("盈亏比 (Win/Loss)", "win_loss_ratio", "{:.2f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("StdDev of Returns", "std_return", "{:.1f}%"),
        ("Max Single Gain", "max_gain", "{:+.1f}%"),
        ("Max Single Loss", "max_loss", "{:+.1f}%"),
        ("Average Win", "avg_win", "{:+.1f}%"),
        ("Average Loss", "avg_loss", "{:+.1f}%"),
        ("Median Hold Days", "med_days", "{:.0f}d"),
        ("Average Hold Days", "avg_days", "{:.0f}d"),
    ]
    for label, key, fmt in metrics_display:
        v3_val = v3_stats[key]
        v4_val = v4_stats[key]
        if isinstance(v3_val, float):
            delta = v4_val - v3_val
            delta_str = f"{delta:+.2f}" if abs(delta) < 100 else f"{delta:+.1f}"
        else:
            delta_str = str(v4_val - v3_val)
        print(f"  {label:<28} {fmt.format(v3_val):>20} {fmt.format(v4_val):>20} {delta_str:>15}")

    # ── Table 2: Return Distribution ──
    print(f"\n  {'─'*120}")
    print(f"  [Return Distribution]")
    print(f"  {'Metric':<28} {'V3':>20} {'V4':>20}")
    print(f"  {'─'*75}")
    for label, key in [("P25 (lower quartile)", "p25_ret"), ("P50 (median)", "med_return"),
                        ("P75 (upper quartile)", "p75_ret"), ("% Returns > 20%", "pct_gt_20"),
                        ("% Returns > 10%", "pct_gt_10"), ("% Returns < -5%", "pct_neg_5"),
                        ("% Returns < -10%", "pct_neg_10")]:
        print(f"  {label:<28} {v3_stats[key]:>+19.1f}% {v4_stats[key]:>+19.1f}%")

    # ── Table 3: Exit Reasons ──
    print(f"\n  {'─'*120}")
    print(f"  [Exit Reason Breakdown]")
    print(f"  {'Reason':<15} {'V3_Count':>8} {'V3_Rate':>8} {'V3_Avg':>10} {'V4_Count':>8} {'V4_Rate':>8} {'V4_Avg':>10}")
    print(f"  {'─'*75}")
    for reason, c3_k, r3_k, a3_k, c4_k, r4_k, a4_k in [
        ("Take Profit", "tp_count", "tp_rate", "tp_avg_ret", "tp_count", "tp_rate", "tp_avg_ret"),
        ("Time Expired", "time_count", "time_rate", "time_avg_ret", "time_count", "time_rate", "time_avg_ret"),
        ("Stop Loss", "cut_count", "cut_rate", "cut_avg_ret", "cut_count", "cut_rate", "cut_avg_ret"),
    ]:
        print(f"  {reason:<15} {v3_stats[c3_k]:>8.0f} {v3_stats[r3_k]:>7.0f}% {v3_stats[a3_k]:>+9.1f}% "
              f"{v4_stats[c4_k]:>8.0f} {v4_stats[r4_k]:>7.0f}% {v4_stats[a4_k]:>+9.1f}%")

    # ── Table 4: By Tier ──
    print(f"\n  {'─'*120}")
    print(f"  [By Signal Tier]")
    print(f"  {'Tier':<8} {'V3_Count':>8} {'V3_Avg':>10} {'V3_Med':>10} {'V4_Count':>8} {'V4_Avg':>10} {'V4_Med':>10}")
    print(f"  {'─'*75}")
    for tier, c3, a3, m3, c4, a4, m4 in [
        ("A", "a_count", "a_avg_ret", "a_med_ret", "a_count", "a_avg_ret", "a_med_ret"),
        ("B", "b_count", "b_avg_ret", "b_med_ret", "b_count", "b_avg_ret", "b_med_ret"),
    ]:
        print(f"  {tier:<8} {v3_stats[c3]:>8.0f} {v3_stats[a3]:>+9.1f}% {v3_stats[m3]:>+9.1f}% "
              f"{v4_stats[c4]:>8.0f} {v4_stats[a4]:>+9.1f}% {v4_stats[m4]:>+9.1f}%")

    # ── Table 5: By Decline Range ──
    print(f"\n  {'─'*120}")
    print(f"  [By Decline Range (the driver of dynamic TP/SL)]")
    print(f"  {'Decline':<15} {'V3_Count':>8} {'V3_Avg':>10} {'V4_Count':>8} {'V4_Avg':>10} {'V4_TP/SL':>15}")
    print(f"  {'─'*80}")
    configs = [("d_lo", "Decline < 12%", "120%/35%"), ("d_mid", "Decline 12-18%", "110%/30%"), ("d_hi", "Decline > 18%", "90%/20%")]
    for key, label, config in configs:
        print(f"  {label:<15} {v3_stats[key+'_count']:>8.0f} {v3_stats[key+'_avg']:>+9.1f}% "
              f"{v4_stats[key+'_count']:>8.0f} {v4_stats[key+'_avg']:>+9.1f}% {config:>15}")

    # ── Summary ──
    print(f"\n  {'='*120}")
    print(f"  Summary")
    print(f"  {'='*120}")
    print(f"  V4 vs V3:")
    print(f"    Sharpe:        {v3_stats['sharpe']:.2f} -> {v4_stats['sharpe']:.2f}  (delta: {v4_stats['sharpe']-v3_stats['sharpe']:+.2f})")
    print(f"    Med Return:    {v3_stats['med_return']:+.1f}% -> {v4_stats['med_return']:+.1f}%  (delta: {v4_stats['med_return']-v3_stats['med_return']:+.1f}pp)")
    print(f"    盈亏比:         {v3_stats['win_loss_ratio']:.2f} -> {v4_stats['win_loss_ratio']:.2f}  (delta: {v4_stats['win_loss_ratio']-v3_stats['win_loss_ratio']:+.2f})")
    print(f"    Profit Factor: {v3_stats['profit_factor']:.2f} -> {v4_stats['profit_factor']:.2f}  (delta: {v4_stats['profit_factor']-v3_stats['profit_factor']:+.2f})")
    print(f"    Avg Win:       {v3_stats['avg_win']:+.1f}% -> {v4_stats['avg_win']:+.1f}%")
    print(f"    Avg Loss:      {v3_stats['avg_loss']:+.1f}% -> {v4_stats['avg_loss']:+.1f}%")
    print(f"    Cut Rate:      {v3_stats['cut_rate']:.0f}% -> {v4_stats['cut_rate']:.0f}%")
    print(f"    Cut Avg Loss:  {v3_stats['cut_avg_ret']:+.1f}% -> {v4_stats['cut_avg_ret']:+.1f}%")
    print()

if __name__ == "__main__":
    main()
