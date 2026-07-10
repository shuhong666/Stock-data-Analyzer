"""
backtest_v5.py -- V5 策略回测 (V4 + 新过滤器)

V5 新增入场过滤器:
  F1: ADX < 20 (最强, 低趋势环境恢复好)
  F2: Skip decline > 18% (71%破位率, 不值得)
  F3: ConsecDown < 5 (急跌比阴跌好)

用法:
  python scripts/backtest_v5.py --samples 300 --seeds 42,123,456,789,999
"""
import argparse, sys, os, random
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def calc_profit_ratio(turn_rates, close_prices, current_price, window=120):
    n = len(close_prices); start = max(0, n - window)
    turns = np.array(turn_rates[start:n], dtype=float)
    closes = np.array(close_prices[start:n], dtype=float)
    if len(turns) < 10: return None
    if np.max(turns) > 10: turns = turns / 100.0
    m = len(turns)
    chip_left = np.zeros(m); cumulative_left = 1.0
    for i in range(m - 1, -1, -1):
        chip_left[i] = turns[i] * cumulative_left
        cumulative_left *= (1.0 - turns[i])
    base_chips = cumulative_left; total_chips = np.sum(chip_left) + base_chips
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
            and 30 <= price_pos <= 60 and profit_ratio < 30): return ("A", 90)
    if 8 <= decline_pct <= 25: return ("B", 70)
    return None


def get_v4_tp_sl(decline_pct):
    if decline_pct < 12: return 1.20, 0.35
    elif decline_pct < 18: return 1.10, 0.30
    else: return 0.90, 0.20


def simulate_exit(entry):
    close = entry["close"]; n = entry["n"]; today = entry["today"]
    peak = entry["peak_price"]; ep = entry["entry_price"]
    tp_ratio, sl_pct = get_v4_tp_sl(entry["decline_pct"])
    tp_price = ep + (peak - ep) * tp_ratio
    for j in range(today + 1, n):
        if close[j] >= tp_price: return (j, "TP", float(close[j]))
        if j - today >= 60: return (j, "Time", float(close[j]))
        if (peak - close[j]) / peak >= sl_pct: return (j, "Cut", float(close[j]))
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
    adx_dict = indicators.adx(high, low, close, 14)

    entries = []
    for today in range(180, n - 65):
        lookback = 40
        seg_high = high[today - lookback:today + 1]
        peak_offset = np.argmax(seg_high)
        peak_idx = today - lookback + peak_offset; peak_price = high[peak_idx]
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
        price_pos = round((close[today] - rng_lo) / (rng_hi - rng_lo) * 100, 1) if rng_hi > rng_lo else None
        rsi_arr = indicators.rsi(close, 14)
        rsi_val = float(rsi_arr[today]) if today < len(rsi_arr) and not np.isnan(rsi_arr[today]) else None
        ma20_arr = indicators.sma(close, 20)
        ma20_dist = round((close[today] - ma20_arr[today]) / ma20_arr[today] * 100, 1) if today < len(ma20_arr) and not np.isnan(ma20_arr[today]) else None
        profit_ratio = calc_profit_ratio(rturn[:today + 1], close[:today + 1], close[today])
        tier_result = classify_tier(decline_pct, price_pos, rsi_val, ma20_dist, profit_ratio)
        if tier_result is None: continue
        tier, score = tier_result
        adx_val = float(adx_dict["adx"][today]) if today < len(adx_dict["adx"]) and not np.isnan(adx_dict["adx"][today]) else None
        # V5 new: consec_down
        consec_down = 0
        for k in range(peak_idx + 1, trough_idx + 1):
            if close[k] < close[k - 1]: consec_down += 1
            else: break

        entries.append({
            "today": today, "entry_date": dates[today],
            "entry_price": float(close[today]),
            "peak_price": float(peak_price), "decline_pct": decline_pct,
            "price_pos": price_pos, "rsi": rsi_val, "ma20_dist": ma20_dist,
            "profit_ratio": profit_ratio, "tier": tier,
            "adx": adx_val, "consec_down": consec_down,
            "close": close, "high": high, "low": low, "n": n,
        })
    return entries


def v5_filter(entry, config):
    """V5 entry filters. Returns True if signal passes."""
    # F1: ADX filter
    if config.get("adx_max") is not None:
        if entry.get("adx") is None or entry["adx"] > config["adx_max"]:
            return False
    # F2: Skip deep decline
    if config.get("max_decline") is not None:
        if entry["decline_pct"] > config["max_decline"]:
            return False
    # F3: Consecutive down days max
    if config.get("max_consec_down") is not None:
        if entry.get("consec_down", 0) > config["max_consec_down"]:
            return False
    return True


def compute_stats(trades):
    if not trades: return None
    rets = [t["return_pct"] for t in trades]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    tp_t = [t for t in trades if t["exit_reason"] == "TP"]
    time_t = [t for t in trades if t["exit_reason"] == "Time"]
    cut_t = [t for t in trades if t["exit_reason"] == "Cut"]
    total = len(trades)
    sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
    avg_w = np.mean(wins) if wins else 0; avg_l = np.mean(losses) if losses else 0
    return {
        "total": total, "win_rate": len(wins)/total*100,
        "med_return": np.median(rets), "avg_return": np.mean(rets),
        "sharpe": sr, "avg_win": avg_w, "avg_loss": avg_l,
        "win_loss_ratio": abs(avg_w/avg_l) if avg_l else 0,
        "profit_factor": sum(wins)/abs(sum(losses)) if sum(losses) else 0,
        "avg_days": np.mean([t["hold_days"] for t in trades]),
        "med_days": np.median([t["hold_days"] for t in trades]),
        "tp_rate": len(tp_t)/total*100, "time_rate": len(time_t)/total*100,
        "cut_rate": len(cut_t)/total*100,
        "tp_avg_ret": np.mean([t["return_pct"] for t in tp_t]) if tp_t else 0,
        "time_avg_ret": np.mean([t["return_pct"] for t in time_t]) if time_t else 0,
        "cut_avg_ret": np.mean([t["return_pct"] for t in cut_t]) if cut_t else 0,
        "p25_ret": np.percentile(rets, 25), "p75_ret": np.percentile(rets, 75),
        "pct_gt_20": len([r for r in rets if r > 20])/total*100,
        "pct_gt_10": len([r for r in rets if r > 10])/total*100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,999")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    db = Database()

    # Configs to test
    configs = {
        "V4 (baseline)": {},
        "V5a: ADX<25": {"adx_max": 25},
        "V5b: ADX<20": {"adx_max": 20},
        "V5c: Skip>18%": {"max_decline": 18},
        "V5d: ConsecDown<5": {"max_consec_down": 5},
        "V5e: ADX<25+Skip>18%": {"adx_max": 25, "max_decline": 18},
        "V5f: ADX<25+Skip>18%+Down<5": {"adx_max": 25, "max_decline": 18, "max_consec_down": 5},
        "V5g: ADX<20+Skip>18%": {"adx_max": 20, "max_decline": 18},
    }

    all_results = {name: [] for name in configs}
    all_entries_all = []

    for seed in seeds:
        random.seed(seed); np.random.seed(seed)
        codes = random.sample(db.get_active_stock_codes(), min(args.samples, len(db.get_active_stock_codes())))
        print(f"\n[Seed {seed}] {len(codes)} stocks...")
        seed_entries = []
        for i, code in enumerate(codes):
            seed_entries.extend(find_entries(code, db))
            if (i+1) % 100 == 0: print(f"  {i+1}/{len(codes)}, {len(seed_entries)} raw signals")
        all_entries_all.extend(seed_entries)
        print(f"  Raw signals: {len(seed_entries)}")

        for name, cfg in configs.items():
            filtered = [e for e in seed_entries if v5_filter(e, cfg)]
            trades = []
            for e in filtered:
                r = simulate_exit(e)
                if r:
                    idx, reason, px = r
                    trades.append({
                        "return_pct": (px - e["entry_price"])/e["entry_price"]*100,
                        "hold_days": idx - e["today"], "exit_reason": reason,
                        "tier": e["tier"],
                    })
            all_results[name].extend(trades)
            if name == "V4 (baseline)":
                print(f"  V4 baseline: {len(trades)} trades")
            else:
                retained = len(filtered)/max(len(seed_entries),1)*100
                print(f"  {name}: {len(trades)} trades ({retained:.0f}% retained)")

    # ═══ Print Report ═══
    print(f"\n{'='*130}")
    print(f"  V5 STRATEGY BACKTEST ({len(seeds)} seeds x {args.samples} stocks)")
    print(f"{'='*130}\n")

    stats = {}
    for name in configs:
        stats[name] = compute_stats(all_results[name])

    v4 = stats["V4 (baseline)"]

    print(f"  {'Strategy':<32} {'N':>5} {'Win%':>7} {'MedRet':>8} {'Sharpe':>7} {'盈亏比':>8} {'TP%':>6} {'Time%':>7} {'Cut%':>6} {'CutAvg':>8} {'AvgDay':>7}")
    print(f"  {'-'*115}")

    for name in configs:
        s = stats[name]
        if s is None: continue
        marker = " << current" if name == "V4 (baseline)" else ""
        print(f"  {name:<32} {s['total']:>5} {s['win_rate']:>6.1f}% {s['med_return']:>+7.1f}% {s['sharpe']:>+6.2f} "
              f"{s['win_loss_ratio']:>7.2f} {s['tp_rate']:>5.0f}% {s['time_rate']:>6.0f}% {s['cut_rate']:>5.0f}% "
              f"{s['cut_avg_ret']:>+7.1f}% {s['med_days']:>6.0f}d{marker}")

    # Detail table
    print(f"\n  {'Strategy':<32} {'AvgWin':>8} {'AvgLoss':>8} {'PF':>7} {'P25':>7} {'P75':>7} {'TP_Avg':>8} {'Time_Avg':>8}")
    print(f"  {'-'*100}")
    for name in configs:
        s = stats[name]
        if s is None: continue
        print(f"  {name:<32} {s['avg_win']:>+7.1f}% {s['avg_loss']:>+7.1f}% {s['profit_factor']:>6.2f} "
              f"{s['p25_ret']:>+6.1f}% {s['p75_ret']:>+6.1f}% {s['tp_avg_ret']:>+7.1f}% {s['time_avg_ret']:>+7.1f}%")

    # Summary
    print(f"\n  {'='*115}")
    print(f"  V5 Recommendation")
    print(f"  {'='*115}")

    # Find best by Sharpe
    best = sorted([(n, s) for n, s in stats.items() if s and n != "V4 (baseline)"],
                  key=lambda x: x[1]["sharpe"], reverse=True)
    print(f"\n  Baseline V4: N={v4['total']}, Sharpe={v4['sharpe']:.2f}, Med={v4['med_return']:+.1f}%, Win={v4['win_rate']:.1f}%")
    for name, s in best:
        d_sh = s["sharpe"] - v4["sharpe"]
        d_med = s["med_return"] - v4["med_return"]
        d_time = s["time_rate"] - v4["time_rate"]
        retained = s["total"] / v4["total"] * 100
        print(f"  {name}: N={s['total']} ({retained:.0f}%), Sharpe={s['sharpe']:.2f} (d={d_sh:+.2f}), "
              f"Med={s['med_return']:+.1f}% (d={d_med:+.1f}), Time%={s['time_rate']:.0f}% (d={d_time:+.0f})")

    print()


if __name__ == "__main__":
    main()
