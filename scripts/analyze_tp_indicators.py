"""
analyze_tp_indicators.py -- 从技术指标角度分析止盈策略

分析维度:
  1. MFE 按入场指标分桶: 哪些指标能预测"会涨多远"
  2. 恢复点指标分析: 回到前高时，哪些特征预示继续涨 vs 反转
  3. 动态止盈规则: 基于入场指标组合动态设定目标
  4. 多指标综合评分: 加权多个指标预测最佳止盈目标

用法:
  python scripts/analyze_tp_indicators.py --samples 300
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
    """Find all valid entry signals with full indicator data."""
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
    raw_v = np.array([r["volume"] for r in rows], dtype=float)

    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_c, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]
    volume = raw_v  # volume doesn't need adjustment
    n = len(close)

    rturn = np.array([r["turn"] or 0 for r in rows], dtype=float)

    # Pre-compute all indicators
    rsi_arr = indicators.rsi(close, 14)
    ma5_arr = indicators.sma(close, 5)
    ma10_arr = indicators.sma(close, 10)
    ma20_arr = indicators.sma(close, 20)
    ma60_arr = indicators.sma(close, 60)
    macd_dict = indicators.macd(close)
    kdj_dict = indicators.kdj(high, low, close)
    bb_dict = indicators.bollinger(close, 20, 2.0)
    atr_arr = indicators.atr(high, low, close, 14)
    adx_dict = indicators.adx(high, low, close, 14)
    vol_ratio_arr = indicators.vol_ratio(raw_v, 5)

    entries = []
    for today in range(180, n - 65):  # need 60+ days of future data
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

        rsi_val = float(rsi_arr[today]) if today < len(rsi_arr) and not np.isnan(rsi_arr[today]) else None
        ma20_val = float(ma20_arr[today]) if today < len(ma20_arr) and not np.isnan(ma20_arr[today]) else 0
        ma20_dist = round((close[today] - ma20_val) / ma20_val * 100, 1) if ma20_val > 0 else None
        ma60_val = float(ma60_arr[today]) if today < len(ma60_arr) and not np.isnan(ma60_arr[today]) else 0
        ma60_dist = round((close[today] - ma60_val) / ma60_val * 100, 1) if ma60_val > 0 else None

        profit_ratio = calc_profit_ratio(rturn[:today + 1], close[:today + 1], close[today])

        # Additional indicators at entry
        kdj_k = float(kdj_dict["k"][today]) if today < len(kdj_dict["k"]) and not np.isnan(kdj_dict["k"][today]) else None
        kdj_d = float(kdj_dict["d"][today]) if today < len(kdj_dict["d"]) and not np.isnan(kdj_dict["d"][today]) else None
        kdj_j = float(kdj_dict["j"][today]) if today < len(kdj_dict["j"]) and not np.isnan(kdj_dict["j"][today]) else None
        macd_dif = float(macd_dict["dif"][today]) if today < len(macd_dict["dif"]) and not np.isnan(macd_dict["dif"][today]) else None
        macd_hist = float(macd_dict["histogram"][today]) if today < len(macd_dict["histogram"]) and not np.isnan(macd_dict["histogram"][today]) else None
        bb_pos = (close[today] - float(bb_dict["lower"][today])) / (float(bb_dict["upper"][today]) - float(bb_dict["lower"][today])) * 100 if today < len(bb_dict["lower"]) and not np.isnan(bb_dict["lower"][today]) else None
        atr_pct = float(atr_arr[today]) / close[today] * 100 if today < len(atr_arr) and not np.isnan(atr_arr[today]) and close[today] > 0 else None
        adx_val = float(adx_dict["adx"][today]) if today < len(adx_dict["adx"]) and not np.isnan(adx_dict["adx"][today]) else None
        vr_val = float(vol_ratio_arr[today]) if today < len(vol_ratio_arr) and not np.isnan(vol_ratio_arr[today]) else None

        # Volume in decline period (shrink ratio)
        decline_days = today - peak_idx
        if decline_days >= 3:
            pre_decline_vol = np.mean(raw_v[max(0, peak_idx - 10):peak_idx + 1])
            decline_vol = np.mean(raw_v[peak_idx:today + 1])
            vol_shrink = decline_vol / pre_decline_vol if pre_decline_vol > 0 else 1.0
        else:
            vol_shrink = 1.0

        tier_result = classify_tier(decline_pct, price_pos, rsi_val, ma20_dist, profit_ratio)
        if tier_result is None:
            continue

        tier, score = tier_result

        # Calculate MFE over the next 60 days
        entry_price = float(close[today])
        recovery_amount = peak_price - entry_price
        mfe_price = entry_price
        mfe_day = today
        end = min(n, today + 61)

        # Also track: did it reach 100%? When? What were indicators then?
        reached_100 = False
        day_of_100 = None
        indicators_at_100 = {}

        for j in range(today + 1, end):
            if close[j] > mfe_price:
                mfe_price = close[j]
                mfe_day = j
            # Check if reached 100% recovery
            if not reached_100 and close[j] >= peak_price:
                reached_100 = True
                day_of_100 = j
                # Capture indicators at recovery point
                indicators_at_100 = {
                    "rsi": float(rsi_arr[j]) if j < len(rsi_arr) and not np.isnan(rsi_arr[j]) else None,
                    "vol_ratio": float(vol_ratio_arr[j]) if j < len(vol_ratio_arr) and not np.isnan(vol_ratio_arr[j]) else None,
                    "ma20_dist": round((close[j] - float(ma20_arr[j])) / float(ma20_arr[j]) * 100, 1) if j < len(ma20_arr) and not np.isnan(ma20_arr[j]) and ma20_arr[j] > 0 else None,
                    "macd_hist": float(macd_dict["histogram"][j]) if j < len(macd_dict["histogram"]) and not np.isnan(macd_dict["histogram"][j]) else None,
                    "bb_pos": (close[j] - float(bb_dict["lower"][j])) / (float(bb_dict["upper"][j]) - float(bb_dict["lower"][j])) * 100 if j < len(bb_dict["lower"]) and not np.isnan(bb_dict["lower"][j]) else None,
                    "adx": float(adx_dict["adx"][j]) if j < len(adx_dict["adx"]) and not np.isnan(adx_dict["adx"][j]) else None,
                    "close": float(close[j]),
                }
            # Stop checking if hit stop-loss
            if (peak_price - close[j]) / peak_price >= 0.30:
                break

        mfe_ratio = (mfe_price - entry_price) / recovery_amount if recovery_amount > 0 else 0
        mfe_pct = (mfe_price - entry_price) / entry_price * 100

        # Determine optimal target category
        if mfe_ratio >= 1.20:
            opt_category = ">=120%"
        elif mfe_ratio >= 1.10:
            opt_category = "110-120%"
        elif mfe_ratio >= 1.00:
            opt_category = "100-110%"
        elif mfe_ratio >= 0.80:
            opt_category = "80-100%"
        else:
            opt_category = "<80%"

        entries.append({
            "today": today,
            "entry_date": dates[today],
            "entry_price": entry_price,
            "peak_price": float(peak_price),
            "decline_pct": decline_pct,
            "price_pos": price_pos,
            "rsi": rsi_val,
            "ma20_dist": ma20_dist,
            "ma60_dist": ma60_dist,
            "profit_ratio": profit_ratio,
            "kdj_k": kdj_k,
            "kdj_j": kdj_j,
            "macd_dif": macd_dif,
            "macd_hist": macd_hist,
            "bb_pos": bb_pos,
            "atr_pct": atr_pct,
            "adx": adx_val,
            "vol_ratio": vr_val,
            "vol_shrink": vol_shrink,
            "tier": tier,
            "mfe_ratio": min(mfe_ratio, 3.0),
            "mfe_pct": mfe_pct,
            "opt_category": opt_category,
            "reached_100": reached_100,
            "day_of_100": day_of_100,
            "indicators_at_100": indicators_at_100,
            "close": close,
            "high": high,
            "n": n,
        })

    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1: MFE distribution by indicator buckets
# ══════════════════════════════════════════════════════════════════════════════

def bucket_analyze(entries, indicator_name, buckets, labels=None):
    """Analyze MFE distribution by indicator buckets."""
    if labels is None:
        labels = [f"{b[0]}-{b[1]}" for b in buckets]

    results = {}
    for i, (lo, hi) in enumerate(buckets):
        if hi is None:
            group = [e for e in entries if (e[indicator_name] or 0) >= lo]
        else:
            group = [e for e in entries if lo <= (e[indicator_name] or 0) < hi]

        if not group:
            continue

        mfes = [e["mfe_ratio"] for e in group]
        cats = defaultdict(int)
        for e in group:
            cats[e["opt_category"]] += 1

        total = len(group)
        results[labels[i]] = {
            "count": total,
            "mfe_median": np.median(mfes),
            "mfe_mean": np.mean(mfes),
            "pct_over_120": cats[">=120%"] / total * 100,
            "pct_over_110": (cats[">=120%"] + cats["110-120%"]) / total * 100,
            "pct_over_100": (cats[">=120%"] + cats["110-120%"] + cats["100-110%"]) / total * 100,
            "pct_under_80": cats["<80%"] / total * 100,
        }

    return results


def print_bucket_table(title, results, sort_key="mfe_median"):
    print(f"\n  {'='*90}")
    print(f"  {title}")
    print(f"  {'='*90}")
    print(f"  {'Bucket':<18} {'N':>6}  {'MFE_Med':>8} {'>120%':>7} {'>110%':>7} {'>100%':>7} {'<80%':>7}")
    print(f"  {'-'*70}")
    for label, r in sorted(results.items(), key=lambda x: x[1][sort_key], reverse=True):
        print(f"  {label:<18} {r['count']:>6}  {r['mfe_median']*100:>7.0f}% {r['pct_over_120']:>6.0f}% {r['pct_over_110']:>6.0f}% {r['pct_over_100']:>6.0f}% {r['pct_under_80']:>6.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Indicators at recovery point (100% reached) → predict continuation
# ══════════════════════════════════════════════════════════════════════════════

def analyze_recovery_point(entries):
    """Among trades that reached 100% recovery, what predicts >110% vs reversal?"""
    recovered = [e for e in entries if e["reached_100"] and e["indicators_at_100"].get("rsi") is not None]

    if not recovered:
        print("  No trades with recovery point data")
        return

    # Split into: continued up (>110% MFE) vs stalled (100-110% MFE) vs reversed (<100% MFE after touching 100%)
    continued = [e for e in recovered if e["mfe_ratio"] >= 1.20]  # went well beyond
    moderate = [e for e in recovered if 1.05 <= e["mfe_ratio"] < 1.20]  # moderate continuation
    stalled = [e for e in recovered if 0.95 <= e["mfe_ratio"] < 1.05]  # just touched peak
    reversed_after = [e for e in recovered if e["mfe_ratio"] < 0.90]  # reversed after touching

    def avg_indicator(group, key):
        vals = [e["indicators_at_100"].get(key) for e in group if e["indicators_at_100"].get(key) is not None]
        return np.mean(vals) if vals else None, np.median(vals) if vals else None

    print(f"\n  {'='*90}")
    print(f"  Analysis 2: Indicators at Recovery Point (100%) → Predict Continuation")
    print(f"  {'='*90}")
    print(f"  Total reaching 100%: {len(recovered)}")
    print(f"    Continued >120%:  {len(continued)} ({len(continued)/len(recovered)*100:.0f}%)")
    print(f"    Moderate 105-120%: {len(moderate)} ({len(moderate)/len(recovered)*100:.0f}%)")
    print(f"    Stalled 95-105%:  {len(stalled)} ({len(stalled)/len(recovered)*100:.0f}%)")
    print(f"    Reversed <90%:    {len(reversed_after)} ({len(reversed_after)/len(recovered)*100:.0f}%)")

    indicators_to_check = ["rsi", "vol_ratio", "ma20_dist", "macd_hist", "bb_pos", "adx"]
    names = {"rsi": "RSI(14)", "vol_ratio": "VolRatio", "ma20_dist": "MA20_dist%",
             "macd_hist": "MACD_Hist", "bb_pos": "BB_Pos%", "adx": "ADX"}

    print(f"\n  {'Indicator':<15} {'Continued>120%':>18} {'Stalled':>12} {'Reversed':>12} {'C/R_Diff':>10}")
    print(f"  {'-'*75}")
    for ind in indicators_to_check:
        c_mean, c_med = avg_indicator(continued, ind)
        s_mean, s_med = avg_indicator(stalled, ind)
        r_mean, r_med = avg_indicator(reversed_after, ind)

        if c_mean is not None and r_mean is not None:
            diff = c_mean - r_mean
            print(f"  {names[ind]:<15} {c_mean:>8.1f}(med={c_med:.0f})  {s_mean:>8.1f}  {r_mean:>8.1f}  {diff:>+9.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3: Dynamic target rules based on indicator combinations
# ══════════════════════════════════════════════════════════════════════════════

def simulate_exit_dynamic(entry, target_ratio):
    """Simulate exit when price recovers target_ratio of the decline."""
    close = entry["close"]
    n = entry["n"]
    today = entry["today"]
    peak_price = entry["peak_price"]
    entry_price = entry["entry_price"]
    target_price = entry_price + (peak_price - entry_price) * target_ratio

    for j in range(today + 1, n):
        if close[j] >= target_price:
            ret = (close[j] - entry_price) / entry_price * 100
            return (j, ret, "TP")
        if j - today >= 60:
            ret = (close[j] - entry_price) / entry_price * 100
            return (j, ret, "Time")
        if (peak_price - close[j]) / peak_price >= 0.30:
            ret = (close[j] - entry_price) / entry_price * 100
            return (j, ret, "Cut")
    return None


def test_dynamic_rules(entries):
    """Test various indicator-based dynamic target rules."""
    rules = {
        "Baseline(100%)": lambda e: 1.00,

        # RSI-based
        "RSI>55→120%_else→100%": lambda e: 1.20 if (e.get("rsi") or 0) > 55 else 1.00,
        "RSI>55→120%_40-55→110%_else→100%": lambda e: 1.20 if (e.get("rsi") or 0) > 55 else (1.10 if (e.get("rsi") or 0) > 40 else 1.00),

        # Decline-based
        "Decline<12→120%_else→100%": lambda e: 1.20 if (e.get("decline_pct") or 99) < 12 else 1.00,
        "Decline<12→120%_12-18→110%_else→90%": lambda e: 1.20 if (e.get("decline_pct") or 99) < 12 else (1.10 if (e.get("decline_pct") or 99) < 18 else 0.90),

        # Composite: RSI + Decline
        "RSI>55&D<12→120%_else→100%": lambda e: 1.20 if ((e.get("rsi") or 0) > 55 and (e.get("decline_pct") or 99) < 12) else 1.00,
        "RSI>50&D<12→120%_else_RSI>40→110%_else→90%": lambda e: (
            1.20 if ((e.get("rsi") or 0) > 50 and (e.get("decline_pct") or 99) < 12)
            else (1.10 if (e.get("rsi") or 0) > 40 else 0.90)),

        # Profit ratio based
        "Profit<30→120%_else→100%": lambda e: 1.20 if (e.get("profit_ratio") or 99) < 30 else 1.00,
        "Profit<20→120%_20-40→110%_else→100%": lambda e: 1.20 if (e.get("profit_ratio") or 99) < 20 else (1.10 if (e.get("profit_ratio") or 99) < 40 else 1.00),

        # MA20 based
        "MA20>0→120%_else→100%": lambda e: 1.20 if (e.get("ma20_dist") or -99) > 0 else 1.00,

        # Price position based
        "Pos>60→120%_30-60→110%_else→100%": lambda e: 1.20 if (e.get("price_pos") or 0) > 60 else (1.10 if (e.get("price_pos") or 0) > 30 else 1.00),

        # Best composite: high confidence multi-indicator
        "HQ:RSI>55&D<12&Profit<30&MA20>0→120%_else→110%": lambda e: (
            1.20 if ((e.get("rsi") or 0) > 55 and (e.get("decline_pct") or 99) < 12
                     and (e.get("profit_ratio") or 99) < 30 and (e.get("ma20_dist") or -99) > 0)
            else 1.10),

        # Aggressive composite
        "Aggro:RSI>50&D<12→130%_RSI>40→110%_else→90%": lambda e: (
            1.30 if ((e.get("rsi") or 0) > 50 and (e.get("decline_pct") or 99) < 12)
            else (1.10 if (e.get("rsi") or 0) > 40 else 0.90)),

        # Volume shrink based
        "VolShrink<0.8→120%_else→100%": lambda e: 1.20 if (e.get("vol_shrink") or 99) < 0.8 else 1.00,
    }

    results = {}
    for name, rule_fn in rules.items():
        trades = []
        for entry in entries:
            target = rule_fn(entry)
            result = simulate_exit_dynamic(entry, target)
            if result is None:
                continue
            exit_idx, ret, reason = result
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
                "target": target,
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            days = [t["hold_days"] for t in trades]
            wins = [r for r in rets if r > 0]
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
            tp_count = len([t for t in trades if t["exit_reason"] == "TP"])
            time_count = len([t for t in trades if t["exit_reason"] == "Time"])
            cut_count = len([t for t in trades if t["exit_reason"] == "Cut"])

            # Group by target used
            target_dist = defaultdict(int)
            for t in trades:
                target_dist[t["target"]] += 1

            results[name] = {
                "total": len(trades),
                "win_rate": len(wins) / len(trades) * 100,
                "avg_return": np.mean(rets),
                "med_return": np.median(rets),
                "avg_days": np.mean(days),
                "med_days": np.median(days),
                "sharpe": sr,
                "tp_count": tp_count,
                "time_count": time_count,
                "cut_count": cut_count,
                "target_dist": dict(target_dist),
            }

    return results


def print_dynamic_table(results):
    print(f"\n  {'='*100}")
    print(f"  Analysis 3: Indicator-Based Dynamic Take-Profit Rules")
    print(f"  {'='*100}")
    print(f"  {'Rule':<50} {'N':>5} {'Win%':>7} {'MedRet':>8} {'AvgDay':>7} {'TP':>5} {'Time':>5} {'Cut':>5} {'Sharpe':>7}")
    print(f"  {'-'*100}")

    # Show baseline first, then sorted by Sharpe
    baseline = results.get("Baseline(100%)")
    if baseline:
        print(f"  {'Baseline(100%)':<50} {baseline['total']:>5} {baseline['win_rate']:>6.1f}% {baseline['med_return']:>+7.1f}% {baseline['avg_days']:>6.0f}d {baseline['tp_count']:>5} {baseline['time_count']:>5} {baseline['cut_count']:>5} {baseline['sharpe']:>+6.2f}")
        print(f"  {'-'*100}")

    others = [(k, v) for k, v in results.items() if k != "Baseline(100%)"]
    for name, r in sorted(others, key=lambda x: x[1]["sharpe"], reverse=True):
        print(f"  {name:<50} {r['total']:>5} {r['win_rate']:>6.1f}% {r['med_return']:>+7.1f}% {r['avg_days']:>6.0f}d {r['tp_count']:>5} {r['time_count']:>5} {r['cut_count']:>5} {r['sharpe']:>+6.2f}")

    # Show target distributions for top 3
    top3 = sorted(others, key=lambda x: x[1]["sharpe"], reverse=True)[:3]
    print(f"\n  Target distribution for top 3 rules:")
    for name, r in top3:
        dist_str = ", ".join([f"{int(k*100)}%:{v}" for k, v in sorted(r["target_dist"].items())])
        print(f"  {name}: {dist_str}")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4: Correlation matrix
# ══════════════════════════════════════════════════════════════════════════════

def print_correlation(entries):
    """Print correlation between entry indicators and MFE ratio."""
    indicators_list = ["decline_pct", "price_pos", "rsi", "ma20_dist", "ma60_dist",
                       "profit_ratio", "kdj_k", "bb_pos", "atr_pct", "adx",
                       "vol_ratio", "vol_shrink", "macd_hist"]
    names = {"decline_pct": "Decline%", "price_pos": "Position%", "rsi": "RSI",
             "ma20_dist": "MA20%", "ma60_dist": "MA60%", "profit_ratio": "Profit%",
             "kdj_k": "KDJ_K", "bb_pos": "BB_Pos", "atr_pct": "ATR%",
             "adx": "ADX", "vol_ratio": "VolRatio", "vol_shrink": "VolShrink",
             "macd_hist": "MACD_Hist"}

    print(f"\n  {'='*90}")
    print(f"  Analysis 4: Correlation: Entry Indicators vs MFE Ratio")
    print(f"  {'='*90}")
    print(f"  (Positive = higher values predict higher ultimate recovery)")
    print(f"\n  {'Indicator':<15} {'Corr_w_MFE':>12} {'Direction':>12}")
    print(f"  {'-'*45}")

    corrs = []
    for ind in indicators_list:
        xs = [e.get(ind) for e in entries if e.get(ind) is not None]
        ys = [e["mfe_ratio"] for e in entries if e.get(ind) is not None]
        if len(xs) > 10:
            corr = np.corrcoef(xs, ys)[0, 1]
            corrs.append((ind, corr))

    for ind, corr in sorted(corrs, key=lambda x: -abs(x[1])):
        direction = "Higher→More" if corr > 0 else "Lower→More"
        print(f"  {names[ind]:<15} {corr:>+11.3f}  {direction:>12}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Indicator-based take-profit analysis")
    parser.add_argument("--samples", type=int, default=300, help="Number of stocks to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    db = Database()
    random.seed(args.seed)
    np.random.seed(args.seed)

    codes = random.sample(db.get_active_stock_codes(), min(args.samples, len(db.get_active_stock_codes())))
    print(f"Scanning {len(codes)} stocks for entry signals (seed={args.seed})...")

    all_entries = []
    for i, code in enumerate(codes):
        entries = find_entries(code, db)
        all_entries.extend(entries)
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(codes)}, {len(all_entries)} signals")

    print(f"\nTotal: {len(all_entries)} valid entry signals")
    a_count = len([e for e in all_entries if e["tier"] == "A"])
    b_count = len([e for e in all_entries if e["tier"] == "B"])
    print(f"  A-grade: {a_count}, B-grade: {b_count}")

    # --- Overall MFE stats ---
    mfes = [e["mfe_ratio"] for e in all_entries]
    cats = defaultdict(int)
    for e in all_entries:
        cats[e["opt_category"]] += 1
    total = len(all_entries)
    print(f"\n  MFE Distribution:")
    print(f"    >=120%: {cats['>=120%']} ({cats['>=120%']/total*100:.0f}%)")
    print(f"    110-120%: {cats['110-120%']} ({cats['110-120%']/total*100:.0f}%)")
    print(f"    100-110%: {cats['100-110%']} ({cats['100-110%']/total*100:.0f}%)")
    print(f"    80-100%: {cats['80-100%']} ({cats['80-100%']/total*100:.0f}%)")
    print(f"    <80%: {cats['<80%']} ({cats['<80%']/total*100:.0f}%)")
    print(f"  MFE Median: {np.median(mfes)*100:.0f}%, Mean: {np.mean(mfes)*100:.0f}%")

    # ===== Analysis 1: MFE by indicator buckets =====

    # RSI
    rsi_results = bucket_analyze(all_entries, "rsi",
        [(0, 40), (40, 45), (45, 50), (50, 55), (55, 65), (65, 100)],
        ["RSI<40", "RSI 40-45", "RSI 45-50", "RSI 50-55", "RSI 55-65", "RSI>65"])
    print_bucket_table("Analysis 1a: MFE by RSI(14) at Entry", rsi_results)

    # Decline
    decline_results = bucket_analyze(all_entries, "decline_pct",
        [(8, 10), (10, 12), (12, 15), (15, 18), (18, 25), (25, 50)],
        ["Decline 8-10%", "Decline 10-12%", "Decline 12-15%", "Decline 15-18%", "Decline 18-25%", "Decline >25%"])
    print_bucket_table("Analysis 1b: MFE by Decline% at Entry", decline_results)

    # Price Position
    pos_results = bucket_analyze(all_entries, "price_pos",
        [(30, 40), (40, 50), (50, 60), (60, 75), (75, 100)],
        ["Pos 30-40%", "Pos 40-50%", "Pos 50-60%", "Pos 60-75%", "Pos 75-100%"])
    print_bucket_table("Analysis 1c: MFE by 60d Price Position", pos_results)

    # Profit Ratio
    profit_results = bucket_analyze(all_entries, "profit_ratio",
        [(0, 15), (15, 30), (30, 40), (40, 50)],
        ["Profit<15%", "Profit 15-30%", "Profit 30-40%", "Profit 40-50%"])
    print_bucket_table("Analysis 1d: MFE by Profit Ratio", profit_results)

    # MA20 distance
    ma20_results = bucket_analyze(all_entries, "ma20_dist",
        [(-10, -4), (-4, 0), (0, 3), (3, 8), (8, 20)],
        ["MA20 <-4%", "MA20 -4~0%", "MA20 0~3%", "MA20 3~8%", "MA20 >8%"])
    print_bucket_table("Analysis 1e: MFE by MA20 Distance", ma20_results)

    # Volume shrink
    vol_results = bucket_analyze(all_entries, "vol_shrink",
        [(0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 10)],
        ["VolShrink<0.5", "VolShrink 0.5-0.8", "VolShrink 0.8-1.0", "VolShrink 1.0-1.5", "VolShrink>1.5"])
    print_bucket_table("Analysis 1f: MFE by Volume Shrink Ratio", vol_results)

    # KDJ-K
    kdj_results = bucket_analyze(all_entries, "kdj_k",
        [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)],
        ["KDJ_K<20", "KDJ_K 20-40", "KDJ_K 40-60", "KDJ_K 60-80", "KDJ_K>80"])
    print_bucket_table("Analysis 1g: MFE by KDJ-K at Entry", kdj_results)

    # ADX
    adx_results = bucket_analyze(all_entries, "adx",
        [(0, 20), (20, 30), (30, 40), (40, 100)],
        ["ADX<20", "ADX 20-30", "ADX 30-40", "ADX>40"])
    print_bucket_table("Analysis 1h: MFE by ADX at Entry", adx_results)

    # ===== Analysis 2: Recovery point indicators =====
    analyze_recovery_point(all_entries)

    # ===== Analysis 3: Dynamic rules =====
    dynamic_results = test_dynamic_rules(all_entries)
    print_dynamic_table(dynamic_results)

    # ===== Analysis 4: Correlation =====
    print_correlation(all_entries)

    # ===== Recommendations =====
    print(f"\n  {'='*90}")
    print(f"  Summary: Indicator-Based Take-Profit Recommendations")
    print(f"  {'='*90}")

    # Find best dynamic rule
    best_dynamic = sorted(
        [(k, v) for k, v in dynamic_results.items() if k != "Baseline(100%)"],
        key=lambda x: x[1]["sharpe"], reverse=True
    )[:3]

    print(f"\n  Top 3 indicator-based rules (by Sharpe):")
    for i, (name, r) in enumerate(best_dynamic):
        vs_base = r["sharpe"] - baseline_sharpe if (baseline_sharpe := dynamic_results.get("Baseline(100%)", {}).get("sharpe", 0)) else 0
        print(f"  {i+1}. {name}")
        print(f"     Sharpe {r['sharpe']:+.2f} (vs baseline {vs_base:+.2f}), MedRet {r['med_return']:+.1f}%, WinRate {r['win_rate']:.1f}%")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
