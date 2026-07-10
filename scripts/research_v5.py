"""
research_v5.py -- V5 策略迭代研究

研究维度:
  R1: TP vs Time 交易入场特征差异 (全指标对比)
  R2: 市场环境 (大盘趋势)
  R3: 成交量模式 (缩量/放量/地量)
  R4: MACD 底背离
  R5: 连续下跌天数 (急跌 vs 阴跌)
  R6: 深度回调(>18%)子集分析
  R7: 反弹确认入场 (延迟入场)
  R8: 板块效应

用法:
  python scripts/research_v5.py --samples 300 --seeds 42,123,456
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
            and 30 <= price_pos <= 60 and profit_ratio < 30): return ("A", 90)
    if 8 <= decline_pct <= 25: return ("B", 70)
    return None


def get_v4_tp_sl(decline_pct):
    """V4 dynamic TP ratio and SL pct."""
    if decline_pct < 12: return 1.20, 0.35
    elif decline_pct < 18: return 1.10, 0.30
    else: return 0.90, 0.20


def simulate_v4(entry):
    """V4 exit simulation. Returns (exit_idx, reason, exit_price) or None."""
    close = entry["close"]; n = entry["n"]; today = entry["today"]
    peak = entry["peak_price"]; ep = entry["entry_price"]
    tp_ratio, sl_pct = get_v4_tp_sl(entry["decline_pct"])
    tp_price = ep + (peak - ep) * tp_ratio
    for j in range(today + 1, n):
        if close[j] >= tp_price: return (j, "TP", float(close[j]))
        if j - today >= 60: return (j, "Time", float(close[j]))
        if (peak - close[j]) / peak >= sl_pct: return (j, "Cut", float(close[j]))
    return None


def find_entries_full(code, db):
    """Find entry signals with ENRICHED indicators for research."""
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),)
    if len(rows) < 180: return []

    dates = [r["trade_date"] for r in rows]
    raw_c = np.array([r["close"] for r in rows], dtype=float)
    raw_h = np.array([r["high"] for r in rows], dtype=float)
    raw_l = np.array([r["low"] for r in rows], dtype=float)
    raw_v = np.array([r["volume"] for r in rows], dtype=float)

    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),)
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_c, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]
    n = len(close)
    rturn = np.array([r["turn"] or 0 for r in rows], dtype=float)

    # Pre-compute all indicators
    rsi_arr = indicators.rsi(close, 14)
    ma5 = indicators.sma(close, 5); ma10 = indicators.sma(close, 10)
    ma20 = indicators.sma(close, 20); ma60 = indicators.sma(close, 60)
    macd_dict = indicators.macd(close)
    kdj_dict = indicators.kdj(high, low, close)
    bb_dict = indicators.bollinger(close, 20, 2.0)
    atr_arr = indicators.atr(high, low, close, 14)
    adx_dict = indicators.adx(high, low, close, 14)

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

        rsi_val = float(rsi_arr[today]) if today < len(rsi_arr) and not np.isnan(rsi_arr[today]) else None
        ma20_dist = round((close[today] - ma20[today]) / ma20[today] * 100, 1) if today < len(ma20) and not np.isnan(ma20[today]) else None
        ma60_dist = round((close[today] - ma60[today]) / ma60[today] * 100, 1) if today < len(ma60) and not np.isnan(ma60[today]) else None

        profit_ratio = calc_profit_ratio(rturn[:today + 1], close[:today + 1], close[today])
        tier_result = classify_tier(decline_pct, price_pos, rsi_val, ma20_dist, profit_ratio)
        if tier_result is None: continue
        tier, score = tier_result

        # ═══ R1: Additional indicators ═══
        # R3: Volume patterns
        vol_5d_avg = np.mean(raw_v[max(0, today - 5):today + 1])
        vol_20d_avg = np.mean(raw_v[max(0, today - 20):today + 1])
        vol_ratio_5 = raw_v[today] / vol_5d_avg if vol_5d_avg > 0 else 1
        vol_ratio_20 = raw_v[today] / vol_20d_avg if vol_20d_avg > 0 else 1
        # Volume in decline period vs pre-decline
        decline_days = today - peak_idx
        if decline_days >= 3:
            pre_vol = np.mean(raw_v[max(0, peak_idx - 10):peak_idx + 1])
            decline_vol = np.mean(raw_v[peak_idx:today + 1])
            vol_shrink = decline_vol / pre_vol if pre_vol > 0 else 1.0
        else:
            vol_shrink = 1.0

        # R4: MACD divergence check
        # Bullish divergence: price makes lower low but MACD histogram makes higher low
        if trough_idx > 20 and today < len(macd_dict["histogram"]):
            prev_low_idx = peak_idx + np.argmin(low[peak_idx:trough_idx]) if trough_idx > peak_idx + 5 else trough_idx
            hist_now = float(macd_dict["histogram"][today])
            hist_trough = float(macd_dict["histogram"][trough_idx]) if trough_idx < len(macd_dict["histogram"]) and not np.isnan(macd_dict["histogram"][trough_idx]) else hist_now
            macd_divergence = 1 if (hist_now > hist_trough and close[today] < close[trough_idx]) else 0
        else:
            macd_divergence = 0

        # R5: Consecutive decline days from peak to trough
        consec_down = 0
        for k in range(peak_idx + 1, trough_idx + 1):
            if close[k] < close[k - 1]:
                consec_down += 1
            else:
                break
        # Decline speed: decline_pct / decline_days
        decline_speed = decline_pct / max(decline_days, 1)

        # R3: Turnover patterns
        turn_5d_avg = np.mean(rturn[max(0, today - 5):today + 1])
        turn_20d_avg = np.mean(rturn[max(0, today - 20):today + 1])
        turn_ratio = rturn[today] / turn_20d_avg if turn_20d_avg > 0 else 1

        # Additional: Bollinger bandwidth (squeeze indicator)
        bb_width = (float(bb_dict["upper"][today]) - float(bb_dict["lower"][today])) / float(bb_dict["mid"][today]) if today < len(bb_dict["mid"]) and not np.isnan(bb_dict["mid"][today]) else 0

        # Additional: MA convergence (MA5 vs MA20 distance)
        ma5_20_dist = (ma5[today] - ma20[today]) / ma20[today] * 100 if today < len(ma5) and not np.isnan(ma5[today]) and not np.isnan(ma20[today]) and ma20[today] > 0 else 0

        # Additional: KDJ divergence
        kdj_j = float(kdj_dict["j"][today]) if today < len(kdj_dict["j"]) and not np.isnan(kdj_dict["j"][today]) else None
        kdj_k = float(kdj_dict["k"][today]) if today < len(kdj_dict["k"]) and not np.isnan(kdj_dict["k"][today]) else None

        # ADX
        adx_val = float(adx_dict["adx"][today]) if today < len(adx_dict["adx"]) and not np.isnan(adx_dict["adx"][today]) else None
        plus_di = float(adx_dict["plus_di"][today]) if today < len(adx_dict["plus_di"]) and not np.isnan(adx_dict["plus_di"][today]) else None
        minus_di = float(adx_dict["minus_di"][today]) if today < len(adx_dict["minus_di"]) and not np.isnan(adx_dict["minus_di"][today]) else None

        # ATR as % of price
        atr_pct = float(atr_arr[today]) / close[today] * 100 if today < len(atr_arr) and not np.isnan(atr_arr[today]) and close[today] > 0 else 0

        # Distance from trough to entry (bounce amount)
        bounce_pct = (close[today] - trough_price) / trough_price * 100

        # Board info (主板/创业板/科创板 as proxy for sector)
        sector_row = db.fetchone("SELECT board FROM stock_basic WHERE code=?", (code,))
        sector = sector_row["board"] if sector_row else ""

        entries.append({
            "today": today, "entry_date": dates[today],
            "entry_price": float(close[today]),
            "peak_price": float(peak_price),
            "trough_price": float(trough_price),
            "decline_pct": decline_pct, "price_pos": price_pos,
            "rsi": rsi_val, "ma20_dist": ma20_dist, "ma60_dist": ma60_dist,
            "profit_ratio": profit_ratio, "tier": tier,
            # New indicators
            "vol_ratio_5": vol_ratio_5, "vol_ratio_20": vol_ratio_20,
            "vol_shrink": vol_shrink, "macd_divergence": macd_divergence,
            "consec_down": consec_down, "decline_speed": decline_speed,
            "decline_days": decline_days, "turn_ratio": turn_ratio,
            "bb_width": bb_width, "ma5_20_dist": ma5_20_dist,
            "kdj_j": kdj_j, "kdj_k": kdj_k, "adx": adx_val,
            "plus_di": plus_di, "minus_di": minus_di,
            "atr_pct": atr_pct, "bounce_pct": bounce_pct,
            "sector": sector,
            "close": close, "high": high, "low": low, "n": n, "raw_v": raw_v,
        })
    return entries


def compare_groups(tp_group, time_group, indicators_list, names):
    """Compare indicator distributions between TP and Time groups."""
    print(f"\n  {'Indicator':<22} {'TP_Mean':>10} {'Time_Mean':>10} {'Diff':>10} {'Direction':>20}")
    print(f"  {'-'*75}")
    diffs = []
    for key, name in zip(indicators_list, names):
        tp_vals = [e.get(key) for e in tp_group if e.get(key) is not None]
        time_vals = [e.get(key) for e in time_group if e.get(key) is not None]
        if len(tp_vals) > 10 and len(time_vals) > 10:
            tp_m = np.mean(tp_vals); tm_m = np.mean(time_vals)
            diff = tp_m - tm_m
            direction = "TP_higher" if diff > 0 else "TP_lower"
            print(f"  {name:<22} {tp_m:>10.3f} {tm_m:>10.3f} {diff:>+10.3f} {direction:>20}")
            diffs.append((name, abs(diff), direction))
    if diffs:
        print(f"\n  Top differentiators (by |diff|):")
        for name, d, direction in sorted(diffs, key=lambda x: -x[1])[:10]:
            print(f"    {name}: |diff|={d:.3f}, {direction}")


def bucket_compare(tp_group, time_group, key, name, buckets, labels):
    """Compare TP rate by indicator buckets."""
    print(f"\n  {'='*70}")
    print(f"  TP Rate by {name} bucket")
    print(f"  {'Bucket':<18} {'Total':>6} {'TP':>5} {'Time':>5} {'Cut':>5} {'TP%':>7} {'AvgRet%':>8}")
    print(f"  {'-'*65}")
    for (lo, hi), label in zip(buckets, labels):
        if hi is None:
            group = [e for e in tp_group + time_group if (e.get(key) or 0) >= lo]
        else:
            group = [e for e in tp_group + time_group if lo <= (e.get(key) or 0) < hi]
        if len(group) < 10: continue
        tp_n = len([e for e in group if e.get("_outcome") == "TP"])
        time_n = len([e for e in group if e.get("_outcome") == "Time"])
        cut_n = len([e for e in group if e.get("_outcome") == "Cut"])
        total = len(group)
        rets = [e["_return"] for e in group]
        print(f"  {label:<18} {total:>6} {tp_n:>5} {time_n:>5} {cut_n:>5} {tp_n/total*100:>6.1f}% {np.mean(rets):>+7.1f}%")


# ══════════════════════════════════════════════════════════════
# R1: TP vs Time feature comparison
# ══════════════════════════════════════════════════════════════

def research_r1(all_entries):
    """Compare TP trades vs Time trades on all indicators."""
    print(f"\n{'='*100}")
    print(f"  R1: TP vs Time Entry Feature Comparison")
    print(f"{'='*100}")

    # Run V4 simulation and tag each entry
    tp_entries = []; time_entries = []; cut_entries = []
    for e in all_entries:
        result = simulate_v4(e)
        if result is None: continue
        idx, reason, px = result
        ret = (px - e["entry_price"]) / e["entry_price"] * 100
        e["_outcome"] = reason
        e["_return"] = ret
        e["_hold_days"] = idx - e["today"]
        if reason == "TP": tp_entries.append(e)
        elif reason == "Time": time_entries.append(e)
        else: cut_entries.append(e)

    print(f"\n  TP: {len(tp_entries)}, Time: {len(time_entries)}, Cut: {len(cut_entries)}")

    # Compare all indicators
    indicators_list = [
        "decline_pct", "price_pos", "rsi", "ma20_dist", "ma60_dist",
        "profit_ratio", "vol_shrink", "vol_ratio_5", "vol_ratio_20",
        "consec_down", "decline_speed", "decline_days",
        "turn_ratio", "bb_width", "atr_pct", "adx",
        "ma5_20_dist", "kdj_k", "kdj_j", "bounce_pct",
    ]
    names = [
        "decline%", "position%", "RSI", "MA20%", "MA60%",
        "profit%", "vol_shrink", "vol_ratio5", "vol_ratio20",
        "consec_down", "decline_speed", "decline_days",
        "turn_ratio", "BB_width", "ATR%", "ADX",
        "MA5-20%", "KDJ_K", "KDJ_J", "bounce%",
    ]
    compare_groups(tp_entries, time_entries, indicators_list, names)

    return tp_entries, time_entries, cut_entries


# ══════════════════════════════════════════════════════════════
# R3: Volume patterns
# ══════════════════════════════════════════════════════════════

def research_r3(all_entries):
    """Volume pattern analysis."""
    print(f"\n{'='*100}")
    print(f"  R3: Volume Pattern Analysis")
    print(f"{'='*100}")
    # Bucket by vol_shrink
    bucket_compare(
        [e for e in all_entries if e.get("_outcome")],
        [], "vol_shrink", "Vol Shrink Ratio",
        [(0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 3), (3, 99)],
        ["<0.5(地量)", "0.5-0.8(缩量)", "0.8-1.0(正常)", "1.0-1.5(放量)", "1.5-3(巨量)", ">3(天量)"],
    )
    # Bucket by vol_ratio_5
    bucket_compare(
        [e for e in all_entries if e.get("_outcome")],
        [], "vol_ratio_5", "Volume/5d_avg",
        [(0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 99)],
        ["<0.5", "0.5-0.8", "0.8-1.2", "1.2-2.0", ">2.0"],
    )


# ══════════════════════════════════════════════════════════════
# R4: MACD divergence
# ══════════════════════════════════════════════════════════════

def research_r4(all_entries):
    """MACD divergence analysis."""
    print(f"\n{'='*100}")
    print(f"  R4: MACD Divergence Analysis")
    print(f"{'='*100}")
    div = [e for e in all_entries if e.get("_outcome") and e.get("macd_divergence") == 1]
    no_div = [e for e in all_entries if e.get("_outcome") and e.get("macd_divergence") == 0]

    for label, group in [("MACD_divergence", div), ("No_divergence", no_div)]:
        if not group: continue
        tp_n = len([e for e in group if e["_outcome"] == "TP"])
        time_n = len([e for e in group if e["_outcome"] == "Time"])
        cut_n = len([e for e in group if e["_outcome"] == "Cut"])
        total = len(group)
        rets = [e["_return"] for e in group]
        print(f"  {label:<22} N={total:>5} TP%={tp_n/total*100:.0f}% Time%={time_n/total*100:.0f}% "
              f"AvgRet={np.mean(rets):+.1f}% MedRet={np.median(rets):+.1f}%")


# ══════════════════════════════════════════════════════════════
# R5: Decline speed
# ══════════════════════════════════════════════════════════════

def research_r5(all_entries):
    """Decline speed (急跌 vs 阴跌)."""
    print(f"\n{'='*100}")
    print(f"  R5: Decline Speed (急跌 vs 阴跌)")
    print(f"{'='*100}")
    bucket_compare(
        [e for e in all_entries if e.get("_outcome")],
        [], "decline_speed", "Decline Speed (%/day)",
        [(0, 1), (1, 2), (2, 3), (3, 5), (5, 50)],
        ["<1%/d(阴跌)", "1-2%/d", "2-3%/d", "3-5%/d(急跌)", ">5%/d(暴跌)"],
    )
    bucket_compare(
        [e for e in all_entries if e.get("_outcome")],
        [], "decline_days", "Decline Duration (days)",
        [(1, 5), (5, 10), (10, 15), (15, 25), (25, 40)],
        ["1-5d", "5-10d", "10-15d", "15-25d", "25-40d"],
    )
    # Consecutive down days
    bucket_compare(
        [e for e in all_entries if e.get("_outcome")],
        [], "consec_down", "Consecutive Down Days",
        [(1, 3), (3, 5), (5, 8), (8, 15), (15, 40)],
        ["1-3d", "3-5d", "5-8d", "8-15d", "15-40d"],
    )


# ══════════════════════════════════════════════════════════════
# R6: Deep decline (>18%) subset
# ══════════════════════════════════════════════════════════════

def research_r6(all_entries):
    """Deep decline analysis — should we skip these?"""
    print(f"\n{'='*100}")
    print(f"  R6: Deep Decline (>18%) Subset Analysis")
    print(f"{'='*100}")
    deep = [e for e in all_entries if e.get("_outcome") and e["decline_pct"] >= 18]
    shallow = [e for e in all_entries if e.get("_outcome") and e["decline_pct"] < 18]

    for label, group in [("Deep(>=18%)", deep), ("Shallow(<18%)", shallow)]:
        if not group: continue
        tp_n = len([e for e in group if e["_outcome"] == "TP"])
        time_n = len([e for e in group if e["_outcome"] == "Time"])
        cut_n = len([e for e in group if e["_outcome"] == "Cut"])
        total = len(group)
        rets = [e["_return"] for e in group]
        print(f"  {label:<20} N={total:>5} TP%={tp_n/total*100:.0f}% Time%={time_n/total*100:.0f}% Cut%={cut_n/total*100:.0f}% "
              f"Avg={np.mean(rets):+.1f}% Med={np.median(rets):+.1f}%")

    # What if we skip deep decline entirely?
    print(f"\n  Impact of skipping >18% decline signals:")
    v4_all = [e["_return"] for e in all_entries if e.get("_return") is not None]
    v4_no_deep = [e["_return"] for e in all_entries if e.get("_return") is not None and e["decline_pct"] < 18]
    print(f"    V4 all:   N={len(v4_all)}, Med={np.median(v4_all):+.1f}%, Avg={np.mean(v4_all):+.1f}%")
    print(f"    V4 <18%:  N={len(v4_no_deep)}, Med={np.median(v4_no_deep):+.1f}%, Avg={np.mean(v4_no_deep):+.1f}%")


# ══════════════════════════════════════════════════════════════
# R7: Bounce confirmation (delayed entry)
# ══════════════════════════════════════════════════════════════

def research_r7(all_entries_raw, db):
    """Test if waiting for bounce confirmation improves results.
    Re-scan with a stricter bounce requirement."""
    print(f"\n{'='*100}")
    print(f"  R7: Bounce Confirmation Entry (delayed entry)")
    print(f"{'='*100}")

    # Test different bounce thresholds
    for min_bounce in [0, 1, 2, 3, 5]:
        filtered = [e for e in all_entries_raw if e.get("bounce_pct", 0) >= min_bounce
                    and e.get("_outcome") is not None]
        if not filtered: continue
        rets = [e["_return"] for e in filtered]
        tp_n = len([e for e in filtered if e["_outcome"] == "TP"])
        time_n = len([e for e in filtered if e["_outcome"] == "Time"])
        total = len(filtered)
        print(f"  Bounce>={min_bounce}%: N={total:>5} TP%={tp_n/total*100:.0f}% Time%={time_n/total*100:.0f}% "
              f"Med={np.median(rets):+.1f}% Avg={np.mean(rets):+.1f}% "
              f"Sharpe={np.mean(rets)/np.std(rets):.2f}" if np.std(rets) > 0 else f"  Bounce>={min_bounce}%: N={total}")


# ══════════════════════════════════════════════════════════════
# R8: Sector analysis
# ══════════════════════════════════════════════════════════════

def research_r8(all_entries):
    """Sector/industry analysis."""
    print(f"\n{'='*100}")
    print(f"  R8: Sector/Industry Analysis")
    print(f"{'='*100}")
    sectors = defaultdict(list)
    for e in all_entries:
        if e.get("_outcome") and e.get("sector"):
            sectors[e["sector"]].append(e)

    print(f"  {'Sector':<25} {'N':>5} {'TP%':>6} {'Time%':>7} {'MedRet':>8}")
    print(f"  {'-'*55}")
    for sector, group in sorted(sectors.items(), key=lambda x: -len(x[1])):
        if len(group) < 20: continue
        tp_n = len([e for e in group if e["_outcome"] == "TP"])
        time_n = len([e for e in group if e["_outcome"] == "Time"])
        rets = [e["_return"] for e in group]
        print(f"  {sector:<25} {len(group):>5} {tp_n/len(group)*100:>5.0f}% {time_n/len(group)*100:>6.0f}% {np.median(rets):>+7.1f}%")


# ══════════════════════════════════════════════════════════════
# New Filter Testing
# ══════════════════════════════════════════════════════════════

def test_filter(all_entries, filter_fn, filter_name):
    """Test a filter: compute stats with and without the filter."""
    filtered = [e for e in all_entries if e.get("_outcome") and filter_fn(e)]
    unfiltered = [e for e in all_entries if e.get("_outcome")]
    if not filtered: return None

    def stats(group):
        rets = [e["_return"] for e in group]
        wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
        tp_n = len([e for e in group if e["_outcome"] == "TP"])
        time_n = len([e for e in group if e["_outcome"] == "Time"])
        cut_n = len([e for e in group if e["_outcome"] == "Cut"])
        total = len(group)
        sr = np.mean(rets)/np.std(rets) if np.std(rets) > 0 else 0
        avg_w = np.mean(wins) if wins else 0; avg_l = np.mean(losses) if losses else 0
        return {
            "n": total, "win_rate": len(wins)/total*100,
            "med": np.median(rets), "avg": np.mean(rets), "sharpe": sr,
            "wl_ratio": abs(avg_w/avg_l) if avg_l else 0,
            "tp_rate": tp_n/total*100, "time_rate": time_n/total*100, "cut_rate": cut_n/total*100,
        }

    f_stats = stats(filtered)
    u_stats = stats(unfiltered)
    return filter_name, u_stats, f_stats


def print_filter_results(filter_results):
    """Print filter comparison table."""
    print(f"\n{'='*120}")
    print(f"  Filter Test Results (sorted by Sharpe improvement)")
    print(f"{'='*120}")
    print(f"  {'Filter':<35} {'VN':>5} {'V_Win%':>7} {'V_Med':>7} {'V_Sh':>6} {'V_Time%':>7} "
          f"{'FN':>5} {'F_Win%':>7} {'F_Med':>7} {'F_Sh':>6} {'F_Time%':>7} {'dSh':>6}")
    print(f"  {'-'*120}")

    valid = [(n, u, f) for n, u, f in filter_results if u and f and f["n"] > 30]
    for name, u, f in sorted(valid, key=lambda x: x[2]["sharpe"] - x[1]["sharpe"], reverse=True):
        d_sh = f["sharpe"] - u["sharpe"]
        print(f"  {name:<35} {u['n']:>5} {u['win_rate']:>6.1f}% {u['med']:>+6.1f}% {u['sharpe']:>5.2f} {u['time_rate']:>6.0f}% "
              f"{f['n']:>5} {f['win_rate']:>6.1f}% {f['med']:>+6.1f}% {f['sharpe']:>5.2f} {f['time_rate']:>6.0f}% {d_sh:>+5.2f}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    db = Database()

    all_entries_all = []

    for seed in seeds:
        random.seed(seed); np.random.seed(seed)
        codes = random.sample(db.get_active_stock_codes(), min(args.samples, len(db.get_active_stock_codes())))
        print(f"\n[Seed {seed}] Scanning {len(codes)} stocks...")
        for i, code in enumerate(codes):
            entries = find_entries_full(code, db)
            all_entries_all.extend(entries)
            if (i+1) % 100 == 0: print(f"  {i+1}/{len(codes)}, {len(all_entries_all)} total")

    print(f"\nTotal enriched entries: {len(all_entries_all)}")

    # ==== Run V4 simulation to tag outcomes ====
    print("\nRunning V4 simulation to tag outcomes...")
    for e in all_entries_all:
        result = simulate_v4(e)
        if result is None: continue
        idx, reason, px = result
        e["_outcome"] = reason
        e["_return"] = (px - e["entry_price"]) / e["entry_price"] * 100
        e["_hold_days"] = idx - e["today"]

    tagged = [e for e in all_entries_all if e.get("_outcome")]
    tp = [e for e in tagged if e["_outcome"] == "TP"]
    time = [e for e in tagged if e["_outcome"] == "Time"]
    cut = [e for e in tagged if e["_outcome"] == "Cut"]
    print(f"Tagged: {len(tagged)} (TP:{len(tp)}, Time:{len(time)}, Cut:{len(cut)})")

    # ==== Run research ====
    research_r1(all_entries_all)
    research_r3(all_entries_all)
    research_r4(all_entries_all)
    research_r5(all_entries_all)
    research_r6(all_entries_all)
    research_r7(all_entries_all, db)
    research_r8(all_entries_all)

    # ==== Test filters ====
    print(f"\n{'='*120}")
    print(f"  Testing New Filters")
    print(f"{'='*120}")

    filters = [
        # Turnover ratio filters
        ("TurnRatio<1.0(冷清)", lambda e: e.get("turn_ratio", 1) < 1.0),
        ("TurnRatio<0.8", lambda e: e.get("turn_ratio", 1) < 0.8),
        ("TurnRatio<0.5", lambda e: e.get("turn_ratio", 1) < 0.5),
        # Volume shrink filters
        ("VolShrink<0.8(缩量)", lambda e: e.get("vol_shrink", 1) < 0.8),
        ("VolShrink<0.5(极度缩量)", lambda e: e.get("vol_shrink", 1) < 0.5),
        ("VolShrink<0.8_AND_TurnRatio<1.0", lambda e: e.get("vol_shrink", 1) < 0.8 and e.get("turn_ratio", 1) < 1.0),
        # Decline speed filters (急跌 better than 阴跌?)
        ("DeclineSpeed>2%/d(急跌)", lambda e: e.get("decline_speed", 0) > 2),
        ("DeclineSpeed>3%/d", lambda e: e.get("decline_speed", 0) > 3),
        ("DeclineDays<10(快跌)", lambda e: e.get("decline_days", 99) < 10),
        # Consecutive down days
        ("ConsecDown<5", lambda e: e.get("consec_down", 99) < 5),
        ("ConsecDown<3", lambda e: e.get("consec_down", 99) < 3),
        # MACD divergence
        ("MACD_Divergence", lambda e: e.get("macd_divergence") == 1),
        # Skip deep decline
        ("Skip_Decline>18%", lambda e: e.get("decline_pct", 0) < 18),
        ("Skip_Decline>20%", lambda e: e.get("decline_pct", 0) < 20),
        # Profit ratio extremes
        ("ProfitRatio<20%(极致低)", lambda e: e.get("profit_ratio", 99) < 20),
        # Volume at entry (not too high)
        ("VolRatio5<1.5", lambda e: e.get("vol_ratio_5", 1) < 1.5),
        ("VolRatio5<1.0(缩量日)", lambda e: e.get("vol_ratio_5", 1) < 1.0),
        # Combined signals
        ("VolShrink<0.8_AND_DeclineSpeed>2", lambda e: e.get("vol_shrink", 1) < 0.8 and e.get("decline_speed", 0) > 2),
        ("VolShrink<0.8_AND_DeclineDays<10", lambda e: e.get("vol_shrink", 1) < 0.8 and e.get("decline_days", 99) < 10),
        # ADX filters
        ("ADX<25(弱趋势)", lambda e: e.get("adx", 99) < 25),
        ("ADX<20", lambda e: e.get("adx", 99) < 20),
        # KDJ oversold
        ("KDJ_J<0(超卖)", lambda e: e.get("kdj_j", 0) < 0),
    ]

    filter_results = []
    for name, fn in filters:
        result = test_filter(all_entries_all, fn, name)
        if result:
            filter_results.append(result)

    print_filter_results(filter_results)

    # ==== Summary ====
    print(f"\n{'='*120}")
    print(f"  V5 Iteration Summary")
    print(f"{'='*120}")
    best = sorted(filter_results, key=lambda x: x[2]["sharpe"] - x[1]["sharpe"], reverse=True)
    if best:
        print(f"\n  Top 5 filters by Sharpe improvement:")
        for i, (name, u, f) in enumerate(best[:5]):
            d_sh = f["sharpe"] - u["sharpe"]
            d_time = f["time_rate"] - u["time_rate"]
            retained = f["n"] / u["n"] * 100
            print(f"  {i+1}. {name}: dSharpe={d_sh:+.2f}  dTimeRate={d_time:+.0f}%  Retained={retained:.0f}%  "
                  f"V4_Med={u['med']:+.1f}% -> Filtered_Med={f['med']:+.1f}%")

    print()


if __name__ == "__main__":
    main()
