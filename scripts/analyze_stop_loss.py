"""
analyze_stop_loss.py -- 止损策略全方位回测

分析维度:
  1. 可变止损阈值: 15%-40% 从峰顶跌幅
  2. MAE (最大不利偏移): 每笔交易最差回撤分布
  3. ATR 动态止损: ATR 倍数 × 入场价
  4. 时间止损: 30/45/60/90/120 天
  5. 入场指标动态止损: 基于跌幅/RSI/获利比例调整
  6. 综合 TP+SL 优化: 最优止盈 + 最优止损组合

用法:
  python scripts/analyze_stop_loss.py --samples 300 --seeds 42,123,456
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
    """Find all valid entry signals."""
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
    n = len(close)

    rturn = np.array([r["turn"] or 0 for r in rows], dtype=float)
    atr_arr = indicators.atr(high, low, close, 14)

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
        atr_val = float(atr_arr[today]) if today < len(atr_arr) and not np.isnan(atr_arr[today]) else close[today] * 0.03

        # Additional indicators
        adx_dict = indicators.adx(high, low, close, 14)
        adx_val = float(adx_dict["adx"][today]) if today < len(adx_dict["adx"]) and not np.isnan(adx_dict["adx"][today]) else None

        entries.append({
            "today": today,
            "entry_date": dates[today],
            "entry_price": float(close[today]),
            "peak_price": float(peak_price),
            "decline_pct": decline_pct,
            "price_pos": price_pos,
            "rsi": rsi_val,
            "ma20_dist": ma20_dist,
            "profit_ratio": profit_ratio,
            "atr": float(atr_val),
            "adx": adx_val,
            "tier": tier,
            "close": close,
            "high": high,
            "low": low,
            "n": n,
        })
    return entries


# ══════════════════════════════════════════════════════════════
# Analysis 1: Variable Stop-Loss Thresholds
# ══════════════════════════════════════════════════════════════

def simulate_exit_sl(entry, sl_pct, tp_ratio=1.00, max_days=60):
    """Simulate exit with variable stop-loss threshold.
    sl_pct: stop-loss as % decline from peak (e.g. 0.30 = 30%)
    tp_ratio: take-profit recovery ratio (1.00 = back to peak)
    """
    close = entry["close"]
    n = entry["n"]
    today = entry["today"]
    peak_price = entry["peak_price"]
    entry_price = entry["entry_price"]
    tp_price = entry_price + (peak_price - entry_price) * tp_ratio

    for j in range(today + 1, n):
        # TP first
        if close[j] >= tp_price:
            return (j, "TP", float(close[j]))
        # Time stop
        if j - today >= max_days:
            return (j, "Time", float(close[j]))
        # Stop-loss: decline from peak >= sl_pct
        if (peak_price - close[j]) / peak_price >= sl_pct:
            return (j, "Cut", float(close[j]))
    return None


def analyze_variable_sl(all_entries, tp_ratio=1.00):
    """Test variable stop-loss thresholds."""
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    results = {}

    for sl in thresholds:
        trades = []
        for entry in all_entries:
            result = simulate_exit_sl(entry, sl, tp_ratio)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
                "tier": entry["tier"],
                "decline_pct": entry["decline_pct"],
            })

        if not trades:
            continue

        rets = [t["return_pct"] for t in trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        tp_trades = [t for t in trades if t["exit_reason"] == "TP"]
        time_trades = [t for t in trades if t["exit_reason"] == "Time"]
        cut_trades = [t for t in trades if t["exit_reason"] == "Cut"]

        total = len(trades)
        sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        pf = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')

        results[sl] = {
            "total": total,
            "win_rate": len(wins) / total * 100,
            "med_return": np.median(rets),
            "avg_return": np.mean(rets),
            "sharpe": sr,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": wl_ratio,
            "profit_factor": pf,
            "med_days": np.median([t["hold_days"] for t in trades]),
            "avg_days": np.mean([t["hold_days"] for t in trades]),
            "tp_rate": len(tp_trades) / total * 100,
            "time_rate": len(time_trades) / total * 100,
            "cut_rate": len(cut_trades) / total * 100,
            "tp_avg_ret": np.mean([t["return_pct"] for t in tp_trades]) if tp_trades else 0,
            "time_avg_ret": np.mean([t["return_pct"] for t in time_trades]) if time_trades else 0,
            "cut_avg_ret": np.mean([t["return_pct"] for t in cut_trades]) if cut_trades else 0,
            "std_return": np.std(rets),
        }

    return results


# ══════════════════════════════════════════════════════════════
# Analysis 2: MAE (Maximum Adverse Excursion)
# ══════════════════════════════════════════════════════════════

def analyze_mae(all_entries):
    """Track the worst drawdown for each trade, categorized by final outcome."""
    tp_mae = []    # MAE for trades that eventually hit TP
    time_mae = []  # MAE for trades that timed out
    cut_mae = []   # MAE for trades that hit stop-loss (using 30% as baseline SL)

    for entry in all_entries:
        close = entry["close"]
        n = entry["n"]
        today = entry["today"]
        peak_price = entry["peak_price"]
        entry_price = entry["entry_price"]
        tp_price = entry_price + (peak_price - entry_price) * 1.00

        max_decline = 0  # track worst decline from peak during hold

        for j in range(today + 1, min(n, today + 61)):
            decline_from_peak = (peak_price - close[j]) / peak_price
            if decline_from_peak > max_decline:
                max_decline = decline_from_peak

            # Determine exit reason using baseline 30% SL
            if close[j] >= tp_price:
                tp_mae.append(max_decline * 100)
                break
            if j - today >= 60:
                time_mae.append(max_decline * 100)
                break
            if decline_from_peak >= 0.30:
                cut_mae.append(max_decline * 100)
                break
        else:
            continue  # didn't exit within data

    def pctiles(arr):
        if not arr:
            return (0, 0, 0, 0, 0)
        return (np.percentile(arr, 25), np.median(arr), np.percentile(arr, 75),
                np.percentile(arr, 90), np.max(arr))

    return {
        "tp_mae": pctiles(tp_mae),
        "time_mae": pctiles(time_mae),
        "cut_mae": pctiles(cut_mae),
        "tp_count": len(tp_mae),
        "time_count": len(time_mae),
        "cut_count": len(cut_mae),
        # How many TP trades would have been killed at various SL levels
        "tp_killed_at_20": len([x for x in tp_mae if x > 20]) / max(len(tp_mae), 1) * 100,
        "tp_killed_at_25": len([x for x in tp_mae if x > 25]) / max(len(tp_mae), 1) * 100,
        "tp_killed_at_30": len([x for x in tp_mae if x > 30]) / max(len(tp_mae), 1) * 100,
        "tp_killed_at_35": len([x for x in tp_mae if x > 35]) / max(len(tp_mae), 1) * 100,
    }


# ══════════════════════════════════════════════════════════════
# Analysis 3: ATR-based Dynamic Stop-Loss
# ══════════════════════════════════════════════════════════════

def simulate_exit_atr_sl(entry, atr_mult, tp_ratio=1.00, max_days=60):
    """Stop-loss = entry_price - ATR * atr_mult"""
    close = entry["close"]
    n = entry["n"]
    today = entry["today"]
    peak_price = entry["peak_price"]
    entry_price = entry["entry_price"]
    tp_price = entry_price + (peak_price - entry_price) * tp_ratio
    atr = entry["atr"]
    sl_price = entry_price - atr * atr_mult

    for j in range(today + 1, n):
        if close[j] >= tp_price:
            return (j, "TP", float(close[j]))
        if j - today >= max_days:
            return (j, "Time", float(close[j]))
        if close[j] <= sl_price:
            return (j, "Cut", float(close[j]))
    return None


def analyze_atr_sl(all_entries, tp_ratio=1.00):
    """Test ATR-based stop-loss with various multipliers."""
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = {}

    for mult in multipliers:
        trades = []
        for entry in all_entries:
            result = simulate_exit_atr_sl(entry, mult, tp_ratio)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            tp_c = len([t for t in trades if t["exit_reason"] == "TP"])
            time_c = len([t for t in trades if t["exit_reason"] == "Time"])
            cut_c = len([t for t in trades if t["exit_reason"] == "Cut"])
            total = len(trades)
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            results[mult] = {
                "total": total, "win_rate": len(wins)/total*100,
                "med_return": np.median(rets), "avg_return": np.mean(rets),
                "sharpe": sr, "avg_win": avg_w, "avg_loss": avg_l,
                "win_loss_ratio": abs(avg_w/avg_l) if avg_l else 0,
                "profit_factor": sum(wins)/abs(sum(losses)) if sum(losses) else 0,
                "tp_rate": tp_c/total*100, "time_rate": time_c/total*100, "cut_rate": cut_c/total*100,
                "cut_avg_ret": np.mean([t["return_pct"] for t in trades if t["exit_reason"]=="Cut"]) if cut_c else 0,
                "med_days": np.median([t["hold_days"] for t in trades]),
            }

    return results


# ══════════════════════════════════════════════════════════════
# Analysis 4: Time Stop Optimization
# ══════════════════════════════════════════════════════════════

def analyze_time_stop(all_entries, sl_pct=0.30, tp_ratio=1.00):
    """Test different max hold periods."""
    periods = [30, 45, 60, 90, 120]
    results = {}

    for max_days in periods:
        trades = []
        for entry in all_entries:
            result = simulate_exit_sl(entry, sl_pct, tp_ratio, max_days)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            tp_c = len([t for t in trades if t["exit_reason"] == "TP"])
            time_c = len([t for t in trades if t["exit_reason"] == "Time"])
            cut_c = len([t for t in trades if t["exit_reason"] == "Cut"])
            total = len(trades)
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            results[max_days] = {
                "total": total, "win_rate": len(wins)/total*100,
                "med_return": np.median(rets), "avg_return": np.mean(rets),
                "sharpe": sr, "avg_win": avg_w, "avg_loss": avg_l,
                "win_loss_ratio": abs(avg_w/avg_l) if avg_l else 0,
                "profit_factor": sum(wins)/abs(sum(losses)) if sum(losses) else 0,
                "tp_rate": tp_c/total*100, "time_rate": time_c/total*100, "cut_rate": cut_c/total*100,
                "time_avg_ret": np.mean([t["return_pct"] for t in trades if t["exit_reason"]=="Time"]) if time_c else 0,
                "med_days": np.median([t["hold_days"] for t in trades]),
            }

    return results


# ══════════════════════════════════════════════════════════════
# Analysis 5: Indicator-based Dynamic Stop-Loss
# ══════════════════════════════════════════════════════════════

def analyze_indicator_sl(all_entries):
    """Dynamic SL based on entry indicators. TP fixed at 100% for baseline comparison."""
    rules = {
        "Baseline_SL30%": {
            "desc": "Fixed 30% SL (current)",
            "fn": lambda e: 0.30,
        },
        "Decline_based": {
            "desc": "Decline<12->SL35%, 12-18->SL30%, >18->SL20%",
            "fn": lambda e: 0.35 if e["decline_pct"] < 12 else (0.30 if e["decline_pct"] < 18 else 0.20),
        },
        "RSI_based": {
            "desc": "RSI>55->SL35%, 40-55->SL30%",
            "fn": lambda e: 0.35 if (e.get("rsi") or 0) > 55 else 0.30,
        },
        "Profit_based": {
            "desc": "Profit<20->SL35%, 20-40->SL30%, >40->SL20%",
            "fn": lambda e: 0.35 if (e.get("profit_ratio") or 99) < 20 else (0.30 if (e.get("profit_ratio") or 99) < 40 else 0.20),
        },
        "Decline+RSI": {
            "desc": "LowDecline+HighRSI->SL35%, DeepDecline->SL25%, else SL30%",
            "fn": lambda e: (0.40 if ((e["decline_pct"] < 12 and (e.get("rsi") or 0) > 50))
                            else (0.25 if e["decline_pct"] > 18 else 0.30)),
        },
        "ADX_based": {
            "desc": "ADX<20->SL35%, 20-30->SL25%, >30->SL30%",
            "fn": lambda e: 0.35 if (e.get("adx") or 99) < 20 else (0.25 if 20 <= (e.get("adx") or 0) <= 30 else 0.30),
        },
    }

    results = {}
    for name, rule in rules.items():
        trades = []
        for entry in all_entries:
            sl_pct = rule["fn"](entry)
            result = simulate_exit_sl(entry, sl_pct, tp_ratio=1.00)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
                "sl_pct": sl_pct,
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            total = len(trades)
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            tp_c = len([t for t in trades if t["exit_reason"] == "TP"])
            time_c = len([t for t in trades if t["exit_reason"] == "Time"])
            cut_c = len([t for t in trades if t["exit_reason"] == "Cut"])

            # SL distribution
            sl_dist = defaultdict(int)
            for t in trades:
                sl_dist[t["sl_pct"]] += 1

            results[name] = {
                "desc": rule["desc"],
                "total": total, "win_rate": len(wins)/total*100,
                "med_return": np.median(rets), "avg_return": np.mean(rets),
                "sharpe": sr, "avg_win": avg_w, "avg_loss": avg_l,
                "win_loss_ratio": abs(avg_w/avg_l) if avg_l else 0,
                "profit_factor": sum(wins)/abs(sum(losses)) if sum(losses) else 0,
                "tp_rate": tp_c/total*100, "time_rate": time_c/total*100, "cut_rate": cut_c/total*100,
                "cut_avg_ret": np.mean([t["return_pct"] for t in trades if t["exit_reason"]=="Cut"]) if cut_c else 0,
                "med_days": np.median([t["hold_days"] for t in trades]),
                "sl_dist": dict(sl_dist),
            }

    return results


# ══════════════════════════════════════════════════════════════
# Analysis 6: Combined TP + SL Optimization
# ══════════════════════════════════════════════════════════════

def get_tp_ratio(entry):
    """Dynamic TP based on decline (best from previous analysis)."""
    d = entry["decline_pct"]
    if d < 12:
        return 1.20
    elif d < 18:
        return 1.10
    else:
        return 0.90


def analyze_combined(all_entries):
    """Test combinations of TP and SL strategies."""
    combos = [
        ("V3_Baseline", 1.00, 0.30),
        ("V4_DynamicTP_Only", "dynamic", 0.30),
        ("DynamicTP+SL25%", "dynamic", 0.25),
        ("DynamicTP+SL35%", "dynamic", 0.35),
        ("DynamicTP+SL40%", "dynamic", 0.40),
        ("DynamicTP+DeclineSL", "dynamic", "decline_sl"),
    ]

    results = {}
    for name, tp, sl in combos:
        trades = []
        for entry in all_entries:
            tp_r = get_tp_ratio(entry) if tp == "dynamic" else tp

            if sl == "decline_sl":
                d = entry["decline_pct"]
                sl_pct = 0.35 if d < 12 else (0.30 if d < 18 else 0.20)
            else:
                sl_pct = sl

            result = simulate_exit_sl(entry, sl_pct, tp_r)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            total = len(trades)
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            tp_c = len([t for t in trades if t["exit_reason"] == "TP"])
            time_c = len([t for t in trades if t["exit_reason"] == "Time"])
            cut_c = len([t for t in trades if t["exit_reason"] == "Cut"])

            results[name] = {
                "total": total, "win_rate": len(wins)/total*100,
                "med_return": np.median(rets), "avg_return": np.mean(rets),
                "sharpe": sr, "avg_win": avg_w, "avg_loss": avg_l,
                "win_loss_ratio": abs(avg_w/avg_l) if avg_l else 0,
                "profit_factor": sum(wins)/abs(sum(losses)) if sum(losses) else 0,
                "tp_rate": tp_c/total*100, "time_rate": time_c/total*100, "cut_rate": cut_c/total*100,
                "tp_avg_ret": np.mean([t["return_pct"] for t in trades if t["exit_reason"]=="TP"]) if tp_c else 0,
                "time_avg_ret": np.mean([t["return_pct"] for t in trades if t["exit_reason"]=="Time"]) if time_c else 0,
                "cut_avg_ret": np.mean([t["return_pct"] for t in trades if t["exit_reason"]=="Cut"]) if cut_c else 0,
                "med_days": np.median([t["hold_days"] for t in trades]),
                "p25_ret": np.percentile(rets, 25),
                "p75_ret": np.percentile(rets, 75),
            }

    return results


# ══════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════

def avg_dicts(dicts):
    """Average a list of result dicts. Assumes all dicts have the same keys. Skips non-numeric."""
    if not dicts:
        return {}
    result = {}
    for key in dicts[0].keys():
        vals = [d[key] for d in dicts if key in d]
        if vals and isinstance(vals[0], (int, float, np.floating, np.integer)):
            result[key] = np.mean(vals)
        else:
            result[key] = vals[0] if vals else ""  # keep first non-numeric as-is
    return result


def main():
    parser = argparse.ArgumentParser(description="Stop-loss strategy analysis")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    db = Database()

    # Collect all results across seeds
    all_sl_results = defaultdict(list)
    all_mae = []
    all_atr = defaultdict(list)
    all_time = defaultdict(list)
    all_indicator = defaultdict(list)
    all_combined = defaultdict(list)

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        codes = random.sample(db.get_active_stock_codes(), min(args.samples, len(db.get_active_stock_codes())))
        print(f"\n[Seed {seed}] Scanning {len(codes)} stocks...")

        all_entries = []
        for i, code in enumerate(codes):
            entries = find_entries(code, db)
            all_entries.extend(entries)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(codes)}, {len(all_entries)} signals")

        print(f"  Total: {len(all_entries)} signals (A:{len([e for e in all_entries if e['tier']=='A'])}, B:{len([e for e in all_entries if e['tier']=='B'])})")

        if len(all_entries) < 10:
            continue

        # Analysis 1: Variable SL
        sl = analyze_variable_sl(all_entries)
        for k, v in sl.items():
            all_sl_results[k].append(v)

        # Analysis 2: MAE
        all_mae.append(analyze_mae(all_entries))

        # Analysis 3: ATR
        atr = analyze_atr_sl(all_entries)
        for k, v in atr.items():
            all_atr[k].append(v)

        # Analysis 4: Time stop
        ts = analyze_time_stop(all_entries)
        for k, v in ts.items():
            all_time[k].append(v)

        # Analysis 5: Indicator dynamic SL
        ind = analyze_indicator_sl(all_entries)
        for k, v in ind.items():
            all_indicator[k].append(v)

        # Analysis 6: Combined
        comb = analyze_combined(all_entries)
        for k, v in comb.items():
            all_combined[k].append(v)

    n_seeds = len([s for s in seeds if True])  # count completed seeds

    # ═══════════════════ Print Reports ═══════════════════

    print("\n" + "=" * 130)
    print("  STOP-LOSS STRATEGY ANALYSIS")
    print("=" * 130)

    # ---- 1: Variable SL ----
    print(f"\n  {'='*120}")
    print(f"  [1] Variable Stop-Loss Thresholds (fixed TP=100%, 60d max)")
    print(f"  {'='*120}")
    print(f"  {'SL_Thr':>8} {'Win%':>7} {'MedRet':>8} {'AvgRet':>8} {'Sharpe':>7} {'盈亏比':>8} {'PF':>7} {'TP%':>6} {'Time%':>7} {'Cut%':>6} {'CutAvg':>8} {'MedDay':>7}")
    print(f"  {'-'*110}")

    baseline_sl = avg_dicts(all_sl_results[0.30])
    for sl in sorted(all_sl_results.keys()):
        r = avg_dicts(all_sl_results[sl])
        marker = " << current" if sl == 0.30 else ""
        print(f"  {f'{int(sl*100)}%':>8} {r['win_rate']:>6.1f}% {r['med_return']:>+7.1f}% {r['avg_return']:>+7.1f}% "
              f"{r['sharpe']:>+6.2f} {r['win_loss_ratio']:>7.2f} {r['profit_factor']:>6.2f} "
              f"{r['tp_rate']:>5.0f}% {r['time_rate']:>6.0f}% {r['cut_rate']:>5.0f}% "
              f"{r['cut_avg_ret']:>+7.1f}% {r['med_days']:>6.0f}d{marker}")

    # ---- 2: MAE ----
    print(f"\n  {'='*120}")
    print(f"  [2] MAE (Maximum Adverse Excursion) - Worst Drawdown by Final Outcome")
    print(f"  {'='*120}")
    print(f"  For trades that eventually hit TP, what was their worst drawdown?")
    print(f"  This tells us how wide the SL needs to be to avoid killing good trades.")
    print()

    if all_mae:
        tp_k20 = np.mean([m["tp_killed_at_20"] for m in all_mae])
        tp_k25 = np.mean([m["tp_killed_at_25"] for m in all_mae])
        tp_k30 = np.mean([m["tp_killed_at_30"] for m in all_mae])
        tp_k35 = np.mean([m["tp_killed_at_35"] for m in all_mae])

        print(f"  Of all TP trades, what % would have been killed by a stop-loss at:")
        print(f"    SL=20%: {tp_k20:.0f}% of TP trades killed (false positive)")
        print(f"    SL=25%: {tp_k25:.0f}% of TP trades killed")
        print(f"    SL=30%: {tp_k30:.0f}% of TP trades killed << current")
        print(f"    SL=35%: {tp_k35:.0f}% of TP trades killed")

        # MAE percentiles by outcome
        print(f"\n  MAE Distribution:")
        print(f"  {'Outcome':<15} {'Count':>7} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Max':>8}")
        print(f"  {'-'*70}")
        for label, key in [("TP Trades", "tp_mae"), ("Time Trades", "time_mae"), ("Cut Trades", "cut_mae")]:
            vals = [m[key] for m in all_mae]
            p25 = np.mean([v[0] for v in vals])
            p50 = np.mean([v[1] for v in vals])
            p75 = np.mean([v[2] for v in vals])
            p90 = np.mean([v[3] for v in vals])
            mx = np.mean([v[4] for v in vals])
            cnt = np.mean([m[f"{key.split('_')[0]}_count"] for m in all_mae])
            print(f"  {label:<15} {cnt:>7.0f} {p25:>7.1f}% {p50:>7.1f}% {p75:>7.1f}% {p90:>7.1f}% {mx:>7.1f}%")

    # ---- 3: ATR SL ----
    print(f"\n  {'='*120}")
    print(f"  [3] ATR-Based Stop-Loss (SL = entry - ATR * multiplier, fixed TP=100%)")
    print(f"  {'='*120}")
    print(f"  {'ATR_Mult':>10} {'Win%':>7} {'MedRet':>8} {'AvgRet':>8} {'Sharpe':>7} {'盈亏比':>8} {'TP%':>6} {'Cut%':>6} {'CutAvg':>8} {'MedDay':>7}")
    print(f"  {'-'*105}")

    for mult in sorted(all_atr.keys()):
        r = avg_dicts(all_atr[mult])
        print(f"  {f'{mult:.1f}x':>10} {r['win_rate']:>6.1f}% {r['med_return']:>+7.1f}% {r['avg_return']:>+7.1f}% "
              f"{r['sharpe']:>+6.2f} {r['win_loss_ratio']:>7.2f} "
              f"{r['tp_rate']:>5.0f}% {r['cut_rate']:>5.0f}% {r['cut_avg_ret']:>+7.1f}% {r['med_days']:>6.0f}d")

    # ---- 4: Time Stop ----
    print(f"\n  {'='*120}")
    print(f"  [4] Time Stop Optimization (fixed SL=30%, TP=100%)")
    print(f"  {'='*120}")
    print(f"  {'MaxDays':>9} {'Win%':>7} {'MedRet':>8} {'AvgRet':>8} {'Sharpe':>7} {'盈亏比':>8} {'TP%':>6} {'Time%':>7} {'TimeAvg':>8} {'MedDay':>7}")
    print(f"  {'-'*100}")

    for days in sorted(all_time.keys()):
        r = avg_dicts(all_time[days])
        marker = " << current" if days == 60 else ""
        print(f"  {f'{days}d':>9} {r['win_rate']:>6.1f}% {r['med_return']:>+7.1f}% {r['avg_return']:>+7.1f}% "
              f"{r['sharpe']:>+6.2f} {r['win_loss_ratio']:>7.2f} "
              f"{r['tp_rate']:>5.0f}% {r['time_rate']:>6.0f}% {r['time_avg_ret']:>+7.1f}% {r['med_days']:>6.0f}d{marker}")

    # ---- 5: Indicator SL ----
    print(f"\n  {'='*120}")
    print(f"  [5] Indicator-Based Dynamic Stop-Loss (fixed TP=100%)")
    print(f"  {'='*120}")
    print(f"  {'Strategy':<35} {'Win%':>7} {'MedRet':>8} {'AvgRet':>8} {'Sharpe':>7} {'盈亏比':>8} {'TP%':>6} {'Cut%':>6} {'CutAvg':>8} {'MedDay':>7}")
    print(f"  {'-'*115}")

    for name in all_indicator.keys():
        r = avg_dicts(all_indicator[name])
        desc = all_indicator[name][0].get("desc", name)
        print(f"  {desc:<35} {r['win_rate']:>6.1f}% {r['med_return']:>+7.1f}% {r['avg_return']:>+7.1f}% "
              f"{r['sharpe']:>+6.2f} {r['win_loss_ratio']:>7.2f} "
              f"{r['tp_rate']:>5.0f}% {r['cut_rate']:>5.0f}% {r['cut_avg_ret']:>+7.1f}% {r['med_days']:>6.0f}d")

    # ---- 6: Combined ----
    print(f"\n  {'='*120}")
    print(f"  [6] Combined TP + SL Optimization")
    print(f"  {'='*120}")
    print(f"  {'Strategy':<28} {'Win%':>7} {'MedRet':>8} {'AvgRet':>8} {'Sharpe':>7} {'盈亏比':>8} {'PF':>7} {'TP%':>6} {'Time%':>7} {'Cut%':>6} {'TP_Avg':>8} {'Cut_Avg':>8} {'MedDay':>7}")
    print(f"  {'-'*125}")

    baseline_comb = avg_dicts(all_combined["V3_Baseline"])
    for name in ["V3_Baseline", "V4_DynamicTP_Only", "DynamicTP+SL25%", "DynamicTP+SL35%", "DynamicTP+SL40%", "DynamicTP+DeclineSL"]:
        if name not in all_combined:
            continue
        r = avg_dicts(all_combined[name])
        marker = ""
        if name == "V3_Baseline":
            marker = " << current"
        elif name == "V4_DynamicTP_Only":
            marker = " << TP only"
        print(f"  {name:<28} {r['win_rate']:>6.1f}% {r['med_return']:>+7.1f}% {r['avg_return']:>+7.1f}% "
              f"{r['sharpe']:>+6.2f} {r['win_loss_ratio']:>7.2f} {r['profit_factor']:>6.2f} "
              f"{r['tp_rate']:>5.0f}% {r['time_rate']:>6.0f}% {r['cut_rate']:>5.0f}% "
              f"{r['tp_avg_ret']:>+7.1f}% {r['cut_avg_ret']:>+7.1f}% {r['med_days']:>6.0f}d{marker}")

    # ---- Summary ----
    if all_combined:
        print(f"\n  {'='*120}")
        print(f"  Summary")
        print(f"  {'='*120}")

        bl = avg_dicts(all_combined["V3_Baseline"])
        best_name = None
        best_sr = bl["sharpe"]
        for name in all_combined:
            r = avg_dicts(all_combined[name])
            if r["sharpe"] > best_sr:
                best_sr = r["sharpe"]
                best_name = name

        br = avg_dicts(all_combined[best_name])
        print(f"\n  Best combined strategy: {best_name}")
        print(f"    Sharpe: {bl['sharpe']:.2f} -> {br['sharpe']:.2f} (delta: {br['sharpe']-bl['sharpe']:+.2f})")
        print(f"    Med Return: {bl['med_return']:+.1f}% -> {br['med_return']:+.1f}% (delta: {br['med_return']-bl['med_return']:+.1f}pp)")
        print(f"    盈亏比: {bl['win_loss_ratio']:.2f} -> {br['win_loss_ratio']:.2f} (delta: {br['win_loss_ratio']-bl['win_loss_ratio']:+.2f})")
        print(f"    Win Rate: {bl['win_rate']:.1f}% -> {br['win_rate']:.1f}%")
        print(f"    Cut Rate: {bl['cut_rate']:.0f}% -> {br['cut_rate']:.0f}%")

    print("\n" + "=" * 130)


if __name__ == "__main__":
    main()
