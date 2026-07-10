"""
analyze_take_profit.py — 止盈策略专项分析

分析维度:
  1. 可变止盈阈值: 测试 70%-110% × 前高
  2. 最大有利偏移 (MFE): 每笔交易期间的最高潜在收益
  3. 移动止盈: 启动阈值 + 追踪距离 组合测试
  4. 时间衰减目标: 持仓越久目标越低
  5. 分级止盈: A/B 级差异化目标
  6. 时间到挽救: 分析"时间到"交易能否通过降低目标获利

用法:
  python scripts/analyze_take_profit.py --samples 200
  python scripts/analyze_take_profit.py --samples 200 --seed 123
"""

import argparse, sys, os, random
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


# ══════════════════════════════════════════════════════════════════════════════
# 筹码获利比例 (复用 backtest_strategy.py 的逻辑)
# ══════════════════════════════════════════════════════════════════════════════

def calc_profit_ratio(turn_rates, close_prices, current_price, window=120):
    """估算筹码获利比例。基于换手率推算筹码分布。"""
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


# ══════════════════════════════════════════════════════════════════════════════
# 入场逻辑 (复用 backtest_strategy.py)
# ══════════════════════════════════════════════════════════════════════════════

def classify_tier(decline_pct, price_pos, rsi, ma20_dist, profit_ratio):
    """Classify buy signal into A/B tier. Same as backtest_strategy.py."""
    if price_pos is None or price_pos < 30:
        return None
    if rsi is None or rsi <= 40:
        return None
    if ma20_dist is not None and ma20_dist < -4:
        return None
    if profit_ratio is None or profit_ratio >= 50:
        return None

    # A 级: 跌幅12-18% + RSI 40-55 + 位置30-60% + 获利<30%
    if (12 <= decline_pct <= 18 and 40 <= rsi <= 55
            and 30 <= price_pos <= 60
            and profit_ratio < 30):
        return ("A", 90)
    # B 级: 跌幅8-25%
    if 8 <= decline_pct <= 25:
        return ("B", 70)
    return None


def find_entries(code, db):
    """Find all valid entry signals for a stock. Returns list of entry dicts."""
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
    for today in range(180, n - 5):
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

        entries.append({
            "today": today,
            "entry_date": dates[today],
            "entry_price": float(close[today]),
            "peak_price": float(peak_price),
            "peak_idx": peak_idx,
            "trough_idx": trough_idx,
            "trough_price": float(trough_price),
            "decline_pct": decline_pct,
            "price_pos": price_pos,
            "rsi": rsi_val,
            "ma20_dist": ma20_dist,
            "profit_ratio": profit_ratio,
            "tier": tier,
            "score": score,
            "close": close,
            "high": high,
            "low": low,
            "n": n,
        })

    return entries


# ══════════════════════════════════════════════════════════════════════════════
# 分析 1: 可变止盈阈值
# ══════════════════════════════════════════════════════════════════════════════

def simulate_exit_fixed_target(entry, recovery_ratio):
    """Simulate exit when price recovers recovery_ratio of the decline.
    recovery_ratio: 0.0-1.0. 1.0 = full recovery to peak.
    target_price = entry_price + (peak_price - entry_price) * recovery_ratio
    """
    close = entry["close"]
    high = entry["high"]
    n = entry["n"]
    today = entry["today"]
    peak_price = entry["peak_price"]
    entry_price = entry["entry_price"]
    recovery_amount = peak_price - entry_price
    target_price = entry_price + recovery_amount * recovery_ratio

    for j in range(today + 1, n):
        # 止盈: 达到恢复目标
        if close[j] >= target_price:
            label = f"恢复{int(recovery_ratio*100)}%"
            return (j, label, float(close[j]))
        # 时间止损
        if j - today >= 60:
            return (j, "时间到", float(close[j]))
        # 破位止损
        if (peak_price - close[j]) / peak_price >= 0.30:
            return (j, "破位止损", float(close[j]))
    return None


def analyze_variable_targets(all_entries):
    """分析1: 可变止盈阈值 (恢复比例: 0.5 = 恢复跌幅的50%)."""
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20]
    results = {}

    for tp in thresholds:
        trades = []
        for entry in all_entries:
            result = simulate_exit_fixed_target(entry, tp)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
                "tier": entry["tier"],
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            days = [t["hold_days"] for t in trades]
            wins = [r for r in rets if r > 0]
            tp_count = len([t for t in trades if "恢复" in t["exit_reason"]])
            time_count = len([t for t in trades if t["exit_reason"] == "时间到"])
            cut_count = len([t for t in trades if t["exit_reason"] == "破位止损"])
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0

            results[tp] = {
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
                "tp_rate": tp_count / len(trades) * 100,
            }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 分析 2: 最大有利偏移 (MFE)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_mfe(all_entries):
    """分析2: 每笔交易的最大潜在恢复."""
    mfe_ratios = []  # recovery ratio: (max_price - entry) / (peak - entry)
    mfe_pcts = []    # absolute gain: (max_price - entry) / entry * 100

    for entry in all_entries:
        close = entry["close"]
        n = entry["n"]
        today = entry["today"]
        peak_price = entry["peak_price"]
        entry_price = entry["entry_price"]
        recovery_amount = peak_price - entry_price  # full recovery would gain this much

        max_price = entry_price
        # 跟踪到出场 (最多 60 天或数据结束)
        end = min(n, today + 61)
        for j in range(today + 1, end):
            if close[j] > max_price:
                max_price = close[j]
            # 如果触发破位止损就停
            if (peak_price - close[j]) / peak_price >= 0.30:
                break

        max_gain = (max_price - entry_price) / entry_price * 100
        mfe_pcts.append(max_gain)

        if recovery_amount > 0:
            mfe_ratio = (max_price - entry_price) / recovery_amount
            mfe_ratios.append(min(mfe_ratio, 3.0))  # cap at 300%

    return {
        "mfe_ratio_p25": np.percentile(mfe_ratios, 25) if mfe_ratios else 0,
        "mfe_ratio_p50": np.percentile(mfe_ratios, 50) if mfe_ratios else 0,
        "mfe_ratio_p75": np.percentile(mfe_ratios, 75) if mfe_ratios else 0,
        "mfe_ratio_p90": np.percentile(mfe_ratios, 90) if mfe_ratios else 0,
        "mfe_ratio_mean": np.mean(mfe_ratios) if mfe_ratios else 0,
        "mfe_pct_p50": np.percentile(mfe_pcts, 50) if mfe_pcts else 0,
        "mfe_pct_p75": np.percentile(mfe_pcts, 75) if mfe_pcts else 0,
        "mfe_pct_mean": np.mean(mfe_pcts) if mfe_pcts else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 分析 3: 移动止盈
# ══════════════════════════════════════════════════════════════════════════════

def simulate_exit_trailing(entry, activate_pct, trail_pct):
    """移动止盈模拟.
    activate_pct: 从入场价涨多少后启动追踪 (e.g. 0.03 = 3%)
    trail_pct: 从最高点回撤多少触发出场 (e.g. 0.05 = 5%)
    """
    close = entry["close"]
    high = entry["high"]
    n = entry["n"]
    today = entry["today"]
    peak_price = entry["peak_price"]
    entry_price = entry["entry_price"]

    trailing_active = False
    high_water = entry_price

    for j in range(today + 1, n):
        # 更新最高水位
        if close[j] > high_water:
            high_water = close[j]

        # 检查是否启动移动止盈
        if not trailing_active:
            if (high_water - entry_price) / entry_price >= activate_pct:
                trailing_active = True
        else:
            # 已启动: 从最高点回撤 trail_pct 就出场
            if (high_water - close[j]) / high_water >= trail_pct:
                return (j, "TrailingTP", float(close[j]))

        # 时间止损
        if j - today >= 60:
            return (j, "时间到", float(close[j]))
        # 破位止损
        if (peak_price - close[j]) / peak_price >= 0.30:
            return (j, "破位止损", float(close[j]))

    return None


def analyze_trailing(all_entries):
    """分析3: 移动止盈 — 测试多组参数."""
    activate_levels = [0.03, 0.05, 0.08, 0.10]
    trail_levels = [0.03, 0.05, 0.08, 0.10]
    results = {}

    for act in activate_levels:
        for trail in trail_levels:
            if trail > act:
                continue  # 追踪距离不能大于启动阈值，否则刚启动就触发
            key = f"激活{int(act*100)}%_追踪{int(trail*100)}%"
            trades = []
            for entry in all_entries:
                result = simulate_exit_trailing(entry, act, trail)
                if result is None:
                    continue
                exit_idx, reason, exit_price = result
                ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
                trades.append({
                    "return_pct": ret,
                    "hold_days": exit_idx - entry["today"],
                    "exit_reason": reason,
                    "tier": entry["tier"],
                })

            if trades:
                rets = [t["return_pct"] for t in trades]
                days = [t["hold_days"] for t in trades]
                wins = [r for r in rets if r > 0]
                sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
                trail_count = len([t for t in trades if t["exit_reason"] == "TrailingTP"])

                results[key] = {
                    "total": len(trades),
                    "win_rate": len(wins) / len(trades) * 100,
                    "avg_return": np.mean(rets),
                    "med_return": np.median(rets),
                    "avg_days": np.mean(days),
                    "med_days": np.median(days),
                    "sharpe": sr,
                    "trail_count": trail_count,
                }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 分析 4: 时间衰减目标
# ══════════════════════════════════════════════════════════════════════════════

def simulate_exit_time_decay(entry, phases):
    """Time-decay take-profit using recovery ratios.
    phases: [(days_threshold, recovery_ratio), ...]
    e.g. [(20, 1.00), (40, 0.90), (60, 0.80)]
    means: 0-20d target 100% recovery, 20-40d 90%, 40-60d 80%
    """
    close = entry["close"]
    n = entry["n"]
    today = entry["today"]
    peak_price = entry["peak_price"]
    entry_price = entry["entry_price"]
    recovery_amount = peak_price - entry_price

    for j in range(today + 1, n):
        hold_days = j - today
        # Determine current phase target
        current_ratio = 1.00
        for days_threshold, ratio in phases:
            if hold_days <= days_threshold:
                current_ratio = ratio
                break
        else:
            current_ratio = phases[-1][1]  # beyond all phases, use last

        target_price = entry_price + recovery_amount * current_ratio

        # TP: reached current target
        if close[j] >= target_price:
            return (j, f"TD-TP({int(current_ratio*100)}%)", float(close[j]))
        # 时间止损
        if hold_days >= 60:
            return (j, "时间到", float(close[j]))
        # 破位止损
        if (peak_price - close[j]) / peak_price >= 0.30:
            return (j, "破位止损", float(close[j]))
    return None


def analyze_time_decay(all_entries):
    """分析4: 测试多组时间衰减方案."""
    schemes = {
        "当前(100%)": [(60, 1.00)],
        "温和衰减": [(20, 1.00), (40, 0.95), (60, 0.85)],
        "激进衰减": [(15, 1.00), (30, 0.90), (60, 0.75)],
        "直接90%": [(60, 0.90)],
        "直接85%": [(60, 0.85)],
        "20天后90%": [(20, 1.00), (60, 0.90)],
        "30天后80%": [(30, 1.00), (60, 0.80)],
    }
    results = {}

    for name, phases in schemes.items():
        trades = []
        for entry in all_entries:
            result = simulate_exit_time_decay(entry, phases)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
                "tier": entry["tier"],
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            days = [t["hold_days"] for t in trades]
            wins = [r for r in rets if r > 0]
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
            tp_count = len([t for t in trades if "TP" in t["exit_reason"] or "恢复" in t["exit_reason"]])
            time_count = len([t for t in trades if t["exit_reason"] == "时间到"])
            cut_count = len([t for t in trades if t["exit_reason"] == "破位止损"])

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
            }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 分析 5: 分级止盈
# ══════════════════════════════════════════════════════════════════════════════

def analyze_tiered(all_entries):
    """分析5: A/B 级使用不同止盈目标 (recovery ratios)."""
    tier_configs = {
        "Current(A100/B100)": {"A": 1.00, "B": 1.00},
        "A105/B95":   {"A": 1.05, "B": 0.95},
        "A110/B90":   {"A": 1.10, "B": 0.90},
        "A110/B85":   {"A": 1.10, "B": 0.85},
        "A100/B90":   {"A": 1.00, "B": 0.90},
        "A100/B85":   {"A": 1.00, "B": 0.85},
        "A105/B90":   {"A": 1.05, "B": 0.90},
        "A120/B90":   {"A": 1.20, "B": 0.90},
    }
    results = {}

    for name, config in tier_configs.items():
        trades = []
        for entry in all_entries:
            target = config.get(entry["tier"], 1.00)
            result = simulate_exit_fixed_target(entry, target)
            if result is None:
                continue
            exit_idx, reason, exit_price = result
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
            trades.append({
                "return_pct": ret,
                "hold_days": exit_idx - entry["today"],
                "exit_reason": reason,
                "tier": entry["tier"],
            })

        if trades:
            rets = [t["return_pct"] for t in trades]
            days = [t["hold_days"] for t in trades]
            wins = [r for r in rets if r > 0]
            sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
            tp_count = len([t for t in trades if "恢复" in t["exit_reason"]])
            time_count = len([t for t in trades if t["exit_reason"] == "时间到"])
            cut_count = len([t for t in trades if t["exit_reason"] == "破位止损"])

            # 按级别细分
            a_trades = [t for t in trades if t["tier"] == "A"]
            b_trades = [t for t in trades if t["tier"] == "B"]

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
                "a_count": len(a_trades),
                "a_avg_ret": np.mean([t["return_pct"] for t in a_trades]) if a_trades else 0,
                "b_count": len(b_trades),
                "b_avg_ret": np.mean([t["return_pct"] for t in b_trades]) if b_trades else 0,
            }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 分析 6: "时间到"挽救分析
# ══════════════════════════════════════════════════════════════════════════════

def analyze_timeout_rescue(all_entries):
    """分析6: 当前"时间到"的交易在不同止盈阈值下的表现 (recovery ratios)."""
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    results = {}

    for ratio in thresholds:
        rescued = 0
        total_timeouts = 0
        rescued_returns = []

        for entry in all_entries:
            close = entry["close"]
            n = entry["n"]
            today = entry["today"]
            peak_price = entry["peak_price"]
            entry_price = entry["entry_price"]
            recovery_amount = peak_price - entry_price
            target_price = entry_price + recovery_amount * ratio

            # 模拟当前策略 (100% 恢复) 看是否会时间到
            baseline_result = simulate_exit_fixed_target(entry, 1.00)
            if baseline_result is None:
                continue
            _, base_reason, _ = baseline_result
            if base_reason != "时间到":
                continue

            total_timeouts += 1

            # 检查是否在 60 天内达到了更低的 target_price
            for j in range(today + 1, min(n, today + 61)):
                if close[j] >= target_price:
                    ret = (close[j] - entry_price) / entry_price * 100
                    rescued += 1
                    rescued_returns.append(ret)
                    break

        results[ratio] = {
            "total_timeouts": total_timeouts,
            "rescued": rescued,
            "rescue_rate": rescued / total_timeouts * 100 if total_timeouts > 0 else 0,
            "avg_rescued_return": np.mean(rescued_returns) if rescued_returns else 0,
            "med_rescued_return": np.median(rescued_returns) if rescued_returns else 0,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 主函数 & 报告
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_var_target_table(results):
    """Print variable target results in a table."""
    print_section("Analysis 1: Variable Take-Profit Thresholds (Recovery Ratio)")
    print(f"  {'Recovery':>10}  {'Trades':>7}  {'WinRate':>8}  {'AvgRet':>8}  {'MedRet':>8}  {'AvgDay':>7}  {'MedDay':>7}  {'TPRate':>7}  {'Sharpe':>7}")
    print(f"  {'-'*90}")
    for tp in sorted(results.keys()):
        r = results[tp]
        print(f"  {f'{int(tp*100)}%':>10}  {r['total']:>7}  {r['win_rate']:>7.1f}%  {r['avg_return']:>+7.1f}%  {r['med_return']:>+7.1f}%  "
              f"{r['avg_days']:>6.0f}d  {r['med_days']:>6.0f}d  {r['tp_rate']:>6.0f}%  {r['sharpe']:>+6.2f}")

    # Find best in each dimension
    best_win = max(results.items(), key=lambda x: x[1]["win_rate"])
    best_med = max(results.items(), key=lambda x: x[1]["med_return"])
    best_sr = max(results.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n  >> Best win rate: {int(best_win[0]*100)}% ({best_win[1]['win_rate']:.1f}%)")
    print(f"  >> Best median return: {int(best_med[0]*100)}% ({best_med[1]['med_return']:+.1f}%)")
    print(f"  >> Best Sharpe: {int(best_sr[0]*100)}% ({best_sr[1]['sharpe']:+.2f})")


def print_mfe_table(mfe):
    """Print MFE analysis."""
    print_section("Analysis 2: Maximum Favorable Excursion (MFE)")
    print(f"  Recovery ratio (relative to full recovery to peak):")
    print(f"    P25: {mfe['mfe_ratio_p25']*100:.0f}%  -- 25% of trades reached {mfe['mfe_ratio_p25']*100:.0f}% recovery")
    print(f"    P50: {mfe['mfe_ratio_p50']*100:.0f}%  -- median max recovery")
    print(f"    P75: {mfe['mfe_ratio_p75']*100:.0f}%  -- 75% of trades reached {mfe['mfe_ratio_p75']*100:.0f}% recovery")
    print(f"    P90: {mfe['mfe_ratio_p90']*100:.0f}%  -- 90% of trades reached {mfe['mfe_ratio_p90']*100:.0f}% recovery")
    print(f"    Mean: {mfe['mfe_ratio_mean']*100:.0f}%")
    print(f"\n  Absolute gain (relative to entry price):")
    print(f"    Median max gain: {mfe['mfe_pct_p50']:+.1f}%")
    print(f"    P75 max gain: {mfe['mfe_pct_p75']:+.1f}%")
    print(f"    Mean max gain: {mfe['mfe_pct_mean']:+.1f}%")


def print_trailing_table(results):
    """Print trailing stop results."""
    print_section("Analysis 3: Trailing Stop")
    print(f"  {'Scheme':<22}  {'Trades':>7}  {'Win%':>7}  {'AvgRet':>8}  {'MedRet':>8}  {'AvgDay':>7}  {'Trail%':>8}  {'Sharpe':>7}")
    print(f"  {'-'*95}")
    # Sort by sharpe
    sorted_items = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for name, r in sorted_items[:15]:  # top 15
        trail_rate = r["trail_count"] / r["total"] * 100
        print(f"  {name:<22}  {r['total']:>7}  {r['win_rate']:>6.1f}%  {r['avg_return']:>+7.1f}%  "
              f"{r['med_return']:>+7.1f}%  {r['avg_days']:>6.0f}d  {trail_rate:>7.0f}%  {r['sharpe']:>+6.2f}")

    if sorted_items:
        best = sorted_items[0]
        print(f"\n  >> Best: {best[0]} (Sharpe {best[1]['sharpe']:+.2f}, MedRet {best[1]['med_return']:+.1f}%)")


def print_time_decay_table(results):
    """Print time-decay results."""
    print_section("Analysis 4: Time-Decay Targets")
    print(f"  {'Scheme':<22}  {'Trades':>7}  {'Win%':>7}  {'AvgRet':>8}  {'MedRet':>8}  {'AvgDay':>7}  {'TP':>6}  {'Time':>6}  {'Cut':>5}  {'Sharpe':>7}")
    print(f"  {'-'*100}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["med_return"], reverse=True):
        print(f"  {name:<22}  {r['total']:>7}  {r['win_rate']:>6.1f}%  {r['avg_return']:>+7.1f}%  "
              f"{r['med_return']:>+7.1f}%  {r['avg_days']:>6.0f}d  {r['tp_count']:>6}  {r['time_count']:>6}  {r['cut_count']:>5}  {r['sharpe']:>+6.2f}")


def print_tiered_table(results):
    """Print tiered take-profit results."""
    print_section("Analysis 5: Tiered Take-Profit")
    print(f"  {'Scheme':<18}  {'Tot':>5}  {'Win%':>7}  {'AvgRet':>8}  {'Med':>7}  {'Sharpe':>7}  {'#A':>5}  {'AvgA':>7}  {'#B':>5}  {'AvgB':>7}")
    print(f"  {'-'*95}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["med_return"], reverse=True):
        print(f"  {name:<18}  {r['total']:>5}  {r['win_rate']:>6.1f}%  {r['avg_return']:>+7.1f}%  "
              f"{r['med_return']:>+6.1f}%  {r['sharpe']:>+6.2f}  "
              f"{r['a_count']:>5}  {r['a_avg_ret']:>+6.1f}%  {r['b_count']:>5}  {r['b_avg_ret']:>+6.1f}%")


def print_rescue_table(results):
    """Print timeout rescue analysis."""
    print_section("Analysis 6: Timeout Rescue (lower target could salvage losses)")
    print(f"  {'Recovery':>10}  {'Timeouts':>10}  {'Rescued':>8}  {'Rescue%':>8}  {'AvgRet':>10}  {'MedRet':>10}")
    print(f"  {'-'*75}")
    for tp in sorted(results.keys()):
        r = results[tp]
        print(f"  {f'{int(tp*100)}%':>10}  {r['total_timeouts']:>10}  {r['rescued']:>8}  "
              f"{r['rescue_rate']:>7.1f}%  {r['avg_rescued_return']:>+9.1f}%  {r['med_rescued_return']:>+9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="止盈策略专项分析")
    parser.add_argument("--samples", type=int, default=200, help="抽样股票数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tier", type=str, default=None, choices=["A", "B"], help="仅分析指定级别")
    args = parser.parse_args()

    db = Database()
    random.seed(args.seed)
    np.random.seed(args.seed)

    codes = random.sample(db.get_active_stock_codes(), min(args.samples, len(db.get_active_stock_codes())))
    print(f"从 {len(codes)} 只股票中寻找入场信号 (seed={args.seed})...")

    all_entries = []
    for i, code in enumerate(codes):
        entries = find_entries(code, db)
        all_entries.extend(entries)
        if (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{len(codes)}, 已发现 {len(all_entries)} 个信号")

    if args.tier:
        all_entries = [e for e in all_entries if e["tier"] == args.tier]

    a_count = len([e for e in all_entries if e["tier"] == "A"])
    b_count = len([e for e in all_entries if e["tier"] == "B"])
    print(f"\n共发现 {len(all_entries)} 个有效入场信号 (A级: {a_count}, B级: {b_count})")

    if len(all_entries) == 0:
        print("没有找到入场信号，请增加样本量或检查数据。")
        return

    # ---- Baseline ----
    print_section("Baseline: Current Strategy (100% recovery = return to peak)")
    var_results = analyze_variable_targets(all_entries)
    base = var_results.get(1.00, {})
    if base:
        print(f"  Trades: {base['total']}  WinRate: {base['win_rate']:.1f}%  AvgRet: {base['avg_return']:+.1f}%  "
              f"Med: {base['med_return']:+.1f}%  Sharpe: {base['sharpe']:+.2f}")
        print(f"  TP: {base['tp_count']}({base['tp_rate']:.0f}%)  "
              f"Time: {base['time_count']}({base['time_count']/base['total']*100:.0f}%)  "
              f"Cut: {base['cut_count']}({base['cut_count']/base['total']*100:.0f}%)")

    # ---- Analysis 1: Variable thresholds ----
    print_var_target_table(var_results)

    # ---- Analysis 2: MFE ----
    mfe = analyze_mfe(all_entries)
    print_mfe_table(mfe)

    # ---- Analysis 3: Trailing stop ----
    trail_results = analyze_trailing(all_entries)
    print_trailing_table(trail_results)

    # ---- Analysis 4: Time-decay ----
    td_results = analyze_time_decay(all_entries)
    print_time_decay_table(td_results)

    # ---- Analysis 5: Tiered take-profit ----
    tiered_results = analyze_tiered(all_entries)
    print_tiered_table(tiered_results)

    # ---- Analysis 6: Timeout rescue ----
    rescue_results = analyze_timeout_rescue(all_entries)
    print_rescue_table(rescue_results)

    # ---- Recommendations ----
    print_section("Recommendations")
    print(f"""
  Based on the above analysis, here are the key findings:

  1. [Variable Targets] MFE median recovery: {mfe['mfe_ratio_p50']*100:.0f}%, P75: {mfe['mfe_ratio_p75']*100:.0f}%.
     If P50 is close to 100%, most trades recover fully -- current 100% target is reasonable.
     If P50 is below 80%, a lower target may improve turnover and overall returns.

  2. [Trailing Stop] Best for signals where the trend continues beyond the peak.
     Compare trailing results vs fixed target -- if Sharpe is higher, trailing adds value.

  3. [Time-Decay] Core trade-off: sacrificing some peak-profit for fewer timeouts.
     Look at how many timeouts get rescued vs how much TP return drops.

  4. [Tiered] A-grade (fast recovery profile) can aim higher; B-grade may benefit from lower targets.

  5. Pick the best 1-2 approaches and validate with --samples 500.
""")

    print("=" * 100)


if __name__ == "__main__":
    main()
