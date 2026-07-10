"""
backtest_strategy.py — 假摔反转策略历史回测 V6

策略:
  前置条件: 60日涨幅<0 + 回调跌幅8-50% + 60日位置>=30% + 获利<50% + MA20>-4% + ADX<25
  市场环境 V6: 沪深300 < MA60 时才交易 (大盘弱势中找企稳个股)
  分级: A/B 两档
  卖出:
    止盈: 跌幅<12%→120%前高, 12-18%→110%前高, >18%→90%前高
    止损: 跌幅<12%→-35%峰顶, 12-18%→-30%峰顶, >18%→-20%峰顶
    时间: 60日

用法:
  python scripts/backtest_strategy.py --samples 100
  python scripts/backtest_strategy.py --samples 200 --compare
"""
import argparse, sys, os, random
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def calc_profit_ratio(turn_rates, close_prices, current_price, window=120):
    """估算筹码获利比例。基于换手率推算筹码分布，计算成本<当前价的筹码占比。

    turn_rates: 全历史日换手率数组 (0-100)
    close_prices: 全历史收盘价数组
    current_price: 当前价格
    window: 回溯窗口
    """
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
    """Classify buy signal into A/B tier. Returns (tier, score) or None.

    V4: 分级逻辑翻转
    - 前置条件由调用方检查: position>=30, RSI>40, profit<50%, MA20>-4%
    - A 级: 跌幅12-18% + RSI 40-55 + 位置30-60% + 获利<30%（回测最大赢家区间）
    - B 级: 跌幅8-25% + RSI>40（覆盖剩余合格信号）
    """
    if price_pos is None or price_pos < 30:
        return None
    if rsi is None or rsi <= 40:
        return None
    if ma20_dist is not None and ma20_dist < -4:       # 均线深度偏离 = 毒药
        return None
    if profit_ratio is None or profit_ratio >= 50:     # 获利盘过半 → 抛压未释放
        return None

    # A 级: 跌幅12-18% + RSI 40-55 + 位置30-60% + 获利<30%（最大赢家区间）
    if (12 <= decline_pct <= 18 and 40 <= rsi <= 55
            and 30 <= price_pos <= 60
            and profit_ratio < 30):
        return ("A", 90)
    # B 级: 跌幅8-25%（覆盖 8-12% 和 12-18% 其余）
    if 8 <= decline_pct <= 25:
        return ("B", 70)

    return None


def backtest_stock(code, db, use_adx_filter=True, market_filter=None):
    """Backtest the strategy on a single stock. Returns list of trade records.
    use_adx_filter: if True, apply V5 ADX<25 filter.
    market_filter: dict date->bool, if provided only enter when market_filter[date] is True."""
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
    raw_o = np.array([r["open"] for r in rows], dtype=float)

    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]
    n = len(close)

    trades = []
    # 从第 180 天开始扫描（确保有足够的回溯数据）
    for today in range(180, n - 5):
        # --- 检查是否在回调中 ---
        lookback = 40
        seg_high = high[today - lookback:today + 1]
        peak_offset = np.argmax(seg_high)
        peak_idx = today - lookback + peak_offset
        peak_price = high[peak_idx]

        # 峰必须在足够远的位置
        if peak_idx >= today - 1:
            continue

        trough_idx = peak_idx + np.argmin(low[peak_idx:today + 1])
        trough_price = low[trough_idx]

        # 低点必须在最近 5 天内，且今天没有明显反弹
        if trough_idx < today - 5:
            continue
        if close[today] > close[today - 1] and close[today] > close[today - 2]:
            continue
        if (close[today] - trough_price) / max(trough_price, 0.01) * 100 > 3:
            continue

        decline_pct = (peak_price - trough_price) / peak_price * 100
        if decline_pct < 8 or decline_pct > 50:
            continue

        # --- 60日涨幅 ---
        gain_60d = round((close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)
        if gain_60d >= 0:
            continue

        # --- 60日价格位置 ---
        rng_hi = float(np.max(high[max(0, today - 60):today + 1]))
        rng_lo = float(np.min(low[max(0, today - 60):today + 1]))
        rng_s = rng_hi - rng_lo
        price_pos = round((close[today] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None

        # --- 技术指标 ---
        rsi_arr = indicators.rsi(close, 14)
        rsi_val = float(rsi_arr[today]) if today < len(rsi_arr) and not np.isnan(rsi_arr[today]) else None
        ma20_arr = indicators.sma(close, 20)
        ma20_dist = round((close[today] - ma20_arr[today]) / ma20_arr[today] * 100, 1) if today < len(ma20_arr) and not np.isnan(ma20_arr[today]) else None
        ma60_arr = indicators.sma(close, 60)
        ma60_dist = round((close[today] - ma60_arr[today]) / ma60_arr[today] * 100, 1) if today < len(ma60_arr) and not np.isnan(ma60_arr[today]) else None

        # --- 筹码获利比例 (仅用 today 及之前的数据) ---
        rturn = np.array([r["turn"] or 0 for r in rows], dtype=float)
        profit_ratio = calc_profit_ratio(rturn[:today + 1], close[:today + 1], close[today])

        # --- V5: ADX filter ---
        adx_dict = indicators.adx(high, low, close, 14)
        adx_val = float(adx_dict["adx"][today]) if today < len(adx_dict["adx"]) and not np.isnan(adx_dict["adx"][today]) else None
        if use_adx_filter and adx_val is not None and adx_val >= 25:
            continue  # V5: skip signals with strong trend (ADX >= 25)

        # --- V6: 市场环境过滤 (大盘弱势才交易) ---
        if market_filter is not None:
            entry_date = dates[today]
            if not market_filter.get(entry_date, True):
                continue  # 大盘 >= MA60, 跳过

        # --- 分级 ---
        tier_result = classify_tier(decline_pct, price_pos, rsi_val, ma20_dist, profit_ratio)
        if tier_result is None:
            continue

        tier, score = tier_result

        # --- 买入 ---
        entry_date = dates[today]
        entry_price = close[today]
        entry_idx = today

        # --- V4 动态止盈止损 (基于跌幅) ---
        # TP: 跌幅<12%→120%, 12-18%→110%, >18%→90%
        # SL: 跌幅<12%→35%,  12-18%→30%,  >18%→20%
        if decline_pct < 12:
            tp_ratio, sl_pct = 1.20, 0.35
        elif decline_pct < 18:
            tp_ratio, sl_pct = 1.10, 0.30
        else:
            tp_ratio, sl_pct = 0.90, 0.20

        tp_price = entry_price + (peak_price - entry_price) * tp_ratio

        exit_idx = None
        exit_reason = None

        for j in range(today + 1, n):
            # 止盈: 动态目标
            if close[j] >= tp_price:
                exit_idx = j
                exit_reason = "止盈"
                break
            # 时间止损: 60 个交易日
            if j - today >= 60:
                exit_idx = j
                exit_reason = "时间到"
                break
            # 动态止损: 从峰顶跌幅 >= sl_pct
            if (peak_price - close[j]) / peak_price >= sl_pct:
                exit_idx = j
                exit_reason = "破位止损"
                break

        if exit_idx is None:
            continue  # 没有足够的后续数据

        exit_date = dates[exit_idx]
        exit_price = close[exit_idx]
        hold_days = exit_idx - entry_idx
        ret = (exit_price - entry_price) / entry_price * 100

        name_row = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
        name = name_row["name"] if name_row else ""

        trades.append({
            "code": code,
            "name": name,
            "tier": tier,
            "score": score,
            "entry_date": entry_date,
            "entry_price": round(float(entry_price), 2),
            "exit_date": exit_date,
            "exit_price": round(float(exit_price), 2),
            "exit_reason": exit_reason,
            "hold_days": hold_days,
            "return_pct": round(ret, 1),
            "peak_date": dates[peak_idx],
            "peak_price": round(float(peak_price), 2),
            "trough_date": dates[trough_idx],
            "trough_price": round(float(trough_price), 2),
            "decline_pct": decline_pct,
            "gain_60d": gain_60d,
            "price_pos": price_pos,
            "rsi": rsi_val,
            "ma20_dist": ma20_dist,
            "ma60_dist": ma60_dist,
            "profit_ratio": profit_ratio,
            "adx": adx_val,
        })

    return trades


def build_market_filter(db):
    """V6: 构建市场环境过滤器。返回 dict: date -> bool (True=允许交易, 即沪深300<MA60)"""
    idx_rows = db.fetchall(
        "SELECT trade_date, close FROM daily_kline WHERE code='sh.000300' ORDER BY trade_date")
    if not idx_rows:
        return None  # no index data, skip filter
    idx_close = np.array([r["close"] for r in idx_rows], dtype=float)
    idx_ma60 = indicators.sma(idx_close, 60)
    idx_dates = [r["trade_date"] for r in idx_rows]
    market_filter = {}
    for i, d in enumerate(idx_dates):
        if i >= 60 and not np.isnan(idx_ma60[i]):
            market_filter[d] = idx_close[i] < idx_ma60[i]  # True = below MA60 = allow
    return market_filter


def run_backtest(sample_count=100, seed=42, tier_filter=None, use_adx_filter=True, use_market_filter=False):
    """Run backtest on sampled stocks."""
    db = Database()
    random.seed(seed)
    np.random.seed(seed)

    market_filter = build_market_filter(db) if use_market_filter else None

    codes = random.sample(db.get_active_stock_codes(), min(sample_count, len(db.get_active_stock_codes())))
    ver = "V6" if use_market_filter else ("V5" if use_adx_filter else "V4")
    print(f"回测 {len(codes)} 只股票 ({ver})...")

    all_trades = []
    for i, code in enumerate(codes):
        trades = backtest_stock(code, db, use_adx_filter, market_filter)
        all_trades.extend(trades)
        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{len(codes)}, 已产生 {len(all_trades)} 笔交易")

    if tier_filter:
        all_trades = [t for t in all_trades if t["tier"] == tier_filter]

    return all_trades


def print_report(trades):
    """Print comprehensive backtest report."""
    if not trades:
        print("无交易记录")
        return

    total = len(trades)
    wins = [t for t in trades if t["return_pct"] > 0]
    losses = [t for t in trades if t["return_pct"] <= 0]
    stopped_out = [t for t in trades if t["exit_reason"] == "止盈"]
    time_out = [t for t in trades if t["exit_reason"] == "时间到"]
    cut_loss = [t for t in trades if t["exit_reason"] == "破位止损"]

    print()
    print("=" * 90)
    print("  回调买入法 — 历史回测报告")
    print("=" * 90)

    # Overall stats
    rets_all = [t["return_pct"] for t in trades]
    rets_win = [t["return_pct"] for t in wins]
    rets_loss = [t["return_pct"] for t in losses]
    avg_win = np.mean(rets_win) if rets_win else 0
    avg_loss = np.mean(rets_loss) if rets_loss else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    profit_factor = sum(rets_win) / abs(sum(rets_loss)) if sum(rets_loss) != 0 else float('inf')

    print(f"\n  总交易: {total}")
    print(f"  胜率: {len(wins)}/{total} = {len(wins)/total*100:.1f}%")
    print(f"  平均收益: {np.mean(rets_all):+.1f}%")
    print(f"  收益中位数: {np.median(rets_all):+.1f}%")
    print(f"  盈亏比 (avg_win/|avg_loss|): {wl_ratio:.2f}")
    print(f"  盈利因子 (total_profit/|total_loss|): {profit_factor:.2f}")
    print(f"  平均盈利: {avg_win:+.1f}%  平均亏损: {avg_loss:+.1f}%")
    print(f"  最大单笔盈利: {max(rets_all):+.1f}%  最大单笔亏损: {min(rets_all):+.1f}%")
    print(f"  收益标准差: {np.std(rets_all):.1f}%")
    print(f"  平均持仓: {np.mean([t['hold_days'] for t in trades]):.0f} 天  中位持仓: {np.median([t['hold_days'] for t in trades]):.0f} 天")

    if wins:
        print(f"\n  盈利交易: {len(wins)} 笔, 平均 +{np.mean([t['return_pct'] for t in wins]):.1f}%, 平均持仓 {np.mean([t['hold_days'] for t in wins]):.0f}d")
    if losses:
        print(f"  亏损交易: {len(losses)} 笔, 平均 {np.mean([t['return_pct'] for t in losses]):.1f}%, 平均持仓 {np.mean([t['hold_days'] for t in losses]):.0f}d")

    # By exit reason
    print(f"\n  按卖出原因:")
    print(f"    止盈: {len(stopped_out)} 笔 ({len(stopped_out)/total*100:.0f}%), 平均 +{np.mean([t['return_pct'] for t in stopped_out]):.1f}%")
    print(f"    时间到 (60d): {len(time_out)} 笔 ({len(time_out)/total*100:.0f}%), 平均 {np.mean([t['return_pct'] for t in time_out]):.1f}%")
    print(f"    破位止损: {len(cut_loss)} 笔 ({len(cut_loss)/total*100:.0f}%), 平均 {np.mean([t['return_pct'] for t in cut_loss]):.1f}%")

    # By tier
    print(f"\n  {'=' * 60}")
    print(f"  按信号分级:")
    print(f"  {'=' * 60}")
    for tier, label, weight in [("A", "A级 (重仓)", 1.0), ("B", "B级 (半仓)", 0.5), ("C", "C级 (轻仓)", 0.25)]:
        g = [t for t in trades if t["tier"] == tier]
        if not g:
            print(f"\n  {label}: 无交易")
            continue
        win_g = [t for t in g if t["return_pct"] > 0]
        loss_g = [t for t in g if t["return_pct"] <= 0]
        tp_g = [t for t in g if t["exit_reason"] == "止盈"]
        rets = [t["return_pct"] for t in g]
        rets_w = [t["return_pct"] for t in win_g]
        rets_l = [t["return_pct"] for t in loss_g]
        days = [t["hold_days"] for t in g]
        avg_w = np.mean(rets_w) if rets_w else 0
        avg_l = np.mean(rets_l) if rets_l else 0
        wl = abs(avg_w / avg_l) if avg_l != 0 else float('inf')
        weighted_rets = [t["return_pct"] * weight for t in g]
        print(f"\n  {label}: {len(g)} 笔")
        print(f"    胜率: {len(win_g)}/{len(g)} = {len(win_g)/len(g)*100:.1f}%")
        print(f"    止盈率: {len(tp_g)}/{len(g)} = {len(tp_g)/len(g)*100:.1f}%")
        print(f"    平均收益: {np.mean(rets):+.1f}%  中位: {np.median(rets):+.1f}%")
        print(f"    盈亏比: {wl:.2f}  均盈 {avg_w:+.1f}% / 均亏 {avg_l:+.1f}%")
        print(f"    最大盈利: {max(rets):+.1f}%  最大亏损: {min(rets):+.1f}%")
        print(f"    平均持仓: {np.mean(days):.0f}d  中位: {np.median(days):.0f}d")

    # By exit reason per tier
    print(f"\n  {'=' * 60}")
    print(f"  各级别卖出原因分布:")
    print(f"  {'=' * 60}")
    for tier, label, _ in [("A", "A级", None), ("B", "B级", None), ("C", "C级", None)]:
        g = [t for t in trades if t["tier"] == tier]
        if not g: continue
        tp = len([t for t in g if t["exit_reason"] == "止盈"])
        to = len([t for t in g if t["exit_reason"] == "时间到"])
        cl = len([t for t in g if t["exit_reason"] == "破位止损"])
        print(f"  {label}: 止盈{tp}({tp/len(g)*100:.0f}%)  时间到{to}({to/len(g)*100:.0f}%)  破位{cl}({cl/len(g)*100:.0f}%)")

    # Sample trades
    print(f"\n  {'=' * 60}")
    print(f"  样本交易 (每级各 5 笔):")
    print(f"  {'=' * 60}")
    for tier in ["A", "B", "C"]:
        g = [t for t in trades if t["tier"] == tier]
        if not g: continue
        print(f"\n  --- {tier}级 ---")
        hdr = f"  {'代码':<12} {'名称':<8} {'入场日':<12} {'入场价':>7} {'出场日':<12} {'出场价':>7} {'收益':>6} {'持仓':>5} {'原因':<8}"
        print(hdr)
        for t in sorted(g, key=lambda x: -x["return_pct"])[:5]:
            print(f"  {t['code']:<12} {t['name']:<8} {t['entry_date']:<12} {t['entry_price']:>7.2f} "
                  f"{t['exit_date']:<12} {t['exit_price']:>7.2f} {t['return_pct']:>+5.1f}% {t['hold_days']:>4}d {t['exit_reason']:<8}")

    # Key metrics summary
    print(f"\n  {'=' * 60}")
    print(f"  策略关键指标:")
    print(f"  {'=' * 60}")
    days_all = [t["hold_days"] for t in trades]
    print(f"  夏普比率 (近似): {np.mean(rets_all)/np.std(rets_all):.2f}" if np.std(rets_all) > 0 else "  夏普比率: N/A")
    print(f"  盈亏比: {wl_ratio:.2f}  盈利因子: {profit_factor:.2f}")
    print(f"  最大单笔回撤: {min(rets_all):+.1f}%")
    print(f"  收益分布: P25={np.percentile(rets_all,25):+.1f}%  P50={np.median(rets_all):+.1f}%  P75={np.percentile(rets_all,75):+.1f}%")
    print(f"  持仓分布: P25={np.percentile(days_all,25):.0f}d  P50={np.median(days_all):.0f}d  P75={np.percentile(days_all,75):.0f}d")

    print()
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="回调买入法历史回测 V5")
    parser.add_argument("--samples", type=int, default=100, help="抽样股票数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tier", type=str, default=None, choices=["A", "B", "C"], help="仅回测指定级别")
    parser.add_argument("--v4", action="store_true", help="使用V4规则(无ADX过滤), 默认V5")
    parser.add_argument("--v6", action="store_true", help="使用V6规则(V5 + 大盘<MA60过滤)")
    parser.add_argument("--compare", action="store_true", help="同时运行 V4/V5/V6 并对比")
    args = parser.parse_args()

    if args.compare:
        print("=" * 90)
        print("  V4 vs V5 vs V6 对比回测")
        print("=" * 90)
        trades_v4 = run_backtest(sample_count=args.samples, seed=args.seed, tier_filter=args.tier, use_adx_filter=False, use_market_filter=False)
        print_report(trades_v4)
        trades_v5 = run_backtest(sample_count=args.samples, seed=args.seed, tier_filter=args.tier, use_adx_filter=True, use_market_filter=False)
        print_report(trades_v5)
        trades_v6 = run_backtest(sample_count=args.samples, seed=args.seed, tier_filter=args.tier, use_adx_filter=True, use_market_filter=True)
        print_report(trades_v6)
        # Print side-by-side summary
        print("\n" + "=" * 90)
        print("  V4 vs V5 vs V6 汇总对比")
        print("=" * 90)
        def quick_stats(trades, label):
            if not trades: return None
            rets = [t["return_pct"] for t in trades]
            wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
            avg_w = np.mean(wins) if wins else 0; avg_l = np.mean(losses) if losses else 0
            wl = abs(avg_w/avg_l) if avg_l else float('inf')
            pf = sum(wins)/abs(sum(losses)) if sum(losses) else float('inf')
            sr = np.mean(rets)/np.std(rets) if np.std(rets) > 0 else 0
            tp = len([t for t in trades if t["exit_reason"]=="止盈"])
            time = len([t for t in trades if t["exit_reason"]=="时间到"])
            cut = len([t for t in trades if t["exit_reason"]=="破位止损"])
            n = len(trades)
            print(f"\n  {label}:")
            print(f"    交易数: {n}  胜率: {len(wins)/n*100:.1f}%  中位收益: {np.median(rets):+.1f}%")
            print(f"    夏普: {sr:.2f}  盈亏比: {wl:.2f}  盈利因子: {pf:.2f}")
            print(f"    均盈: {avg_w:+.1f}%  均亏: {avg_l:+.1f}%")
            print(f"    止盈: {tp}({tp/n*100:.0f}%)  时间到: {time}({time/n*100:.0f}%)  破位: {cut}({cut/n*100:.0f}%)")
            return {"n": n, "win%": len(wins)/n*100, "med": np.median(rets), "sharpe": sr, "wl": wl, "pf": pf}
        s4 = quick_stats(trades_v4, "V4 (无ADX过滤)")
        s5 = quick_stats(trades_v5, "V5 (ADX<25)")
        s6 = quick_stats(trades_v6, "V6 (ADX<25 + 大盘<MA60)")
        print(f"\n  V4→V5: 夏普 {s4['sharpe']:.2f}→{s5['sharpe']:.2f}  V5→V6: 夏普 {s5['sharpe']:.2f}→{s6['sharpe']:.2f}")
        print("=" * 90)
    else:
        use_adx = not args.v4
        use_mkt = args.v6
        trades = run_backtest(sample_count=args.samples, seed=args.seed, tier_filter=args.tier, use_adx_filter=use_adx, use_market_filter=use_mkt)
        print_report(trades)


if __name__ == "__main__":
    main()
