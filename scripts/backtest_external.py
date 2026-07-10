"""
backtest_external.py -- 外部策略回测验证

支持的策略:
  S1: 双均线交叉 (MA20/60)
  S2: RSI(2) 均值回归 (Larry Connors)
  S3: 布林带均值回归
  S4: 52周新高突破 (动量)
  S5: 缩量回调反弹
  S6: 短期反转 (5日)

用法:
  python scripts/backtest_external.py --strategy S2 --samples 200
  python scripts/backtest_external.py --all --samples 200 --seeds 42,123,456
"""
import argparse, sys, os, random
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


def get_price_data(code, db):
    """Get adjusted price arrays for a stock."""
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),)
    if len(rows) < 250: return None
    dates = [r["trade_date"] for r in rows]
    raw_c = np.array([r["close"] for r in rows], dtype=float)
    raw_h = np.array([r["high"] for r in rows], dtype=float)
    raw_l = np.array([r["low"] for r in rows], dtype=float)
    raw_v = np.array([r["volume"] for r in rows], dtype=float)
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),)
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_c, dates, afs)
    return {
        "dates": dates, "close": adj["close"], "high": adj["high"],
        "low": adj["low"], "volume": raw_v, "n": len(rows),
    }


def compute_stats(trades):
    if not trades: return None
    rets = [t["return_pct"] for t in trades]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    total = len(trades)
    sr = np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
    avg_w = np.mean(wins) if wins else 0; avg_l = np.mean(losses) if losses else 0
    wl = abs(avg_w / avg_l) if avg_l else float('inf')
    pf = sum(wins) / abs(sum(losses)) if sum(losses) else float('inf')
    return {
        "total": total, "win_rate": len(wins) / total * 100,
        "med_return": np.median(rets), "avg_return": np.mean(rets),
        "sharpe": sr, "wl_ratio": wl, "profit_factor": pf,
        "avg_win": avg_w, "avg_loss": avg_l,
        "max_gain": max(rets), "max_loss": min(rets),
        "avg_days": np.mean([t["hold_days"] for t in trades]),
        "p25_ret": np.percentile(rets, 25), "p75_ret": np.percentile(rets, 75),
    }


# ══════════════════════════════════════════════════════════════
# S1: 双均线交叉 MA20/60
# ══════════════════════════════════════════════════════════════
def backtest_s1(code, db):
    data = get_price_data(code, db)
    if data is None: return []
    close, n = data["close"], data["n"]
    ma20 = indicators.sma(close, 20); ma60 = indicators.sma(close, 60)
    trades = []
    in_position = False; entry_idx = 0; entry_price = 0
    for i in range(250, n - 5):
        if np.isnan(ma20[i]) or np.isnan(ma60[i]): continue
        # Buy: MA20 crosses above MA60 + price above MA60
        if not in_position and ma20[i] > ma60[i] and ma20[i-1] <= ma60[i-1] and close[i] > ma60[i]:
            entry_idx = i; entry_price = close[i]; in_position = True
        elif in_position:
            hold = i - entry_idx
            # Sell: MA20 crosses below MA60, or hold > 60 days
            if (ma20[i] < ma60[i] and ma20[i-1] >= ma60[i-1]) or hold >= 60:
                ret = (close[i] - entry_price) / entry_price * 100
                reason = "卖出信号" if hold < 60 else "时间到"
                trades.append({"return_pct": ret, "hold_days": hold, "exit_reason": reason})
                in_position = False
    return trades


# ══════════════════════════════════════════════════════════════
# S2: RSI(2) 均值回归 (Larry Connors)
# ══════════════════════════════════════════════════════════════
def backtest_s2(code, db):
    data = get_price_data(code, db)
    if data is None: return []
    close, n = data["close"], data["n"]
    rsi2 = indicators.rsi(close, 2)
    trades = []
    in_position = False; entry_idx = 0; entry_price = 0
    for i in range(250, n - 5):
        if np.isnan(rsi2[i]): continue
        # Buy: RSI(2) < 10 (extreme oversold)
        if not in_position and rsi2[i] < 10:
            entry_idx = i; entry_price = close[i]; in_position = True
        elif in_position:
            hold = i - entry_idx
            # Sell: RSI(2) > 50, or hold > 5 days, or stop loss -5%
            loss_pct = (close[i] - entry_price) / entry_price * 100
            if rsi2[i] > 50 or hold >= 5 or loss_pct < -5:
                ret = (close[i] - entry_price) / entry_price * 100
                reason = "RSI>50" if rsi2[i] > 50 else ("时间到" if hold >= 5 else "止损")
                trades.append({"return_pct": ret, "hold_days": hold, "exit_reason": reason})
                in_position = False
    return trades


# ══════════════════════════════════════════════════════════════
# S3: 布林带均值回归
# ══════════════════════════════════════════════════════════════
def backtest_s3(code, db):
    data = get_price_data(code, db)
    if data is None: return []
    close, n = data["close"], data["n"]
    bb = indicators.bollinger(close, 20, 2.0)
    ma20 = indicators.sma(close, 20)
    trades = []
    in_position = False; entry_idx = 0; entry_price = 0
    for i in range(250, n - 5):
        if np.isnan(bb["lower"][i]): continue
        # Buy: close at or below lower band (second day is better)
        at_lower = close[i] <= bb["lower"][i] * 1.01
        if not in_position and at_lower:
            entry_idx = i; entry_price = close[i]; in_position = True
        elif in_position:
            hold = i - entry_idx
            mid = float(bb["mid"][i]) if not np.isnan(bb["mid"][i]) else close[i]
            # Sell: close >= mid band, or hold > 20 days, or stop loss -10%
            loss_pct = (close[i] - entry_price) / entry_price * 100
            if close[i] >= mid or hold >= 20 or loss_pct < -10:
                ret = (close[i] - entry_price) / entry_price * 100
                reason = "回到中轨" if close[i] >= mid else ("时间到" if hold >= 20 else "止损")
                trades.append({"return_pct": ret, "hold_days": hold, "exit_reason": reason})
                in_position = False
    return trades


# ══════════════════════════════════════════════════════════════
# S4: 52周新高突破 (动量)
# ══════════════════════════════════════════════════════════════
def backtest_s4(code, db):
    data = get_price_data(code, db)
    if data is None: return []
    close, high, n = data["close"], data["high"], data["n"]
    ma10 = indicators.sma(close, 10)
    trades = []
    in_position = False; entry_idx = 0; entry_price = 0
    for i in range(250, n - 5):
        high_250 = np.max(high[max(0, i-250):i])
        # Buy: close >= 250-day high (new high breakout)
        if not in_position and close[i] >= high_250 * 0.995:  # 0.5% tolerance
            entry_idx = i; entry_price = close[i]; in_position = True
        elif in_position:
            hold = i - entry_idx
            # Sell: close <= MA10, or hold > 40 days, or stop loss -10%
            loss_pct = (close[i] - entry_price) / entry_price * 100
            exit_signal = (not np.isnan(ma10[i]) and close[i] <= ma10[i])
            if exit_signal or hold >= 40 or loss_pct < -10:
                ret = (close[i] - entry_price) / entry_price * 100
                reason = "跌破MA10" if exit_signal else ("时间到" if hold >= 40 else "止损")
                trades.append({"return_pct": ret, "hold_days": hold, "exit_reason": reason})
                in_position = False
    return trades


# ══════════════════════════════════════════════════════════════
# S5: 缩量回调反弹 (A股经验)
# ══════════════════════════════════════════════════════════════
def backtest_s5(code, db):
    data = get_price_data(code, db)
    if data is None: return []
    close, high, volume, n = data["close"], data["high"], data["volume"], data["n"]
    vol_ma20 = indicators.sma(volume, 20)
    trades = []
    in_position = False; entry_idx = 0; entry_price = 0; peak_price = 0
    for i in range(250, n - 5):
        if np.isnan(vol_ma20[i]): continue
        # Find local peak in last 40 days
        lookback = min(40, i)
        peak_idx = i - lookback + np.argmax(high[i-lookback:i+1])
        peak = float(high[peak_idx])
        decline = (peak - close[i]) / peak * 100
        vol_shrink = volume[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1
        # Buy: decline 5-15% + volume < 80% of 20d avg
        if not in_position and 5 <= decline <= 15 and vol_shrink < 0.8:
            entry_idx = i; entry_price = close[i]; peak_price = peak; in_position = True
        elif in_position:
            hold = i - entry_idx
            # Sell: back to peak, or hold > 30 days, or decline from peak > 15%
            loss_from_peak = (peak_price - close[i]) / peak_price
            if close[i] >= peak_price or hold >= 30 or loss_from_peak >= 0.15:
                ret = (close[i] - entry_price) / entry_price * 100
                reason = "回到前高" if close[i] >= peak_price else ("时间到" if hold >= 30 else "止损")
                trades.append({"return_pct": ret, "hold_days": hold, "exit_reason": reason})
                in_position = False
    return trades


# ══════════════════════════════════════════════════════════════
# S6: 短期反转 (5日)
# ══════════════════════════════════════════════════════════════
def backtest_s6(code, db):
    data = get_price_data(code, db)
    if data is None: return []
    close, n = data["close"], data["n"]
    trades = []
    for i in range(250, n - 10):
        # 5-day return
        ret_5d = (close[i] - close[i-5]) / close[i-5] * 100
        # Today is a green candle
        is_green = close[i] > close[i-1]
        # Buy: 5d return < -8% + green candle today (bounce starting)
        if ret_5d < -8 and is_green:
            entry_price = close[i]
            # Hold for exactly 5 days or until +5% profit (whichever comes first)
            for j in range(i+1, min(i+6, n)):
                ret = (close[j] - entry_price) / entry_price * 100
                if ret >= 5 or j == min(i+5, n-1):
                    trades.append({"return_pct": ret, "hold_days": j - i, "exit_reason": "止盈" if ret >= 5 else "时间到"})
                    break
    return trades


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

STRATEGIES = {
    "S1": {"name": "双均线交叉 MA20/60", "fn": backtest_s1},
    "S2": {"name": "RSI(2) 均值回归", "fn": backtest_s2},
    "S3": {"name": "布林带均值回归", "fn": backtest_s3},
    "S4": {"name": "52周新高突破", "fn": backtest_s4},
    "S5": {"name": "缩量回调反弹", "fn": backtest_s5},
    "S6": {"name": "短期反转(5日)", "fn": backtest_s6},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None, help="Strategy ID (S1-S6)")
    parser.add_argument("--all", action="store_true", help="Run all strategies")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()

    if not args.all and not args.strategy:
        print("Usage: --strategy S2 or --all"); return

    strat_ids = list(STRATEGIES.keys()) if args.all else [args.strategy]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    db = Database()

    all_strat_results = {}

    for sid in strat_ids:
        strat = STRATEGIES[sid]
        print(f"\n{'='*90}")
        print(f"  {sid}: {strat['name']}")
        print(f"{'='*90}")

        all_seed_trades = []
        for seed in seeds:
            random.seed(seed); np.random.seed(seed)
            codes = random.sample(db.get_active_stock_codes(), min(args.samples, len(db.get_active_stock_codes())))
            trades = []
            for i, code in enumerate(codes):
                trades.extend(strat["fn"](code, db))
                if (i+1) % 50 == 0:
                    print(f"  Seed {seed}: {i+1}/{len(codes)}, {len(trades)} trades")
            all_seed_trades.extend(trades)
            print(f"  Seed {seed} done: {len(trades)} trades")

        stats = compute_stats(all_seed_trades)
        if stats:
            all_strat_results[sid] = stats
            print(f"\n  Results ({len(seeds)} seeds x {args.samples} stocks):")
            print(f"    Trades:    {stats['total']}")
            print(f"    Win Rate:  {stats['win_rate']:.1f}%")
            print(f"    Med Return:{stats['med_return']:+.2f}%")
            print(f"    Avg Return:{stats['avg_return']:+.2f}%")
            print(f"    Sharpe:    {stats['sharpe']:.2f}")
            print(f"    盈亏比:     {stats['wl_ratio']:.2f}")
            print(f"    Profit Factor: {stats['profit_factor']:.2f}")
            print(f"    Avg Win:   {stats['avg_win']:+.2f}%")
            print(f"    Avg Loss:  {stats['avg_loss']:+.2f}%")
            print(f"    Max Gain:  {stats['max_gain']:+.1f}%")
            print(f"    Max Loss:  {stats['max_loss']:+.1f}%")
            print(f"    Avg Days:  {stats['avg_days']:.0f}d")
            print(f"    P25/P75:   {stats['p25_ret']:+.1f}% / {stats['p75_ret']:+.1f}%")

    # ═══ Summary Table ═══
    if args.all and all_strat_results:
        print(f"\n{'='*110}")
        print(f"  STRATEGY COMPARISON ({len(seeds)} seeds x {args.samples} stocks)")
        print(f"{'='*110}")
        print(f"  {'Strategy':<28} {'N':>6} {'Win%':>7} {'MedRet':>8} {'AvgRet':>8} {'Sharpe':>7} {'WL':>6} {'PF':>6} {'AvgWin':>8} {'AvgLoss':>8} {'Days':>5}")
        print(f"  {'-'*105}")

        # Include V5 as reference
        print(f"  {'[Ref] V5 假摔反转':<28} {'~1700':>6} {'74.8%':>7} {'+12.2%':>8} {' +6.9%':>8} {'0.60':>7} {'1.27':>6} {'3.91':>6} {'+14.7%':>8} {'-12.1%':>8} {'35d':>5}")
        print(f"  {'-'*105}")

        for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
            if sid not in all_strat_results: continue
            s = all_strat_results[sid]
            print(f"  {sid+' '+STRATEGIES[sid]['name']:<28} {s['total']:>6} {s['win_rate']:>6.1f}% {s['med_return']:>+7.1f}% {s['avg_return']:>+7.1f}% {s['sharpe']:>+6.2f} {s['wl_ratio']:>5.2f} {s['profit_factor']:>5.2f} {s['avg_win']:>+7.1f}% {s['avg_loss']:>+7.1f}% {s['avg_days']:>4.0f}d")

        # Grade
        print(f"\n  Sharpe Grade (target ~1.0):")
        for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
            if sid not in all_strat_results: continue
            s = all_strat_results[sid]
            sr = s["sharpe"]
            grade = "优秀" if sr >= 0.6 else ("良好" if sr >= 0.4 else ("及格" if sr >= 0.25 else "不及格"))
            print(f"    {sid}: Sharpe={sr:.2f} [{grade}]")

    print()


if __name__ == "__main__":
    main()
