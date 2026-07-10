"""
find_pullbacks.py — 历史上涨回调筛选器 (v3)

两层算法 + 动态阈值:
  Layer 1 — 大趋势检测 (动态 trailing-stop):
    回撤容忍度 = min(trend_base + 前期涨幅 * trend_ratio, trend_max)
    涨得越多 → 容忍越深的回调 → 趋势不容易被震出
  Layer 2 — 回调检测 (mini zigzag):
    在大趋势内，回撤超过 pb_threshold% 标记为回调

用法:
  python scripts/find_pullbacks.py --code sh.601138
  python scripts/find_pullbacks.py --code sz.002491 --trend-base 10 --trend-ratio 0.3
  python scripts/find_pullbacks.py --all
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.storage.database import Database

logger = logging.getLogger("find_pullbacks")

# ====================================================================
# 默认参数
# ====================================================================

DEFAULT_TREND_BASE = 10.0      # 基础回撤容忍度 (%)
DEFAULT_TREND_RATIO = 0.2      # 动态系数: 每1%前期涨幅增加0.2%容忍度
DEFAULT_TREND_MAX = 30.0       # 容忍度上限 (%)
DEFAULT_PB_THRESHOLD = 2.0     # 回调敏感度 (%)
DEFAULT_MIN_TREND_RETURN = 10.0 # 最小趋势涨幅 (%)
DEFAULT_NEW_HIGH_LOOKBACK = 60 # 新高验证: 峰值须超过此前N日内的最高价
DEFAULT_MAX_RETRACE_RATIO = 12.0  # 最大回撤比: 超过此值的回调丢弃
DEFAULT_MAX_WORKERS = 8


# ====================================================================
# Layer 1: 动态 trailing-stop 大趋势检测
# ====================================================================

def find_major_uptrends(
    dates: list[str],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    trend_base: float,
    trend_ratio: float,
    trend_max: float,
    min_return: float,
    new_high_lookback: int = 60,
) -> list[dict]:
    """
    Dynamic trailing-stop: tolerance = min(base + prior_gain * ratio, max)

    From each local low, track highest close. Trend ends when
    drawdown from highest_close exceeds the dynamic threshold.
    """
    n = len(close)
    if n < 10:
        return []

    lookback = 5
    local_lows = []
    for i in range(lookback, n - lookback):
        if low[i] == min(low[i - lookback:i + lookback + 1]):
            local_lows.append(i)

    if not local_lows:
        return []

    trends = []
    used_until = 0

    for start_idx in local_lows:
        if start_idx < used_until:
            continue

        start_price = close[start_idx]
        highest_close = start_price
        highest_idx = start_idx
        trend_alive = True

        for j in range(start_idx + 1, n):
            if close[j] > highest_close:
                highest_close = close[j]
                highest_idx = j

            # Dynamic threshold based on prior gain
            prior_gain = (highest_close - start_price) / start_price * 100
            dyn_threshold = min(trend_base + prior_gain * trend_ratio, trend_max)
            drop = (highest_close - close[j]) / highest_close * 100

            if drop >= dyn_threshold:
                trend_alive = False
                used_until = j
                break

        # Advance used_until to prevent overlapping trends
        if trend_alive:
            used_until = max(used_until, highest_idx)
        # else: used_until already set at break point

        if highest_idx <= start_idx:
            continue

        ret = (highest_close - start_price) / start_price * 100
        if ret < min_return:
            continue

        # Filter: peak must make a new high vs. the preceding lookback window.
        # This discards dead-cat bounces within larger downtrends.
        lookback_start = max(0, start_idx - new_high_lookback)
        prior_max = float(np.max(close[lookback_start:start_idx])) if start_idx > 0 else 0
        if highest_close <= prior_max:
            continue

        trends.append({
            "start_date": dates[start_idx],
            "start_price": round(float(start_price), 2),
            "peak_date": dates[highest_idx],
            "peak_price": round(float(highest_close), 2),
            "end_date": dates[used_until] if not trend_alive else dates[-1],
            "return_pct": round(ret, 1),
            "duration_days": highest_idx - start_idx,
            "start_idx": start_idx,
            "peak_idx": highest_idx,
            "pullbacks": [],
        })

    return trends


# ====================================================================
# Layer 2: 趋势内回调检测
# ====================================================================

def find_pullbacks_in_trend(
    dates: list[str],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    trend_start_idx: int,
    trend_peak_idx: int,
    pb_threshold: float,
) -> list[dict]:
    """
    Within an uptrend, find all minor pullbacks.
    Tracks depth, duration, recovery time, and volume contraction.
    """
    pullbacks = []
    i = trend_start_idx
    running_high = close[trend_start_idx]
    running_high_idx = trend_start_idx

    while i <= trend_peak_idx:
        if close[i] > running_high:
            running_high = close[i]
            running_high_idx = i
            i += 1
            continue

        drawdown = (running_high - low[i]) / running_high * 100
        if drawdown < pb_threshold:
            i += 1
            continue

        # Enter pullback
        pb_start_idx = running_high_idx
        pb_high_price = running_high
        pb_low_price = low[i]
        pb_low_idx = i
        recovered_idx = None  # bar where price recovers back to pb_high_price

        for j in range(i, trend_peak_idx + 1):
            if low[j] < pb_low_price:
                pb_low_price = low[j]
                pb_low_idx = j

            if close[j] >= pb_high_price:
                recovered_idx = j
                i = j + 1
                running_high = max(pb_high_price, close[j])
                running_high_idx = j if close[j] >= pb_high_price else running_high_idx
                break

            if close[j] > running_high:
                running_high = close[j]
                running_high_idx = j
                i = j + 1
                break
        else:
            i = trend_peak_idx + 1

        depth = (pb_high_price - pb_low_price) / pb_high_price * 100
        pb_days = pb_low_idx - pb_start_idx
        if depth >= pb_threshold and pb_days >= 1:
            # Volume contraction: PB period avg vol vs pre-PB avg vol
            pre_pb_vol = float(np.mean(volume[max(trend_start_idx, pb_start_idx - 5):pb_start_idx])) if pb_start_idx > trend_start_idx else 0
            pb_vol = float(np.mean(volume[pb_start_idx:pb_low_idx + 1]))
            vol_contraction = round((1 - pb_vol / pre_pb_vol) * 100, 1) if pre_pb_vol > 0 and pb_vol > 0 else 0

            recovery = recovered_idx - pb_low_idx if recovered_idx else None
            pullbacks.append({
                "peak_date": dates[pb_start_idx],
                "peak_price": round(float(pb_high_price), 2),
                "trough_date": dates[pb_low_idx],
                "trough_price": round(float(pb_low_price), 2),
                "depth_pct": round(depth, 1),
                "pullback_days": pb_days,
                "recovery_days": recovery,
                "recovery_speed": round(depth / recovery, 1) if recovery and recovery > 0 else None,
                "vol_contraction_pct": vol_contraction,
            })

    return pullbacks


# ====================================================================
# 单只股票分析
# ====================================================================

def analyze_stock(
    code: str, db: Database,
    trend_base: float = DEFAULT_TREND_BASE,
    trend_ratio: float = DEFAULT_TREND_RATIO,
    trend_max: float = DEFAULT_TREND_MAX,
    pb_threshold: float = DEFAULT_PB_THRESHOLD,
    min_return: float = DEFAULT_MIN_TREND_RETURN,
    new_high_lookback: int = DEFAULT_NEW_HIGH_LOOKBACK,
    max_retrace_ratio: float = DEFAULT_MAX_RETRACE_RATIO,
) -> dict | None:
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume "
        "FROM daily_kline WHERE code = ? ORDER BY trade_date",
        (code,),
    )
    if len(rows) < 30:
        return None

    dates = [r["trade_date"] for r in rows]
    close = np.array([r["close"] for r in rows], dtype=float)
    high = np.array([r["high"] for r in rows], dtype=float)
    low = np.array([r["low"] for r in rows], dtype=float)
    vol = np.array([r["volume"] or 0 for r in rows], dtype=float)

    info = db.fetchone("SELECT name FROM stock_basic WHERE code = ?", (code,))
    name = info["name"] if info else ""

    trends = find_major_uptrends(
        dates, close, high, low,
        trend_base, trend_ratio, trend_max, min_return, new_high_lookback,
    )
    if not trends:
        return None

    for t in trends:
        t["pullbacks"] = find_pullbacks_in_trend(
            dates, close, high, low, vol,
            t["start_idx"], t["peak_idx"], pb_threshold,
        )
        # Derived features
        trend_return = t["return_pct"]
        trend_dur = t["duration_days"]
        for p in t["pullbacks"]:
            p["retracement_ratio"] = round(p["depth_pct"] / trend_return * 100, 1) if trend_return > 0 else None
            p["duration_ratio"] = round(p["pullback_days"] / trend_dur * 100, 1) if trend_dur > 0 else None

        # Filter low-quality pullbacks
        t["pullbacks"] = [
            p for p in t["pullbacks"]
            if p.get("retracement_ratio") is not None and p["retracement_ratio"] <= max_retrace_ratio
        ]

        del t["start_idx"]
        del t["peak_idx"]

    trends_with_pb = [t for t in trends if t["pullbacks"]]
    if not trends_with_pb:
        return None

    return {
        "code": code,
        "name": name,
        "total_bars": len(rows),
        "date_range": [dates[0], dates[-1]],
        "params": {
            "trend_base_pct": trend_base,
            "trend_ratio": trend_ratio,
            "trend_max_pct": trend_max,
            "pb_threshold_pct": pb_threshold,
            "min_trend_return_pct": min_return,
        },
        "trend_segments": trends_with_pb,
    }


# ====================================================================
# 全市场扫描
# ====================================================================

def scan_all(
    db: Database,
    trend_base: float = DEFAULT_TREND_BASE,
    trend_ratio: float = DEFAULT_TREND_RATIO,
    trend_max: float = DEFAULT_TREND_MAX,
    pb_threshold: float = DEFAULT_PB_THRESHOLD,
    min_return: float = DEFAULT_MIN_TREND_RETURN,
    new_high_lookback: int = DEFAULT_NEW_HIGH_LOOKBACK,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress_callback=None,
) -> dict:
    codes = db.get_active_stock_codes()
    total_codes = len(codes)
    logger.info(f"Scanning {total_codes} stocks, base={trend_base}%, ratio={trend_ratio}, max={trend_max}%")

    start_time = time.time()
    results = {}
    stats = {"scanned": 0, "with_pb": 0, "total_trends": 0, "total_pb": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(analyze_stock, code, db, trend_base, trend_ratio,
                        trend_max, pb_threshold, min_return, new_high_lookback): code
            for code in codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                r = future.result()
                stats["scanned"] += 1
                if r is not None:
                    results[code] = r
                    stats["with_pb"] += 1
                    stats["total_trends"] += len(r["trend_segments"])
                    stats["total_pb"] += sum(len(t["pullbacks"]) for t in r["trend_segments"])
                if progress_callback:
                    progress_callback(stats["scanned"], total_codes)
            except Exception as e:
                logger.warning(f"{code}: {e}")

    elapsed = time.time() - start_time
    return {
        "params": {
            "trend_base_pct": trend_base,
            "trend_ratio": trend_ratio,
            "trend_max_pct": trend_max,
            "pb_threshold_pct": pb_threshold,
            "min_trend_return_pct": min_return,
        },
        "summary": {
            "total_scanned": stats["scanned"],
            "with_pullbacks": stats["with_pb"],
            "total_trends": stats["total_trends"],
            "total_pullbacks": stats["total_pb"],
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": results,
    }


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="历史上涨回调筛选器 v3 — 动态阈值 + 两层算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--code", type=str, default=None, help="单只股票代码")
    parser.add_argument("--all", action="store_true", help="全市场扫描")
    parser.add_argument("--trend-base", type=float, default=DEFAULT_TREND_BASE,
                        help=f"基础回撤容忍度 %% (default: {DEFAULT_TREND_BASE})")
    parser.add_argument("--trend-ratio", type=float, default=DEFAULT_TREND_RATIO,
                        help=f"动态系数 (default: {DEFAULT_TREND_RATIO})")
    parser.add_argument("--trend-max", type=float, default=DEFAULT_TREND_MAX,
                        help=f"容忍度上限 %% (default: {DEFAULT_TREND_MAX})")
    parser.add_argument("--pb-threshold", type=float, default=DEFAULT_PB_THRESHOLD,
                        help=f"回调检测敏感度 %% (default: {DEFAULT_PB_THRESHOLD})")
    parser.add_argument("--min-return", type=float, default=DEFAULT_MIN_TREND_RETURN,
                        help=f"最小趋势涨幅 %% (default: {DEFAULT_MIN_TREND_RETURN})")
    parser.add_argument("--new-high-lookback", type=int, default=DEFAULT_NEW_HIGH_LOOKBACK,
                        help=f"新高验证回溯天数 (default: {DEFAULT_NEW_HIGH_LOOKBACK})")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 文件")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"并发线程数 (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--compact", action="store_true", help="紧凑 JSON")

    args = parser.parse_args()

    if not args.code and not args.all:
        parser.error("需要 --code 或 --all")

    db = Database()
    db.create_tables()

    if args.code:
        result = analyze_stock(args.code, db, args.trend_base, args.trend_ratio,
                               args.trend_max, args.pb_threshold, args.min_return,
                               args.new_high_lookback)
        if result is None:
            print(f"Not found: {args.code}")
            return
        output = {"params": result["params"], "result": result}
    else:
        def progress(current, total):
            print(f"\r  Progress: {current}/{total} ({current/total*100:.0f}%)",
                  end="", file=sys.stderr)

        output = scan_all(db, args.trend_base, args.trend_ratio, args.trend_max,
                          args.pb_threshold, args.min_return, args.new_high_lookback,
                          args.max_workers, progress)
        print(file=sys.stderr)

    indent = None if args.compact else 2
    json_str = json.dumps(output, ensure_ascii=False, indent=indent, default=str)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Saved: {args.output} ({len(json_str)} bytes)")
    else:
        print(json_str)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
