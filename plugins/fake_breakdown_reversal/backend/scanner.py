"""
scanner.py — 假摔反转 V5 策略扫描器

V6 买入条件 (5+1个前置条件):
  1. 60日涨幅 < 0 (前期充分调整)
  2. 当前处于回调中 (前高→低点 >= 3%)
  3. 60日价格位置 >= 30%
  4. 筹码获利比例 < 50%
  5. ADX(14) < 25 (弱趋势环境, 回调恢复更可靠)
  6. 沪深300 < MA60 (大盘弱势中找企稳个股)

分级 (不变):
  A级: 位置>=50% + 跌<12% + RSI>50 + MA20上方 + 获利<30%
  B级: 位置>=30% + 跌<18%

V6 改进 (vs V5):
  +大盘<MA60 过滤: 夏普0.56→1.14, 胜率+12.5pp, 盈利因子3.39→9.89
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.storage.database import Database
from src.server.plugin_mgr import indicators


# ====================================================================
# 筹码获利比例
# ====================================================================

def calc_profit_ratio(turn_rates, close_prices, current_price, window=120):
    """基于换手率估算获利筹码占比。仅用 current_price 之前的数据(无未来泄露)。

    Returns 0-100 的百分比, 或 None。
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
    total_chips = float(np.sum(chip_left)) + base_chips

    if total_chips < 0.001:
        return None

    profit_chips = float(np.sum(chip_left[closes < current_price]))
    if base_chips > 0:
        base_cost = float(np.median(closes[:max(1, m // 5)]))
        if base_cost < current_price:
            profit_chips += base_chips

    result = round(profit_chips / total_chips * 100, 1)
    return max(0.0, min(100.0, result))  # clamp to [0, 100]


# ====================================================================
# 分级
# ====================================================================

def classify_tier(decline_pct, price_pos, rsi, ma20_dist, profit_ratio):
    """V5 分级。返回 ("A", score) / ("B", score) / None。ADX 过滤在 scan_stock 中完成。"""
    if price_pos is None or price_pos < 30:
        return None
    if rsi is None or rsi <= 40:
        return None
    if profit_ratio is None or profit_ratio >= 50:
        return None

    # A 级
    if (price_pos >= 50 and decline_pct < 12 and rsi > 50
            and (ma20_dist is not None and ma20_dist > 0)
            and profit_ratio < 30):
        return ("A", 90)

    # B 级
    if decline_pct < 18:
        return ("B", 70)

    return None


# ====================================================================
# 单只扫描
# ====================================================================

def scan_stock(code, db):
    """扫描单只股票, 返回 dict 或 None。"""
    rows = db.fetchall(
        "SELECT trade_date, open, high, low, close, volume, turn "
        "FROM daily_kline WHERE code=? ORDER BY trade_date",
        (code,),
    )
    if len(rows) < 180:
        return None

    dates = [r["trade_date"] for r in rows]
    raw_c = np.array([r["close"] for r in rows], dtype=float)
    raw_h = np.array([r["high"] for r in rows], dtype=float)
    raw_l = np.array([r["low"] for r in rows], dtype=float)
    raw_o = np.array([r["open"] for r in rows], dtype=float)
    rturn = np.array([r["turn"] or 0 for r in rows], dtype=float)
    n = len(raw_c)

    # 前复权
    afs = db.fetchall(
        "SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date",
        (code,),
    )
    adj = indicators.forward_adjust(raw_c, raw_h, raw_l, raw_o, dates, afs)
    close, high, low = adj["close"], adj["high"], adj["low"]

    # --- 找最近的峰→谷 ---
    lookback = 40
    today = n - 1
    seg_high = high[today - lookback:today + 1]
    peak_offset = int(np.argmax(seg_high))
    peak_idx = today - lookback + peak_offset
    peak_price = float(high[peak_idx])

    if peak_idx >= today - 1:
        return None

    trough_idx = peak_idx + int(np.argmin(low[peak_idx:today + 1]))
    trough_price = float(low[trough_idx])

    # 低点必须最近, 且没反弹
    if trough_idx < today - 5:
        return None
    if close[today] > close[today - 1] and close[today] > close[today - 2]:
        return None
    if (close[today] - trough_price) / max(trough_price, 0.01) * 100 > 8:
        return None

    decl = round((peak_price - trough_price) / peak_price * 100, 1)
    if decl < 3 or decl > 50:
        return None

    # --- 60日涨幅 ---
    gain_60d = round(
        (close[peak_idx] - close[max(0, peak_idx - 60)]) / close[max(0, peak_idx - 60)] * 100, 1)
    if gain_60d >= 0:
        return None

    # --- 60日价格位置 ---
    rng_hi = float(np.max(high[max(0, today - 60):today + 1]))
    rng_lo = float(np.min(low[max(0, today - 60):today + 1]))
    rng_s = rng_hi - rng_lo
    price_pos = round((close[today] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None

    # --- 技术指标 ---
    rsi_arr = indicators.rsi(close, 14)
    rsi_val = float(rsi_arr[today]) if today < len(rsi_arr) and not np.isnan(rsi_arr[today]) else None

    ma20_arr = indicators.sma(close, 20)
    ma20_dist = (round((close[today] - ma20_arr[today]) / ma20_arr[today] * 100, 1)
                 if today < len(ma20_arr) and not np.isnan(ma20_arr[today]) else None)

    # --- 筹码获利比例 (仅用 today 及之前的数据) ---
    profit_ratio = calc_profit_ratio(rturn[:today + 1], close[:today + 1], close[today])

    # --- V5: ADX filter ---
    adx_dict = indicators.adx(high, low, close, 14)
    adx_val = float(adx_dict["adx"][today]) if today < len(adx_dict["adx"]) and not np.isnan(adx_dict["adx"][today]) else None
    if adx_val is not None and adx_val >= 25:
        return None  # V5: skip signals with strong trend

    # --- V6: 市场环境 (沪深300 < MA60) ---
    idx_row = db.fetchone(
        "SELECT close FROM daily_kline WHERE code='sh.000300' ORDER BY trade_date DESC LIMIT 60")
    if idx_row:
        idx_60 = db.fetchall(
            "SELECT close FROM daily_kline WHERE code='sh.000300' ORDER BY trade_date DESC LIMIT 60")
        if len(idx_60) >= 60:
            idx_closes = [r["close"] for r in idx_60]
            idx_ma60 = sum(idx_closes) / 60
            if idx_closes[0] >= idx_ma60:  # 最新收盘 >= MA60 → 大盘强势, 不交易
                return None

    # --- 分级 ---
    tier_result = classify_tier(decl, price_pos, rsi_val, ma20_dist, profit_ratio)
    if tier_result is None:
        return None
    tier, score = tier_result

    # --- 基本面 ---
    fund_row = db.fetchone(
        "SELECT pe_ttm, pb_mrq, total_mv, turn FROM daily_kline "
        "WHERE code=? AND total_mv IS NOT NULL ORDER BY trade_date DESC LIMIT 1",
        (code,))
    if not fund_row:
        fund_row = db.fetchone(
            "SELECT pe_ttm, pb_mrq, total_mv, turn FROM daily_kline "
            "WHERE code=? ORDER BY trade_date DESC LIMIT 1", (code,))

    pe = round(fund_row["pe_ttm"], 1) if fund_row and fund_row["pe_ttm"] else None
    pb = round(fund_row["pb_mrq"], 2) if fund_row and fund_row["pb_mrq"] else None
    mv = round(fund_row["total_mv"] / 1e8, 1) if fund_row and fund_row["total_mv"] else None
    turn_latest = round(fund_row["turn"], 1) if fund_row and fund_row["turn"] else None

    # --- 名称 ---
    name_row = db.fetchone("SELECT name FROM stock_basic WHERE code=?", (code,))
    name = name_row["name"] if name_row else ""

    # 去前缀
    short_code = code.replace("sh.", "").replace("sz.", "")

    # 警告
    warnings = []
    if pe is not None and pe < 0:
        warnings.append("PE为负")
    if pe is not None and pe > 80:
        warnings.append(f"PE过高({pe:.0f})")
    if turn_latest is not None and turn_latest > 15:
        warnings.append(f"换手高({turn_latest:.0f}%)")
    if gain_60d < -30:
        warnings.append(f"前期跌深({gain_60d:.0f}%)")

    # 盘中标注
    is_intraday = dates[-1] == dates[-1]  # always true for latest bar
    # Check if today is a trading day and time < 15:00
    from datetime import datetime
    now = datetime.now()
    is_trading_hour = (now.hour >= 9 and now.hour < 15) or (now.hour == 15 and now.minute < 5)

    return {
        "code": short_code,
        "full_code": code,
        "name": name,
        "tier": tier,
        "score": score,
        "latest_date": dates[-1],
        "peak_date": dates[peak_idx],
        "peak_price": round(float(peak_price), 2),
        "trough_date": dates[trough_idx],
        "trough_price": round(float(trough_price), 2),
        "decline_pct": decl,
        "gain_60d": gain_60d,
        "price_pos": price_pos,
        "rsi": rsi_val,
        "adx": round(adx_val, 1) if adx_val is not None else None,
        "ma20_dist": ma20_dist,
        "profit_ratio": profit_ratio,
        "pe_ttm": pe,
        "pb_mrq": pb,
        "total_mv": mv,
        "turn": turn_latest,
        "warnings": warnings,
        "is_intraday": is_trading_hour,
        "sixty_low": round(rng_lo, 2),
        "stop_price": round(rng_lo * 0.95, 2),
    }


# ====================================================================
# 全市场扫描
# ====================================================================

def scan_all(max_workers=8, progress_cb=None):
    """全市场并发扫描, 返回排序后的结果列表。"""
    db = Database()
    codes = db.get_active_stock_codes()
    total = len(codes)
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(scan_stock, code, db): code for code in codes}
        for future in as_completed(futures):
            done += 1
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception:
                pass
            if progress_cb:
                progress_cb(done, total)

    # 排序: A 优先, 然后按概率/收益潜力
    results.sort(key=lambda r: (
        0 if r["tier"] == "A" else 1,
        -(r["profit_ratio"] or 50),
        r["decline_pct"],
    ))
    return results
