"""
Stock V0.2 — 技术指标库

所有方法接收 numpy 数组或 pandas Series，返回 numpy 数组。
插件通过 sdk 调用这些指标，无需重复实现。

约定:
  - close/high/low/open/volume 均为 numpy float 数组，按时间升序排列
  - 返回值长度与输入相同，前 N 个元素为 np.nan（表示计算未就绪）
  - 筹码类指标返回 dict 而非数组
"""

import numpy as np


# ====================================================================
# 均线
# ====================================================================

def sma(values: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    result[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均"""
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result
    alpha = 2 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


# ====================================================================
# MACD
# ====================================================================

def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    返回: dif, dea, histogram (均为 numpy 数组)
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    histogram = 2 * (dif - dea)
    return {"dif": dif, "dea": dea, "histogram": histogram}


# ====================================================================
# RSI
# ====================================================================

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """相对强弱指标"""
    result = np.full(len(close), np.nan)
    if len(close) < period + 1:
        return result

    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period + 1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            result[i] = 100 - 100 / (1 + avg_gain / avg_loss)

    return result


# ====================================================================
# KDJ
# ====================================================================

def kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        n: int = 9, m1: int = 3, m2: int = 3) -> dict:
    """
    返回: k, d, j
    """
    n_len = len(close)
    k_vals = np.full(n_len, np.nan)
    d_vals = np.full(n_len, np.nan)

    if n_len < n:
        return {"k": k_vals, "d": d_vals, "j": np.full(n_len, np.nan)}

    rsv = np.full(n_len, np.nan)
    for i in range(n - 1, n_len):
        hh = np.max(high[i - n + 1 : i + 1])
        ll = np.min(low[i - n + 1 : i + 1])
        if hh != ll:
            rsv[i] = (close[i] - ll) / (hh - ll) * 100
        else:
            rsv[i] = 50.0

    # K, D 初始值用 50
    k_prev, d_prev = 50.0, 50.0
    for i in range(n - 1, n_len):
        if not np.isnan(rsv[i]):
            k_prev = (m1 - 1) / m1 * k_prev + 1 / m1 * rsv[i]
            d_prev = (m2 - 1) / m2 * d_prev + 1 / m2 * k_prev
        k_vals[i] = k_prev
        d_vals[i] = d_prev

    j_vals = 3 * k_vals - 2 * d_vals
    return {"k": k_vals, "d": d_vals, "j": j_vals}


# ====================================================================
# OBV
# ====================================================================

def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """能量潮 (On-Balance Volume)"""
    result = np.zeros(len(close))
    if len(close) < 2:
        return result
    result[0] = 0
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]
    return result


# ====================================================================
# ATR
# ====================================================================

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """平均真实波幅"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    result = sma(tr, period)
    # Wilder's ATR smoothing
    if n > period:
        result[period] = np.mean(tr[:period])
        for i in range(period + 1, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


# ====================================================================
# Bollinger Bands
# ====================================================================

def bollinger(close: np.ndarray, period: int = 20, k: float = 2.0) -> dict:
    """布林带"""
    mid = sma(close, period)
    std = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i - period + 1 : i + 1], ddof=1)
    upper = mid + k * std
    lower = mid - k * std
    # 带宽
    width = (upper - lower) / mid * 100
    return {"upper": upper, "mid": mid, "lower": lower, "width": width}


# ====================================================================
# ADX / DMI
# ====================================================================

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> dict:
    """平均趋向指数"""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder smoothing
    atr14 = np.full(n, np.nan)
    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    adx_vals = np.full(n, np.nan)

    if n > period:
        atr14[period] = np.sum(tr[:period + 1])
        pdi14_sum = np.sum(plus_dm[:period + 1])
        mdi14_sum = np.sum(minus_dm[:period + 1])
        pdi[period] = pdi14_sum / atr14[period] * 100 if atr14[period] > 0 else 0
        mdi[period] = mdi14_sum / atr14[period] * 100 if atr14[period] > 0 else 0

        for i in range(period + 1, n):
            atr14[i] = atr14[i - 1] - atr14[i - 1] / period + tr[i]
            pdi14_sum = pdi14_sum - pdi14_sum / period + plus_dm[i]
            mdi14_sum = mdi14_sum - mdi14_sum / period + minus_dm[i]
            pdi[i] = pdi14_sum / atr14[i] * 100 if atr14[i] > 0 else 0
            mdi[i] = mdi14_sum / atr14[i] * 100 if atr14[i] > 0 else 0

        # ADX = smoothed |+DI - -DI| / (+DI + -DI)
        dx = np.abs(pdi - mdi) / (pdi + mdi) * 100
        adx_vals[period * 2 - 1] = np.mean(dx[period:period * 2])
        for i in range(period * 2, n):
            adx_vals[i] = (adx_vals[i - 1] * (period - 1) + dx[i]) / period

    return {"adx": adx_vals, "plus_di": pdi, "minus_di": mdi}


# ====================================================================
# 量比 (Volume Ratio)
# ====================================================================

def vol_ratio(volume: np.ndarray, period: int = 5) -> np.ndarray:
    """当日成交量 / 过去 N 日均量"""
    avg_vol = sma(volume, period)
    result = np.full(len(volume), np.nan)
    for i in range(period - 1, len(volume)):
        if avg_vol[i] > 0:
            result[i] = volume[i] / avg_vol[i]
    return result


# ====================================================================
# VWAP (成交量加权均价)
# ====================================================================

def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray) -> np.ndarray:
    """日内 VWAP 累计线 (从第 0 个元素开始累加)"""
    typical = (high + low + close) / 3
    cum_pv = np.cumsum(typical * volume)
    cum_vol = np.cumsum(volume)
    result = np.full(len(close), np.nan)
    mask = cum_vol > 0
    result[mask] = cum_pv[mask] / cum_vol[mask]
    return result


# ====================================================================
# 筹码指标 (CYQ 简化版)
# ====================================================================

def _chip_distribution(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, turn: np.ndarray,
    bins: int = 100, decay_days: int = 200,
) -> dict:
    """
    计算筹码分布。

    原理: 每日成交量按价格区间 (low~high) 均匀分配，逐日累加。
    旧筹码按 decay_days 线性衰减。

    返回:
      distribution: (bins,)  每个价格区间的筹码量
      price_grid:   (bins,)  价格区间中心
      profit_ratio: float    当前收盘价以下的筹码占比 (获利比例)
      concentration: float   筹码集中度 (越高越集中)
      avg_cost:      float   加权平均成本
    """
    n = len(close)
    if n == 0:
        return {"distribution": np.array([]), "price_grid": np.array([]),
                "profit_ratio": np.nan, "concentration": np.nan, "avg_cost": np.nan}

    # 价格范围
    price_min = np.min(low)
    price_max = np.max(high)
    if price_max <= price_min:
        price_max = price_min + 0.01
    grid = np.linspace(price_min, price_max, bins + 1)
    price_centers = (grid[:-1] + grid[1:]) / 2
    chip = np.zeros(bins)

    for i in range(n):
        if volume[i] <= 0:
            continue
        p_low, p_high = low[i], high[i]
        if p_high <= p_low:
            p_high = p_low + 0.01

        # 成交量均匀分配到价格区间
        vol_per_bin = np.zeros(bins)
        for j in range(bins):
            bin_low, bin_high = grid[j], grid[j + 1]
            overlap = max(0, min(p_high, bin_high) - max(p_low, bin_low))
            if overlap > 0:
                vol_per_bin[j] = overlap / (p_high - p_low) * volume[i]

        # 衰减 + 累加
        decay = 1.0
        if decay_days > 0:
            age = n - 1 - i
            decay = max(0, 1 - age / decay_days)
        chip += vol_per_bin * decay

    total = np.sum(chip)
    if total <= 0:
        return {"distribution": chip, "price_grid": price_centers,
                "profit_ratio": np.nan, "concentration": np.nan, "avg_cost": np.nan}

    # 获利比例: 当前价以下筹码 / 总筹码
    current_price = close[-1]
    below_mask = price_centers <= current_price
    profit_ratio = np.sum(chip[below_mask]) / total * 100

    # 平均成本
    avg_cost = np.sum(chip * price_centers) / total

    # 集中度: 峰值的相对高度和宽度
    # 集中度 = 峰值筹码占比 / 峰值区间价格跨度
    max_chip = np.max(chip)
    # 找到筹码 > 50% 峰值的区间宽度
    significant = chip > max_chip * 0.5
    if np.any(significant):
        peak_indices = np.where(significant)[0]
        peak_width = price_centers[peak_indices[-1]] - price_centers[peak_indices[0]]
        if peak_width > 0:
            concentration = (max_chip / total) / peak_width * price_centers[-1]
        else:
            concentration = max_chip / total * 100
    else:
        concentration = 0.0

    return {
        "distribution": chip.tolist(),
        "price_grid": price_centers.tolist(),
        "profit_ratio": round(profit_ratio, 2),
        "concentration": round(float(concentration), 2),
        "avg_cost": round(float(avg_cost), 2),
    }


def profit_ratio(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 volume: np.ndarray, turn: np.ndarray) -> float:
    """获利比例: 当前价以下筹码占比 (%)"""
    d = _chip_distribution(close, high, low, volume, turn)
    return d["profit_ratio"]


def chip_concentration(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       volume: np.ndarray, turn: np.ndarray) -> float:
    """筹码集中度"""
    d = _chip_distribution(close, high, low, volume, turn)
    return d["concentration"]


def avg_cost(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, turn: np.ndarray) -> float:
    """平均成本"""
    d = _chip_distribution(close, high, low, volume, turn)
    return d["avg_cost"]


def chip_full(close: np.ndarray, high: np.ndarray, low: np.ndarray,
              volume: np.ndarray, turn: np.ndarray) -> dict:
    """完整的筹码分布数据"""
    return _chip_distribution(close, high, low, volume, turn)

# ====================================================================
# 复权工具
# ====================================================================

def forward_adjust(close, high, low, open_, dates, adj_factors):
    """
    将不复权 OHLC 转为前复权。

    Baostock foreAdjustFactor 是累计值。
    公式: 前复权 = raw_price × 该日期之前最近一个事件的 fore_factor.

    adj_factors: list[dict], 含 trade_date, fore_factor, 按日期升序
    若 fore_factor 为空则用 adj_factor 近似。
    """
    import numpy as np
    if not adj_factors:
        return {"close": close, "high": high, "low": low, "open": open_}

    n = len(close)
    factors = np.ones(n)

    # Build intervals: [event_date, next_event_date) -> fore_factor
    for i, af in enumerate(adj_factors):
        start_d = af["trade_date"]
        end_d = adj_factors[i + 1]["trade_date"] if i + 1 < len(adj_factors) else "2099-12-31"
        ff = af.get("fore_factor")
        if ff is None:
            adj = af.get("adj_factor", 1.0) or 1.0
            ff = 1.0 / adj if adj > 1.0 else adj
        for j, d in enumerate(dates):
            if start_d <= d < end_d:
                factors[j] = ff

    # Before first event: use first event's fore_factor
    first_d = adj_factors[0]["trade_date"]
    first_ff = adj_factors[0].get("fore_factor")
    if first_ff is None:
        adj = adj_factors[0].get("adj_factor", 1.0) or 1.0
        first_ff = 1.0 / adj if adj > 1.0 else adj
    for j, d in enumerate(dates):
        if d < first_d:
            factors[j] = first_ff

    return {
        "close": close * factors,
        "high": high * factors,
        "low": low * factors,
        "open": open_ * factors,
    }