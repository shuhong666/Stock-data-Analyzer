"""
Stock V0.2 — 插件数据 SDK

插件通过此 SDK 访问核心数据，不直接 import core 模块。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.core.storage.database import Database
from src.server.plugin_mgr import indicators

logger = logging.getLogger(__name__)

# 默认并发数 (I/O 密集型，线程数可适当放大)
_MAX_WORKERS = 8

# 每个请求创建新的 SDK 实例，绑定到当前数据库连接
# 插件代码中: sdk = get_sdk() → sdk.query("SELECT ...")


class DataSDK:
    """插件数据访问 SDK

    封装数据库查询，插件只需调用 SDK 方法，不关心底层存储。
    """

    def __init__(self, db: Database = None):
        self._db = db or Database()

    # ------------------------------------------------------------------
    # K线数据
    # ------------------------------------------------------------------

    def get_kline(self, code: str, start: str = None, end: str = None) -> list[dict]:
        """获取单只股票日K线"""
        if start and end:
            return self._db.fetchall(
                "SELECT * FROM daily_kline WHERE code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                (code, start, end),
            )
        return self._db.fetchall(
            "SELECT * FROM daily_kline WHERE code = ? ORDER BY trade_date", (code,)
        )

    def get_kline_page(self, page: int = 1, page_size: int = 50, code: str = None,
                       start: str = None, end: str = None) -> dict:
        """分页查询K线数据"""
        conditions = ["1=1"]
        params = []
        if code:
            conditions.append("code = ?")
            params.append(code)
        if start:
            conditions.append("trade_date >= ?")
            params.append(start)
        if end:
            conditions.append("trade_date <= ?")
            params.append(end)

        where = " AND ".join(conditions)
        total = self._db.fetchone(f"SELECT COUNT(*) AS cnt FROM daily_kline WHERE {where}", params)["cnt"]
        offset = (page - 1) * page_size
        rows = self._db.fetchall(
            f"SELECT * FROM daily_kline WHERE {where} ORDER BY code, trade_date LIMIT ? OFFSET ?",
            params + [page_size, offset],
        )
        return {"total": total, "page": page, "page_size": page_size, "rows": rows}

    # ------------------------------------------------------------------
    # 股票基础信息
    # ------------------------------------------------------------------

    def get_stocks(self, active_only: bool = True) -> list[dict]:
        """获取股票列表"""
        if active_only:
            return self._db.fetchall("SELECT * FROM stock_basic WHERE delist_date IS NULL ORDER BY code")
        return self._db.fetchall("SELECT * FROM stock_basic ORDER BY code")

    def get_stock_info(self, code: str) -> dict | None:
        """获取单只股票信息"""
        return self._db.fetchone("SELECT * FROM stock_basic WHERE code = ?", (code,))


    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def is_trading_day(self, date_str: str = None) -> bool:
        """判断是否为交易日"""
        from src.core.scheduler.calendar import TradingCalendar
        cal = TradingCalendar(self._db)
        return cal.is_trading_day(date_str)

    # ------------------------------------------------------------------
    # 原始查询 (高级用途)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 技术指标
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 并发计算
    # ------------------------------------------------------------------

    def batch_compute(
        self, codes: list[str], func, *,
        max_workers: int = _MAX_WORKERS,
        on_error: str = "skip",
        **func_kwargs,
    ) -> dict[str, any]:
        """
        对多只股票并发执行同一计算。

        Args:
            codes:      股票代码列表
            func:       计算函数，签名为 func(sdk, code, **kwargs) → any
            max_workers: 最大并发线程数
            on_error:   "skip" 跳过错误 | "raise" 抛出

        Returns:
            {code: result, ...}  — 出错时 on_error="skip" 则不含该 code
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(func, self, code, **func_kwargs): code for code in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result()
                except Exception as e:
                    if on_error == "raise":
                        raise
                    logger.warning(f"batch_compute {code}: {e}")
        return results

    def batch_indicator(
        self, codes: list[str], indicator_name: str, *,
        max_workers: int = _MAX_WORKERS,
        **params,
    ) -> dict[str, any]:
        """
        对多只股票并发计算同一指标。

        indicator_name: "rsi" / "macd" / "kdj" / "profit_ratio" / ...
        params:         传给指标方法的参数 (如 period=14)
        """
        method = getattr(self, indicator_name, None)
        if method is None:
            raise ValueError(f"未知指标: {indicator_name}")

        def _compute(sdk, code, **kw):
            return method(code, **kw)

        return self.batch_compute(codes, _compute, max_workers=max_workers, **params)

    # ------------------------------------------------------------------
    # OHLCV 数组缓存
    # ------------------------------------------------------------------

    def _get_arrays(self, code: str) -> dict:
        """获取单只股票的 OHLCV 数组，线程级缓存避免重复 DB 读取"""
        if not hasattr(self, "_array_cache"):
            self._array_cache: dict[str, dict] = {}
        if code in self._array_cache:
            return self._array_cache[code]

        rows = self._db.fetchall(
            "SELECT open, high, low, close, volume, turn, pct_chg "
            "FROM daily_kline WHERE code = ? ORDER BY trade_date",
            (code,),
        )
        if not rows:
            result = {}
        else:
            result = {
                "open": np.array([r["open"] for r in rows], dtype=float),
                "high": np.array([r["high"] for r in rows], dtype=float),
                "low": np.array([r["low"] for r in rows], dtype=float),
                "close": np.array([r["close"] for r in rows], dtype=float),
                "volume": np.array([r["volume"] for r in rows], dtype=float),
                "turn": np.array([r["turn"] or 0 for r in rows], dtype=float),
            }
        self._array_cache[code] = result
        return result

    def ma(self, code: str, period: int = 20, ma_type: str = "sma") -> np.ndarray:
        """均线"""
        arr = self._get_arrays(code).get("close")
        if arr is None: return np.array([])
        if ma_type == "ema":
            return indicators.ema(arr, period)
        return indicators.sma(arr, period)

    def macd(self, code: str, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """MACD → {dif, dea, histogram}"""
        arr = self._get_arrays(code).get("close")
        if arr is None: return {}
        return indicators.macd(arr, fast, slow, signal)

    def rsi(self, code: str, period: int = 14) -> np.ndarray:
        """RSI"""
        arr = self._get_arrays(code).get("close")
        if arr is None: return np.array([])
        return indicators.rsi(arr, period)

    def kdj(self, code: str, n: int = 9, m1: int = 3, m2: int = 3) -> dict:
        """KDJ → {k, d, j}"""
        a = self._get_arrays(code)
        if not a: return {}
        return indicators.kdj(a["high"], a["low"], a["close"], n, m1, m2)

    def obv(self, code: str) -> np.ndarray:
        """OBV"""
        a = self._get_arrays(code)
        if not a: return np.array([])
        return indicators.obv(a["close"], a["volume"])

    def atr(self, code: str, period: int = 14) -> np.ndarray:
        """ATR"""
        a = self._get_arrays(code)
        if not a: return np.array([])
        return indicators.atr(a["high"], a["low"], a["close"], period)

    def bollinger(self, code: str, period: int = 20, k: float = 2.0) -> dict:
        """Bollinger Bands → {upper, mid, lower, width}"""
        arr = self._get_arrays(code).get("close")
        if arr is None: return {}
        return indicators.bollinger(arr, period, k)

    def adx(self, code: str, period: int = 14) -> dict:
        """ADX → {adx, plus_di, minus_di}"""
        a = self._get_arrays(code)
        if not a: return {}
        return indicators.adx(a["high"], a["low"], a["close"], period)

    def vol_ratio(self, code: str, period: int = 5) -> np.ndarray:
        """量比"""
        arr = self._get_arrays(code).get("volume")
        if arr is None: return np.array([])
        return indicators.vol_ratio(arr, period)

    def vwap(self, code: str) -> np.ndarray:
        """VWAP"""
        a = self._get_arrays(code)
        if not a: return np.array([])
        return indicators.vwap(a["high"], a["low"], a["close"], a["volume"])

    def profit_ratio(self, code: str) -> float:
        """获利比例 (%)"""
        a = self._get_arrays(code)
        if not a: return np.nan
        return indicators.profit_ratio(a["close"], a["high"], a["low"], a["volume"], a["turn"])

    def chip_concentration(self, code: str) -> float:
        """筹码集中度"""
        a = self._get_arrays(code)
        if not a: return np.nan
        return indicators.chip_concentration(a["close"], a["high"], a["low"], a["volume"], a["turn"])

    def avg_cost(self, code: str) -> float:
        """平均成本"""
        a = self._get_arrays(code)
        if not a: return np.nan
        return indicators.avg_cost(a["close"], a["high"], a["low"], a["volume"], a["turn"])

    def chip_full(self, code: str) -> dict:
        """完整筹码分布"""
        a = self._get_arrays(code)
        if not a: return {}
        return indicators.chip_full(a["close"], a["high"], a["low"], a["volume"], a["turn"])

    # ------------------------------------------------------------------
    # 原始查询
    # ------------------------------------------------------------------

    def query(self, sql: str, params: list = None) -> list[dict]:
        """执行只读查询"""
        return self._db.fetchall(sql, params or [])

    def query_one(self, sql: str, params: list = None) -> dict | None:
        """执行只读查询（单条）"""
        return self._db.fetchone(sql, params or [])


# 全局单例 (开发环境下每次请求可创建新实例)
_sdk_instance: DataSDK | None = None


def get_sdk() -> DataSDK:
    global _sdk_instance
    if _sdk_instance is None:
        _sdk_instance = DataSDK()
    return _sdk_instance
