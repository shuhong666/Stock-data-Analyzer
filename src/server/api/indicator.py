"""
Stock V0.2 — 技术指标 REST API
"""

from typing import Optional

from fastapi import APIRouter, Query
from src.server.plugin_mgr.sdk import get_sdk

router = APIRouter(prefix="/api/indicator", tags=["indicator"])


@router.get("/{name}")
def compute_indicator(
    name: str,
    code: str,
    period: Optional[int] = None,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
    k: float = 2.0,
):
    """
    计算单个技术指标。

    路径参数:
      name: ma / macd / rsi / kdj / obv / atr / bollinger /
            adx / vol_ratio / vwap / profit_ratio / chip_concentration / avg_cost / chip_full
    """
    sdk = get_sdk()
    method = getattr(sdk, name, None)
    if method is None:
        return {"error": f"未知指标: {name}"}

    kwargs = {}
    if period is not None:
        kwargs["period"] = period
    if name == "macd":
        kwargs.update(fast=fast, slow=slow, signal=signal)
    elif name == "kdj":
        kwargs.update(n=n, m1=m1, m2=m2)
    elif name == "bollinger":
        kwargs.update(period=period or 20, k=k)

    result = method(code, **kwargs)
    return _serialize(result)


@router.post("/batch/{name}")
def batch_indicator(name: str, codes: list[str], period: Optional[int] = None):
    """批量并发计算指标 → {code: result, ...}"""
    sdk = get_sdk()
    kwargs = {}
    if period is not None:
        kwargs["period"] = period
    results = sdk.batch_indicator(codes, name, **kwargs)
    return {"count": len(results), "results": {c: _serialize(r) for c, r in results.items()}}


def _serialize(val):
    """将 numpy 数组和字典转为 JSON 可序列化"""
    import numpy as np
    if isinstance(val, np.ndarray):
        arr = val.tolist()
        return [None if (isinstance(x, float) and np.isnan(x)) else x for x in arr]
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    if isinstance(val, float) and np.isnan(val):
        return None
    return val
