"""
Stock V0.2 — 数据查询 API

提供 K线、股票信息、趋势标注、回调特征等只读接口。
"""

import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Query, BackgroundTasks
from src.server.plugin_mgr.sdk import get_sdk

# 快照进程 PID 文件
_PID_FILE = Path(__file__).resolve().parent.parent.parent.parent / "logs" / "snapshot.pid"

router = APIRouter(prefix="/api/data", tags=["data"])


# ------------------------------------------------------------------
# 股票信息
# ------------------------------------------------------------------

@router.get("/stocks")
def list_stocks(active_only: bool = True):
    """获取股票列表"""
    sdk = get_sdk()
    stocks = sdk.get_stocks(active_only=active_only)
    return {"total": len(stocks), "rows": stocks}


@router.get("/stocks/{code}")
def get_stock(code: str):
    """获取单只股票信息"""
    sdk = get_sdk()
    info = sdk.get_stock_info(code)
    if not info:
        return {"error": f"未找到 {code}"}
    return info


# ------------------------------------------------------------------
# K线数据
# ------------------------------------------------------------------

@router.get("/kline/{code}")
def get_kline(code: str, start: str = None, end: str = None,
              page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=1000)):
    """分页查询指定股票的日K线"""
    sdk = get_sdk()
    return sdk.get_kline_page(code=code, start=start, end=end, page=page, page_size=page_size)


@router.get("/kline")
def list_kline(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=1000),
               code: str = None, start: str = None, end: str = None):
    """分页查询全部K线"""
    sdk = get_sdk()
    return sdk.get_kline_page(code=code, start=start, end=end, page=page, page_size=page_size)


# ------------------------------------------------------------------
# 交易日历
# ------------------------------------------------------------------

@router.get("/calendar")
def check_trading_day(date: str = None):
    """检查是否为交易日"""
    sdk = get_sdk()
    return {"date": date, "is_trading_day": sdk.is_trading_day(date)}


# ------------------------------------------------------------------
# 数据覆盖统计
# ------------------------------------------------------------------

@router.get("/coverage")
def get_coverage():
    """获取每只股票的 K线 覆盖情况"""
    sdk = get_sdk()
    rows = sdk.query("""
        SELECT
            d.code,
            s.name,
            MIN(d.trade_date) AS first_date,
            MAX(d.trade_date) AS last_date,
            COUNT(*) AS total_days
        FROM daily_kline d
        JOIN stock_basic s ON s.code = d.code
        WHERE s.delist_date IS NULL
        GROUP BY d.code
        ORDER BY last_date DESC, total_days DESC
    """)
    return {"total": len(rows), "rows": rows}


# ------------------------------------------------------------------
# 数据补全 (Baostock backfill)
# ------------------------------------------------------------------

_backfill_status = {"running": False, "current": "", "progress": "0/0", "log": []}


@router.get("/backfill/status")
def backfill_status():
    """获取补全任务状态"""
    return _backfill_status


@router.post("/backfill/{code}")
def backfill_stock(code: str, background_tasks: BackgroundTasks):
    """补全单只股票的日K线数据"""
    if _backfill_status["running"]:
        return {"error": "已有补全任务在运行"}

    def _run():
        import logging
        logger = logging.getLogger("backfill")
        _backfill_status["running"] = True
        _backfill_status["current"] = code
        _backfill_status["progress"] = "0/1"
        _backfill_status["log"] = []
        try:
            from src.core.collectors.baostock import BaostockCollector
            from src.core.storage.database import Database
            db = Database()
            collector = BaostockCollector(db)
            collector.login()
            collector.backfill_all_daily(codes=[code])
            collector.logout()
            _backfill_status["log"].append(f"[OK] {code} 补全完成")
        except Exception as e:
            _backfill_status["log"].append(f"[ERR] {code}: {e}")
        _backfill_status["running"] = False
        _backfill_status["current"] = ""

    background_tasks.add_task(_run)
    return {"status": "started", "code": code}


@router.post("/backfill")
def backfill_all(background_tasks: BackgroundTasks):
    """补全全部股票的日K线数据"""
    if _backfill_status["running"]:
        return {"error": "已有补全任务在运行"}

    def _run():
        import logging
        logger = logging.getLogger("backfill")
        _backfill_status["running"] = True
        _backfill_status["current"] = "全部"
        _backfill_status["log"] = []
        try:
            from src.core.collectors.baostock import BaostockCollector
            from src.core.storage.database import Database
            db = Database()
            collector = BaostockCollector(db)
            collector.login()
            codes = db.get_active_stock_codes()
            total = len(codes)
            for i, code in enumerate(codes):
                _backfill_status["progress"] = f"{i+1}/{total}"
                _backfill_status["current"] = code
                try:
                    collector.backfill_all_daily(codes=[code])
                except Exception as e:
                    _backfill_status["log"].append(f"[ERR] {code}: {e}")
            collector.logout()
            _backfill_status["log"].append(f"[OK] {total} 只股票补全完成")
        except Exception as e:
            _backfill_status["log"].append(f"[FATAL] {e}")
        _backfill_status["running"] = False
        _backfill_status["current"] = ""

    background_tasks.add_task(_run)
    return {"status": "started"}


# ------------------------------------------------------------------
# 快照采集控制
# ------------------------------------------------------------------

from src.core.scheduler.runner import is_snapshot_daemon_running, stop_snapshot_daemon


@router.get("/snapshot/status")
def snapshot_status():
    """获取快照采集状态"""
    running = is_snapshot_daemon_running()
    return {"running": running, "mode": "daemon" if running else "stopped"}


@router.post("/snapshot/start")
def snapshot_start():
    """启动快照采集（守护模式下随服务器自动启动，此接口保留用于手动控制）"""
    if is_snapshot_daemon_running():
        return {"status": "already_running"}

    # 守护线程由服务器主进程管理，此处提示通过重启服务器启动
    return {
        "status": "not_running",
        "message": "快照守护随服务器自动启动，如未运行请重启服务器"
    }


@router.post("/snapshot/stop")
def snapshot_stop():
    """停止快照采集"""
    if not is_snapshot_daemon_running():
        return {"status": "not_running"}

    ok = stop_snapshot_daemon()
    if ok:
        return {"status": "stopping"}
    return {"status": "error", "message": "无法发送停止信号"}
