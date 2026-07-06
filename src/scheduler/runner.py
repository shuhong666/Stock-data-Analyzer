"""
Stock V0.1 — 采集调度器

提供统一入口，管理:
  - 快照采集启动/停止
  - 盘后自动归档
  - 周K/月K 聚合
"""

import logging
from datetime import datetime

from src.config import DB_PATH
from src.storage.database import Database
from src.storage.archiver import SnapshotArchiver
from src.collectors.tencent import TencentSnapshot
from src.scheduler.calendar import TradingCalendar

logger = logging.getLogger(__name__)


def aggregate_weekly_kline(db: Database):
    """从 daily_kline 聚合周K线（存周期首日=周一）

    采用两阶段聚合避免子查询中的 MIN/MAX 冲突：
    1. 先查出每组的首日/末日
    2. 再 JOIN 回 daily_kline 取 OHLCV
    """
    sql = """
        INSERT OR REPLACE INTO weekly_kline (
            code, trade_date, open, high, low, close, preclose,
            volume, amount, data_source, updated_at
        )
        SELECT
            g.code,
            g.week_start,
            d_open.open,
            g.high,
            g.low,
            d_close.close,
            d_open.preclose,
            g.volume,
            g.amount,
            'aggregated',
            datetime('now')
        FROM (
            SELECT
                code,
                MIN(trade_date) AS week_start,
                MAX(trade_date) AS week_end,
                SUM(volume) AS volume,
                SUM(amount) AS amount,
                MAX(high) AS high,
                MIN(low) AS low
            FROM daily_kline
            WHERE trade_date IS NOT NULL
            GROUP BY code, strftime('%Y-%W', trade_date)
        ) g
        LEFT JOIN daily_kline d_open
            ON d_open.code = g.code AND d_open.trade_date = g.week_start
        LEFT JOIN daily_kline d_close
            ON d_close.code = g.code AND d_close.trade_date = g.week_end
    """
    db.execute(sql)
    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM weekly_kline")
    count = cur.fetchone()[0]
    logger.info(f"周K聚合完成，共 {count} 条")


def aggregate_monthly_kline(db: Database):
    """从 daily_kline 聚合月K线（存周期首日=1日）"""
    sql = """
        INSERT OR REPLACE INTO monthly_kline (
            code, trade_date, open, high, low, close, preclose,
            volume, amount, data_source, updated_at
        )
        SELECT
            g.code,
            g.month_start,
            d_open.open,
            g.high,
            g.low,
            d_close.close,
            d_open.preclose,
            g.volume,
            g.amount,
            'aggregated',
            datetime('now')
        FROM (
            SELECT
                code,
                MIN(trade_date) AS month_start,
                MAX(trade_date) AS month_end,
                SUM(volume) AS volume,
                SUM(amount) AS amount,
                MAX(high) AS high,
                MIN(low) AS low
            FROM daily_kline
            WHERE trade_date IS NOT NULL
            GROUP BY code, strftime('%Y-%m', trade_date)
        ) g
        LEFT JOIN daily_kline d_open
            ON d_open.code = g.code AND d_open.trade_date = g.month_start
        LEFT JOIN daily_kline d_close
            ON d_close.code = g.code AND d_close.trade_date = g.month_end
    """
    db.execute(sql)
    logger.info("月K聚合完成")


def run_snapshot(db: Database = None, stock_codes: list[str] = None):
    """启动盘中快照采集"""
    if db is None:
        db = Database()
    db.create_tables()

    snapshot = TencentSnapshot(db)
    try:
        snapshot.run(stock_codes)
    except KeyboardInterrupt:
        logger.info("收到中断信号")
        snapshot.stop()


def run_post_close(db: Database = None):
    """执行收盘后任务：归档 + 周K/月K聚合"""
    if db is None:
        db = Database()
    db.create_tables()

    calendar = TradingCalendar(db)
    if not calendar.is_trading_day():
        logger.info("非交易日，跳过盘后任务")
        return

    # 1. 归档旧快照
    archiver = SnapshotArchiver(db)
    archiver.archive()

    # 2. 聚合周K / 月K
    aggregate_weekly_kline(db)
    aggregate_monthly_kline(db)

    logger.info("盘后任务完成")


def init_database(db: Database = None):
    """初始化数据库：建表 + 股票池 + 交易日历"""
    if db is None:
        db = Database()
    db.create_tables()

    from src.collectors.baostock import BaostockCollector

    collector = BaostockCollector(db)
    collector.login()

    logger.info("初始化股票基础信息...")
    collector.init_stock_basic()

    logger.info("拉取交易日历...")
    collector.fetch_trade_calendar()

    collector.logout()
    logger.info("数据库初始化完成")
