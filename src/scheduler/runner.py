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
    """从 daily_kline 聚合周K线（存周期首日=周一）"""
    sql = """
        INSERT OR REPLACE INTO weekly_kline (
            code, trade_date, open, high, low, close, preclose,
            volume, amount, turn, pct_chg, pe_ttm, pb_mrq,
            ps_ttm, pcf_ttm, total_mv, circ_mv, amplitude,
            vol_ratio, avg_price, limit_up, limit_down, is_st,
            data_source, updated_at
        )
        SELECT
            code,
            date(trade_date, 'weekday 1', '-7 days') AS week_start,
            FIRST_VALUE(open) OVER w AS open,
            MAX(high) OVER w AS high,
            MIN(low) OVER w AS low,
            FIRST_VALUE(close) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
            FIRST_VALUE(preclose) OVER w AS preclose,
            SUM(volume) OVER w AS volume,
            SUM(amount) OVER w AS amount,
            SUM(turn) OVER w AS turn,
            NULL AS pct_chg,
            FIRST_VALUE(pe_ttm) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS pe_ttm,
            FIRST_VALUE(pb_mrq) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS pb_mrq,
            FIRST_VALUE(ps_ttm) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS ps_ttm,
            FIRST_VALUE(pcf_ttm) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS pcf_ttm,
            FIRST_VALUE(total_mv) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total_mv,
            FIRST_VALUE(circ_mv) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS circ_mv,
            FIRST_VALUE(amplitude) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS amplitude,
            FIRST_VALUE(vol_ratio) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS vol_ratio,
            FIRST_VALUE(avg_price) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS avg_price,
            FIRST_VALUE(limit_up) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS limit_up,
            FIRST_VALUE(limit_down) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS limit_down,
            FIRST_VALUE(is_st) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS is_st,
            'aggregated',
            datetime('now')
        FROM daily_kline
        WHERE trade_date IS NOT NULL
        WINDOW w AS (PARTITION BY code, strftime('%Y-%W', trade_date) ORDER BY trade_date)
    """
    # 上面的窗口函数思路在 SQLite 中实现较复杂，
    # 这里使用简化方案：按 code + week 分组聚合
    simple_sql = """
        INSERT OR REPLACE INTO weekly_kline (
            code, trade_date, open, high, low, close, preclose,
            volume, amount, data_source, updated_at
        )
        SELECT
            code,
            MIN(trade_date) AS week_start,
            (SELECT d2.open FROM daily_kline d2 WHERE d2.code = d.code AND d2.trade_date = MIN(d.trade_date)) AS open,
            MAX(high),
            MIN(low),
            (SELECT d3.close FROM daily_kline d3 WHERE d3.code = d.code AND d3.trade_date = MAX(d.trade_date)) AS close,
            (SELECT d4.preclose FROM daily_kline d4 WHERE d4.code = d.code AND d4.trade_date = MIN(d.trade_date)) AS preclose,
            SUM(volume),
            SUM(amount),
            'aggregated',
            datetime('now')
        FROM daily_kline d
        WHERE trade_date IS NOT NULL
        GROUP BY code, strftime('%Y-%W', trade_date)
    """
    db.execute(simple_sql)
    count = len(db.fetchall("SELECT COUNT(*) AS c FROM weekly_kline"))
    logger.info(f"周K聚合完成，共 {count} 条（估）")


def aggregate_monthly_kline(db: Database):
    """从 daily_kline 聚合月K线（存周期首日=1日）"""
    sql = """
        INSERT OR REPLACE INTO monthly_kline (
            code, trade_date, open, high, low, close, preclose,
            volume, amount, data_source, updated_at
        )
        SELECT
            code,
            MIN(trade_date) AS month_start,
            (SELECT d2.open FROM daily_kline d2 WHERE d2.code = d.code AND d2.trade_date = MIN(d.trade_date)) AS open,
            MAX(high),
            MIN(low),
            (SELECT d3.close FROM daily_kline d3 WHERE d3.code = d.code AND d3.trade_date = MAX(d.trade_date)) AS close,
            (SELECT d4.preclose FROM daily_kline d4 WHERE d4.code = d.code AND d4.trade_date = MIN(d.trade_date)) AS preclose,
            SUM(volume),
            SUM(amount),
            'aggregated',
            datetime('now')
        FROM daily_kline d
        WHERE trade_date IS NOT NULL
        GROUP BY code, strftime('%Y-%m', trade_date)
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
