"""
Stock V0.1 — 交易日历工具
"""

from datetime import datetime, time

from src.config import (
    SNAPSHOT_MARKET_OPEN,
    SNAPSHOT_MARKET_CLOSE_AM,
    SNAPSHOT_MARKET_OPEN_PM,
    SNAPSHOT_MARKET_CLOSE_PM,
)
from src.storage.database import Database


class TradingCalendar:
    """交易日历与交易时段管理"""

    def __init__(self, db: Database):
        self.db = db

    def is_trading_day(self, date_str: str = None) -> bool:
        """判断是否为交易日"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        row = self.db.fetchone("SELECT is_trading FROM trade_calendar WHERE trade_date = ?", (date_str,))
        return row is not None and row["is_trading"] == 1

    def is_trading_session(self) -> bool:
        """判断当前是否在交易时段内 (含 11:30 和 15:00 尾盘时刻)"""
        now = datetime.now()
        t = now.time()

        morning_open = time.fromisoformat(SNAPSHOT_MARKET_OPEN)
        morning_close = time.fromisoformat(SNAPSHOT_MARKET_CLOSE_AM)
        afternoon_open = time.fromisoformat(SNAPSHOT_MARKET_OPEN_PM)
        afternoon_close = time.fromisoformat(SNAPSHOT_MARKET_CLOSE_PM)

        in_morning = morning_open <= t <= morning_close
        in_afternoon = afternoon_open <= t <= afternoon_close
        return in_morning or in_afternoon

    def next_open_time(self) -> datetime | None:
        """获取下一个交易时段开始时间 (用于启动时判断等待多久)"""
        now = datetime.now()
        t = now.time()
        today_str = now.strftime("%Y-%m-%d")

        if not self.is_trading_day(today_str):
            return None

        morning_open = time.fromisoformat(SNAPSHOT_MARKET_OPEN)
        afternoon_open = time.fromisoformat(SNAPSHOT_MARKET_OPEN_PM)

        if t < morning_open:
            return datetime.combine(now.date(), morning_open)
        elif t < afternoon_open:
            return datetime.combine(now.date(), afternoon_open)
        return None

    def wait_until_session(self):
        """等待直到进入交易时段"""
        import time as _time
        next_open = self.next_open_time()
        if next_open and next_open > datetime.now():
            wait_sec = (next_open - datetime.now()).total_seconds()
            if wait_sec > 0:
                _time.sleep(wait_sec)
