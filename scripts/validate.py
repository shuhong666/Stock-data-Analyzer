"""
数据校验：对比 daily_kline（snapshot 来源）与 Baostock 官方日K
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.database import Database
from src.validate import validate_daily_kline


def main():
    db = Database()
    db.create_tables()
    validate_daily_kline(db)
    print("校验完成，详见日志")


if __name__ == "__main__":
    main()
