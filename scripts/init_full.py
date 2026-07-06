"""
全量初始化：行业分类 + 复权因子 + 周K/月K聚合
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.database import Database
from src.collectors.baostock import BaostockCollector
from src.scheduler.runner import aggregate_weekly_kline, aggregate_monthly_kline


def main():
    db = Database()
    db.create_tables()

    collector = BaostockCollector(db)
    collector.login()

    try:
        # 1. 行业分类（快速）
        print(">>> 拉取行业分类...")
        collector.fetch_industry()

        # 2. 复权因子（慢，每只股票一个请求）
        print(">>> 拉取复权因子...")
        collector.fetch_all_adjust_factors()

    finally:
        collector.logout()

    # 3. 周K/月K聚合（从 daily_kline）
    print(">>> 聚合周K...")
    aggregate_weekly_kline(db)

    print(">>> 聚合月K...")
    aggregate_monthly_kline(db)

    print("=== 全量初始化完成 ===")


if __name__ == "__main__":
    main()
