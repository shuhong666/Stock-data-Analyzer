"""
Baostock 补全日K数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.database import Database
from src.collectors.baostock import BaostockCollector


def main():
    db = Database()
    db.create_tables()

    collector = BaostockCollector(db)
    collector.login()

    try:
        collector.backfill_all_daily()
    finally:
        collector.logout()

    print("补全完成")


if __name__ == "__main__":
    main()
