"""
初始化数据库：建表 + 股票基础信息 + 交易日历
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scheduler.runner import init_database

if __name__ == "__main__":
    init_database()
    print("数据库初始化完成")
