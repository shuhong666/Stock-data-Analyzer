"""
启动盘中快照采集（Ctrl+C 停止）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scheduler.runner import run_snapshot

if __name__ == "__main__":
    run_snapshot()
