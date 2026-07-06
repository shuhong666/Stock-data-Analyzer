"""
启动盘中快照采集（Ctrl+C 停止）
"""

import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LOG_DIR, LOG_FILE_LEVEL, LOG_CONSOLE_LEVEL
from src.scheduler.runner import run_snapshot

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(LOG_DIR, "snapshot.log"),
                encoding="utf-8",
            ),
        ],
    )
    # 文件日志级别提到 WARNING
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.FileHandler):
            h.setLevel(getattr(logging, LOG_FILE_LEVEL))

    run_snapshot()
