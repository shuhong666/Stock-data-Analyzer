"""
Stock V0.1 — 全局配置常量
"""

import os

# ============================================================
# 数据库
# ============================================================
DB_PATH = os.environ.get("STOCK_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "stock.db"))

# ============================================================
# 快照采集 (Tencent)
# ============================================================
SNAPSHOT_BATCH_SIZE = 400              # 每批股票数
SNAPSHOT_INTERVAL = 30                 # 轮询间隔（秒）
SNAPSHOT_MARKET_OPEN = "09:30"
SNAPSHOT_MARKET_CLOSE_AM = "11:30"
SNAPSHOT_MARKET_OPEN_PM = "13:00"
SNAPSHOT_MARKET_CLOSE_PM = "15:00"

# 腾讯接口
TENCENT_API_URL = "http://qt.gtimg.cn/q="
TENCENT_TIMEOUT = 15                   # 请求超时（秒）

# ============================================================
# 数据保留
# ============================================================
SNAPSHOT_RETAIN_DAYS = 5               # 快照在 DB 中保留交易日数
SNAPSHOT_ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "archive")

# ============================================================
# Baostock
# ============================================================
BAOSTOCK_RETRY_COUNT = 3
BAOSTOCK_RETRY_INTERVAL = 2            # 重试间隔（秒）
BAOSTOCK_REQUEST_GAP = 0.5             # 请求间隔（秒），避免限流
BAOSTOCK_RECONNECT_GAP = 5             # 断点续跑重连等待（秒）
BAOSTOCK_DEFAULT_DAYS = 500            # 首次拉取默认交易日数

# ============================================================
# 数据校验
# ============================================================
VALIDATE_DIFF_THRESHOLD = 0.005        # OHLCV 差异阈值 (0.5%)
VALIDATE_DEFAULT_DAYS = 5              # 默认校验最近 N 个交易日

# ============================================================
# 日志
# ============================================================
LOG_DIR = "logs"
LOG_FILE_LEVEL = "WARNING"
LOG_CONSOLE_LEVEL = "INFO"
LOG_RETENTION_DAYS = 30
