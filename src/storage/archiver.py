"""
Stock V0.1 — 快照数据归档器

收盘后自动将超过 5 个交易日的 snapshot_raw 数据导出为 Parquet，
验证完整性后从 SQLite 删除。
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from src.config import SNAPSHOT_RETAIN_DAYS, SNAPSHOT_ARCHIVE_DIR
from src.storage.database import Database

logger = logging.getLogger(__name__)


class SnapshotArchiver:
    """快照归档器"""

    def __init__(self, db: Database):
        self.db = db
        os.makedirs(SNAPSHOT_ARCHIVE_DIR, exist_ok=True)

    def archive(self, before_date: str = None) -> int:
        """将指定日期之前的快照导出为 Parquet 并从 DB 删除

        Args:
            before_date: 导出该日期之前的数据，默认为 (今天 - RETAIN_DAYS)

        Returns:
            归档记录数
        """
        if before_date is None:
            before_date = (datetime.now() - timedelta(days=SNAPSHOT_RETAIN_DAYS)).strftime("%Y-%m-%d")

        # 1. 查询需归档的数据，按 trade_date 分组
        dates = self.db.fetchall(
            "SELECT DISTINCT trade_date FROM snapshot_raw WHERE trade_date < ? ORDER BY trade_date",
            (before_date,),
        )
        if not dates:
            logger.info("无待归档数据")
            return 0

        total_archived = 0

        for row in dates:
            trade_date = row["trade_date"]
            count = self._archive_date(trade_date)
            total_archived += count

        logger.info(f"归档完成: {total_archived} 条 → {len(dates)} 个 Parquet 文件")
        return total_archived

    def _archive_date(self, trade_date: str) -> int:
        """归档单个交易日的数据"""
        # 1. 从 DB 读取该日全部快照
        data = self.db.fetchall(
            "SELECT * FROM snapshot_raw WHERE trade_date = ?",
            (trade_date,),
        )
        if not data:
            return 0

        df = pd.DataFrame(data)
        file_path = os.path.join(SNAPSHOT_ARCHIVE_DIR, f"snapshot_{trade_date}.parquet")

        # 2. 写 Parquet
        try:
            df.to_parquet(file_path, index=False)
        except Exception as e:
            logger.error(f"Parquet 写入失败 {trade_date}: {e}")
            return 0

        # 3. 验证：能读回来且行数一致
        try:
            df_read = pd.read_parquet(file_path)
            if len(df_read) != len(df):
                logger.error(f"Parquet 验证失败 {trade_date}: 行数不匹配 ({len(df_read)} vs {len(df)})")
                return 0
        except Exception as e:
            logger.error(f"Parquet 读取验证失败 {trade_date}: {e}")
            return 0

        # 4. 从 DB 删除
        self.db.execute("DELETE FROM snapshot_raw WHERE trade_date = ?", (trade_date,))
        logger.info(f"归档 {trade_date}: {len(df)} 条 → {file_path}")
        return len(df)
