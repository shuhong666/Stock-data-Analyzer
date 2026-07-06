"""
Stock V0.1 — Akshare 数据采集器

采集内容:
  - 财务摘要（同花顺）
  - 业绩快报（东方财富）
  - 指数成分股（中证）
均为按需拉取，返回值全字段 JSON 入库。
"""

import json
import logging
from datetime import datetime

from src.storage.database import Database

logger = logging.getLogger(__name__)


class AkshareCollector:
    """Akshare 数据采集器，按需拉取"""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # 代码格式转换
    # ------------------------------------------------------------------

    @staticmethod
    def to_akshare_code(internal_code: str) -> str:
        """sh.600000 → 600000（Akshare 不需要前缀）"""
        return internal_code.replace("sh.", "").replace("sz.", "")

    # ------------------------------------------------------------------
    # 财务摘要（同花顺）
    # ------------------------------------------------------------------

    def fetch_financial_summary(self, code: str) -> int:
        """按需拉取单只股票财务摘要

        Args:
            code: 内部代码格式 sh.600000

        Returns:
            插入/更新的记录数
        """
        import akshare as ak

        ak_code = self.to_akshare_code(code)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 同花顺财务摘要接口
            df = ak.stock_financial_abstract_ths(symbol=ak_code, indicator="按报告期")
        except Exception as e:
            try:
                # 备选：按年度
                df = ak.stock_financial_abstract_ths(symbol=ak_code, indicator="按年度")
            except Exception as e2:
                logger.error(f"财务摘要拉取失败 {code}: {e2}")
                return 0

        if df is None or df.empty:
            logger.info(f"财务摘要无数据: {code}")
            return 0

        rows = []
        for _, row_data in df.iterrows():
            # 报告期处理
            report_date = str(row_data.get("报告期", row_data.iloc[0] if len(row_data) > 0 else ""))
            # 整体 JSON 入库
            data_json = json.dumps(row_data.to_dict(), ensure_ascii=False, default=str)
            rows.append({
                "code": code,
                "report_date": report_date,
                "data_json": data_json,
                "updated_at": now,
            })

        if rows:
            self.db.executemany(
                "INSERT OR REPLACE INTO financial_summary (code, report_date, data_json, updated_at) "
                "VALUES (:code, :report_date, :data_json, :updated_at)",
                rows,
            )

        logger.info(f"财务摘要: {code} → {len(rows)} 条")
        return len(rows)

    # ------------------------------------------------------------------
    # 业绩快报（东方财富）
    # ------------------------------------------------------------------

    def fetch_performance_express(self, code: str) -> int:
        """按需拉取单只股票业绩快报

        Args:
            code: 内部代码格式 sh.600000

        Returns:
            插入/更新的记录数
        """
        import akshare as ak

        ak_code = self.to_akshare_code(code)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            df = ak.stock_yjkb_em(symbol=ak_code)
        except Exception as e:
            logger.error(f"业绩快报拉取失败 {code}: {e}")
            return 0

        if df is None or df.empty:
            logger.info(f"业绩快报无数据: {code}")
            return 0

        rows = []
        for _, row_data in df.iterrows():
            report_date = str(row_data.get("报告期", row_data.iloc[0] if len(row_data) > 0 else ""))
            data_json = json.dumps(row_data.to_dict(), ensure_ascii=False, default=str)
            rows.append({
                "code": code,
                "report_date": report_date,
                "data_json": data_json,
                "updated_at": now,
            })

        if rows:
            self.db.executemany(
                "INSERT OR REPLACE INTO performance_express (code, report_date, data_json, updated_at) "
                "VALUES (:code, :report_date, :data_json, :updated_at)",
                rows,
            )

        logger.info(f"业绩快报: {code} → {len(rows)} 条")
        return len(rows)

    # ------------------------------------------------------------------
    # 指数成分股（中证）
    # ------------------------------------------------------------------

    # 默认核心宽基
    DEFAULT_INDICES = {
        "000300": "沪深300",
        "000905": "中证500",
        "000852": "中证1000",
        "000985": "中证全指",
    }

    def fetch_index_constituents(self, index_code: str) -> int:
        """拉取中证指数成分股

        Args:
            index_code: 指数代码，如 000300=沪深300

        Returns:
            写入的成分股数量
        """
        import akshare as ak

        now = datetime.now().strftime("%Y-%m-%d")

        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
        except Exception as e:
            logger.error(f"指数成分股拉取失败 {index_code}: {e}")
            return 0

        if df is None or df.empty:
            logger.info(f"指数成分股无数据: {index_code}")
            return 0

        rows = []
        for _, row_data in df.iterrows():
            constituent_code = str(row_data.get("成分券代码", row_data.get("stock_code", "")))
            # 转为内部格式
            if constituent_code.startswith("6"):
                code = f"sh.{constituent_code}"
            else:
                code = f"sz.{constituent_code}"

            in_date = str(row_data.get("纳入日期", row_data.get("in_date", now)))

            rows.append({
                "index_code": index_code,
                "code": code,
                "in_date": in_date,
                "out_date": None,
            })

        if rows:
            self.db.executemany(
                "INSERT OR REPLACE INTO index_constituent (index_code, code, in_date, out_date) "
                "VALUES (:index_code, :code, :in_date, :out_date)",
                rows,
            )

        logger.info(f"指数成分股: {index_code} → {len(rows)} 只")
        return len(rows)

    def fetch_all_default_indices(self):
        """拉取全部默认宽基成分股"""
        for code, name in self.DEFAULT_INDICES.items():
            logger.info(f"拉取 {name}({code}) 成分股...")
            self.fetch_index_constituents(code)
