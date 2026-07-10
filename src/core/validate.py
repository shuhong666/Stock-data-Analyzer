"""
Stock V0.1 — 数据校验

对比 daily_kline（snapshot 来源）与 Baostock 官方日K，
发现差异后记录日志（不自动覆盖）。
"""

import logging
import time
from datetime import datetime, timedelta

from src.core.config import (
    VALIDATE_DIFF_THRESHOLD,
    VALIDATE_DEFAULT_DAYS,
    BAOSTOCK_REQUEST_GAP,
)
from src.core.storage.database import Database
from src.core.collectors.baostock import BaostockCollector

logger = logging.getLogger(__name__)

# OHLCV 对比字段及其阈值
VALIDATE_FIELDS = ["open", "high", "low", "close", "volume", "amount"]


def validate_daily_kline(
    db: Database,
    codes: list[str] = None,
    start_date: str = None,
    end_date: str = None,
) -> None:
    """对比 snapshot 来源日K 与 Baostock 官方数据

    Args:
        db: 数据库实例
        codes: 待校验股票代码，默认全部活跃股票
        start_date: 校验起始日期，默认最近 VALIDATE_DEFAULT_DAYS 个自然日
        end_date: 校验结束日期，默认今天
    """
    if codes is None:
        codes = db.get_active_stock_codes()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=VALIDATE_DEFAULT_DAYS)).strftime("%Y-%m-%d")

    total = len(codes)
    matched = 0
    diff_count = 0
    error_count = 0
    diffs: list[dict] = []

    logger.info(f"开始数据校验: {total} 只股票 [{start_date} ~ {end_date}]")

    collector = BaostockCollector(db)
    collector.login()

    for i, code in enumerate(codes):
        try:
            # 1. 从本地读取 snapshot 来源的日K
            local_rows = db.fetchall(
                "SELECT * FROM daily_kline WHERE code = ? AND trade_date BETWEEN ? AND ?",
                (code, start_date, end_date),
            )
            if not local_rows:
                continue

            # 2. 从 Baostock 拉取同日官方日K
            baostock_rows = collector.fetch_kline(code, start_date, end_date, "d", "baostock_ref")
            bs_map = {r["trade_date"]: r for r in baostock_rows} if baostock_rows else {}

            # 3. 逐日对比
            for local in local_rows:
                td = local["trade_date"]
                if td not in bs_map:
                    continue
                bs = bs_map[td]

                for field in VALIDATE_FIELDS:
                    v_local = local.get(field)
                    v_bs = bs.get(field)
                    if v_local is None or v_bs is None or v_bs == 0:
                        continue
                    diff_pct = abs(v_local - v_bs) / abs(v_bs)
                    if diff_pct > VALIDATE_DIFF_THRESHOLD:
                        diff_entry = {
                            "code": code,
                            "trade_date": td,
                            "field": field,
                            "local": v_local,
                            "baostock": v_bs,
                            "diff_pct": round(diff_pct * 100, 2),
                        }
                        diffs.append(diff_entry)
                        logger.warning(
                            f"差异 {code} {td} {field}: "
                            f"本地={v_local}, Baostock={v_bs}, 偏差={diff_pct*100:.2f}%"
                        )
                        diff_count += 1

            matched += 1

        except Exception as e:
            logger.error(f"校验失败 {code}: {e}")
            error_count += 1

        time.sleep(BAOSTOCK_REQUEST_GAP)

        if (i + 1) % 100 == 0:
            logger.info(f"校验进度: {i+1}/{total}, 已发现 {diff_count} 处差异")

    collector.logout()

    # 汇总日志
    logger.info(
        f"校验完成: {total} 只, 已校验 {matched} 只, "
        f"{diff_count} 处差异 (> {VALIDATE_DIFF_THRESHOLD*100}%), "
        f"{error_count} 只失败"
    )

    # 如果有差异，集中输出摘要
    if diffs:
        logger.warning(f"=== 差异摘要 ({len(diffs)} 处) ===")
        # 按股票+日期分组统计
        from collections import Counter
        by_code = Counter(d["code"] for d in diffs)
        for code, cnt in by_code.most_common(20):
            logger.warning(f"  {code}: {cnt} 处差异")
