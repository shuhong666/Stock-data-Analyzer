"""
Stock V0.1 — Baostock 数据采集器

采集内容:
  - 股票基础信息 (stock_basic)
  - 交易日历 (trade_calendar)
  - 日/周/月K线 (daily_kline / weekly_kline / monthly_kline)
  - 行业分类 (industry_class)
  - 复权因子 (adjust_factor)
"""

import logging
import time
from datetime import datetime, timedelta

import baostock as bs

from src.config import (
    BAOSTOCK_RETRY_COUNT,
    BAOSTOCK_RETRY_INTERVAL,
    BAOSTOCK_REQUEST_GAP,
    BAOSTOCK_RECONNECT_GAP,
    BAOSTOCK_DEFAULT_DAYS,
)
from src.storage.database import Database

logger = logging.getLogger(__name__)


class BaostockCollector:
    """Baostock 数据采集器"""

    def __init__(self, db: Database):
        self.db = db
        self._logged_in = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def login(self):
        if not self._logged_in:
            bs.login()
            self._logged_in = True
            logger.info("Baostock 登录成功")

    def logout(self):
        if self._logged_in:
            try:
                bs.logout()
            except Exception:
                pass
            self._logged_in = False
            logger.info("Baostock 登出")

    def _force_reconnect(self):
        """强制重置连接：登出 + 等待 + 登入"""
        try:
            self.logout()
        except Exception:
            pass
        self._logged_in = False
        time.sleep(BAOSTOCK_RECONNECT_GAP)
        return self._ensure_login()

    def _ensure_login(self):
        """确保登录，失败自动重试（指数退避）"""
        for attempt in range(BAOSTOCK_RETRY_COUNT):
            try:
                self.login()
                return True
            except Exception as e:
                wait = BAOSTOCK_RETRY_INTERVAL * (2 ** attempt)
                logger.warning(f"Baostock 登录失败 (第 {attempt+1} 次): {e}, 等待 {wait}s")
                if attempt < BAOSTOCK_RETRY_COUNT - 1:
                    time.sleep(wait)
        return False

    def _safe_query(self, func, *args, **kwargs):
        """带指数退避自动重连的安全查询"""
        for attempt in range(BAOSTOCK_RETRY_COUNT):
            try:
                if not self._logged_in:
                    if not self._force_reconnect():
                        wait = BAOSTOCK_RECONNECT_GAP * (2 ** attempt)
                        logger.warning(f"Baostock 重登失败, 等待 {wait}s")
                        time.sleep(wait)
                        continue
                rs = func(*args, **kwargs)
                if rs.error_code != "0":
                    raise Exception(rs.error_msg)
                return rs
            except (ConnectionError, TimeoutError, OSError) as e:
                # 网络层错误：强制重连 + 指数退避
                wait = BAOSTOCK_RECONNECT_GAP * (2 ** attempt)
                logger.warning(f"Baostock 网络错误 (第 {attempt+1} 次): {e}, 等待 {wait}s")
                self._logged_in = False
                time.sleep(wait)
            except Exception as e:
                # 业务层错误（error_code 非 0 等）
                wait = BAOSTOCK_RETRY_INTERVAL * (2 ** attempt)
                logger.warning(f"Baostock 查询失败 (第 {attempt+1} 次): {e}, 等待 {wait}s")
                if attempt < BAOSTOCK_RETRY_COUNT - 1:
                    time.sleep(wait)
        return None

    # ------------------------------------------------------------------
    # 股票基础信息
    # ------------------------------------------------------------------

    def init_stock_basic(self):
        """初始化股票基础信息表，仅纳入上证主板 + 深证主板"""
        self._ensure_login()
        rs = self._safe_query(bs.query_stock_basic)
        if rs is None:
            logger.error("获取股票基础信息失败")
            return

        rows = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        while rs.next():
            row = rs.get_row_data()
            # 字段: code, code_name, ipoDate, outDate, type, status
            # type=1 表示股票（非指数）
            if row[4] != "1":
                continue
            code = row[0]          # 格式: sh.600000
            # 仅保留上证主板 (sh.60xxxx) 和深证主板 (sz.00xxxx)
            if not (code.startswith("sh.60") or code.startswith("sz.00")):
                continue
            rows.append({
                "code": code,
                "name": row[1],
                "ipo_date": row[2],
                "delist_date": row[3] if row[3] else None,
                "board": "sh_main" if code.startswith("sh.") else "sz_main",
                "updated_at": now,
            })

        self.db.executemany(
            "INSERT OR REPLACE INTO stock_basic (code, name, ipo_date, delist_date, board, updated_at) "
            "VALUES (:code, :name, :ipo_date, :delist_date, :board, :updated_at)",
            rows,
        )
        logger.info(f"股票基础信息写入完成，共 {len(rows)} 只")

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def fetch_trade_calendar(self, start_date: str = "1990-01-01", end_date: str = None):
        """拉取交易日历"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        self._ensure_login()
        rs = self._safe_query(bs.query_trade_dates, start_date=start_date, end_date=end_date)
        if rs is None:
            logger.error("获取交易日历失败")
            return

        rows = []
        while rs.next():
            row = rs.get_row_data()
            rows.append({
                "trade_date": row[0],
                "is_trading": int(row[1] == "1"),
            })

        self.db.executemany(
            "INSERT OR REPLACE INTO trade_calendar (trade_date, is_trading) VALUES (:trade_date, :is_trading)",
            rows,
        )
        logger.info(f"交易日历写入完成，共 {len(rows)} 天")

    def is_trading_day(self, date_str: str = None) -> bool:
        """判断是否为交易日"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        row = self.db.fetchone("SELECT is_trading FROM trade_calendar WHERE trade_date = ?", (date_str,))
        return row is not None and row["is_trading"] == 1

    def get_recent_trading_days(self, n: int) -> list[str]:
        """获取最近 N 个交易日日期列表"""
        rows = self.db.fetchall(
            "SELECT trade_date FROM trade_calendar WHERE is_trading = 1 AND trade_date <= date('now') "
            "ORDER BY trade_date DESC LIMIT ?",
            (n,),
        )
        return [r["trade_date"] for r in rows]

    # ------------------------------------------------------------------
    # K线数据
    # ------------------------------------------------------------------

    KLINE_FIELDS = (
        "date,code,open,high,low,close,preclose,volume,amount,"
        "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
    )

    def _parse_kline_row(self, row: list[str], data_source: str, updated_at: str) -> dict:
        """将 Baostock K线行转换为 daily_kline 记录"""
        def _f(val, default=None):
            if val is None or val == "":
                return default
            try:
                return float(val)
            except ValueError:
                return default

        return {
            "code": row[1],
            "trade_date": row[0],
            "open": _f(row[2]),
            "high": _f(row[3]),
            "low": _f(row[4]),
            "close": _f(row[5]),
            "preclose": _f(row[6]),
            "volume": _f(row[7]),
            "amount": _f(row[8]),
            "turn": _f(row[10]),
            "pct_chg": _f(row[12]),
            "pe_ttm": _f(row[13]),
            "pb_mrq": _f(row[14]),
            "ps_ttm": _f(row[15]),
            "pcf_ttm": _f(row[16]),
            "total_mv": None,       # Baostock 不提供
            "circ_mv": None,
            "amplitude": None,
            "vol_ratio": None,
            "avg_price": None,
            "limit_up": None,
            "limit_down": None,
            "is_st": int(row[17]) if row[17] in ("0", "1") else 0,
            "data_source": data_source,
            "updated_at": updated_at,
        }

    def fetch_kline(self, code: str, start_date: str, end_date: str,
                    frequency: str = "d", data_source: str = "baostock") -> list[dict]:
        """拉取单只股票 K线，返回记录列表

        Args:
            frequency: d=日, w=周, m=月
        """
        self._ensure_login()
        rs = self._safe_query(
            bs.query_history_k_data_plus,
            code, self.KLINE_FIELDS,
            start_date=start_date, end_date=end_date,
            frequency=frequency, adjustflag="3",
        )
        if rs is None:
            logger.warning(f"K线查询返回空: {code} [{start_date} ~ {end_date}]")
            return []

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        while rs.next():
            rows.append(self._parse_kline_row(rs.get_row_data(), data_source, now))

        logger.info(f"K线: {code} {frequency} [{start_date} ~ {end_date}] → {len(rows)} 条")
        return rows

    # ------------------------------------------------------------------
    # 批量操作
    # ------------------------------------------------------------------

    def backfill_daily(self, code: str, start_date: str, end_date: str) -> int:
        """补全单只股票日K"""
        rows = self.fetch_kline(code, start_date, end_date, "d", "baostock")
        if rows:
            self.db.upsert_daily_kline(rows)
        return len(rows)

    def backfill_all_daily(self, start_date: str = None):
        """断点续跑：补全所有股票缺失的日K

        对每只股票，查找 daily_kline 最大日期，
        从该日期次日开始拉取至今日。
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=BAOSTOCK_DEFAULT_DAYS * 2)).strftime("%Y-%m-%d")

        end_date = datetime.now().strftime("%Y-%m-%d")
        codes = self.db.get_active_stock_codes()
        total = len(codes)
        logger.info(f"开始批量补全日K，共 {total} 只股票")

        for i, code in enumerate(codes):
            max_date = self.db.get_kline_max_date(code)
            if max_date and max_date >= end_date:
                # 数据已是最新，跳过
                if (i + 1) % 500 == 0:
                    logger.info(f"进度: {i+1}/{total}")
                continue

            fetch_start = max_date if max_date else start_date
            # 从最后一个交易日次日开始
            if max_date:
                next_day = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                fetch_start = next_day

            try:
                rows = self.fetch_kline(code, fetch_start, end_date, "d", "baostock")
                if rows:
                    self.db.upsert_daily_kline(rows)
            except Exception as e:
                logger.error(f"补全失败 {code}: {e}")

            # 限流控制
            time.sleep(BAOSTOCK_REQUEST_GAP)

            if (i + 1) % 100 == 0:
                logger.info(f"进度: {i+1}/{total}")
            if (i + 1) % 500 == 0:
                # 定期重连防止会话过期
                self.logout()
                time.sleep(BAOSTOCK_RECONNECT_GAP)
                self._ensure_login()

        logger.info(f"批量补全日K完成: {total} 只")

    def fetch_all_weekly(self, start_date: str, end_date: str = None):
        """拉取全量周K，写入 weekly_kline"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        codes = self.db.get_active_stock_codes()
        logger.info(f"开始拉取周K，共 {len(codes)} 只")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, code in enumerate(codes):
            try:
                rows = self.fetch_kline(code, start_date, end_date, "w", "baostock")
                if rows:
                    self.db.executemany(
                        "INSERT OR REPLACE INTO weekly_kline (...) VALUES (...)",  # 简化：用 daily 同结构
                        rows,
                    )
            except Exception as e:
                logger.error(f"周K拉取失败 {code}: {e}")
            time.sleep(BAOSTOCK_REQUEST_GAP)

    def fetch_all_monthly(self, start_date: str, end_date: str = None):
        """拉取全量月K，写入 monthly_kline"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        codes = self.db.get_active_stock_codes()
        logger.info(f"开始拉取月K，共 {len(codes)} 只")

        for i, code in enumerate(codes):
            try:
                rows = self.fetch_kline(code, start_date, end_date, "m", "baostock")
                if rows:
                    self.db.executemany(
                        "INSERT OR REPLACE INTO monthly_kline (...) VALUES (...)",
                        rows,
                    )
            except Exception as e:
                logger.error(f"月K拉取失败 {code}: {e}")
            time.sleep(BAOSTOCK_REQUEST_GAP)

    # ------------------------------------------------------------------
    # 行业分类 (申万一级)
    # ------------------------------------------------------------------

    def fetch_industry(self):
        """拉取行业分类（Baostock 提供的是证监会行业分类）"""
        self._ensure_login()
        rs = self._safe_query(bs.query_stock_industry)
        if rs is None:
            logger.error("查询行业分类失败")
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        while rs.next():
            row = rs.get_row_data()
            # fields: updateDate, code, code_name, industry, industryClassification
            code = row[1]
            if not (code.startswith("sh.60") or code.startswith("sz.00")):
                continue
            industry_name = row[3] if row[3] else ""
            if not industry_name:
                continue
            rows.append({
                "code": code,
                "level": "一级",
                "industry": industry_name,
                "updated_at": now,
            })

        self.db.executemany(
            "INSERT OR REPLACE INTO industry_class (code, level, industry, updated_at) "
            "VALUES (:code, :level, :industry, :updated_at)",
            rows,
        )
        logger.info(f"行业分类写入完成，共 {len(rows)} 条")

    # ------------------------------------------------------------------
    # 复权因子
    # ------------------------------------------------------------------

    def fetch_adjust_factors(self, code: str, start_date: str = "1990-01-01",
                             end_date: str = None):
        """拉取单只股票复权因子"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        self._ensure_login()
        rs = self._safe_query(
            bs.query_adjust_factor,
            code,
            start_date=start_date,
            end_date=end_date,
        )
        if rs is None:
            return

        rows = []
        while rs.next():
            row = rs.get_row_data()
            # fields: code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor
            rows.append({
                "code": code,
                "trade_date": row[1],          # dividOperateDate
                "adj_factor": float(row[4]) if row[4] else None,        # adjustFactor
                "adj_factor_after": float(row[3]) if row[3] else None,  # backAdjustFactor
            })

        if rows:
            self.db.executemany(
                "INSERT OR REPLACE INTO adjust_factor (code, trade_date, adj_factor, adj_factor_after) "
                "VALUES (:code, :trade_date, :adj_factor, :adj_factor_after)",
                rows,
            )

    def fetch_all_adjust_factors(self, start_date: str = "1990-01-01"):
        """拉取全部股票复权因子（已存在的跳过）"""
        codes = self.db.get_active_stock_codes()
        total = len(codes)
        logger.info(f"开始拉取复权因子，共 {total} 只")

        consecutive_fails = 0
        skipped = 0
        backoff_level = 0
        for i, code in enumerate(codes):
            # 已有数据则跳过
            existing = self.db.fetchone("SELECT COUNT(*) AS c FROM adjust_factor WHERE code = ?", (code,))
            if existing and existing["c"] > 0:
                skipped += 1
                continue

            try:
                self.fetch_adjust_factors(code, start_date)
                new_count = len(self.db.fetchall("SELECT COUNT(*) AS c FROM adjust_factor WHERE code = ?", (code,)))
                if new_count > 0:
                    consecutive_fails = 0
                    backoff_level = 0
                else:
                    consecutive_fails += 1
            except Exception as e:
                logger.error(f"复权因子拉取失败 {code}: {e}")
                consecutive_fails += 1

            # 指数退避：每 10 次连续失败增加等待时间
            if consecutive_fails > 0 and consecutive_fails % 10 == 0:
                backoff_level += 1
                wait = min(BAOSTOCK_RECONNECT_GAP * (2 ** backoff_level), 120)
                logger.warning(
                    f"连续 {consecutive_fails} 次失败，"
                    f"等待 {wait}s 后强制重连..."
                )
                time.sleep(wait)
                self._force_reconnect()
            else:
                time.sleep(BAOSTOCK_REQUEST_GAP)

            if (i + 1) % 100 == 0:
                logger.info(f"复权因子进度: {i+1}/{total} ({skipped} 跳过, {consecutive_fails} 连续失败)")
