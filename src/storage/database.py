"""
Stock V0.1 — SQLite 数据库连接管理与基础 CRUD
"""

import sqlite3
import os
import threading
from contextlib import contextmanager
from src.config import DB_PATH


class Database:
    """SQLite 数据库管理，线程安全的连接池（每线程一个连接）"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._local = threading.local()
        self._ensure_dir()

    def _ensure_dir(self):
        d = os.path.dirname(self.db_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    @contextmanager
    def cursor(self):
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str, params=None):
        with self.cursor() as cur:
            cur.execute(sql, params or [])
            return cur

    def executemany(self, sql: str, seq):
        with self.cursor() as cur:
            cur.executemany(sql, seq)
            return cur

    def fetchall(self, sql: str, params=None) -> list[dict]:
        cur = self.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def fetchone(self, sql: str, params=None) -> dict | None:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    # ================================================================
    # 建表
    # ================================================================

    def create_tables(self):
        self._create_stock_basic()
        self._create_daily_kline()
        self._create_weekly_kline()
        self._create_monthly_kline()
        self._create_adjust_factor()
        self._create_industry_class()
        self._create_financial_summary()
        self._create_performance_express()
        self._create_index_constituent()
        self._create_snapshot_raw()
        self._create_trade_calendar()
        self._create_run_log()

    def _create_stock_basic(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS stock_basic (
                code        TEXT PRIMARY KEY,
                name        TEXT,
                ipo_date    TEXT,
                delist_date TEXT,
                board       TEXT,
                updated_at  TEXT
            )
        """)

    def _create_daily_kline(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS daily_kline (
                code        TEXT,
                trade_date  TEXT,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                preclose    REAL,
                volume      REAL,
                amount      REAL,
                turn        REAL,
                pct_chg     REAL,
                pe_ttm      REAL,
                pb_mrq      REAL,
                ps_ttm      REAL,
                pcf_ttm     REAL,
                total_mv    REAL,
                circ_mv     REAL,
                amplitude   REAL,
                vol_ratio   REAL,
                avg_price   REAL,
                limit_up    REAL,
                limit_down  REAL,
                is_st       INTEGER,
                data_source TEXT,
                updated_at  TEXT,
                PRIMARY KEY (code, trade_date)
            )
        """)

    def _create_weekly_kline(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS weekly_kline (
                code        TEXT,
                trade_date  TEXT,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                preclose    REAL,
                volume      REAL,
                amount      REAL,
                turn        REAL,
                pct_chg     REAL,
                pe_ttm      REAL,
                pb_mrq      REAL,
                ps_ttm      REAL,
                pcf_ttm     REAL,
                total_mv    REAL,
                circ_mv     REAL,
                amplitude   REAL,
                vol_ratio   REAL,
                avg_price   REAL,
                limit_up    REAL,
                limit_down  REAL,
                is_st       INTEGER,
                data_source TEXT,
                updated_at  TEXT,
                PRIMARY KEY (code, trade_date)
            )
        """)

    def _create_monthly_kline(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS monthly_kline (
                code        TEXT,
                trade_date  TEXT,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                preclose    REAL,
                volume      REAL,
                amount      REAL,
                turn        REAL,
                pct_chg     REAL,
                pe_ttm      REAL,
                pb_mrq      REAL,
                ps_ttm      REAL,
                pcf_ttm     REAL,
                total_mv    REAL,
                circ_mv     REAL,
                amplitude   REAL,
                vol_ratio   REAL,
                avg_price   REAL,
                limit_up    REAL,
                limit_down  REAL,
                is_st       INTEGER,
                data_source TEXT,
                updated_at  TEXT,
                PRIMARY KEY (code, trade_date)
            )
        """)

    def _create_adjust_factor(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS adjust_factor (
                code            TEXT,
                trade_date      TEXT,
                adj_factor      REAL,
                adj_factor_after REAL,
                PRIMARY KEY (code, trade_date)
            )
        """)

    def _create_industry_class(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS industry_class (
                code       TEXT,
                level      TEXT,
                industry   TEXT,
                updated_at TEXT,
                PRIMARY KEY (code, level)
            )
        """)

    def _create_financial_summary(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS financial_summary (
                code        TEXT,
                report_date TEXT,
                data_json   TEXT,
                updated_at  TEXT,
                PRIMARY KEY (code, report_date)
            )
        """)

    def _create_performance_express(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS performance_express (
                code        TEXT,
                report_date TEXT,
                data_json   TEXT,
                updated_at  TEXT,
                PRIMARY KEY (code, report_date)
            )
        """)

    def _create_index_constituent(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS index_constituent (
                index_code TEXT,
                code       TEXT,
                in_date    TEXT,
                out_date   TEXT,
                PRIMARY KEY (index_code, code)
            )
        """)

    def _create_snapshot_raw(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS snapshot_raw (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                code        TEXT,
                snap_time   TEXT,
                trade_date  TEXT,
                price       REAL,
                open        REAL,
                high        REAL,
                low         REAL,
                preclose    REAL,
                volume      REAL,
                amount      REAL,
                turn        REAL,
                pct_chg     REAL,
                pe_ttm      REAL,
                pb          REAL,
                total_mv    REAL,
                circ_mv     REAL,
                vol_ratio   REAL,
                amplitude   REAL,
                avg_price   REAL,
                limit_up    REAL,
                limit_down  REAL,
                bid1_price  REAL,
                bid1_vol    REAL,
                bid2_price  REAL,
                bid2_vol    REAL,
                bid3_price  REAL,
                bid3_vol    REAL,
                bid4_price  REAL,
                bid4_vol    REAL,
                bid5_price  REAL,
                bid5_vol    REAL,
                ask1_price  REAL,
                ask1_vol    REAL,
                ask2_price  REAL,
                ask2_vol    REAL,
                ask3_price  REAL,
                ask3_vol    REAL,
                ask4_price  REAL,
                ask4_vol    REAL,
                ask5_price  REAL,
                ask5_vol    REAL,
                outside     REAL,
                inside      REAL,
                trade_status TEXT
            )
        """)
        # 加速按日期清理和查询
        self.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_trade_date ON snapshot_raw(trade_date)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_code ON snapshot_raw(code)")

    def _create_trade_calendar(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS trade_calendar (
                trade_date  TEXT PRIMARY KEY,
                is_trading  INTEGER DEFAULT 1
            )
        """)

    def _create_run_log(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS run_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type        TEXT,
                start_time      TEXT,
                end_time        TEXT,
                total_stocks    INTEGER,
                total_batches   INTEGER,
                failed_batches  INTEGER,
                rows_inserted   INTEGER,
                status          TEXT,
                message         TEXT
            )
        """)

    # ================================================================
    # daily_kline 专用 UPSERT
    # ================================================================

    DAILY_KLINE_INSERT_SQL = """
        INSERT OR REPLACE INTO daily_kline (
            code, trade_date, open, high, low, close, preclose,
            volume, amount, turn, pct_chg, pe_ttm, pb_mrq,
            ps_ttm, pcf_ttm, total_mv, circ_mv, amplitude,
            vol_ratio, avg_price, limit_up, limit_down, is_st,
            data_source, updated_at
        ) VALUES (
            :code, :trade_date, :open, :high, :low, :close, :preclose,
            :volume, :amount, :turn, :pct_chg, :pe_ttm, :pb_mrq,
            :ps_ttm, :pcf_ttm, :total_mv, :circ_mv, :amplitude,
            :vol_ratio, :avg_price, :limit_up, :limit_down, :is_st,
            :data_source, :updated_at
        )
    """

    def upsert_daily_kline(self, rows: list[dict]):
        """批量 upsert 日K线，快照和 Baostock 共用"""
        self.executemany(self.DAILY_KLINE_INSERT_SQL, rows)

    def get_kline_max_date(self, code: str) -> str | None:
        row = self.fetchone("SELECT MAX(trade_date) AS d FROM daily_kline WHERE code = ?", (code,))
        return row["d"] if row else None

    def get_stock_codes(self, board: str = None) -> list[str]:
        if board:
            return [r["code"] for r in self.fetchall("SELECT code FROM stock_basic WHERE board = ?", (board,))]
        return [r["code"] for r in self.fetchall("SELECT code FROM stock_basic")]

    def get_active_stock_codes(self) -> list[str]:
        return [r["code"] for r in self.fetchall("SELECT code FROM stock_basic WHERE delist_date IS NULL")]
