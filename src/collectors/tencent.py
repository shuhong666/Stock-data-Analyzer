"""
Stock V0.1 — Tencent 实时快照采集器

从 qt.gtimg.cn 拉取实时行情快照，30 秒/轮，400 只/批。
实时更新 daily_kline，同时写入 snapshot_raw。
"""

import logging
import re
import threading
import time
from datetime import datetime

import requests

from src.config import (
    TENCENT_API_URL,
    TENCENT_TIMEOUT,
    SNAPSHOT_BATCH_SIZE,
    SNAPSHOT_INTERVAL,
)
from src.storage.database import Database
from src.scheduler.calendar import TradingCalendar

logger = logging.getLogger(__name__)

# 腾讯返回编码
TENCENT_ENCODING = "gbk"


class TencentSnapshot:
    """腾讯实时快照采集器"""

    def __init__(self, db: Database):
        self.db = db
        self.calendar = TradingCalendar(db)
        self._stop_event = threading.Event()
        self._running = False

    # ------------------------------------------------------------------
    # 代码转换
    # ------------------------------------------------------------------

    @staticmethod
    def to_tencent_format(code: str) -> str:
        """sh.600000 → sh600000"""
        return code.replace(".", "")

    @staticmethod
    def from_tencent_market(market_code: str, stock_code: str) -> str:
        """腾讯市场标识 + 代码 → sh.600000"""
        prefix = "sh" if market_code == "1" else "sz"
        return f"{prefix}.{stock_code}"

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> list[dict]:
        """解析腾讯行情接口返回文本，返回快照记录列表"""
        results = []
        # 匹配: v_sh600000="..."
        pattern = r'v_(\w+)="([^"]*)"'
        now = datetime.now()
        snap_time = now.strftime("%Y-%m-%d %H:%M:%S")
        trade_date = now.strftime("%Y-%m-%d")

        for m in re.finditer(pattern, text):
            tencent_code = m.group(1)  # sh600000
            raw_data = m.group(2)
            if not raw_data or raw_data == "":
                continue
            fields = raw_data.split("~")
            if len(fields) < 50:
                continue

            try:
                record = self._parse_single(tencent_code, fields, snap_time, trade_date)
                if record:
                    results.append(record)
            except Exception as e:
                logger.debug(f"解析行情失败 {tencent_code}: {e}")

        return results

    def _parse_single(self, tencent_code: str, f: list[str], snap_time: str, trade_date: str) -> dict | None:
        """解析单只股票快照，腾讯字段索引参考社区文档"""
        def _f(idx, default=None):
            if idx >= len(f) or f[idx] == "" or f[idx] is None:
                return default
            try:
                return float(f[idx])
            except (ValueError, TypeError):
                return default

        market = f[0]   # 1=SH, 51=SZ
        code = f[2]     # 纯数字代码: 600000
        name = f[1]
        price = _f(3)   # 最新价

        # 若最新价为空，可能是停牌或未上市
        if price is None:
            return None

        full_code = self.from_tencent_market(market, code)

        return {
            # --- 公用字段 ---
            "code": full_code,
            "snap_time": snap_time,
            "trade_date": trade_date,
            # --- OHLCV ---
            "price": price,
            "open": _f(5),
            "high": _f(33),
            "low": _f(34),
            "preclose": _f(4),
            "volume": _f(6, 0) * 100 if _f(6) is not None else None,    # 手 → 股
            "amount": _f(37, 0) * 10000 if _f(37) is not None else None,  # 万 → 元
            # --- 估值 ---
            "turn": _f(38),
            "pct_chg": _f(32),
            "pe_ttm": _f(39),
            "pb": _f(46),
            "total_mv": _f(45),
            "circ_mv": _f(44),
            "vol_ratio": _f(49),
            "amplitude": _f(43),
            "avg_price": _f(51),
            "limit_up": _f(47),
            "limit_down": _f(48),
            # --- 盘口五档 ---
            "bid1_price": _f(9),
            "bid1_vol": _f(10),
            "bid2_price": _f(11),
            "bid2_vol": _f(12),
            "bid3_price": _f(13),
            "bid3_vol": _f(14),
            "bid4_price": _f(15),
            "bid4_vol": _f(16),
            "bid5_price": _f(17),
            "bid5_vol": _f(18),
            "ask1_price": _f(19),
            "ask1_vol": _f(20),
            "ask2_price": _f(21),
            "ask2_vol": _f(22),
            "ask3_price": _f(23),
            "ask3_vol": _f(24),
            "ask4_price": _f(25),
            "ask4_vol": _f(26),
            "ask5_price": _f(27),
            "ask5_vol": _f(28),
            # --- 成交明细 ---
            "outside": _f(7),
            "inside": _f(8),
            "trade_status": "1" if price > 0 and _f(33) and _f(33) > 0 else "0",
        }

    # ------------------------------------------------------------------
    # 数据写入
    # ------------------------------------------------------------------

    SNAPSHOT_INSERT_SQL = """
        INSERT INTO snapshot_raw (
            code, snap_time, trade_date,
            price, open, high, low, preclose, volume, amount,
            turn, pct_chg, pe_ttm, pb, total_mv, circ_mv,
            vol_ratio, amplitude, avg_price, limit_up, limit_down,
            bid1_price, bid1_vol, bid2_price, bid2_vol,
            bid3_price, bid3_vol, bid4_price, bid4_vol,
            bid5_price, bid5_vol,
            ask1_price, ask1_vol, ask2_price, ask2_vol,
            ask3_price, ask3_vol, ask4_price, ask4_vol,
            ask5_price, ask5_vol,
            outside, inside, trade_status
        ) VALUES (
            :code, :snap_time, :trade_date,
            :price, :open, :high, :low, :preclose, :volume, :amount,
            :turn, :pct_chg, :pe_ttm, :pb, :total_mv, :circ_mv,
            :vol_ratio, :amplitude, :avg_price, :limit_up, :limit_down,
            :bid1_price, :bid1_vol, :bid2_price, :bid2_vol,
            :bid3_price, :bid3_vol, :bid4_price, :bid4_vol,
            :bid5_price, :bid5_vol,
            :ask1_price, :ask1_vol, :ask2_price, :ask2_vol,
            :ask3_price, :ask3_vol, :ask4_price, :ask4_vol,
            :ask5_price, :ask5_vol,
            :outside, :inside, :trade_status
        )
    """

    DAILY_UPDATE_SQL = """
        INSERT INTO daily_kline (
            code, trade_date, open, high, low, close, preclose,
            volume, amount, turn, pct_chg, pe_ttm, pb_mrq,
            total_mv, circ_mv, amplitude, vol_ratio, avg_price,
            limit_up, limit_down, data_source, updated_at
        ) VALUES (
            :code, :trade_date, :open, :high, :low, :close, :preclose,
            :volume, :amount, :turn, :pct_chg, :pe_ttm, :pb_mrq,
            :total_mv, :circ_mv, :amplitude, :vol_ratio, :avg_price,
            :limit_up, :limit_down, 'snapshot', :updated_at
        ) ON CONFLICT(code, trade_date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            preclose = excluded.preclose,
            volume = excluded.volume,
            amount = excluded.amount,
            turn = excluded.turn,
            pct_chg = excluded.pct_chg,
            pe_ttm = excluded.pe_ttm,
            pb_mrq = excluded.pb_mrq,
            total_mv = excluded.total_mv,
            circ_mv = excluded.circ_mv,
            amplitude = excluded.amplitude,
            vol_ratio = excluded.vol_ratio,
            avg_price = excluded.avg_price,
            limit_up = excluded.limit_up,
            limit_down = excluded.limit_down,
            data_source = 'snapshot',
            updated_at = excluded.updated_at
    """

    def _save_round(self, snapshots: list[dict]):
        """保存一轮快照数据：写入 snapshot_raw + 更新 daily_kline"""
        if not snapshots:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. 写入 snapshot_raw
        self.db.executemany(self.SNAPSHOT_INSERT_SQL, snapshots)

        # 2. 更新 daily_kline (18 个字段)
        daily_rows = []
        for s in snapshots:
            daily_rows.append({
                "code": s["code"],
                "trade_date": s["trade_date"],
                "open": s["open"],
                "high": s["high"],
                "low": s["low"],
                "close": s["price"],        # 最新价 → close
                "preclose": s["preclose"],
                "volume": s["volume"],
                "amount": s["amount"],
                "turn": s["turn"],
                "pct_chg": s["pct_chg"],
                "pe_ttm": s["pe_ttm"],
                "pb_mrq": s["pb"],
                "total_mv": s["total_mv"],
                "circ_mv": s["circ_mv"],
                "amplitude": s["amplitude"],
                "vol_ratio": s["vol_ratio"],
                "avg_price": s["avg_price"],
                "limit_up": s["limit_up"],
                "limit_down": s["limit_down"],
                "updated_at": now,
            })
        self.db.executemany(self.DAILY_UPDATE_SQL, daily_rows)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, stock_codes: list[str] = None):
        """启动快照采集主循环（阻塞，需手动 Ctrl+C 停止）

        Args:
            stock_codes: 股票代码列表，默认从 stock_basic 获取全部活跃股票
        """
        if stock_codes is None:
            stock_codes = self.db.get_active_stock_codes()

        total = len(stock_codes)
        batch_size = SNAPSHOT_BATCH_SIZE
        batches = [stock_codes[i:i + batch_size] for i in range(0, total, batch_size)]

        logger.info(f"快照采集启动: {total} 只股票, {len(batches)} 批/轮, {SNAPSHOT_INTERVAL}s 间隔")
        self._running = True

        # 非交易日直接退出
        if not self.calendar.is_trading_day():
            logger.info("非交易日，快照采集退出")
            self._running = False
            return

        while not self._stop_event.is_set():
            # 检查交易时段
            if not self.calendar.is_trading_session():
                # 不在交易时段：判断是等待还是退出
                next_open = self.calendar.next_open_time()
                if next_open is None:
                    # 当天没有更多交易时段（已过 15:00 或非交易日）
                    logger.info("今日交易时段已结束，快照采集退出")
                    break
                else:
                    wait_sec = (next_open - datetime.now()).total_seconds()
                    if wait_sec > 0:
                        logger.info(f"等待开盘: {next_open.strftime('%H:%M:%S')} ({wait_sec:.0f}s)")
                        self._stop_event.wait(wait_sec)
                        continue

            round_start = time.time()
            total_snapshots = 0
            failed_batches = 0

            for batch_codes in batches:
                if self._stop_event.is_set():
                    break

                # 腾讯接口格式: sh600000,sz000001
                tc_codes = [self.to_tencent_format(c) for c in batch_codes]
                url = TENCENT_API_URL + ",".join(tc_codes)

                try:
                    resp = requests.get(url, timeout=TENCENT_TIMEOUT)
                    resp.encoding = TENCENT_ENCODING
                    snapshots = self._parse_response(resp.text)
                    self._save_round(snapshots)
                    total_snapshots += len(snapshots)
                except requests.Timeout:
                    failed_batches += 1
                    logger.warning(f"请求超时: 批次 {len(tc_codes)} 只")
                except requests.RequestException as e:
                    failed_batches += 1
                    logger.warning(f"请求失败: {e}")
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"解析/写入失败: {e}")

            # 连续多批次失败 → 警告
            if failed_batches >= 3:
                logger.warning(f"连续 {failed_batches} 批次失败，行情源可能异常")

            round_elapsed = time.time() - round_start
            logger.info(f"轮次完成: {total_snapshots} 条快照, {failed_batches} 批失败, 耗时 {round_elapsed:.1f}s")

            # 等待下一轮
            sleep_time = SNAPSHOT_INTERVAL - round_elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

        self._running = False
        logger.info("快照采集已停止")

    def stop(self):
        """优雅停止（完成当前轮后退出）"""
        logger.info("收到停止信号，完成当前轮后退出...")
        self._stop_event.set()

    @property
    def running(self) -> bool:
        return self._running
