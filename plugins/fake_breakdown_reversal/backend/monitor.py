"""
monitor.py — 后台持仓监控 (V5: 动态止盈止损 + ADX入场过滤)

交易日 9:30-15:00 每分钟轮询, 检查卖出条件:
  - 止盈 V5: 跌幅<12%→120%前高, 12-18%→110%前高, >18%→90%前高
  - 止损 V5: 跌幅<12%→-35%峰顶, 12-18%→-30%峰顶, >18%→-20%峰顶
  - 时间止损: 持仓 >= 60 交易日
"""
import logging
import threading
import time
from datetime import datetime

from src.core.storage.database import Database
from . import portfolio

logger = logging.getLogger(__name__)

_monitor_thread = None
_stop_flag = False
_last_alerts = []  # (pos_id, code, name, reason, detail) for UI polling


def get_last_alerts():
    """返回最近一次的提醒列表, 供前端轮询。"""
    global _last_alerts
    alerts = list(_last_alerts)
    return alerts


def clear_alerts():
    """前端取走提醒后清空。"""
    global _last_alerts
    _last_alerts = []


def is_trading_time(db=None):
    """判断当前是否在交易时段: 交易日 + 9:30-15:00。"""
    if db is None:
        db = Database()
    today = datetime.now().strftime("%Y-%m-%d")
    row = db.fetchone(
        "SELECT is_trading FROM trade_calendar WHERE trade_date=?", (today,))
    if not row or row["is_trading"] != 1:
        return False

    now = datetime.now()
    h, m = now.hour, now.minute
    return (h == 9 and m >= 30) or (h >= 10 and h < 15)


def _get_dynamic_targets(pos):
    """V5: 根据跌幅返回 (tp_recovery_ratio, sl_decline_pct)。
    tp_recovery_ratio: 止盈目标 = entry + (peak - entry) * ratio
    sl_decline_pct: 止损线 = 从峰顶跌 sl_decline_pct 触发
    """
    d = pos.get("decline_pct", 15) or 15  # default to mid-range
    if d < 12:
        return (1.20, 0.35)   # shallow pullback: high target, wide stop
    elif d < 18:
        return (1.10, 0.30)   # moderate pullback: standard
    else:
        return (0.90, 0.20)   # deep pullback: conservative target, tight stop


def check_position(pos, db=None):
    """检查单条持仓是否触发卖出条件 (V5 动态规则)。返回 (triggered, reason, detail) 或 None。"""
    if db is None:
        db = Database()

    code = pos["code"]
    # 组装完整代码 (自动补 sh./sz. 前缀)
    full_code = code
    if not code.startswith("sh.") and not code.startswith("sz."):
        for prefix in ["sh.", "sz."]:
            check = db.fetchone(
                "SELECT code FROM stock_basic WHERE code=?", (f"{prefix}{code}",))
            if check:
                full_code = check["code"]
                break

    # 取最新行情
    row = db.fetchone(
        "SELECT trade_date, close FROM daily_kline WHERE code=? ORDER BY trade_date DESC LIMIT 1",
        (full_code,),
    )
    if not row:
        return None

    latest_close = row["close"]
    latest_date = row["trade_date"]

    # T+1: 买入当天不检查
    if latest_date == pos["entry_date"]:
        return None

    # 时间止损
    from datetime import datetime as dt
    entry_dt = dt.strptime(pos["entry_date"], "%Y-%m-%d")
    now = dt.now()
    hold_days = (now - entry_dt).days

    if hold_days >= 60:
        return (True, "时间到",
                f"持仓 {hold_days} 天，已达 60 天时间止损线")

    # V4 动态止盈止损
    tp_ratio, sl_pct = _get_dynamic_targets(pos)
    entry_price = pos["entry_price"]
    peak_price = pos.get("peak_price") or entry_price
    tp_price = entry_price + (peak_price - entry_price) * tp_ratio
    sl_price = peak_price * (1.0 - sl_pct)

    # 止盈 (动态目标)
    if peak_price and latest_close >= tp_price:
        pnl = (latest_close - entry_price) / entry_price * 100
        decline_label = f"跌幅{pos.get('decline_pct', '?')}%"
        return (True, "止盈",
                f"{decline_label}, 目标{int(tp_ratio*100)}%前高: "
                f"{latest_close:.2f} >= {tp_price:.2f}, 盈利 {pnl:+.1f}%")

    # 止损 (动态线)
    if latest_close < sl_price:
        pnl = (latest_close - entry_price) / entry_price * 100
        decline_label = f"跌幅{pos.get('decline_pct', '?')}%"
        return (True, "止损",
                f"{decline_label}, SL{int(sl_pct*100)}%峰顶: "
                f"{latest_close:.2f} < {sl_price:.2f}, 亏损 {pnl:+.1f}%")

    return None


def _monitor_loop():
    """后台监控主循环。"""
    global _stop_flag, _last_alerts
    logger.info("监控线程启动")

    db = Database()
    portfolio.ensure_table(db)

    while not _stop_flag:
        try:
            if not is_trading_time(db):
                time.sleep(60)
                continue

            positions = portfolio.get_open_positions(db)
            for pos in positions:
                # 只检查"监控中"的, "已触发"等待用户确认
                if pos["status"] != "监控中":
                    continue

                result = check_position(pos, db)
                if result and result[0]:
                    triggered, reason, detail = result
                    portfolio.set_alerted(pos["id"], reason, db)
                    _last_alerts.append((
                        pos["id"], pos["code"], pos["name"], reason, detail,
                    ))
                    logger.info(f"卖出提醒: {pos['code']} {pos['name']} — {reason}: {detail}")

            time.sleep(60)

        except Exception as e:
            logger.error(f"监控异常: {e}")
            time.sleep(60)

    logger.info("监控线程退出")


def start_monitor():
    """启动后台监控线程 (幂等)。"""
    global _monitor_thread, _stop_flag
    if _monitor_thread and _monitor_thread.is_alive():
        return
    _stop_flag = False
    _monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
    _monitor_thread.start()


def stop_monitor():
    """停止监控线程。"""
    global _stop_flag
    _stop_flag = True
