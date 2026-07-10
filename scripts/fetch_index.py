"""
fetch_index.py -- 拉取大盘指数日K线到 daily_kline 表

指数列表:
  sh.000001 上证综指 (全市场基准)
  sh.000300 沪深300  (大盘蓝筹)
  sz.399001 深证成指 (深圳市场)
  sz.399006 创业板指 (小盘成长)

用法: python scripts/fetch_index.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.storage.database import Database

INDICES = [
    ("sh.000001", "上证综指"),
    ("sh.000300", "沪深300"),
    ("sz.399001", "深证成指"),
    ("sz.399006", "创业板指"),
]

def fetch():
    try:
        import baostock as bs
    except ImportError:
        print("需要安装 baostock: pip install baostock")
        return

    db = Database()

    # 注册指数到 stock_basic
    now = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for code, name in INDICES:
        db.execute(
            "INSERT OR REPLACE INTO stock_basic (code, name, ipo_date, board, updated_at) VALUES (?, ?, ?, 'index', ?)",
            (code, name, "1990-01-01", now))
    db.conn.commit()
    print("已注册指数到 stock_basic")

    # 登录
    lg = bs.login()
    if lg.error_code != '0':
        print(f"Baostock 登录失败: {lg.error_msg}")
        return
    print(f"Baostock 登录成功")

    for code, name in INDICES:
        print(f"\n拉取 {code} {name}...")
        # 查询已有数据范围
        existing = db.fetchone(
            "SELECT MAX(trade_date) as max_d FROM daily_kline WHERE code=?", (code,))
        start_date = existing["max_d"] if existing and existing["max_d"] else "2023-01-01"
        end_date = __import__('datetime').datetime.now().strftime("%Y-%m-%d")

        if existing and existing["max_d"]:
            # 增量更新
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)
            start_date = start_dt.strftime("%Y-%m-%d")

        if start_date >= end_date:
            print(f"  已是最新 ({existing['max_d']}), 跳过")
            continue

        fields = "date,open,high,low,close,volume,amount,turn"
        rs = bs.query_history_k_data_plus(
            code, fields,
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="1")
        if rs.error_code != '0':
            print(f"  查询失败: {rs.error_msg}")
            continue

        count = 0
        while rs.next():
            row = rs.get_row_data()
            if not row or row[0] == '' or row[4] == '' or float(row[4]) == 0:
                continue
            try:
                db.execute(
                    "INSERT OR REPLACE INTO daily_kline "
                    "(code, trade_date, open, high, low, close, volume, amount, turn) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (code, row[0],
                     float(row[1]) if row[1] else 0,
                     float(row[2]) if row[2] else 0,
                     float(row[3]) if row[3] else 0,
                     float(row[4]) if row[4] else 0,
                     float(row[5]) if row[5] else 0,
                     float(row[6]) if row[6] else 0,
                     float(row[7]) if row[7] else 0))
                count += 1
            except Exception as e:
                print(f"  写入失败: {e}")

        db.conn.commit()
        print(f"  写入 {count} 条 (从 {start_date})")
        time.sleep(0.5)  # 速率限制

    bs.logout()
    print("\n完成")


if __name__ == "__main__":
    fetch()
