"""
portfolio.py — 持仓管理 (增删改查)
"""
from src.core.storage.database import Database


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    name TEXT,
    tier TEXT,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    peak_price REAL,
    peak_date TEXT,
    sixty_low REAL,
    stop_price REAL,
    decline_pct REAL DEFAULT 0,
    exit_date TEXT,
    exit_price REAL,
    exit_reason TEXT,
    status TEXT DEFAULT '监控中',
    alert_reason TEXT,
    alert_date TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime'))
)
"""


def ensure_table(db=None):
    """建表(幂等)。"""
    if db is None:
        db = Database()
    db.execute(TABLE_SQL)
    db.conn.commit()


def add_position(code, name, tier, entry_date, entry_price,
                 peak_price, peak_date, sixty_low, stop_price,
                 decline_pct=0, db=None):
    """新增持仓。返回新行 id。"""
    if db is None:
        db = Database()
    # migrate: add decline_pct column if missing
    try:
        db.execute("ALTER TABLE trade_portfolio ADD COLUMN decline_pct REAL DEFAULT 0")
        db.conn.commit()
    except Exception:
        pass
    db.execute(
        "INSERT INTO trade_portfolio (code, name, tier, entry_date, entry_price, "
        "peak_price, peak_date, sixty_low, stop_price, decline_pct, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '监控中')",
        (code, name, tier, entry_date, entry_price,
         peak_price, peak_date, sixty_low, stop_price, decline_pct),
    )
    db.conn.commit()
    row = db.fetchone("SELECT last_insert_rowid() as id")
    return row["id"] if row else None


def get_open_positions(db=None):
    """获取所有未平仓的持仓 (监控中 + 已触发)。"""
    if db is None:
        db = Database()
    return db.fetchall(
        "SELECT * FROM trade_portfolio WHERE status IN ('监控中', '已触发') ORDER BY entry_date DESC"
    )


def get_all_positions(db=None):
    """获取全部持仓记录。"""
    if db is None:
        db = Database()
    return db.fetchall("SELECT * FROM trade_portfolio ORDER BY entry_date DESC")


def get_position_by_id(pos_id, db=None):
    """按 ID 获取单条持仓。"""
    if db is None:
        db = Database()
    return db.fetchone("SELECT * FROM trade_portfolio WHERE id=?", (pos_id,))


def set_alerted(pos_id, reason, db=None):
    """标记为已触发提醒。"""
    if db is None:
        db = Database()
    from datetime import datetime
    alert_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    db.execute(
        "UPDATE trade_portfolio SET status='已触发', alert_reason=?, alert_date=? WHERE id=?",
        (reason, alert_date, pos_id),
    )
    db.conn.commit()


def confirm_sell(pos_id, exit_date, exit_price, exit_reason, db=None):
    """确认卖出, 记录卖出价和原因。"""
    if db is None:
        db = Database()
    db.execute(
        "UPDATE trade_portfolio SET status='已卖出', exit_date=?, exit_price=?, exit_reason=? WHERE id=?",
        (exit_date, exit_price, exit_reason, pos_id),
    )
    db.conn.commit()


def delete_position(pos_id, db=None):
    """删除持仓记录。"""
    if db is None:
        db = Database()
    db.execute("DELETE FROM trade_portfolio WHERE id=?", (pos_id,))
    db.conn.commit()


def reset_alert(pos_id, db=None):
    """重置提醒状态回监控中。"""
    if db is None:
        db = Database()
    db.execute(
        "UPDATE trade_portfolio SET status='监控中', alert_reason=NULL, alert_date=NULL WHERE id=?",
        (pos_id,),
    )
    db.conn.commit()
