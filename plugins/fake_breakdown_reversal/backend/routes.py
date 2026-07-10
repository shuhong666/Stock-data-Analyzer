"""
routes.py — API + UI for 假摔反转
"""
import threading
import logging
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# ====================================================================
# 全局状态
# ====================================================================
_scan_state = {"running": False, "progress": (0, 0), "results": [], "elapsed": 0}


def _run_scan(max_workers=8):
    global _scan_state
    import time
    from .scanner import scan_all

    _scan_state["running"] = True
    _scan_state["results"] = []
    _scan_state["progress"] = (0, 0)

    def progress(done, total):
        _scan_state["progress"] = (done, total)

    t0 = time.time()
    try:
        _scan_state["results"] = scan_all(max_workers=max_workers, progress_cb=progress)
    except Exception as e:
        logger.error(f"扫描失败: {e}")
    finally:
        _scan_state["elapsed"] = round(time.time() - t0, 1)
        _scan_state["running"] = False


# ====================================================================
# API — 扫描
# ====================================================================

@router.get("/scan")
def start_scan(max_workers: int = Query(8, ge=1, le=16)):
    if _scan_state["running"]:
        return {"status": "already_running"}
    t = threading.Thread(target=_run_scan, kwargs={"max_workers": max_workers}, daemon=True)
    t.start()
    return {"status": "started"}


@router.get("/scan/status")
def scan_status():
    return {
        "running": _scan_state["running"],
        "progress": list(_scan_state["progress"]),
        "elapsed": _scan_state["elapsed"],
        "result_count": len(_scan_state["results"]),
    }


@router.get("/scan/results")
def scan_results(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    results = _scan_state["results"]
    total = len(results)
    start = (page - 1) * page_size
    return {"total": total, "page": page, "rows": results[start:start + page_size]}


# ====================================================================
# API — 持仓
# ====================================================================

@router.get("/portfolio")
def get_portfolio():
    from .portfolio import get_open_positions, ensure_table
    from src.core.storage.database import Database
    db = Database()
    ensure_table(db)
    positions = get_open_positions(db)

    # 补最新价
    rows = []
    for p in positions:
        full_code = p["code"]
        if not full_code.startswith("sh.") and not full_code.startswith("sz."):
            for prefix in ["sh.", "sz."]:
                check = db.fetchone("SELECT code FROM stock_basic WHERE code=?", (f"{prefix}{full_code}",))
                if check:
                    full_code = check["code"]
                    break

        kline = db.fetchone(
            "SELECT trade_date, close FROM daily_kline WHERE code=? ORDER BY trade_date DESC LIMIT 1",
            (full_code,))
        latest_close = kline["close"] if kline else None
        latest_date = kline["trade_date"] if kline else None

        pnl = None
        if latest_close and p["entry_price"]:
            pnl = round((latest_close - p["entry_price"]) / p["entry_price"] * 100, 1)

        rows.append({
            "id": p["id"], "code": p["code"], "name": p["name"], "tier": p["tier"],
            "entry_date": p["entry_date"], "entry_price": p["entry_price"],
            "peak_price": p["peak_price"], "peak_date": p["peak_date"],
            "stop_price": p["stop_price"],
            "latest_close": latest_close, "latest_date": latest_date,
            "pnl_pct": pnl,
            "status": p["status"],
            "alert_reason": p["alert_reason"],
            "alert_date": p["alert_date"],
        })
    return {"rows": rows}


@router.post("/portfolio/buy")
def buy_stock(data: dict):
    """买入。data: {code, name, tier, entry_date, entry_price, peak_price, peak_date, sixty_low, stop_price, decline_pct}"""
    from .portfolio import add_position, ensure_table
    from src.core.storage.database import Database
    db = Database()
    ensure_table(db)
    pos_id = add_position(
        code=data["code"], name=data["name"], tier=data["tier"],
        entry_date=data["entry_date"], entry_price=float(data["entry_price"]),
        peak_price=float(data.get("peak_price", 0)),
        peak_date=data.get("peak_date", ""),
        sixty_low=float(data.get("sixty_low", 0)),
        stop_price=float(data.get("stop_price", 0)),
        decline_pct=float(data.get("decline_pct", 0)),
        db=db,
    )
    return {"id": pos_id}


@router.post("/portfolio/sell")
def sell_stock(data: dict):
    """确认卖出。data: {id, exit_date, exit_price, exit_reason}"""
    from .portfolio import confirm_sell
    from src.core.storage.database import Database
    db = Database()
    confirm_sell(
        pos_id=int(data["id"]),
        exit_date=data["exit_date"],
        exit_price=float(data["exit_price"]),
        exit_reason=data["exit_reason"],
        db=db,
    )
    return {"ok": True}


@router.delete("/portfolio/{pos_id}")
def delete_position(pos_id: int):
    from .portfolio import delete_position
    from src.core.storage.database import Database
    db = Database()
    delete_position(pos_id, db)
    return {"ok": True}


@router.get("/portfolio/history")
def get_history():
    """获取已平仓交易记录。"""
    from .portfolio import get_all_positions, ensure_table
    from src.core.storage.database import Database
    db = Database()
    ensure_table(db)
    all_pos = get_all_positions(db)
    closed = [p for p in all_pos if p["status"] == "已卖出"]
    rows = []
    for p in closed:
        pnl = None
        hold = None
        if p["entry_price"] and p["exit_price"]:
            pnl = round((p["exit_price"] - p["entry_price"]) / p["entry_price"] * 100, 1)
        if p["entry_date"] and p["exit_date"]:
            from datetime import datetime
            d1 = datetime.strptime(p["entry_date"], "%Y-%m-%d")
            d2 = datetime.strptime(p["exit_date"], "%Y-%m-%d")
            hold = (d2 - d1).days
        rows.append({
            "id": p["id"], "code": p["code"], "name": p["name"], "tier": p["tier"],
            "entry_date": p["entry_date"], "entry_price": p["entry_price"],
            "exit_date": p["exit_date"], "exit_price": p["exit_price"],
            "exit_reason": p["exit_reason"],
            "pnl_pct": pnl, "hold_days": hold,
        })
    # Summary stats
    wins = [r for r in rows if r["pnl_pct"] is not None and r["pnl_pct"] > 0]
    summary = {}
    if rows:
        import numpy as np
        pnls = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]
        holds = [r["hold_days"] for r in rows if r["hold_days"] is not None]
        summary = {
            "total": len(rows),
            "win_rate": round(len(wins) / len(rows) * 100, 1),
            "avg_pnl": round(np.mean(pnls), 1) if pnls else 0,
            "median_pnl": round(np.median(pnls), 1) if pnls else 0,
            "avg_hold": round(np.mean(holds), 0) if holds else 0,
            "total_pnl": round(sum(pnls), 1) if pnls else 0,
        }
    return {"rows": rows, "summary": summary}


@router.post("/portfolio/reset")
def reset_alert(data: dict):
    """重置提醒。data: {id}"""
    from .portfolio import reset_alert
    from src.core.storage.database import Database
    db = Database()
    reset_alert(int(data["id"]), db)
    return {"ok": True}


# ====================================================================
# API — 提醒
# ====================================================================

@router.get("/alerts")
def get_alerts():
    from .monitor import get_last_alerts, clear_alerts
    alerts = get_last_alerts()
    clear_alerts()
    return {"alerts": [{"id": a[0], "code": a[1], "name": a[2], "reason": a[3], "detail": a[4]} for a in alerts]}


# ====================================================================
# API — 启动监控
# ====================================================================

@router.post("/monitor/start")
def api_start_monitor():
    from .monitor import start_monitor
    start_monitor()
    return {"ok": True}


# ====================================================================
# UI
# ====================================================================

@router.get("/")
def serve_ui():
    return HTMLResponse(content=UI_HTML)


UI_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>假摔反转</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1419;color:#e7e9ea;display:flex;height:100vh;overflow:hidden}
/* ==== LEFT PANEL: Scan ==== */
.left{width:58%;display:flex;flex-direction:column;border-right:1px solid #2f3336;overflow:hidden}
.left-header{padding:12px 16px;border-bottom:1px solid #2f3336;flex-shrink:0}
.left-header h1{font-size:18px;margin-bottom:2px}
.left-header .sub{font-size:12px;color:#8899a6}
.controls{display:flex;gap:8px;align-items:center;padding:10px 16px;flex-shrink:0;flex-wrap:wrap}
button{padding:7px 14px;border:none;border-radius:5px;cursor:pointer;font-size:14px;font-weight:600}
.btn-scan{background:#00ba7c;color:#fff}
.btn-scan:disabled{opacity:.5;cursor:not-allowed}
.btn-buy{background:#1d9bf0;color:#fff;padding:5px 12px;font-size:12px}
.btn-sell{background:#f4212e;color:#fff;padding:5px 12px;font-size:12px}
.btn-reset{background:#8899a6;color:#fff;padding:5px 12px;font-size:12px}
.btn-del{background:#333;color:#f4212e;padding:5px 12px;font-size:12px}
select,input[type=number],input[type=text],input[type=date]{padding:5px 8px;border:1px solid #333;border-radius:5px;background:#1a1f26;color:#e7e9ea;font-size:13px}
.progress-bar{height:3px;background:#00ba7c;border-radius:1px;transition:width .3s;margin:0 16px 8px;flex-shrink:0}
.scan-stats{display:flex;gap:16px;padding:0 16px 6px;font-size:13px;color:#8899a6;flex-shrink:0}
.scan-stats span{color:#e7e9ea;font-weight:600}
.summary-cards{display:flex;gap:8px;padding:0 16px 8px;flex-shrink:0;flex-wrap:wrap}
.card{background:#1a1f26;border-radius:6px;padding:8px 14px;min-width:80px}
.card .val{font-size:22px;font-weight:700;color:#00ba7c}
.card .lbl{font-size:11px;color:#8899a6}
.table-wrap{flex:1;overflow:auto;padding:0 0 16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px 6px;border-bottom:1px solid #2f3336;color:#8899a6;font-weight:500;position:sticky;top:0;background:#0f1419;cursor:pointer;white-space:nowrap}
th:hover{color:#e7e9ea}
td{padding:6px;border-bottom:1px solid #1a1f26;white-space:nowrap}
tr:hover{background:#1a1f26}
.tag{padding:2px 5px;border-radius:3px;font-size:11px;font-weight:600}
.tag-a{background:#ffd70022;color:#ffd700}
.tag-b{background:#1d9bf022;color:#1d9bf0}
.tag-ok{background:#00ba7c22;color:#00ba7c}
.tag-warn{background:#f4212e22;color:#f4212e}
.tag-info{background:#1d9bf022;color:#1d9bf0}
.dec-ok{color:#00ba7c}
.dec-warn{color:#ffd700}
.dec-bad{color:#f4212e}
/* ==== RIGHT PANEL: Portfolio ==== */
.right{width:42%;display:flex;flex-direction:column;overflow:hidden}
.right-header{padding:12px 16px;border-bottom:1px solid #2f3336;flex-shrink:0}
.right-header h2{font-size:18px;margin-bottom:2px}
.right-header .sub{font-size:12px;color:#8899a6}
.right-table-wrap{flex:1;overflow:auto}
/* ==== Modal ==== */
.modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.6);z-index:999;justify-content:center;align-items:center}
.modal-overlay.show{display:flex}
.modal{background:#1a1f26;border:1px solid #333;border-radius:10px;padding:20px;min-width:340px;max-width:420px}
.modal h3{font-size:15px;margin-bottom:14px}
.modal .field{display:flex;align-items:center;margin-bottom:10px}
.modal .field label{width:80px;font-size:12px;color:#8899a6;flex-shrink:0}
.modal .field input{flex:1}
.modal .btns{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
/* ==== Toast ==== */
#toast{position:fixed;bottom:20px;right:20px;padding:12px 20px;border-radius:8px;font-size:14px;display:none;z-index:9999}
.toast-ok{background:#00ba7c;color:#fff}
.toast-err{background:#f4212e;color:#fff}
.intra-note{font-size:11px;color:#ffd700;margin-left:4px}
.metric-tip{cursor:help;border-bottom:1px dotted #555}
/* ==== Validation Panel ==== */
.val-panel{border-bottom:1px solid #2f3336;flex-shrink:0}
.val-header{display:flex;justify-content:space-between;align-items:center;padding:6px 16px;cursor:pointer;user-select:none;font-size:12px;color:#8899a6}
.val-header:hover{color:#e7e9ea;background:#1a1f26}
.val-body{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:0 16px 10px}
.val-body.collapsed{display:none}
.val-item{font-size:11px;color:#8899a6;line-height:1.5}
.val-item b{color:#e7e9ea}
.val-good{color:#00ba7c}
.val-bad{color:#f4212e}
.val-warn{color:#ffd700}
</style>
</head>
<body>

<!-- ===== LEFT ===== -->
<div class="left">
  <div class="left-header">
    <h1>假摔反转 — 买入扫描</h1>
    <div class="sub">V5策略: 60日没涨 + 抛压枯竭 + 区间中高位 + ADX<25 | 获利<50%</div>
  </div>
  <div class="val-panel">
    <div class="val-header" onclick="var b=document.getElementById('valBody');b.classList.toggle('collapsed');this.querySelector('span').textContent=b.classList.contains('collapsed')?'▶ 回测数据':'▼ 回测数据'">
      <span>▼ 回测数据</span>
      <span style="font-size:10px">100只/512笔 | 胜率70.7% | 中位+5.5% | 夏普0.28</span>
    </div>
    <div class="val-body" id="valBody">
      <div class="val-item"><b>V5策略回测</b>: 1703笔 胜率<b class="val-good">75.6%</b> 中位<b class="val-good">+12.0%</b> 夏普0.66 盈亏比1.32</div>
      <div class="val-item"><b>A级条件</b>: 位置≥50+跌<12%+RSI>50+MA20上+获利<30% → 恢复率<b class="val-good">99%</b></div>
      <div class="val-item"><b>60日位置</b>: ≥30% → 恢复率87% | <5% → <b class="val-bad">59%</b>(趴底危险)</div>
      <div class="val-item"><b>获利比例</b>: <30%+位>30% → 恢复率<b class="val-good">99%</b> | >70% → <b class="val-bad">54%</b></div>
      <div class="val-item"><b>跌幅</b>: 3-8%→恢复率93% | 8-12%→87% | >18%→<b class="val-warn">67%</b></div>
      <div class="val-item"><b>RSI</b>: >55→中位恢复<b class="val-good">8d</b> | 40-55→16d | <30→146d</div>
    </div>
  </div>
  <div class="controls">
    <button class="btn-scan" id="btnScan" onclick="startScan()">开始扫描</button>
    <label>线程: <input type="number" id="workers" value="8" min="1" max="16" style="width:50px"></label>
  </div>
  <div class="progress-bar" id="progressBar" style="width:0%"></div>
  <div class="scan-stats">
    进度: <span id="statScanned">待扫描</span> | 符合: <span id="statFound">-</span> | 耗时: <span id="statElapsed">-</span>s
  </div>
  <div class="summary-cards" id="summaryCards" style="display:none">
    <div class="card"><div class="val" id="cardTotal">-</div><div class="lbl">A+B 总数</div></div>
    <div class="card"><div class="val" id="cardA">-</div><div class="lbl">A 级</div></div>
    <div class="card"><div class="val" id="cardB">-</div><div class="lbl">B 级</div></div>
    <div class="card"><div class="val" id="cardDecline">-</div><div class="lbl">平均跌幅</div></div>
  </div>
  <div class="table-wrap">
    <table id="scanTbl"><thead><tr>
      <th onclick="sortScan('tier')">级</th><th onclick="sortScan('code')">代码</th><th>名称</th>
      <th onclick="sortScan('decline_pct')">跌幅</th><th onclick="sortScan('price_pos')">位置</th><th>ADX</th>
      <th>获利%</th><th>RSI</th><th>概率</th><th>PE</th><th>换手%</th>
      <th>前高日</th><th>低点日</th><th>操作</th>
    </tr></thead><tbody></tbody></table>
  </div>
</div>

<!-- ===== RIGHT ===== -->
<div class="right">
  <div class="right-header">
    <h2>我的持仓</h2>
    <div class="sub">
      <span id="tabOpen" style="color:#00ba7c;cursor:pointer;font-weight:600" onclick="switchTab('open')">持仓中</span>
      &nbsp;|&nbsp;
      <span id="tabClosed" style="color:#8899a6;cursor:pointer" onclick="switchTab('closed')">交易记录</span>
      &nbsp; T+1 次日监控 | 卖出提醒实时通知
    </div>
  </div>
  <div class="controls" style="padding:6px 16px">
    <button class="btn-scan" onclick="loadPortfolio()">刷新</button>
    <span id="alertBadge" style="display:none;background:#f4212e;color:#fff;padding:3px 8px;border-radius:10px;font-size:11px;font-weight:600;animation:pulse 2s infinite"></span>
    <span id="historyStats" style="font-size:11px;color:#8899a6;margin-left:auto;display:none"></span>
  </div>
  <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}</style>
  <div class="right-table-wrap" id="posTableWrap">
    <table id="posTbl"><thead><tr>
      <th>代码</th><th>名称</th><th>级</th><th>买入日</th><th>买入价</th><th>前高</th><th>止损</th><th>最新价</th><th>盈亏%</th><th>天数</th><th>状态</th><th>操作</th>
    </tr></thead><tbody></tbody></table>
  </div>
  <div class="right-table-wrap" id="histTableWrap" style="display:none">
    <table id="histTbl"><thead><tr>
      <th>代码</th><th>名称</th><th>级</th><th>买入日</th><th>买入价</th><th>卖出日</th><th>卖出价</th><th>盈亏%</th><th>持仓天</th><th>原因</th>
    </tr></thead><tbody></tbody></table>
  </div>
</div>

<!-- ===== BUY MODAL ===== -->
<div class="modal-overlay" id="buyModal">
  <div class="modal">
    <h3>确认买入</h3>
    <div class="field"><label>代码</label><input type="text" id="bmCode" readonly></div>
    <div class="field"><label>名称</label><input type="text" id="bmName" readonly></div>
    <div class="field"><label>级别</label><input type="text" id="bmTier" readonly></div>
    <div class="field"><label>买入日期</label><input type="date" id="bmDate"></div>
    <div class="field"><label>买入价</label><input type="number" id="bmPrice" step="0.01"></div>
    <div class="field"><label>前高</label><input type="number" id="bmPeak" step="0.01" readonly></div>
    <div class="field"><label>前高日</label><input type="text" id="bmPeakDate" readonly></div>
    <div class="field"><label>止损价</label><input type="number" id="bmStop" step="0.01" readonly></div>
    <div class="btns">
      <button class="btn-del" onclick="closeBuyModal()">取消</button>
      <button class="btn-scan" onclick="confirmBuy()">确认买入</button>
    </div>
  </div>
</div>

<!-- ===== SELL MODAL ===== -->
<div class="modal-overlay" id="sellModal">
  <div class="modal">
    <h3>确认卖出</h3>
    <div class="field"><label>代码</label><input type="text" id="smCode" readonly></div>
    <div class="field"><label>名称</label><input type="text" id="smName" readonly></div>
    <div class="field"><label>触发原因</label><input type="text" id="smReason" readonly></div>
    <div class="field"><label>卖出日期</label><input type="date" id="smDate"></div>
    <div class="field"><label>卖出价</label><input type="number" id="smPrice" step="0.01"></div>
    <div class="btns">
      <button class="btn-reset" onclick="closeSellModal()">取消</button>
      <button class="btn-sell" onclick="confirmSell()">确认卖出</button>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
const API = '/api/plugin/fake_breakdown_reversal';
let scanResults = [], scanSortKey = 'tier', scanSortDir = 1, pollTimer = null, alertTimer = null;

// ==================== SCAN ====================

async function startScan() {
  document.getElementById('btnScan').disabled = true;
  document.getElementById('progressBar').style.width = '0%';
  document.getElementById('summaryCards').style.display = 'none';
  scanResults = [];
  renderScanTable();
  const workers = document.getElementById('workers').value;
  const resp = await fetch(API + '/scan?max_workers=' + workers);
  const data = await resp.json();
  if (data.status === 'started') pollTimer = setInterval(pollScanStatus, 1000);
}

async function pollScanStatus() {
  const resp = await fetch(API + '/scan/status');
  const s = await resp.json();
  const done = s.progress[0], total = s.progress[1];
  document.getElementById('progressBar').style.width = (total > 0 ? done / total * 100 : 0) + '%';
  document.getElementById('statScanned').textContent = done + '/' + total;
  document.getElementById('statElapsed').textContent = s.elapsed;
  if (!s.running) {
    clearInterval(pollTimer);
    document.getElementById('btnScan').disabled = false;
    document.getElementById('progressBar').style.width = '100%';
    loadScanResults();
  }
}

async function loadScanResults() {
  let all = [], page = 1;
  while (true) {
    const resp = await fetch(API + '/scan/results?page=' + page + '&page_size=200');
    const data = await resp.json();
    all = all.concat(data.rows);
    if (all.length >= data.total) break;
    page++;
  }
  scanResults = all;
  document.getElementById('statFound').textContent = all.length;
  updateScanSummary();
  sortScanData();
}

function updateScanSummary() {
  if (scanResults.length === 0) return;
  document.getElementById('summaryCards').style.display = 'flex';
  document.getElementById('cardTotal').textContent = scanResults.length;
  document.getElementById('cardA').textContent = scanResults.filter(r => r.tier === 'A').length;
  document.getElementById('cardB').textContent = scanResults.filter(r => r.tier === 'B').length;
  const avgD = (scanResults.reduce((s, r) => s + r.decline_pct, 0) / scanResults.length).toFixed(1);
  document.getElementById('cardDecline').textContent = avgD + '%';
}

function sortScanData() {
  scanResults.sort((a, b) => {
    let va = a[scanSortKey], vb = b[scanSortKey];
    if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (scanSortKey === 'tier') { va = va === 'A' ? 0 : 1; vb = vb === 'A' ? 0 : 1; }
    return (va > vb ? 1 : va < vb ? -1 : 0) * scanSortDir;
  });
  renderScanTable();
}

function sortScan(key) {
  if (scanSortKey === key) scanSortDir *= -1; else { scanSortKey = key; scanSortDir = key === 'tier' ? 1 : -1; }
  sortScanData();
}

function declineClass(d) { if (d < 8) return 'dec-ok'; if (d < 18) return 'dec-warn'; return 'dec-bad'; }
function declineTip(d) { if (d < 8) return '3-8%:恢复率93%中位19d'; if (d < 12) return '8-12%:恢复率87%中位33d'; if (d < 18) return '12-18%:恢复率79%中位36d'; return '>18%:恢复率67%中位55d'; }
function posClass(p) { if (p==null) return ''; if (p<5) return 'dec-bad'; if (p<15) return 'dec-warn'; if (p<30) return 'dec-warn'; return 'dec-ok'; }
function posTip(p) { if (p==null) return ''; if (p<5) return '<5%:恢复率59%趴底危险'; if (p<15) return '5-15%:恢复率76%'; if (p<30) return '15-30%:恢复率85%'; return '>30%:恢复率87-97%'; }
function profitClass(pr) { if (pr==null) return ''; if (pr<30) return 'dec-ok'; if (pr<50) return 'dec-warn'; return 'dec-bad'; }
function profitTip(pr) { if (pr==null) return ''; if (pr<30) return '<30%:恢复率99%抛压枯竭'; if (pr<50) return '30-50%:恢复率93%'; return '>50%:恢复率54-69%抛压未释'; }
function rsiClass(r) { if (r==null) return ''; if (r>55) return 'dec-ok'; if (r>=40) return 'dec-warn'; return 'dec-bad'; }
function rsiTip(r) { if (r==null) return ''; if (r>55) return '>55:中位恢复8d假摔'; if (r>=40) return '40-55:中位恢复16d'; return '<40:中位恢复43d+'; }

function renderScanTable() {
  const tbody = document.querySelector('#scanTbl tbody');
  tbody.innerHTML = scanResults.map(r => {
    const tierTag = r.tier === 'A' ? '<span class="tag tag-a">A</span>' : '<span class="tag tag-b">B</span>';
    const intra = r.is_intraday ? '<span class="intra-note">盘中</span>' : '';
    const rsi = r.rsi != null ? r.rsi.toFixed(0) : '-';
    const pr = r.profit_ratio != null ? r.profit_ratio.toFixed(0) + '%' : '-';
    const pos = r.price_pos != null ? r.price_pos.toFixed(0) + '%' : '-';
    const pe = r.pe_ttm != null ? r.pe_ttm.toFixed(1) : '-';
    const turn = r.turn != null ? r.turn.toFixed(1) : '-';
    const probColor = r.score >= 80 ? '#00ba7c' : r.score >= 60 ? '#1d9bf0' : '#ffd700';
    const probBar = '<span style="display:inline-block;width:' + (r.score / 4) + 'px;height:6px;border-radius:3px;background:' + probColor + ';vertical-align:middle"></span>';
    return '<tr>' +
      '<td>' + tierTag + '</td><td>' + r.code + '</td><td>' + r.name + intra + '</td>' +
      '<td class="' + declineClass(r.decline_pct) + '"><span class="metric-tip" title="' + declineTip(r.decline_pct) + '"><b>' + r.decline_pct + '%</b></span></td>' +
      '<td class="' + posClass(r.price_pos) + '"><span class="metric-tip" title="' + posTip(r.price_pos) + '">' + pos + '</span></td>' +
      '<td>' + (r.adx != null ? r.adx.toFixed(1) : '-') + '</td>' +
      '<td class="' + profitClass(r.profit_ratio) + '"><span class="metric-tip" title="' + profitTip(r.profit_ratio) + '">' + pr + '</span></td>' +
      '<td class="' + rsiClass(r.rsi) + '"><span class="metric-tip" title="' + rsiTip(r.rsi) + '">' + rsi + '</span></td>' +
      '<td>' + probBar + '</td>' +
      '<td>' + pe + '</td><td>' + turn + '</td>' +
      '<td>' + (r.peak_date || '') + '</td><td>' + (r.trough_date || '') + '</td>' +
      '<td><button class="btn-buy" onclick="openBuyModal(\'' + r.code + '\')">买入</button></td></tr>';
  }).join('');
}

// ==================== BUY MODAL ====================

function openBuyModal(code) {
  const r = scanResults.find(r => r.code === code);
  if (!r) return;
  document.getElementById('bmCode').value = r.code;
  document.getElementById('bmName').value = r.name;
  document.getElementById('bmTier').value = r.tier + '级';
  const today = new Date().toISOString().slice(0, 10);
  document.getElementById('bmDate').value = today;
  document.getElementById('bmPrice').value = r.trough_price || r.latest_close || '';
  document.getElementById('bmPeak').value = r.peak_price || '';
  document.getElementById('bmPeakDate').value = r.peak_date || '';
  document.getElementById('bmStop').value = r.stop_price || '';
  document.getElementById('buyModal').classList.add('show');
}

function closeBuyModal() {
  document.getElementById('buyModal').classList.remove('show');
}

async function confirmBuy() {
  const code = document.getElementById('bmCode').value;
  const r = scanResults.find(r => r.code === code);
  const body = {
    code: code,
    name: document.getElementById('bmName').value,
    tier: document.getElementById('bmTier').value.replace('级', ''),
    entry_date: document.getElementById('bmDate').value,
    entry_price: parseFloat(document.getElementById('bmPrice').value),
    peak_price: parseFloat(document.getElementById('bmPeak').value),
    peak_date: document.getElementById('bmPeakDate').value,
    stop_price: parseFloat(document.getElementById('bmStop').value),
    decline_pct: r ? r.decline_pct : 0,
    sixty_low: 0,
  };
  const resp = await fetch(API + '/portfolio/buy', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await resp.json();
  closeBuyModal();
  showToast('已加入持仓 #' + data.id, true);
  loadPortfolio();
}

// ==================== PORTFOLIO ====================

async function loadPortfolio() {
  const resp = await fetch(API + '/portfolio');
  const data = await resp.json();
  renderPortfolioTable(data.rows);
}

function renderPortfolioTable(rows) {
  const tbody = document.querySelector('#posTbl tbody');
  tbody.innerHTML = rows.map(r => {
    const tierTag = r.tier === 'A' ? '<span class="tag tag-a">A</span>' : '<span class="tag tag-b">B</span>';
    const pnlCls = (r.pnl_pct || 0) >= 0 ? 'dec-ok' : 'dec-bad';
    const pnl = r.pnl_pct != null ? '<span class="' + pnlCls + '">' + (r.pnl_pct > 0 ? '+' : '') + r.pnl_pct.toFixed(1) + '%</span>' : '-';
    const statusTag = r.status === '已触发'
      ? '<span class="tag tag-warn">已触发:' + (r.alert_reason || '') + '</span>'
      : '<span class="tag tag-ok">监控中</span>';
    const days = r.entry_date ? Math.floor((new Date() - new Date(r.entry_date)) / 86400000) : '-';
    const btns = [];
    if (r.status === '已触发') {
      btns.push('<button class="btn-sell" onclick="openSellModal(' + r.id + ')">卖出</button>');
      btns.push('<button class="btn-reset" onclick="resetAlert(' + r.id + ')">忽略</button>');
    }
    btns.push('<button class="btn-del" onclick="deletePos(' + r.id + ')">删</button>');
    return '<tr>' +
      '<td>' + r.code + '</td><td>' + r.name + '</td><td>' + tierTag + '</td>' +
      '<td>' + (r.entry_date || '') + '</td><td>' + (r.entry_price || '-') + '</td>' +
      '<td>' + (r.peak_price || '-') + '</td><td>' + (r.stop_price || '-') + '</td>' +
      '<td>' + (r.latest_close || '-') + '</td><td>' + pnl + '</td>' +
      '<td>' + days + '</td><td>' + statusTag + '</td>' +
      '<td>' + btns.join(' ') + '</td></tr>';
  }).join('');
}

async function deletePos(id) {
  if (!confirm('确定删除持仓 #' + id + '?')) return;
  await fetch(API + '/portfolio/' + id, { method: 'DELETE' });
  loadPortfolio();
}

async function resetAlert(id) {
  await fetch(API + '/portfolio/reset', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }),
  });
  loadPortfolio();
}

// ==================== SELL MODAL ====================

function openSellModal(id) {
  document.getElementById('sellModal').dataset.posId = id;
  const today = new Date().toISOString().slice(0, 10);
  document.getElementById('smDate').value = today;
  document.getElementById('smPrice').value = '';
  document.getElementById('smCode').value = '';
  document.getElementById('smName').value = '';
  document.getElementById('smReason').value = '';
  // fill from portfolio data
  fetch(API + '/portfolio').then(r => r.json()).then(data => {
    const pos = data.rows.find(p => p.id === id);
    if (pos) {
      document.getElementById('smCode').value = pos.code;
      document.getElementById('smName').value = pos.name;
      document.getElementById('smReason').value = pos.alert_reason || '';
      document.getElementById('smPrice').value = pos.latest_close || '';
    }
  });
  document.getElementById('sellModal').classList.add('show');
}

function closeSellModal() {
  document.getElementById('sellModal').classList.remove('show');
}

async function confirmSell() {
  const id = parseInt(document.getElementById('sellModal').dataset.posId);
  const body = {
    id: id,
    exit_date: document.getElementById('smDate').value,
    exit_price: parseFloat(document.getElementById('smPrice').value),
    exit_reason: document.getElementById('smReason').value,
  };
  await fetch(API + '/portfolio/sell', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  closeSellModal();
  showToast('已卖出 #' + id, true);
  loadPortfolio();
}

// ==================== ALERTS POLLING ====================

async function pollAlerts() {
  try {
    const resp = await fetch(API + '/alerts');
    const data = await resp.json();
    if (data.alerts && data.alerts.length > 0) {
      for (const a of data.alerts) {
        showToast('卖出提醒: ' + a.code + ' ' + a.name + ' — ' + a.reason, false);
        if (Notification.permission === 'granted') {
          new Notification('假摔反转 — 卖出提醒', {
            body: a.code + ' ' + a.name + '\n' + a.reason + ': ' + a.detail,
            icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">📈</text></svg>',
          });
        }
      }
      loadPortfolio();
      document.getElementById('alertBadge').style.display = 'inline';
      document.getElementById('alertBadge').textContent = data.alerts.length;
      setTimeout(() => { document.getElementById('alertBadge').style.display = 'none'; }, 10000);
    }
  } catch (e) { /* ignore */ }
}

// ==================== UTILS ====================

function showToast(msg, ok) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = ok ? 'toast-ok' : 'toast-err';
  t.style.display = 'block';
  setTimeout(() => { t.style.display = 'none'; }, 4000);
}

// ==================== HISTORY ====================

let currentTab = 'open';

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tabOpen').style.color = tab === 'open' ? '#00ba7c' : '#8899a6';
  document.getElementById('tabClosed').style.color = tab === 'closed' ? '#00ba7c' : '#8899a6';
  document.getElementById('posTableWrap').style.display = tab === 'open' ? '' : 'none';
  document.getElementById('histTableWrap').style.display = tab === 'closed' ? '' : 'none';
  if (tab === 'closed') loadHistory();
}

async function loadHistory() {
  const resp = await fetch(API + '/portfolio/history');
  const data = await resp.json();
  const rows = data.rows || [];
  const s = data.summary;
  // Render stats
  if (s && rows.length > 0) {
    document.getElementById('historyStats').style.display = '';
    document.getElementById('historyStats').innerHTML =
      '<b>' + rows.length + '</b>笔 | 胜率<b style=\"color:' + (s.win_rate >= 60 ? '#00ba7c' : '#ffd700') + '\">' + s.win_rate + '%</b> | ' +
      '均收益<b style=\"color:' + (s.avg_pnl >= 0 ? '#00ba7c' : '#f4212e') + '\">' + (s.avg_pnl > 0 ? '+' : '') + s.avg_pnl + '%</b> | ' +
      '中位<b>' + (s.median_pnl > 0 ? '+' : '') + s.median_pnl + '%</b> | ' +
      '均持' + s.avg_hold + 'd | 累计<b>' + (s.total_pnl > 0 ? '+' : '') + s.total_pnl + '%</b>';
  } else {
    document.getElementById('historyStats').style.display = 'none';
  }
  // Render table
  const tbody = document.querySelector('#histTbl tbody');
  tbody.innerHTML = rows.map(r => {
    const pnlCls = (r.pnl_pct || 0) >= 0 ? 'dec-ok' : 'dec-bad';
    const pnl = r.pnl_pct != null ? '<span class="' + pnlCls + '">' + (r.pnl_pct > 0 ? '+' : '') + r.pnl_pct.toFixed(1) + '%</span>' : '-';
    const reasonTag = r.exit_reason === '止盈' ? '<span class="tag tag-ok">止盈</span>'
      : r.exit_reason === '止损' ? '<span class="tag tag-warn">止损</span>'
      : '<span class="tag tag-info">' + (r.exit_reason || '-') + '</span>';
    return '<tr>' +
      '<td>' + r.code + '</td><td>' + r.name + '</td>' +
      '<td>' + (r.tier === 'A' ? '<span class="tag tag-a">A</span>' : '<span class="tag tag-b">B</span>') + '</td>' +
      '<td>' + (r.entry_date || '') + '</td><td>' + (r.entry_price || '-') + '</td>' +
      '<td>' + (r.exit_date || '') + '</td><td>' + (r.exit_price || '-') + '</td>' +
      '<td>' + pnl + '</td><td>' + (r.hold_days || '-') + 'd</td><td>' + reasonTag + '</td></tr>';
  }).join('');
}

// ==================== INIT ====================

(async function init() {
  // Request notification permission
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
  }
  // Start monitor
  await fetch(API + '/monitor/start', { method: 'POST' });
  // Load existing data
  loadPortfolio();
  // Poll alerts every 30s
  alertTimer = setInterval(pollAlerts, 30000);
  pollAlerts();
  // Load scan results if available
  const resp = await fetch(API + '/scan/status');
  const s = await resp.json();
  if (s.result_count > 0) loadScanResults();
  else document.getElementById('statScanned').textContent = '待扫描';
})();
</script>
</body>
</html>"""
