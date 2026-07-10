"""Pullback scanner API routes."""
import threading
import logging
from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)
router = APIRouter()

# Global scan state
_scan_state = {
    "running": False,
    "progress": (0, 0),  # (done, total)
    "results": [],
    "elapsed": 0,
}


def _run_scan(min_probability=50, max_workers=8):
    """Run scan in background thread."""
    global _scan_state
    import time
    from plugins.pullback_scanner.backend.scanner import scan_all

    _scan_state["running"] = True
    _scan_state["results"] = []
    _scan_state["progress"] = (0, 0)

    def progress(done, total):
        _scan_state["progress"] = (done, total)

    t0 = time.time()
    try:
        results = scan_all(max_workers=max_workers, min_probability=min_probability, progress_cb=progress)
        _scan_state["results"] = results
    except Exception as e:
        logger.error(f"Scan failed: {e}")
    finally:
        _scan_state["elapsed"] = round(time.time() - t0, 1)
        _scan_state["running"] = False


@router.get("/scan")
def start_scan(min_probability: int = Query(50, ge=0, le=100), max_workers: int = Query(8, ge=1, le=16)):
    """Start a full-market pullback scan (async)."""
    global _scan_state
    if _scan_state["running"]:
        return {"status": "already_running", "progress": _scan_state["progress"]}

    thread = threading.Thread(
        target=_run_scan,
        kwargs={"min_probability": min_probability, "max_workers": max_workers},
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@router.get("/status")
def scan_status():
    """Check scan progress."""
    return {
        "running": _scan_state["running"],
        "progress": list(_scan_state["progress"]),
        "elapsed": _scan_state["elapsed"],
        "result_count": len(_scan_state["results"]),
    }


@router.get("/results")
def scan_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    min_probability: int = Query(0, ge=0, le=100),
    hide_warnings: bool = Query(False),
):
    """Get scan results (paginated, sorted)."""
    results = _scan_state["results"]

    # Filter
    if min_probability > 0:
        results = [r for r in results if r["probability"] >= min_probability]
    if hide_warnings:
        results = [r for r in results if not r["warnings"]]

    total = len(results)
    start = (page - 1) * page_size
    page_data = results[start:start + page_size]

    # Simplify output
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": [
            {
                "code": r["code"],
                "name": r["name"],
                "latest_date": r["latest_date"],
                "peak_date": r["peak_date"],
                "peak_price": r["peak_price"],
                "trough_date": r["trough_date"],
                "trough_price": r["trough_price"],
                "decline_pct": r["features"]["decline_pct"],
                "probability": r["probability"],
                "rsi": r["features"]["rsi"],
                "rsi_label": r.get("rsi_label"),
                "gain_60d": r["features"]["gain_60d"],
                "pe_ttm": r["fundamentals"]["pe_ttm"],
                "pb_mrq": r["fundamentals"]["pb_mrq"],
                "total_mv": r["fundamentals"]["total_mv"],
                "turn": r["fundamentals"]["turn"],
                "warnings": r["warnings"],
            }
            for r in page_data
        ],
    }


@router.get("/")
def serve_ui():
    """Serve the scanner UI page."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=UI_HTML)


# ====================================================================
# "前期没涨就跌" historical retrieval endpoints
# ====================================================================

@router.get("/no-prior-gain/scan")
def start_no_prior_gain_scan(
    min_probability: int = Query(80, ge=0, le=100),
    min_decline: float = Query(3, ge=1, le=20),
    max_decline: float = Query(50, ge=10, le=60),
    max_workers: int = Query(8, ge=1, le=16),
):
    """Start async full-market scan for 前期没涨就跌 pullbacks."""
    global _no_prior_gain_state
    if _no_prior_gain_state["running"]:
        return {"status": "already_running", "progress": _no_prior_gain_state["progress"]}

    thread = threading.Thread(
        target=_run_no_prior_gain_scan,
        kwargs={
            "min_probability": min_probability,
            "min_decline": min_decline,
            "max_decline": max_decline,
            "max_workers": max_workers,
        },
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@router.get("/no-prior-gain/status")
def no_prior_gain_status():
    """Check scan progress."""
    return {
        "running": _no_prior_gain_state["running"],
        "progress": list(_no_prior_gain_state["progress"]),
        "elapsed": _no_prior_gain_state["elapsed"],
        "result_count": len(_no_prior_gain_state["results"]),
    }


@router.get("/no-prior-gain/results")
def no_prior_gain_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    min_probability: int = Query(0, ge=0, le=100),
):
    """Get scan results (paginated)."""
    results = _no_prior_gain_state["results"]
    if min_probability > 0:
        results = [r for r in results if r["probability"] >= min_probability]

    total = len(results)
    start = (page - 1) * page_size
    page_data = results[start:start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": [
            {
                "code": r["code"],
                "name": r["name"],
                "peak_date": r["peak_date"],
                "peak_price": r["peak_price"],
                "trough_date": r["trough_date"],
                "trough_price": r["trough_price"],
                "decline_pct": r["features"]["decline_pct"],
                "probability": r["probability"],
                "rsi": r["features"]["rsi"],
                "rsi_label": r.get("rsi_label"),
                "gain_60d": r["features"]["gain_60d"],
                "recovery_days": r["recovery"]["recovery_days"] if r["recovery"] else None,
                "recovery_date": r["recovery"]["recovery_date"] if r["recovery"] else None,
                "forward_5d": r["forward_returns"].get("r5d"),
                "forward_10d": r["forward_returns"].get("r10d"),
                "forward_20d": r["forward_returns"].get("r20d"),
                "forward_60d": r["forward_returns"].get("r60d"),
            }
            for r in page_data
        ],
    }


@router.get("/no-prior-gain/summary")
def no_prior_gain_summary():
    """Get aggregated summary statistics for current results."""
    from plugins.pullback_scanner.backend.scanner import retrieve_no_prior_gain_summary
    results = _no_prior_gain_state["results"]
    return retrieve_no_prior_gain_summary(results)


# Global state for no-prior-gain scan
_no_prior_gain_state = {
    "running": False,
    "progress": (0, 0),
    "results": [],
    "elapsed": 0,
}


def _run_no_prior_gain_scan(min_probability=80, min_decline=3, max_decline=50, max_workers=8):
    """Run no-prior-gain scan in background thread."""
    global _no_prior_gain_state
    import time
    from plugins.pullback_scanner.backend.scanner import retrieve_all_no_prior_gain

    _no_prior_gain_state["running"] = True
    _no_prior_gain_state["results"] = []
    _no_prior_gain_state["progress"] = (0, 0)

    def progress(done, total):
        _no_prior_gain_state["progress"] = (done, total)

    t0 = time.time()
    try:
        output = retrieve_all_no_prior_gain(
            max_workers=max_workers,
            min_probability=min_probability,
            min_decline=min_decline,
            max_decline=max_decline,
            progress_cb=progress,
        )
        _no_prior_gain_state["results"] = output["results"]
    except Exception as e:
        logger.error(f"No-prior-gain scan failed: {e}")
    finally:
        _no_prior_gain_state["elapsed"] = round(time.time() - t0, 1)
        _no_prior_gain_state["running"] = False


# ====================================================================
# "前期没涨就跌" CURRENT-day real-time scanner
# ====================================================================

@router.get("/no-prior-gain/current")
def serve_no_prior_gain_ui():
    """Serve the no-prior-gain current scanner UI."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=NO_PRIOR_GAIN_UI)


@router.get("/no-prior-gain/current/scan")
def start_no_prior_gain_current_scan(
    min_probability: int = Query(80, ge=0, le=100),
    max_workers: int = Query(8, ge=1, le=16),
):
    """Start async scan for stocks CURRENTLY in a no-prior-gain pullback."""
    global _no_prior_gain_current_state
    if _no_prior_gain_current_state["running"]:
        return {"status": "already_running", "progress": _no_prior_gain_current_state["progress"]}

    thread = threading.Thread(
        target=_run_no_prior_gain_current_scan,
        kwargs={"min_probability": min_probability, "max_workers": max_workers},
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@router.get("/no-prior-gain/current/status")
def no_prior_gain_current_status():
    """Check current scan progress."""
    return {
        "running": _no_prior_gain_current_state["running"],
        "progress": list(_no_prior_gain_current_state["progress"]),
        "elapsed": _no_prior_gain_current_state["elapsed"],
        "result_count": len(_no_prior_gain_current_state["results"]),
    }


@router.get("/no-prior-gain/current/results")
def no_prior_gain_current_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    min_probability: int = Query(0, ge=0, le=100),
    hide_warnings: bool = Query(False),
):
    """Get current scan results (paginated, sorted by probability desc)."""
    results = _no_prior_gain_current_state["results"]
    if min_probability > 0:
        results = [r for r in results if r["probability"] >= min_probability]
    if hide_warnings:
        results = [r for r in results if not r["warnings"]]

    total = len(results)
    start = (page - 1) * page_size
    page_data = results[start:start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": [
            {
                "code": r["code"],
                "name": r["name"],
                "latest_date": r["latest_date"],
                "peak_date": r["peak_date"],
                "peak_price": r["peak_price"],
                "trough_date": r["trough_date"],
                "trough_price": r["trough_price"],
                "decline_pct": r["features"]["decline_pct"],
                "probability": r["probability"],
                "rsi": r["features"]["rsi"],
                "rsi_label": r.get("rsi_label"),
                "gain_60d": r["features"]["gain_60d"],
                "gain_20d": r["features"]["gain_20d"],
                "kdj_k": r["features"]["kdj_k"],
                "bb_pos": r["features"]["bb_pos"],
                "ma20_dist": r["features"]["ma20_dist"],
                "ma60_dist": r["features"]["ma60_dist"],
                "adx": r["features"]["adx"],
                "pe_ttm": r["fundamentals"]["pe_ttm"],
                "pb_mrq": r["fundamentals"]["pb_mrq"],
                "total_mv": r["fundamentals"]["total_mv"],
                "turn": r["fundamentals"]["turn"],
                "warnings": r["warnings"],
            }
            for r in page_data
        ],
    }


# Global state for no-prior-gain CURRENT scan
_no_prior_gain_current_state = {
    "running": False,
    "progress": (0, 0),
    "results": [],
    "elapsed": 0,
}


def _run_no_prior_gain_current_scan(min_probability=80, max_workers=8):
    """Run no-prior-gain CURRENT scan in background thread."""
    global _no_prior_gain_current_state
    import time
    from plugins.pullback_scanner.backend.scanner import scan_all_no_prior_gain_current

    _no_prior_gain_current_state["running"] = True
    _no_prior_gain_current_state["results"] = []
    _no_prior_gain_current_state["progress"] = (0, 0)

    def progress(done, total):
        _no_prior_gain_current_state["progress"] = (done, total)

    t0 = time.time()
    try:
        results = scan_all_no_prior_gain_current(
            max_workers=max_workers,
            min_probability=min_probability,
            progress_cb=progress,
        )
        _no_prior_gain_current_state["results"] = results
    except Exception as e:
        logger.error(f"No-prior-gain current scan failed: {e}")
    finally:
        _no_prior_gain_current_state["elapsed"] = round(time.time() - t0, 1)
        _no_prior_gain_current_state["running"] = False


# ====================================================================
# UI: 前期没涨就跌 当前扫描
# ====================================================================

NO_PRIOR_GAIN_UI = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>前期没涨就跌 Scanner</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1419;color:#e7e9ea;padding:20px}
h1{font-size:18px;margin-bottom:4px}
.subtitle{font-size:12px;color:#8899a6;margin-bottom:16px}
.controls{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
button{padding:8px 16px;border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:600}
.btn-scan{background:#00ba7c;color:#fff}
.btn-scan:disabled{opacity:.5;cursor:not-allowed}
select,input{padding:6px 10px;border:1px solid #333;border-radius:6px;background:#1a1f26;color:#e7e9ea;font-size:13px}
.progress-bar{height:4px;background:#00ba7c;border-radius:2px;transition:width .3s;margin-bottom:8px}
.stats{display:flex;gap:20px;margin-bottom:12px;font-size:13px;color:#8899a6}
.stats span{color:#e7e9ea;font-weight:600}
.summary-cards{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.card{background:#1a1f26;border-radius:8px;padding:12px 16px;min-width:100px}
.card .val{font-size:22px;font-weight:700;color:#00ba7c}
.card .lbl{font-size:11px;color:#8899a6;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:8px 6px;border-bottom:1px solid #2f3336;color:#8899a6;font-weight:500;position:sticky;top:0;background:#0f1419;cursor:pointer;white-space:nowrap}
th:hover{color:#e7e9ea}
td{padding:6px;border-bottom:1px solid #1a1f26;white-space:nowrap}
tr:hover{background:#1a1f26}
.tag{padding:2px 5px;border-radius:3px;font-size:10px;font-weight:600}
.tag-safe{background:#00ba7c22;color:#00ba7c}
.tag-warn{background:#f4212e22;color:#f4212e}
.tag-info{background:#1d9bf022;color:#1d9bf0}
.prob-cell{font-weight:700}
.prob-high{color:#00ba7c}
.prob-mid{color:#1d9bf0}
.gain-neg{color:#00ba7c}
.decline-ok{color:#00ba7c}
.decline-warn{color:#ffd700}
.decline-danger{color:#f4212e}
.warn-row td{background:#f4212e06}
#toast{position:fixed;bottom:20px;right:20px;padding:12px 20px;border-radius:8px;font-size:14px;display:none;z-index:999}
.toast-ok{background:#00ba7c;color:#fff}
.toast-err{background:#f4212e;color:#fff}
.note{font-size:11px;color:#8899a6;margin-top:8px;padding:8px;background:#1a1f26;border-radius:6px}
.validation-panel{border:1px solid #2f3336;border-radius:8px;margin-bottom:12px;overflow:hidden}
.validation-header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#1a1f26;cursor:pointer;user-select:none}
.validation-header:hover{background:#1f2a33}
.validation-header .title{font-size:13px;font-weight:600;color:#e7e9ea}
.validation-header .toggle{font-size:11px;color:#8899a6}
.validation-body{padding:14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.validation-body.collapsed{display:none}
.verify-card{background:#1a1f26;border-radius:6px;padding:12px}
.verify-card h4{font-size:12px;color:#8899a6;margin-bottom:8px;font-weight:500}
.verify-card .big-num{font-size:28px;font-weight:700;color:#00ba7c;line-height:1}
.verify-card .big-num .unit{font-size:14px;font-weight:400;color:#8899a6}
.verify-table{width:100%;font-size:11px;border-collapse:collapse;margin-top:6px}
.verify-table td{padding:3px 6px;border-bottom:1px solid #1d2330;white-space:nowrap}
.verify-table .hdr td{color:#8899a6;font-size:10px;border-bottom:1px solid #2f3336}
.verify-table .hl td{color:#00ba7c;font-weight:600}
.verify-table .wr td{color:#ffd700}
.verify-table .dg td{color:#f4212e}
.cond-tags{display:flex;gap:6px;flex-wrap:wrap}
.cond-tag{padding:3px 8px;border-radius:4px;font-size:11px;font-weight:600;background:#00ba7c22;color:#00ba7c}
</style>
</head>
<body>
<h1>前期没涨就跌 — 当日扫描</h1>
<div class="subtitle">条件: 60日涨幅 &lt; 0 + 当前回调中 + 概率 ≥ 阈值</div>

<!-- ==== Validation Stats Panel ==== -->
<div class="validation-panel" id="valPanel">
  <div class="validation-header" onclick="var b=document.getElementById('valBody');var t=this.querySelector('.toggle');b.classList.toggle('collapsed');t.textContent=b.classList.contains('collapsed')?'▶':'▼'">
    <span class="title">📊 历史回测验证数据 (点击展开/收起)</span>
    <span class="toggle">▼</span>
  </div>
  <div class="validation-body" id="valBody">
    <!-- Card 1: Overall -->
    <div class="verify-card">
      <h4>📈 总体 (20只抽查, 513次, 概率≥80%)</h4>
      <div class="big-num">79.7<span class="unit">%</span></div>
      <div style="font-size:11px;color:#8899a6;margin-top:4px">恢复率 · 中位恢复 <b style="color:#e7e9ea">38天</b></div>
      <div style="font-size:11px;color:#8899a6">P25=22d · P75=69d · P90=230d</div>
    </div>
    <!-- Card 2: By decline -->
    <div class="verify-card">
      <h4>📉 按跌幅 — 恢复率 &amp; 中位恢复天数</h4>
      <table class="verify-table">
        <tr class="hdr"><td>跌幅</td><td>样本</td><td>恢复率</td><td>中位</td></tr>
        <tr class="hl"><td>3-8%</td><td>20</td><td>100%</td><td>19d</td></tr>
        <tr class="hl"><td>8-12%</td><td>95</td><td>93.7%</td><td>33d</td></tr>
        <tr class="wr"><td>12-18%</td><td>235</td><td>79.1%</td><td>36d</td></tr>
        <tr class="dg"><td>&gt;18%</td><td>163</td><td>69.9%</td><td>55d</td></tr>
      </table>
    </div>
    <!-- Card 3: Forward returns -->
    <div class="verify-card">
      <h4>🚀 低点买入后前向收益 (513次)</h4>
      <table class="verify-table">
        <tr class="hdr"><td>持有</td><td>均值</td><td>中位</td><td>胜率</td></tr>
        <tr class="hl"><td>5日</td><td>+1.7%</td><td>+1.7%</td><td>68%</td></tr>
        <tr class="hl"><td>10日</td><td>+2.0%</td><td>+1.9%</td><td>66%</td></tr>
        <tr class="hl"><td>20日</td><td>+4.2%</td><td>+3.0%</td><td>61%</td></tr>
        <tr class="hl"><td><b>60日</b></td><td><b>+11.5%</b></td><td><b>+9.6%</b></td><td><b>66%</b></td></tr>
      </table>
    </div>
    <!-- Card 4: 3-8% deep dive -->
    <div class="verify-card">
      <h4>🔬 跌幅3-8%专项 (全市场5905次)</h4>
      <div class="big-num">76.8<span class="unit">%</span></div>
      <div style="font-size:11px;color:#8899a6;margin-top:4px">恢复率 · 中位恢复 <b style="color:#e7e9ea">26天</b></div>
      <div style="font-size:11px;color:#8899a6;margin-top:6px">60日收益: 均值+6.5% · 中位+3.0% · 胜率60%</div>
      <div style="font-size:11px;color:#8899a6">分布: 1-5d(4%) 5-10d(10%) 10-20d(19%) <b>20-40d(33%)</b> 40-80d(20%) 80d+(14%)</div>
    </div>
    <!-- Card 5: Rules -->
    <div class="verify-card">
      <h4>✅ 实战要点</h4>
      <div style="font-size:11px;color:#8899a6;line-height:1.7">
        • <b style="color:#00ba7c">跌幅 &lt; 12%</b> → 恢复率 93%+，最安全<br>
        • <b style="color:#ffd700">跌幅 12-18%</b> → 恢复率 79%，需观察<br>
        • <b style="color:#f4212e">跌幅 &gt; 18%</b> → 恢复率骤降至 70%，红线<br>
        • 60日持有平均收益 <b style="color:#00ba7c">+11.5%</b><br>
        • 短期(5-20日)效果不明显，中期(60日)确定性强
      </div>
    </div>
    <!-- Card 6: Definitions -->
    <div class="verify-card">
      <h4>📖 指标说明</h4>
      <div style="font-size:11px;color:#8899a6;line-height:1.7">
        <b style="color:#e7e9ea">跌幅</b>: 前40交易日内最高点 → 当前最低点<br>
        <span style="color:#666">回撤幅度，非固定天数。跨度取决于实际走势，范围 1~40 天。</span><br>
        <b style="color:#e7e9ea;display:inline-block;margin-top:6px">RSI &gt; 55</b>: <span style="color:#00ba7c">下跌中保持动能 = 假摔</span><br>
        <span style="color:#666">历史验证 RSI&gt;55 的"没涨就跌"回调: 恢复率 <b style="color:#00ba7c">100%</b>，中位恢复仅 <b style="color:#00ba7c">8 天</b>。说明跌不动的才是真回调。</span><br>
        <b style="color:#e7e9ea;display:inline-block;margin-top:6px">60日涨幅</b>: 前高日的收盘价 vs 60个交易日前<br>
        <span style="color:#666">&lt; 0 意味着前期没有获利盘，跌下来是洗盘而非出货。</span>
      </div>
    </div>
  </div>
</div>

<div class="cond-tags" style="margin-bottom:12px">
  <span class="cond-tag">60日涨幅 &lt; 0</span>
  <span class="cond-tag">当前回调中</span>
  <span class="cond-tag">概率 ≥ 80%</span>
</div>

<div class="controls">
  <button class="btn-scan" id="btnScan" onclick="startScan()">开始扫描</button>
  <label>最低概率: <input type="number" id="minProb" value="80" min="0" max="100" style="width:60px">%</label>
  <label>线程: <input type="number" id="workers" value="8" min="1" max="16" style="width:45px"></label>
  <label><input type="checkbox" id="hideWarn" onchange="renderTable()"> 隐藏有警告的</label>
</div>

<div class="progress-bar" id="progressBar" style="width:0%"></div>
<div class="stats">
  已扫描: <span id="statScanned">-</span> | 符合条件: <span id="statFound">-</span> | 耗时: <span id="statElapsed">-</span>s
</div>

<div class="summary-cards" id="summaryCards" style="display:none">
  <div class="card"><div class="val" id="cardTotal">-</div><div class="lbl">符合条件</div></div>
  <div class="card"><div class="val" id="cardDecline">-</div><div class="lbl">平均跌幅</div></div>
  <div class="card"><div class="val" id="cardRSI">-</div><div class="lbl">平均 RSI</div></div>
  <div class="card"><div class="val" id="cardClean">-</div><div class="lbl">无警告</div></div>
</div>

<div style="max-height:calc(100vh - 460px);overflow:auto">
<table id="tbl"><thead><tr>
  <th onclick="sortBy('probability')">概率</th>
  <th onclick="sortBy('code')">代码</th>
  <th>名称</th>
  <th onclick="sortBy('decline_pct')">跌幅</th>
  <th>RSI</th>
  <th>MA20%</th>
  <th>60日涨</th>
  <th>PE</th>
  <th>换手%</th>
  <th>市值(亿)</th>
  <th>前高日</th>
  <th>低点日</th>
  <th>警告</th>
</tr></thead><tbody></tbody></table>
</div>
<div id="toast"></div>

<script>
let results=[],sortKey='probability',sortDir=-1,pollTimer=null;

const API='/api/plugin/pullback_scanner/no-prior-gain/current';

async function startScan(){
  document.getElementById('btnScan').disabled=true;
  document.getElementById('progressBar').style.width='0%';
  document.getElementById('summaryCards').style.display='none';
  results=[];
  renderTable();
  const minProb=document.getElementById('minProb').value;
  const workers=document.getElementById('workers').value;
  const resp=await fetch(API+'/scan?min_probability='+minProb+'&max_workers='+workers);
  const data=await resp.json();
  if(data.status==='started'||data.status==='already_running'){
    pollTimer=setInterval(pollStatus,1000);
  }
}

async function pollStatus(){
  const resp=await fetch(API+'/status');
  const s=await resp.json();
  const done=s.progress[0],total=s.progress[1];
  document.getElementById('progressBar').style.width=(total>0?done/total*100:0)+'%';
  document.getElementById('statScanned').textContent=done+'/'+total;
  document.getElementById('statElapsed').textContent=s.elapsed;
  if(!s.running){
    clearInterval(pollTimer);
    document.getElementById('btnScan').disabled=false;
    document.getElementById('progressBar').style.width='100%';
    loadResults();
  }
}

async function loadResults(){
  const hideWarn=document.getElementById('hideWarn').checked;
  let all=[],page=1;
  while(true){
    const resp=await fetch(API+'/results?page='+page+'&page_size=200&hide_warnings='+hideWarn);
    const data=await resp.json();
    all=all.concat(data.rows);
    if(all.length>=data.total) break;
    page++;
  }
  results=all;
  document.getElementById('statFound').textContent=all.length;
  updateSummary();
  sortData();
}

function updateSummary(){
  if(results.length===0) return;
  document.getElementById('summaryCards').style.display='flex';
  document.getElementById('cardTotal').textContent=results.length;
  const avgDecline=(results.reduce((s,r)=>s+r.decline_pct,0)/results.length).toFixed(1);
  document.getElementById('cardDecline').textContent=avgDecline+'%';
  const rsis=results.filter(r=>r.rsi!=null).map(r=>r.rsi);
  const avgRSI=rsis.length>0?(rsis.reduce((s,v)=>s+v,0)/rsis.length).toFixed(0):'-';
  document.getElementById('cardRSI').textContent=avgRSI;
  const clean=results.filter(r=>!r.warnings||r.warnings.length===0).length;
  document.getElementById('cardClean').textContent=clean+'/'+results.length;
}

function sortData(){
  results.sort((a,b)=>{
    let va=a[sortKey],vb=b[sortKey];
    if(typeof va==='string') va=va.toLowerCase(),vb=vb.toLowerCase();
    return (va>vb?1:va<vb?-1:0)*sortDir;
  });
  renderTable();
}

function sortBy(key){
  if(sortKey===key) sortDir*=-1; else{sortKey=key;sortDir=-1;}
  sortData();
}

function declineClass(d){
  if(d<8) return 'decline-ok';
  if(d<18) return 'decline-warn';
  return 'decline-danger';
}

function probClass(p){
  if(p>=90) return 'prob-high';
  if(p>=80) return 'prob-mid';
  return '';
}

function renderTable(){
  const tbody=document.querySelector('#tbl tbody');
  const hideWarn=document.getElementById('hideWarn').checked;
  let rows=results;
  if(hideWarn) rows=rows.filter(r=>!r.warnings||r.warnings.length===0);
  tbody.innerHTML=rows.map(r=>{
    const tags=r.warnings&&r.warnings.length>0?r.warnings.map(w=>'<span class=\"tag tag-warn\">'+w+'</span>').join(' '):'<span class=\"tag tag-safe\">OK</span>';
    const rsi=r.rsi!=null?r.rsi.toFixed(0):'-';
    const ma20=r.ma20_dist!=null?(r.ma20_dist>0?'+':'')+r.ma20_dist.toFixed(1)+'%':'-';
    const gain60=r.gain_60d!=null?(r.gain_60d>0?'+':'')+r.gain_60d.toFixed(1)+'%':'-';
    const gain60cls=r.gain_60d!=null&&r.gain_60d<0?'gain-neg':'';
    const pe=r.pe_ttm!=null?r.pe_ttm.toFixed(1):'-';
    const turn=r.turn!=null?r.turn.toFixed(1):'-';
    const mv=r.total_mv!=null?r.total_mv.toFixed(1):'-';
    return '<tr class=\"'+(r.warnings&&r.warnings.length>0?'warn-row':'')+'\">'+
      '<td class=\"prob-cell '+probClass(r.probability)+'\">'+r.probability+'%</td>'+
      '<td>'+r.code+'</td><td>'+r.name+'</td>'+
      '<td class=\"'+declineClass(r.decline_pct)+'\"><b>'+r.decline_pct+'%</b></td>'+
      '<td>'+rsi+'</td><td>'+ma20+'</td>'+
      '<td class=\"'+gain60cls+'\">'+gain60+'</td>'+
      '<td>'+pe+'</td><td>'+turn+'</td><td>'+mv+'</td>'+
      '<td>'+r.peak_date+'</td><td>'+r.trough_date+'</td>'+
      '<td>'+tags+'</td></tr>';
  }).join('');
}

// Auto-load results on page open
(async function(){
  const resp=await fetch(API+'/status');
  const s=await resp.json();
  if(s.result_count>0) loadResults();
  else document.getElementById('statScanned').textContent='idle';
})();
</script>
</body>
</html>"""


UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pullback Scanner</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1419;color:#e7e9ea;padding:20px}
h1{font-size:20px;margin-bottom:16px}
.controls{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
button{padding:8px 16px;border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:600}
.btn-scan{background:#1d9bf0;color:#fff}
.btn-scan:disabled{opacity:.5;cursor:not-allowed}
.btn-stop{background:#f4212e;color:#fff}
select,input{padding:6px 10px;border:1px solid #333;border-radius:6px;background:#1a1f26;color:#e7e9ea;font-size:13px}
.progress-bar{height:4px;background:#1d9bf0;border-radius:2px;transition:width .3s;margin-bottom:8px}
.stats{display:flex;gap:20px;margin-bottom:12px;font-size:13px;color:#8899a6}
.stats span{color:#e7e9ea;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:10px 8px;border-bottom:1px solid #2f3336;color:#8899a6;font-weight:500;position:sticky;top:0;background:#0f1419;cursor:pointer}
th:hover{color:#e7e9ea}
td{padding:8px;border-bottom:1px solid #1a1f26}
tr:hover{background:#1a1f26}
.warn{color:#f4212e;font-weight:600}
.tag{padding:2px 6px;border-radius:4px;font-size:11px;font-weight:600}
.tag-warn{background:#f4212e22;color:#f4212e}
.tag-ok{background:#00ba7c22;color:#00ba7c}
.tag-info{background:#1d9bf022;color:#1d9bf0}
.prob-bar{display:inline-block;height:6px;border-radius:3px;margin-right:4px;vertical-align:middle}
.tooltip{cursor:help;border-bottom:1px dotted #8899a6}
.warn-row td{background:#f4212e08}
#toast{position:fixed;bottom:20px;right:20px;padding:12px 20px;border-radius:8px;font-size:14px;display:none;z-index:999}
.toast-ok{background:#00ba7c;color:#fff}
.toast-err{background:#f4212e;color:#fff}
</style>
</head>
<body>
<h1>Pullback Scanner</h1>
<div class="controls">
  <button class="btn-scan" id="btnScan" onclick="startScan()">Start Scan</button>
  <label>Min Probability: <input type="number" id="minProb" value="70" min="0" max="100" style="width:70px">%</label>
  <label>Workers: <input type="number" id="workers" value="8" min="1" max="16" style="width:50px"></label>
  <label><input type="checkbox" id="hideWarn" onchange="renderTable()"> Hide warnings</label>
</div>
<div class="progress-bar" id="progressBar" style="width:0%"></div>
<div class="stats">
  Scanned: <span id="statScanned">-</span> | Found: <span id="statFound">-</span> | Elapsed: <span id="statElapsed">-</span>s
</div>
<div style="max-height:calc(100vh - 220px);overflow:auto">
<table id="tbl"><thead><tr>
  <th onclick="sortBy('probability')">Prob</th>
  <th onclick="sortBy('code')">Code</th>
  <th>Name</th>
  <th onclick="sortBy('decline_pct')">Decline</th><th>RSI</th>
  <th>PE</th><th>PB</th><th>Turn%</th><th>Gain60d</th>
  <th>Peak Date</th><th>Trough Date</th>
  <th>Warnings</th>
</tr></thead><tbody></tbody></table>
</div>
<div id="toast"></div>

<script>
let results=[],sortKey='probability',sortDir=-1,pollTimer=null;

const API='/api/plugin/pullback_scanner';

async function startScan(){
  document.getElementById('btnScan').disabled=true;
  document.getElementById('progressBar').style.width='0%';
  results=[];
  renderTable();
  const minProb=document.getElementById('minProb').value;
  const workers=document.getElementById('workers').value;
  const resp=await fetch(API+'/scan?min_probability='+minProb+'&max_workers='+workers);
  const data=await resp.json();
  if(data.status==='started'||data.status==='already_running'){
    pollTimer=setInterval(pollStatus,1000);
  }
}

async function pollStatus(){
  const resp=await fetch(API+'/status');
  const s=await resp.json();
  const done=s.progress[0],total=s.progress[1];
  document.getElementById('progressBar').style.width=(total>0?done/total*100:0)+'%';
  document.getElementById('statScanned').textContent=done+'/'+total;
  document.getElementById('statElapsed').textContent=s.elapsed;
  if(!s.running){
    clearInterval(pollTimer);
    document.getElementById('btnScan').disabled=false;
    document.getElementById('progressBar').style.width='100%';
    loadResults();
  }
}

async function loadResults(){
  const hideWarn=document.getElementById('hideWarn').checked;
  let all=[],page=1;
  while(true){
    const resp=await fetch(API+'/results?page='+page+'&page_size=200&hide_warnings='+hideWarn);
    const data=await resp.json();
    all=all.concat(data.rows);
    if(all.length>=data.total) break;
    page++;
  }
  results=all;
  document.getElementById('statFound').textContent=all.length;
  sortData();
}

function sortData(){
  results.sort((a,b)=>{
    let va=a[sortKey],vb=b[sortKey];
    if(typeof va==='string') va=va.toLowerCase(),vb=vb.toLowerCase();
    return (va>vb?1:va<vb?-1:0)*sortDir;
  });
  renderTable();
}

function sortBy(key){
  if(sortKey===key) sortDir*=-1; else{sortKey=key;sortDir=-1;}
  sortData();
}

function renderTable(){
  const tbody=document.querySelector('#tbl tbody');
  const hideWarn=document.getElementById('hideWarn').checked;
  let rows=results;
  if(hideWarn) rows=rows.filter(r=>!r.warnings||r.warnings.length===0);
  tbody.innerHTML=rows.map(r=>{
    const probColor=r.probability>=90?'#00ba7c':r.probability>=75?'#1d9bf0':r.probability>=50?'#ffd700':'#f4212e';
    const tags=r.warnings&&r.warnings.length>0?r.warnings.map(w=>'<span class=\"tag tag-warn\">'+w+'</span>').join(' '):'<span class=\"tag tag-ok\">clean</span>';
    return '<tr class=\"'+(r.warnings&&r.warnings.length>0?'warn-row':'')+'\">'+
      '<td><span class=\"prob-bar\" style=\"width:'+(r.probability/4)+'px;background:'+probColor+'\"></span><b style=\"color:'+probColor+'\">'+r.probability+'%</b></td>'+
      '<td>'+r.code+'</td><td>'+r.name+'</td>'+
      '<td>'+r.decline_pct+'%</td>'+
      '<td><span class="tooltip" title="'+(r.rsi_label||'')+'">'+(r.rsi!=null?r.rsi.toFixed(0):'-')+'</span></td>'+'<td>'+(r.pe_ttm!=null?r.pe_ttm.toFixed(1):'-')+'</td>'+
      '<td>'+(r.pb_mrq!=null?r.pb_mrq.toFixed(2):'-')+'</td>'+
      '<td>'+(r.turn!=null?r.turn.toFixed(1):'-')+'%</td>'+
      '<td>'+(r.gain_60d!=null?(r.gain_60d>0?'+':'')+r.gain_60d.toFixed(1)+'%':'-')+'</td>'+
      '<td>'+r.peak_date+'</td><td>'+r.trough_date+'</td>'+
      '<td>'+tags+'</td></tr>';
  }).join('');
}

// Auto-load results on page open
(async function(){
  const resp=await fetch(API+'/status');
  const s=await resp.json();
  document.getElementById('statScanned').textContent=(s.progress[1]>0?s.progress[0]+'/'+s.progress[1]:'idle');
  if(s.result_count>0) loadResults();
})();
</script>
</body>
</html>"""
