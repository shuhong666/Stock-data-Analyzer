"""
Stock V0.2 — FastAPI 主入口

启动: python -m src.server.main
      或: uvicorn src.server.main:app --reload --port 8000
"""

import logging
import os
import sys
import threading
from contextlib import asynccontextmanager

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.server.api.data import router as data_router
from src.server.api.indicator import router as indicator_router
from src.server.api.plugin import router as plugin_router
from src.server.plugin_mgr.loader import plugin_manager, PROJECT_ROOT

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 快照采集守护线程
# ------------------------------------------------------------------

_snapshot_thread: threading.Thread | None = None
_snapshot_stop_event: threading.Event | None = None


def _start_snapshot_daemon():
    """在后台线程启动快照采集守护"""
    global _snapshot_thread, _snapshot_stop_event

    from src.core.scheduler.runner import run_snapshot_daemon

    _snapshot_stop_event = threading.Event()
    _snapshot_thread = threading.Thread(
        target=run_snapshot_daemon,
        kwargs={"stop_event": _snapshot_stop_event},
        name="snapshot-daemon",
        daemon=True,
    )
    _snapshot_thread.start()
    logger.info("快照采集守护线程已启动")


def _stop_snapshot_daemon():
    """停止快照采集守护线程"""
    global _snapshot_thread, _snapshot_stop_event

    if _snapshot_thread is None or not _snapshot_thread.is_alive():
        logger.info("快照守护线程未运行，无需停止")
        return

    logger.info("正在停止快照采集守护线程...")
    _snapshot_stop_event.set()
    _snapshot_thread.join(timeout=10)
    if _snapshot_thread.is_alive():
        logger.warning("快照守护线程未能在 10s 内退出（daemon 线程将随进程终止）")
    else:
        logger.info("快照采集守护线程已停止")


# ------------------------------------------------------------------
# 生命周期
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时开启快照采集，关闭时停止"""
    logger.info("服务器启动 — 开启快照采集守护")
    _start_snapshot_daemon()
    yield
    logger.info("服务器关闭 — 停止快照采集守护")
    _stop_snapshot_daemon()


# ------------------------------------------------------------------
# FastAPI 应用
# ------------------------------------------------------------------

app = FastAPI(
    title="Stock V0.2",
    description="股票数据分析平台 — Web UI + 插件系统",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — 开发环境下允许 Vue dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册核心路由
app.include_router(data_router)
app.include_router(indicator_router)
app.include_router(plugin_router)

# 发现并加载插件
plugin_manager.discover()
for p in plugin_manager.list_all():
    plugin_manager.load_backend(p, app)

# 静态文件 — 前端构建产物
static_dir = os.path.join(os.path.dirname(__file__), "static")
has_static = os.path.exists(static_dir) and os.path.exists(os.path.join(static_dir, "index.html"))

if has_static:
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/favicon.svg")
    async def serve_favicon():
        path = os.path.join(static_dir, "favicon.svg")
        if os.path.exists(path):
            return FileResponse(path)


# ------------------------------------------------------------------
# 健康检查
# ------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "plugins": plugin_manager.count,
    }


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import webbrowser

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    host = "127.0.0.1"
    port = 8000

    has_frontend = os.path.exists(static_dir) and os.path.exists(
        os.path.join(static_dir, "index.html")
    )

    if has_frontend:
        print(f"\n  Stock V0.2 — http://{host}:{port}\n")
        webbrowser.open(f"http://{host}:{port}")
    else:
        print(f"\n  Stock V0.2 API — http://{host}:{port}/docs\n"
              f"  前端未构建, 请运行: cd src/frontend && npm run dev\n")
        webbrowser.open(f"http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
