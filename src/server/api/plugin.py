"""
Stock V0.2 — 插件管理 API
"""

from fastapi import APIRouter
from src.server.plugin_mgr.loader import plugin_manager

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


@router.get("")
def list_plugins():
    """列出所有已发现的插件"""
    plugins = plugin_manager.get_frontend_plugins()
    return {"total": len(plugins), "plugins": plugins}


@router.get("/{name}")
def get_plugin(name: str):
    """获取单个插件信息"""
    manifest = plugin_manager.get(name)
    if not manifest:
        return {"error": f"插件 {name} 不存在"}
    return {
        "name": manifest.name,
        "label": manifest.label,
        "version": manifest.version,
        "description": manifest.description,
        "author": manifest.author,
        "menu": manifest.menu,
        "permissions": manifest.permissions,
        "depends": manifest.depends,
        "loaded": manifest.loaded,
    }
