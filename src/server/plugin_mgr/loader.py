"""
Stock V0.2 — 插件加载器

扫描 plugins/ 目录，读取 manifest.json，注册插件。
"""

import json
import logging
import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 项目根目录 (stock_V0.1/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"


@dataclass
class PluginManifest:
    """插件清单"""
    name: str
    version: str
    label: str
    description: str = ""
    author: str = ""
    menu: dict = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)
    depends: list[str] = field(default_factory=list)

    path: str = ""
    loaded: bool = False
    error: str = ""


class PluginManager:
    """插件管理器"""

    def __init__(self):
        self._plugins: dict[str, PluginManifest] = {}

    def discover(self) -> list[PluginManifest]:
        """扫描插件目录"""
        self._plugins.clear()

        if not PLUGINS_DIR.exists():
            logger.warning(f"插件目录不存在: {PLUGINS_DIR}")
            return []

        for entry in sorted(PLUGINS_DIR.iterdir()):
            if not entry.is_dir():
                continue

            manifest_path = entry / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = self._load_manifest(manifest_path, str(entry))
                self._plugins[manifest.name] = manifest
                logger.info(f"发现插件: {manifest.name} v{manifest.version} — {manifest.label}")
            except Exception as e:
                logger.error(f"加载插件失败 {entry.name}: {e}")

        return list(self._plugins.values())

    def load_backend(self, manifest: PluginManifest, app):
        """为 FastAPI app 注册插件后端路由"""
        routes_path = f"plugins.{manifest.name}.backend.routes"
        try:
            module = import_module(routes_path)
            if hasattr(module, "router"):
                prefix = f"/api/plugin/{manifest.name}"
                app.include_router(module.router, prefix=prefix)
                manifest.loaded = True
                logger.info(f"插件 [{manifest.name}] 路由已注册: {prefix}")
            else:
                logger.info(f"插件 [{manifest.name}] 无后端路由")
        except ImportError:
            logger.info(f"插件 [{manifest.name}] 无后端路由 (这是正常的)")

    def get_frontend_plugins(self) -> list[dict]:
        """返回前端组件清单"""
        return [{
            "name": m.name, "label": m.label, "version": m.version,
            "description": m.description, "menu": m.menu,
            "loaded": m.loaded, "error": m.error,
        } for m in self._plugins.values()]

    def get(self, name: str) -> Optional[PluginManifest]:
        return self._plugins.get(name)

    def list_all(self) -> list[PluginManifest]:
        return list(self._plugins.values())

    @property
    def count(self) -> int:
        return len(self._plugins)

    @staticmethod
    def _load_manifest(manifest_path: Path, plugin_dir: str) -> PluginManifest:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ["name", "version", "label"]:
            if key not in data:
                raise ValueError(f"manifest.json 缺少必要字段: {key}")
        return PluginManifest(
            name=data["name"], version=data["version"], label=data["label"],
            description=data.get("description", ""), author=data.get("author", ""),
            menu=data.get("menu", {}), permissions=data.get("permissions", []),
            depends=data.get("depends", []), path=plugin_dir,
        )


# 全局单例
plugin_manager = PluginManager()
