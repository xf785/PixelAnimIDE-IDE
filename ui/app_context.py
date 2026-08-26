"""应用上下文：在页面之间共享的配置管理器与用户设置。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.api_config import APIConfigManager
from config.settings import UI_SETTINGS_FILE

logger = logging.getLogger("PixelAnimIDE.ui.context")


class UISettings:
    """轻量 UI 偏好存储（主题、默认输出目录等），JSON 落盘。"""

    def __init__(self, path: Path | str = UI_SETTINGS_FILE):
        self.path = Path(path)
        self.data: dict = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("UI 设置读取失败: %s", exc)
                self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()


@dataclass
class AppContext:
    """全局共享对象。"""

    api: APIConfigManager
    ui_settings: UISettings
