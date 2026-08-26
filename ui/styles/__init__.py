"""主题加载：dark/light QSS。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

_STYLES_DIR = Path(__file__).resolve().parent

THEMES = ("dark", "light")


def load_qss(name: str) -> str:
    """读取主题 QSS 文件内容。"""
    name = name if name in THEMES else "dark"
    path = _STYLES_DIR / f"{name}.qss"
    return path.read_text(encoding="utf-8")


def apply_theme(app: QApplication, name: str) -> None:
    """应用主题到全局 QApplication。"""
    app.setStyleSheet(load_qss(name))
