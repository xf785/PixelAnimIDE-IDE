"""全局 UI 布局比例（设置 -> 界面布局比例）。

固定像素尺寸统一经 scaled() 取值，随全局比例同步缩放；
MainWindow 在启动与设置变更时调用 set_ui_scale() 更新。
"""
from __future__ import annotations

_CURRENT: float = 1.0


def set_ui_scale(scale: float) -> None:
    """设置全局 UI 比例（0.7 ~ 1.6）。"""
    global _CURRENT
    _CURRENT = max(0.7, min(1.6, float(scale)))


def ui_scale() -> float:
    return _CURRENT


def scaled(value) -> int:
    """把基准像素值按全局比例换算为实际像素。"""
    return int(round(float(value) * _CURRENT))
