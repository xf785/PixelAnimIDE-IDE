"""建筑/墙体艺术管线的共用小工具（原先在 walls.py 与 buildings.py 各写了一份）。

- :func:`key_white` —— 白键抠背景（建筑外区域透明，可叠放）；
- :func:`outline_color` —— 由不透明像素推导轮廓色（最暗 15% 的中位数）。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from core.processing import background as bg

#: 轮廓色兜底值（图块全透明时使用；建筑管线用纯黑，墙体管线用近黑）
DEFAULT_OUTLINE = (34, 30, 32)


def key_white(img: Image.Image, tolerance: int = 42) -> Image.Image:
    """白键抠除背景：接近白色的像素透明（建筑外区域透明，可叠放）。"""
    return bg.remove_background(
        img.convert("RGBA"), key_color=(255, 255, 255), tolerance=tolerance,
        feather=0, edge_clean=True, mode="hybrid",
    )


def outline_color(img: Image.Image, fallback: Tuple[int, int, int] = DEFAULT_OUTLINE) -> Tuple[int, int, int]:
    """轮廓色 = 不透明像素中最暗 15% 的中位数（稳健、贴近描边色）。"""
    arr = np.asarray(img.convert("RGBA"))
    mask = arr[..., 3] > 32
    if not mask.any():
        return tuple(fallback)
    rgb = arr[mask][..., :3].astype(np.float32)
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    k = max(1, int(len(lum) * 0.15))
    return tuple(int(c) for c in np.median(rgb[np.argsort(lum)[:k]], axis=0))
