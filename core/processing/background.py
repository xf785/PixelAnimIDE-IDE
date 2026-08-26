"""背景去除：颜色键抠图（默认白色背景），输出带 Alpha 通道的图像。

步骤（对应文档 10.2）：
1. 转 RGBA。
2. 按颜色距离阈值生成背景掩膜。
3. 形态学开/闭运算清理掩膜（去噪点、填空洞）。
4. 可选羽化：掩膜边界像素做半透明过渡，减少硬边。
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("PixelAnimIDE.processing.background")

RGB = Tuple[int, int, int]


def remove_background(
    img: Image.Image,
    key_color: RGB = (255, 255, 255),
    tolerance: int = 30,
    feather: int = 0,
    edge_clean: bool = True,
    erode: int = 0,
) -> Image.Image:
    """把接近 key_color 的像素设为透明，返回 RGBA 图像。

    erode>0 时对前景掩膜做「内缩」（形态学腐蚀）N 像素，消掉对象边缘
    残留的白边/白晕（halo）。
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.int16)
    rgb = arr[..., :3]
    key = np.array(key_color, dtype=np.int16)
    dist = np.abs(rgb - key).sum(axis=-1)  # L1 颜色距离

    mask = dist <= tolerance
    if edge_clean:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        # 开运算：腐蚀->膨胀，去掉背景掩膜中的孤立噪点
        mask_img = mask_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        # 闭运算：膨胀->腐蚀，填掉前景内部的小洞
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        mask = np.array(mask_img) > 127

    if erode > 0:
        # 前景内缩 N 像素：去掉边缘白边/白晕
        mask = ~_erode_fg(mask, erode)

    alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
    alpha[mask] = 0

    if feather > 0:
        # 边界过渡带：距离在 (tolerance, tolerance+feather] 的像素做半透明
        band = (dist > tolerance) & (dist <= tolerance + feather)
        frac = 1.0 - (dist[band] - tolerance) / float(feather)
        alpha[band] = (frac * 255).astype(np.uint8)

    out = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA")


def _erode_fg(mask: np.ndarray, erode: int) -> np.ndarray:
    """前景（非背景）掩膜做形态学腐蚀 N 像素。"""
    fg = Image.fromarray((~mask * 255).astype(np.uint8))
    if erode > 0:
        fg = fg.filter(ImageFilter.MinFilter(2 * erode + 1))
    return np.array(fg) > 127


def remove_white_background(img: Image.Image, tolerance: int = 30, feather: int = 0, edge_clean: bool = True) -> Image.Image:
    """去除白色背景的便捷入口。"""
    return remove_background(img, (255, 255, 255), tolerance=tolerance, feather=feather, edge_clean=edge_clean)


# --------------------------------------------------------------------------- #
# 自适应背景归一化（强制纯色背景：白 or 黑）
# --------------------------------------------------------------------------- #
def _box_mean3(rgb: np.ndarray) -> np.ndarray:
    """3x3 盒式均值（快速平坦度检测）。"""
    p = np.pad(rgb.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = p.shape[0] - 2, p.shape[1] - 2
    acc = np.zeros((h, w, p.shape[2]), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += p[dy : dy + h, dx : dx + w]
    return acc / 9.0


def _flood_from_border(cand: np.ndarray) -> np.ndarray:
    """从四条边 flood-fill 候选掩膜，返回连通区域。"""
    h, w = cand.shape
    visited = np.zeros_like(cand)
    stack = deque()
    for x in range(w):
        if cand[0, x]:
            visited[0, x] = True
            stack.append((0, x))
        if cand[h - 1, x]:
            visited[h - 1, x] = True
            stack.append((h - 1, x))
    for y in range(h):
        if cand[y, 0]:
            visited[y, 0] = True
            stack.append((y, 0))
        if cand[y, w - 1]:
            visited[y, w - 1] = True
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and cand[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))
    return visited


def normalize_background(
    img: Image.Image,
    bg_tol: int = 48,
    flat_tol: int = 26,
    subject_lum_threshold: float = 0.74,
    max_coverage: float = 0.92,
) -> Tuple[Image.Image, Optional[RGB], Optional[np.ndarray]]:
    """自适应背景归一化：把背景强制为纯白或纯黑（RGBA 输出）。

    背景判定：与图像边缘连通、局部平坦、且颜色接近「边缘主色」的区域
    （平坦度能防止浅色主体被误当成背景吃掉）。

    填充颜色自适应：
    - 主体（非背景区域）平均亮度偏高（浅色系，甚至极淡色）→ 背景强制纯黑；
    - 否则背景强制纯白。

    返回 (归一化图像, 填充色或 None, 背景掩膜或 None)。
    无法可靠识别主体（如整图纯色）时返回 (原图 RGBA, None, None) 不处理。
    """
    rgba = img.convert("RGBA")
    rgb = np.array(rgba.convert("RGB"), dtype=np.int16)
    h, w = rgb.shape[:2]

    border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]])
    bg_color = np.median(border, axis=0)  # 边缘主色（抗噪）
    dist = np.abs(rgb - bg_color).sum(axis=-1)
    flat = np.abs(rgb - _box_mean3(rgb)).sum(axis=-1) <= flat_tol
    cand = flat & (dist <= bg_tol)

    visited = _flood_from_border(cand)
    if not visited.any() or visited.mean() > max_coverage:
        # 没有可识别的背景/没有主体（整图近一色）→ 不处理
        return rgba, None, None

    subject = rgb[~visited]
    lum = (
        0.299 * subject[..., 0] + 0.587 * subject[..., 1] + 0.114 * subject[..., 2]
    ).mean() / 255.0
    fill: RGB = (0, 0, 0) if lum > subject_lum_threshold else (255, 255, 255)
    logger.debug("背景归一化：主体平均亮度 %.2f -> 填充 %s", lum, fill)

    out = rgb.copy()
    out[visited] = fill
    result = np.dstack([out.astype(np.uint8), np.array(rgba)[..., 3]])
    return Image.fromarray(result, "RGBA"), fill, visited


def apply_background_mask(img: Image.Image, mask: np.ndarray, feather: int = 8, erode: int = 0) -> Image.Image:
    """按背景掩膜设透明（比颜色键抠图更精确，不误伤同色主体/描边）。

    feather>0 时对掩膜边缘做半透明过渡；erode>0 时前景内缩 N 像素去白边。
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3].astype(np.int16)
    if erode > 0:
        mask = ~_erode_fg(mask, erode)
    alpha[mask] = 0
    if feather > 0:
        m = Image.fromarray((mask * 255).astype(np.uint8))
        dilated = np.array(m.filter(ImageFilter.MaxFilter(3))) > 127
        band = dilated & ~mask
        alpha[band] = 140  # 边缘带半透明
    arr[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def whiten_background(img: Image.Image, **kwargs) -> Image.Image:
    """兼容入口：自适应背景归一化（主体浅色时自动黑底），返回图像。"""
    out, _, _ = normalize_background(img, **kwargs)
    return out
