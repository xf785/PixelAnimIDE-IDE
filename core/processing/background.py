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
    mode: str = "global",
    non_contiguous_tolerance: Optional[int] = None,
    large_region_threshold: int = 256,
    large_region_bonus: int = 24,
) -> Image.Image:
    """把接近 key_color 的像素设为透明，返回 RGBA 图像。

    erode>0 时对前景掩膜做「内缩」（形态学腐蚀）N 像素，消掉对象边缘
    残留的白边/白晕（halo）。

    mode 抠图策略（借鉴 FrameRonin 的色度键分级容差思路）：
    - "global"：全局容差（默认，原行为）。主体内部接近背景色的像素也会被删。
    - "contiguous"：只删除与图像边缘连通的背景区域（flood-fill），
      主体内部被包围的同色像素（如白眼球、白色高光）不受影响。
    - "hybrid"：连通背景用 tolerance，非连通区域用更小的
      non_contiguous_tolerance（默认 tolerance//2，下限 4）——大块背景容差大、
      主体内部同色像素容差小，兼顾干净去背与细节保护。
    - "adaptive"：hybrid 基础上，非连通候选区域按连通域大小分级——
      像素数 > large_region_threshold 的大区域（如背景里的色块/水印）额外获得
      large_region_bonus 容差，小区域维持小容差。
    """
    if mode not in ("global", "contiguous", "hybrid", "adaptive"):
        raise ValueError(f"未知抠图模式: {mode!r}")
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.int16)
    rgb = arr[..., :3]
    key = np.array(key_color, dtype=np.int16)
    dist = np.abs(rgb - key).sum(axis=-1)  # L1 颜色距离

    tol_eff = np.full(dist.shape, tolerance, dtype=np.int16)
    mask = dist <= tolerance

    if mode != "global":
        tol2 = max(4, tolerance // 2) if non_contiguous_tolerance is None else int(non_contiguous_tolerance)
        conn = _flood_from_border(mask)
        if mode == "contiguous":
            mask = conn
        elif mode == "hybrid":
            mask = conn | (dist <= tol2)
            tol_eff = np.where(conn, tolerance, tol2)
        else:  # adaptive
            cand = (dist <= tolerance + large_region_bonus) & ~conn
            labels, sizes = _label_region_sizes(cand)
            eff = np.where(conn, tolerance, tol2).astype(np.int16)
            for region_id, size in enumerate(sizes):
                if size > large_region_threshold:
                    eff[labels == region_id] = tolerance + large_region_bonus
            tol_eff = eff
            mask = conn | (dist <= eff)

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
        # 边界过渡带：距离在 (tol_eff, tol_eff+feather] 的像素做半透明；
        # 非全局模式下只在被删除掩膜的邻域内羽化（不羽化主体内部同色像素）
        band = (dist > tol_eff) & (dist <= tol_eff + feather)
        if mode != "global":
            band = band & _dilate_mask(mask)
        frac = 1.0 - (dist[band] - tol_eff[band]) / float(feather)
        alpha[band] = (frac * 255).astype(np.uint8)

    out = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA")


def _label_region_sizes(cand: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """4 邻域连通域标记候选掩膜，返回 (区域 id 图, 各区域像素数)。

    区域 id 从 0 起；非候选像素为 -1。用于 adaptive 抠图的分级容差。
    """
    h, w = cand.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    sizes: List[int] = []
    lab = 0
    ys, xs = np.nonzero(cand)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if labels[y, x] != -1:
            continue
        stack = [(y, x)]
        labels[y, x] = lab
        cnt = 0
        while stack:
            cy, cx = stack.pop()
            cnt += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and cand[ny, nx] and labels[ny, nx] == -1:
                    labels[ny, nx] = lab
                    stack.append((ny, nx))
        sizes.append(cnt)
        lab += 1
    return labels, sizes


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """3x3 膨胀掩膜（用于羽毛边界限制在删除区域邻域）。"""
    m = Image.fromarray((mask * 255).astype(np.uint8))
    return np.array(m.filter(ImageFilter.MaxFilter(3))) > 127


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

    feather>0 时对掩膜边缘做「渐晕过渡」：前景 alpha 做半径为 feather/2 的
    BoxBlur，得到约 feather 像素宽的平滑渐变带（feather 数值真正生效，
    而非固定半透明值）；erode>0 时前景内缩 N 像素去白边。
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3].astype(np.int16)
    if erode > 0:
        mask = ~_erode_fg(mask, erode)
    # 前景保留原 alpha，背景置 0
    fg = ~mask
    base = np.where(fg, alpha, 0).astype(np.uint8)
    if feather > 0:
        r = max(1, int(round(feather / 2.0)))
        # 限制半径：避免半径接近图像尺寸时整图被羽化掉
        r = min(r, max(1, min(img.width, img.height) // 4))
        base = np.array(Image.fromarray(base).filter(ImageFilter.BoxBlur(r))).astype(np.uint8)
    arr[..., 3] = base
    return Image.fromarray(arr, "RGBA")


def whiten_background(img: Image.Image, **kwargs) -> Image.Image:
    """兼容入口：自适应背景归一化（主体浅色时自动黑底），返回图像。"""
    out, _, _ = normalize_background(img, **kwargs)
    return out


# --------------------------------------------------------------------------- #
# 统一背景处理入口（solo / ide / sprite 共用）
# --------------------------------------------------------------------------- #
def border_key_color(img: Image.Image) -> Optional[RGB]:
    """从图像四周边缘主色推断背景键色（中位数抗噪，忽略全透明边缘像素）。

    用于背景归一化无法给出精确掩膜时的颜色键回退：对 AI 生成图常见的浅灰/
    米白背景，键色贴合实际背景比固定纯白抠得更干净、白边残留更少。
    图像过小或边缘全部透明时返回 None。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        return None
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    if int(alpha.min()) < 255:
        # 已含透明：只统计不透明边缘像素，避免透明区颜色干扰键色估计
        parts = [
            rgb[0, :][alpha[0, :] > 0],
            rgb[-1, :][alpha[-1, :] > 0],
            rgb[:, 0][alpha[:, 0] > 0],
            rgb[:, -1][alpha[:, -1] > 0],
        ]
        border = np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.int16)
    else:
        border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]])
    if border.shape[0] == 0:
        return None
    med = np.median(border, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def process_background(
    img: Image.Image,
    *,
    force_pure_bg: bool = False,
    remove_bg: bool = False,
    key_color: Optional[RGB] = None,
    tolerance: int = 30,
    feather: int = 8,
    erode: int = 0,
) -> Tuple[Image.Image, bool]:
    """统一背景处理入口（solo / ide / sprite 共用），逐帧调用。

    - force_pure_bg：自适应背景归一化（主体浅色→黑底、否则白底），
      成功时得到精确背景掩膜，返回 normalized=True；
    - remove_bg：
        * 有精确掩膜 -> apply_background_mask 抠图（feather/erode 生效，
          不误伤主体内部同色像素）；
        * 无掩膜（归一化失败或未开启）-> 颜色键抠图：键色取 key_color，
          未指定时由 border_key_color 按图像边缘主色自动推断（贴合 AI 生成
          的发灰/米白背景，白边残留更少），并统一 hybrid 模式——连通背景大
          容差、主体内部同色像素小容差保护；feather/erode 同样生效。

    返回 (处理后的图像, 是否完成了背景归一化)。
    """
    out = img
    normalized = False
    mask: Optional[np.ndarray] = None
    if force_pure_bg:
        out, _fill, mask = normalize_background(out)
        normalized = mask is not None
    if remove_bg:
        if mask is not None:
            out = apply_background_mask(out, mask, feather=feather, erode=erode)
        else:
            key = key_color if key_color is not None else (border_key_color(out) or (255, 255, 255))
            out = remove_background(
                out,
                key_color=key,
                tolerance=tolerance,
                non_contiguous_tolerance=None,
                feather=feather,
                erode=erode,
                mode="hybrid",
            )
    return out, normalized
