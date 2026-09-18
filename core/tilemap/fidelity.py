"""画质修正：AI 底图 → 瓦片纹理改为「整数倍块众数降采样 + 1px 边缘焊合」。

原实现的问题（就是"成品比文生图差很多"的根因）：
1. `make_tile_texture` 先偏移错位缝合（整图 roll 半格 + 十字带交叉淡化），缩放后又缝合一次 ——
   两次 roll 把整幅图平移了 25%，并淡化掉约 25% 的像素（32 格上 8 列 + 8 行），细节被"抹平/重影"✗；
2. inset 裁切后的尺寸与瓦片尺寸往往不是整数倍（例：115 → 32），`resize_tile` 只能退回最近邻，
   像素格色被采丢、出现锯齿 ✗；
3. 九格中值纹理同样再走一遍缝合与量化 ✗。

新实现：
- **整数倍窗口**：从 AI 格子里选一个 `k × target` 的窗口（k = 整数倍率），用块众数降采样一次到位
  （AI 按 k 倍画像素时能**精确还原**原始像素格）；
- **最佳偏移**：在 k×k 个可选窗口里挑「左右 / 上下边缘最接近」的那个，让焊缝本来就几乎看不见；
- **1px 焊合**：只把最外一圈行列改成两侧边缘的均值（左右相等、上下相等），除这 2 列 2 行外
  **其余像素与 AI 原图逐像素一致** —— 既保持「相邻瓦片共享边逐像素相等」的不变量，又不再毁画质；
- **不再量化**（默认 `max_colors=0`）：像素画本身色数有限，量化只会破坏渐变/抖动。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger("PixelFoundry.tilemap.seamless")


def _crop(img: Image.Image, box) -> Image.Image:
    return img.convert("RGBA").crop(box)


def _block_mode_downscale(arr: np.ndarray, target: int) -> Optional[np.ndarray]:
    """整数倍块众数降采样（保持像素格色；非整数倍返回 None）。"""
    from .tiles import _mode_downscale

    h, w = arr.shape[:2]
    if target <= 0 or h % target or w % target or h == target:
        return None
    return _mode_downscale(arr, target, target)


def _edge_cost(arr: np.ndarray) -> float:
    """窗口的左右 / 上下边缘差异（越小越接近"天生无缝"）。"""
    left, right = arr[:, 0].astype(np.float32), arr[:, -1].astype(np.float32)
    top, bottom = arr[0, :].astype(np.float32), arr[-1, :].astype(np.float32)
    return float(np.abs(left - right).mean() + np.abs(top - bottom).mean())


def faithful_tile_texture(
    tile: Image.Image,
    target: int,
    inset_frac: float = 0.04,
    max_colors: int = 0,
) -> Image.Image:
    """AI 一格 → `target×target` 无缝纹理（**最大化保真**，见模块文档）。"""
    rgba = tile.convert("RGBA")
    src = np.asarray(rgba)
    h, w = src.shape[:2]
    # 优先：整格正好是瓦片的整数倍（我们请求的生图尺寸就是 cells×cell_px，通常成立）
    #         —— 此时用**整格**、不裁边、倍率精确，保真度最高（AI 按 k 倍画的像素格被逐格还原）
    full = min(h, w)
    if full >= target and full % target == 0:
        k = full // target
        win = full
        inset = 0
    else:
        # 回退：留一点边（甩掉残留格线）再取最大整数倍窗口
        inset = int(round(full * max(0.0, min(0.2, inset_frac))))
        k = max(1, (full - 2 * inset) // max(1, target))
        win = k * target
        if win > full:
            k = max(1, full // max(1, target))
            win = k * target
    # 在可选窗口里挑边缘最接近的一个（焊缝更不可见）
    best_box, best_cost, best_arr = None, None, None
    # 关键：窗口偏移必须是 k 的整数倍，才能与 AI 的像素格对齐
    #（否则块众数降采样会跨格取值，颜色/形状都会错位 —— 这正是"成品比底图差"的另一半原因）
    inset_k = (inset // max(1, k)) * max(1, k)
    ys = list(range(inset_k, h - win + 1, max(1, k))) or [max(0, ((h - win) // 2 // max(1, k)) * max(1, k))]
    xs = list(range(inset_k, w - win + 1, max(1, k))) or [max(0, ((w - win) // 2 // max(1, k)) * max(1, k))]
    for y0 in ys:
        for x0 in xs:
            cand = src[y0:y0 + win, x0:x0 + win]
            small = _block_mode_downscale(cand, target)
            probe = small if small is not None else cand[::max(1, win // target), ::max(1, win // target)][:target, :target]
            cost = _edge_cost(probe)
            if best_cost is None or cost < best_cost:
                best_box, best_cost, best_arr = (x0, y0, x0 + win, y0 + win), cost, cand
    if best_arr is None:
        best_box = (0, 0, min(h, win), min(w, win))
        best_arr = src[:win, :win]
    small = _block_mode_downscale(best_arr, target)
    if small is None:                                # 非整数倍：面积平均（抗锯齿但不引入重影）
        img = Image.fromarray(best_arr, "RGBA").resize((target, target), Image.Resampling.BOX)
        small = np.asarray(img)
    if max_colors and max_colors > 0:
        small = np.asarray(Image.fromarray(small, "RGBA").convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=int(max_colors)).convert("RGBA"))
    return weld_edges(Image.fromarray(small, "RGBA"))


def weld_edges(img: Image.Image) -> Image.Image:
    """1px 焊合：左右边缘逐像素相等、上下边缘逐像素相等（其余像素原样保留）。"""
    arr = np.asarray(img.convert("RGBA")).copy()
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return Image.fromarray(arr, "RGBA")
    left, right = arr[:, 0].astype(np.int32), arr[:, -1].astype(np.int32)
    seam = ((left + right + 1) // 2).astype(np.uint8)
    arr[:, 0] = seam
    arr[:, -1] = seam
    top, bottom = arr[0, :].astype(np.int32), arr[-1, :].astype(np.int32)
    seam2 = ((top + bottom + 1) // 2).astype(np.uint8)
    arr[0, :] = seam2
    arr[-1, :] = seam2
    return Image.fromarray(arr, "RGBA")


def make_tile_texture(
    tile: Image.Image,
    target: int,
    inset_frac: float = 0.04,
    max_colors: int = 0,
) -> Image.Image:
    """兼容旧签名：AI 一格 → 无缝瓦片纹理（内部改用保真提取 + 边缘焊合）。"""
    return faithful_tile_texture(tile, int(target), inset_frac=inset_frac, max_colors=max_colors)


def median_tile_texture(
    tiles: Sequence[Image.Image],
    target: int,
    inset_frac: float = 0.04,
    max_colors: int = 0,
) -> Image.Image:
    """九格逐像素中位数（抹掉只出现在个别格里的文字/贴花）→ 保真降采样 + 边缘焊合。

    注意：中值本身是"投票去噪"，之后**不再**做偏移缝合与量化，避免二次毁画质。
    """
    crops: List[np.ndarray] = []
    for t in tiles:
        rgba = t.convert("RGBA")
        inset = int(round(min(rgba.size) * max(0.0, min(0.2, inset_frac))))
        box = (inset, inset, rgba.size[0] - inset, rgba.size[1] - inset)
        crops.append(np.asarray(rgba.crop(box), dtype=np.float32))
    h = min(c.shape[0] for c in crops)
    w = min(c.shape[1] for c in crops)
    med = np.median(np.stack([c[:h, :w] for c in crops], axis=0), axis=0).astype(np.uint8)
    return faithful_tile_texture(Image.fromarray(med, "RGBA"), int(target),
                                 inset_frac=0.0, max_colors=max_colors)
