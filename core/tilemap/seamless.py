"""无缝化算法（确定性、无 AI 依赖）。

1. 全向纹理无缝：镜像相位平均（水平镜像均值 + 垂直镜像均值再平均）——
   数学上保证 left==right、top==bottom 逐像素相等（周期平铺零接缝），
   两条镜像中线的残影互相稀释；随后重量化回原调色板恢复像素硬边与颜色数。
2. 墙面/边界瓦片轴向无缝：把边界朝向旋转到顶部，仅沿边界方向做镜像平均
   （边界线带不动），再把边界线带统一为同一颜色（四张边瓦片共享线色），
   保证长墙无限重复、四个朝向风格一致。
3. 转角瓦片：不直接修 AI 的角瓦片，而是由「无缝中心纹理 + 统一线色」按
   位掩码构图推导（见 autotile.derive_corners），构造上保证与墙面/中心
   零接缝；AI 角瓦片仅可选地以低权重混合内部细节。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from core.processing.pixelizer import extract_dominant_palette, map_to_palette
from .tiles import EDGE_NAMES, BaseTileSet

logger = logging.getLogger("PixelAnimIDE.tilemap.seamless")

RGB = Tuple[int, int, int]


def _requantize_rgba(img: Image.Image, source: Image.Image, extra: Optional[RGB] = None, max_colors: int = 31) -> Image.Image:
    """重量化到 source 主色（+ 可选附加色），恢复像素画硬边与帧间一致性。"""
    palette = list(extract_dominant_palette(source.convert("RGB"), max_colors))
    if extra is not None:
        lc = (int(extra[0]), int(extra[1]), int(extra[2]))
        if lc not in palette:
            palette.append(lc)
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    q = map_to_palette(rgba.convert("RGB"), palette).convert("RGBA")
    q.putalpha(alpha)
    return q


def make_texture_seamless(img: Image.Image, band: Optional[int] = None, max_colors: int = 32) -> Image.Image:
    """把纹理修正为全向无缝瓦片（确定性；输入输出同尺寸 RGBA）。

    偏移错位缝合（Offset Quilt）：把纹理平移半格，仅对中央十字接缝的
    窄带（默认每侧 tile//8，≥2px）做交叉淡化，再平移回来——接缝变成
    新的左右/上下边缘，两侧像素完全相等（周期平铺零接缝）；纹理主体
    细节原样保留，不再出现全图镜像平均导致的「渐变+色块」退化。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    s = arr.shape[0]
    b = max(2, int(band or max(2, s // 8)))
    b = min(b, s // 2 - 1)

    shifted = np.roll(np.roll(arr, s // 2, axis=0), s // 2, axis=1)
    out = shifted.copy()
    c = s // 2

    def _blend(axis: int) -> None:
        """沿 axis=1（垂直缝）或 axis=0（水平缝）交叉淡化中央带。

        接缝两侧第 i 列（0 起）：i=0 严格 50/50（保证新边缘逐像素相等），
        i≥1 时以 t=i/(2b) 线性回到原纹理（带宽外的像素完全不动）。
        垂直缝先处理（源=shifted），水平缝随后处理（源=已融合的 out），
        这样第二次融合不会破坏第一次的列相等性。
        """
        src = shifted if axis == 1 else out
        for i in range(b):
            t = 0.0 if i == 0 else i / (2.0 * b)  # i=0 为接缝对，严格 50/50
            if axis == 1:
                a_idx = (c - 1 - i) % s
                b_idx = (c + i) % s
                a = src[:, a_idx].copy()
                bb = src[:, b_idx].copy()
                out[:, a_idx] = (0.5 + t) * a + (0.5 - t) * bb
                out[:, b_idx] = (0.5 - t) * a + (0.5 + t) * bb
            else:
                a_idx = (c - 1 - i) % s
                b_idx = (c + i) % s
                a = src[a_idx].copy()
                bb = src[b_idx].copy()
                out[a_idx] = (0.5 + t) * a + (0.5 - t) * bb
                out[b_idx] = (0.5 - t) * a + (0.5 + t) * bb

    _blend(1)
    _blend(0)

    # 平移回来：接缝带成为新边缘（左右/上下逐像素相等）
    out = np.roll(np.roll(out, -s // 2, axis=0), -s // 2, axis=1)
    rgb = np.clip(out[..., :3], 0, 255).astype(np.uint8)
    alpha = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
    merged = Image.fromarray(rgb, "RGB")
    q = _requantize_rgba(merged, rgba, max_colors=max_colors)
    q.putalpha(Image.fromarray(alpha, "L"))
    return q


def make_edge_seamless(img_top: Image.Image, line_width: int = 2) -> Tuple[Image.Image, RGB]:
    """把「边界朝上」的墙面瓦片修正为沿边界方向无缝。

    边界线带（顶部 line_width 行）保持原样并统一为带内中位数颜色；
    其余部分沿水平轴镜像平均（左右边缘相等）。返回 (瓦片, 线带颜色)。
    """
    s = img_top.size[0]
    lw = max(1, min(int(line_width), max(1, s // 4)))
    rgba = img_top.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    band = arr[:lw, :, :3].reshape(-1, 3)
    band_color = tuple(int(c) for c in np.median(band, axis=0))
    out = 0.5 * (arr + arr[:, ::-1, :])  # 左右无缝
    out[:lw, :, :3] = np.array(band_color, dtype=np.float32)
    out[:lw, :, 3] = 255.0
    rgb = np.clip(out[..., :3], 0, 255).astype(np.uint8)
    alpha = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
    merged = Image.fromarray(rgb, "RGB")
    q = _requantize_rgba(merged, rgba, extra=band_color, max_colors=31)
    q.putalpha(Image.fromarray(alpha, "L"))
    return q, band_color


# 旋转：把四张边瓦片统一朝向（边界朝上）处理，再转回
_EDGE_ROT = {
    "top": (None, None),
    "right": (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270),
    "bottom": (Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_180),
    "left": (Image.Transpose.ROTATE_270, Image.Transpose.ROTATE_90),
}


def make_axis_seamless(img: Image.Image, axis: str = "h", band: Optional[int] = None) -> Image.Image:
    """仅沿单轴做偏移错位缝合（保持另一轴内容完全不变）。

    用于地块/建筑边瓦片：top/bottom 沿水平重复、left/right 沿垂直重复，
    同时保留 AI 绘制的过渡艺术（不做线带统一化，避免破坏纹理）。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    s = arr.shape[0]
    b = max(2, int(band or max(2, s // 8)))
    b = min(b, s // 2 - 1)
    c = s // 2
    if axis == "h":
        shifted = np.roll(arr, c, axis=1)
        out = shifted.copy()
        for i in range(b):
            t = 0.0 if i == 0 else i / (2.0 * b)
            a = shifted[:, (c - 1 - i) % s].copy()
            bb = shifted[:, (c + i) % s].copy()
            out[:, (c - 1 - i) % s] = (0.5 + t) * a + (0.5 - t) * bb
            out[:, (c + i) % s] = (0.5 - t) * a + (0.5 + t) * bb
        out = np.roll(out, -c, axis=1)
    else:
        shifted = np.roll(arr, c, axis=0)
        out = shifted.copy()
        for i in range(b):
            t = 0.0 if i == 0 else i / (2.0 * b)
            a = shifted[(c - 1 - i) % s].copy()
            bb = shifted[(c + i) % s].copy()
            out[(c - 1 - i) % s] = (0.5 + t) * a + (0.5 - t) * bb
            out[(c + i) % s] = (0.5 - t) * a + (0.5 + t) * bb
        out = np.roll(out, -c, axis=0)
    rgb = np.clip(out[..., :3], 0, 255).astype(np.uint8)
    alpha = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
    q = _requantize_rgba(Image.fromarray(rgb, "RGB"), rgba, max_colors=48)
    q.putalpha(Image.fromarray(alpha, "L"))
    return q


def _union_palette(tiles, cap: int = 64):
    """多张瓦片主色的并集（保持帧间/瓦片间颜色一致，按频率截断）。"""
    out = []
    for t in tiles:
        for c in extract_dominant_palette(t.convert("RGB"), 16):
            if c not in out:
                out.append(c)
        if len(out) >= cap:
            break
    return out[:cap]


def prepare_terrain_set(base: BaseTileSet, max_colors: int = 64, band: Optional[int] = None) -> BaseTileSet:
    """地块/特征瓦片组预处理（保留 AI 艺术，仅做必要无缝化）。

    - 中心：全向偏移错位缝合（纹理细节保留）；
    - 四边：沿边界方向单轴缝合（过渡艺术不动、不做线带统一）；
    - 四角：原样保留（构图时直接取四分之一块）；
    - 全部重量化到九宫格并集调色板，保证颜色家族一致。
    """
    center = make_texture_seamless(base.center, band=band, max_colors=max_colors)
    edges: Dict[str, Image.Image] = {}
    for name in EDGE_NAMES:
        rot_in, rot_out = _EDGE_ROT[name]
        oriented = base.edges[name].transpose(rot_in) if rot_in else base.edges[name]
        processed = make_axis_seamless(oriented, axis="h", band=band)
        edges[name] = processed.transpose(rot_out) if rot_out else processed
    palette = _union_palette([center] + list(edges.values()) + list(base.corners.values()), cap=max_colors)
    quant = lambda im: _requantize_rgba_palette(im, palette)
    return BaseTileSet(
        size=base.size,
        center=quant(center),
        edges={n: quant(t) for n, t in edges.items()},
        corners={n: quant(t) for n, t in base.corners.items()},
        line_color=base.line_color,
        line_width=base.line_width,
    )


def _requantize_rgba_palette(img: Image.Image, palette) -> Image.Image:
    """按给定调色板量化（保留 alpha）。"""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    q = map_to_palette(rgba.convert("RGB"), palette).convert("RGBA")
    q.putalpha(alpha)
    return q


def process_base_set(
    base: BaseTileSet,
    max_colors: int = 32,
    detail_keep: float = 0.3,
    line_width: Optional[int] = None,
) -> BaseTileSet:
    """把 AI 裁出的 9 张基础瓦片处理成自洽无缝瓦片组。

    - 中心：全向无缝；
    - 四边：沿边界方向无缝 + 统一边界线色；
    - 四角：由中心纹理 + 统一线色构图推导（与墙面/中心零接缝），
      可选混合 AI 角瓦片内部细节；
    - 全部重量化到「中心主色 + 线色」调色板，保证整组颜色一致。
    """
    from .autotile import derive_corners

    s = base.size
    lw = max(1, int(line_width or base.line_width or max(1, s // 16)))
    center = make_texture_seamless(base.center, max_colors=max_colors)

    edge_tops: Dict[str, Image.Image] = {}
    band_colors: Dict[str, RGB] = {}
    for name in EDGE_NAMES:
        rot_in, rot_out = _EDGE_ROT[name]
        oriented = base.edges[name].transpose(rot_in) if rot_in else base.edges[name]
        processed, band_color = make_edge_seamless(oriented, line_width=lw)
        edge_tops[name] = processed.transpose(rot_out) if rot_out else processed
        band_colors[name] = band_color

    # 统一线色：四边线带颜色取中位数，再回填到全部边瓦片线带
    colors = np.array([band_colors[n] for n in EDGE_NAMES], dtype=np.float32)
    line_color: RGB = tuple(int(c) for c in np.median(colors, axis=0))
    edges: Dict[str, Image.Image] = {}
    for name in EDGE_NAMES:
        tile = edge_tops[name]
        arr = np.asarray(tile.convert("RGBA")).copy()
        band = {"top": (slice(0, lw), slice(None)),
                "bottom": (slice(s - lw, s), slice(None)),
                "left": (slice(None), slice(0, lw)),
                "right": (slice(None), slice(s - lw, s))}[name]
        arr[band][..., :3] = np.array(line_color, dtype=np.uint8)
        arr[band][..., 3] = 255
        edges[name] = _requantize_rgba(
            Image.fromarray(arr, "RGBA"), center, extra=line_color, max_colors=max_colors
        )

    corners = derive_corners(
        center, line_color, line_width=lw,
        ai_corners=base.corners, detail_keep=detail_keep,
    )
    return BaseTileSet(
        size=s,
        center=center,
        edges=edges,
        corners=corners,
        line_color=line_color,
        line_width=lw,
    )
