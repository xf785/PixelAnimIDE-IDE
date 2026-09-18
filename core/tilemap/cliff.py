"""俯视 2.5D 地形：把「高台」与「崖壁」拆成两层瓦片。

俯视 2.5D 的关键不是把地形画成斜视，而是**规划出两套瓦片**：

1. **顶面层**（已有的地块瓦片）：高台格子照常画该地形的无缝顶面 —— 高台与平地用同一套
   47-tile 族，只是高台格子在边界处会露出崖壁；
2. **崖壁层**（本模块新增）：高台**南侧**与低地相接时，在**低地那一格的上半部分**画
   "崖壁立面"（下半部分透明，露出低地地面），因此看起来高台是"立"在低地上的。
   崖壁同样用 **16-tile 族**（左右端头/连续/上下相连），保证长崖壁逐格连续、两端收口。

`CliffArt` 由该地形的实测艺术推导（不需要额外文生图）：立面 = 地形纹理压暗 + 垂直条纹 +
顶部亮边 + 底部接触阴影 + 1px 描边，与地块层同源因此颜色天然协调。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .autotile import _disc_dilate, _side_seed, _wobble
from .tiles import BaseTileSet
from .walls import W16, W16_SLOTS

logger = logging.getLogger("PixelFoundry.tilemap.cliff")


@dataclass
class CliffArt:
    """崖壁艺术（由地形实测推导）。"""

    size: int
    texture: Image.Image                      # 地形顶面纹理（用于崖顶亮边）
    face: Image.Image                         # 崖壁立面纹理（压暗 + 条纹）
    outline: Tuple[int, int, int] = (20, 18, 20)
    face_h: int = 0                           # 立面高度（0 = 自动取 55% 瓦片高）
    edge_noise: int = 0
    art_meta: Dict = field(default_factory=dict)


def cliff_art_from_terrain(tset: BaseTileSet, edge_noise: int = 0) -> CliffArt:
    """从地块艺术推导崖壁艺术（同源配色，避免额外生成）。"""
    s = int(tset.size)
    top = tset.center.convert("RGBA").resize((s, s), Image.Resampling.NEAREST)
    arr = np.asarray(top).astype(np.float32).copy()
    # 立面：整体压暗 + 垂直条纹（模拟岩层/土层），并略带自下而上的加深
    face = arr.copy()
    face[..., :3] *= 0.72
    ys = np.arange(s)[:, None].astype(np.float32)
    face[..., :3] *= (0.86 + 0.14 * (1.0 - ys / max(1, s - 1)))[..., None]
    # 垂直条纹（岩层/土层感）：用乘法因子而不是布尔索引，避免在 (s,s,3) 视图上广播失败
    stripe = np.where(((np.arange(s)[None, :] // 3) % 2) == 0, 0.90, 1.0).astype(np.float32)
    face[..., :3] *= stripe[..., None]
    meta = getattr(tset, "art_meta", {}) or {}
    outline = meta.get("outline") or [[24, 22, 24]]
    dark = outline[0] if isinstance(outline[0], (list, tuple)) else (24, 22, 24)
    return CliffArt(
        size=s,
        texture=top,
        face=Image.fromarray(np.clip(face, 0, 255).astype(np.uint8), "RGBA"),
        outline=tuple(int(c) for c in dark[:3]),
        face_h=max(6, int(round(s * 0.55))),
        edge_noise=max(0, min(int(edge_noise), s // 8)),
        art_meta={"tile_size": s, "family": "cliff-16", "face_px": max(6, int(round(s * 0.55)))},
    )


def cliff_footprint(art: CliffArt, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """崖壁足迹：上方 `face_h` 的立面带（下半部分透明，露出低地）。

    `bits`（16-tile 族）：哪些方向还有相邻崖壁（N=上方同一条崖壁的下一格、
    W/E=左右延伸、S=下方继续）。
    """
    s = art.size
    h = max(4, min(int(art.face_h or s // 2), s - 2))
    ys, xs = np.mgrid[0:s, 0:s]
    amp = int(art.edge_noise)
    taper = max(2, s // 4)
    top = _wobble(s, _side_seed(bits, "cliff_top"), amp, taper) if amp else np.zeros(s, dtype=np.int32)
    solid = (ys >= top[None, :]) & (ys < h)
    if not (bits & W16["w"]):                       # 左端收口：左侧让出 2px 做侧壁
        solid &= xs >= 2
    if not (bits & W16["e"]):
        solid &= xs < s - 2
    return solid, ~solid


def compose_cliff_piece(art: CliffArt, bits: int) -> Image.Image:
    """合成一块崖壁拼件（RGBA，下半部分完全透明 → 叠在低地瓦片上）。"""
    s = art.size
    solid, empty = cliff_footprint(art, bits)
    face = np.asarray(art.face.convert("RGBA"))
    top_px = np.asarray(art.texture.convert("RGBA"))
    out = np.zeros((s, s, 4), dtype=np.uint8)
    out[solid] = face[solid]

    def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        pad = np.pad(mask, 1, mode="constant", constant_values=False)
        return pad[1 + dy:1 + dy + s, 1 + dx:1 + dx + s]

    # 相连侧不画描边（长崖壁逐格连续）；顶部亮边 + 底部接触阴影
    pad = np.pad(solid, 1, mode="constant", constant_values=False)
    if bits & W16["n"]:
        pad[0, :] = True
    if bits & W16["s"]:
        pad[-1, :] = True
    if bits & W16["w"]:
        pad[:, 0] = True
    if bits & W16["e"]:
        pad[:, -1] = True
    exposed = ~pad[1:-1, 1:-1]
    if solid.any():
        row0 = int(np.nonzero(solid.any(axis=1))[0][0])
        lit = solid & ~shifted(solid, -1, 0) & (np.arange(s)[:, None] < row0 + 2)
    else:
        lit = np.zeros_like(solid)
    out[lit] = np.clip(top_px[lit].astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
    out[solid & _disc_dilate(exposed, 1)] = np.array([*art.outline, 255], dtype=np.uint8)
    out[empty] = (0, 0, 0, 0)
    return Image.fromarray(out, "RGBA")


def build_cliff_set(art: CliffArt) -> Dict[str, Image.Image]:
    """崖壁 16-tile 族（键为 16-tile 槽位名，与墙体族同一命名）。"""
    return {W16_SLOTS[bits]: compose_cliff_piece(art, bits) for bits in range(16)}


def cliff_bits_for(occupied: Dict[Tuple[int, int], bool], x: int, y: int) -> int:
    """按「相邻崖壁格」求 16-tile 位（N=上方、W/E=左右、S=下方）。"""
    bits = 0
    for name, (dx, dy) in (("n", (0, -1)), ("e", (1, 0)), ("s", (0, 1)), ("w", (-1, 0))):
        if occupied.get((x + dx, y + dy)):
            bits |= W16[name]
    return bits
