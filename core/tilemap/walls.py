"""建筑类拼件生成：**墙体 16-tile 族 + 透明外部**（可叠加在地块层上）。

## 为什么用 16-tile

墙体是「一条有厚度的带子」，一个格子只需要知道**四个方向是否与同类墙相连**
（N/E/S/W 四位），全部组合恰好 16 种，正是经典墙体的 16-tile 族：

| 位 | 形状 | 位 | 形状 |
|---|---|---|---|
| `none` | 孤立墙块 | `ns` / `ew` | 直墙（竖/横） |
| `n` `e` `s` `w` | 端头（4 向） | `ne` `es` `sw` `wn` | 转角（内角/外角同时出现在一块里） |
| `nes` `esw` `swn` `wne` | T 形（4 向） | `nesw` | 十字路口 |

对角位（47 图块那套）对墙体没有意义：墙是一维路径，不存在「对角连通」的画法。
本模块在此之上只多保留一块 **`solid`**（八邻全满时用）——用于把整片实心建筑画成
不透明大块，避免十字件在实心区域留下透明角。

## 几何（为什么这样切才无缝）

- 带子厚度 = `s - 2*margin`，即墙体带占据中间那条 `[margin, s-margin)` 的条；
- 每条**相连的边**伸出一条「臂」：N 臂 = 带子列 × 上半格，E 臂 = 带子行 × 右半格……
  四臂在格子中心汇合，于是直墙/转角/T 形/十字都是同一条规则的自然结果；
- **让位区（外部）alpha=0**：没有臂覆盖到的地方完全透明，叠加在地块上就露出地面；
- **接缝一致**：A 的 E 位与 B 的 W 位同时为满时，两块在共享边上的墙/透明分界
  完全相同（都由 `[margin, s-margin)` 决定），且取同一份网格对齐的墙纹理
  → 共享边逐像素相等（含透明段），墙带天然连续；
- 转角件把弯折的**外角倒圆**（与两条墙面相切的圆弧）、**内角做小圆角**，
  两者都在瓦片内部，不接触共享边；
- 层次：顶面提亮带（AI 顶面纹理优先）→ 立面压暗带 → 1px 描边 →
  外部再无白边（白底先经白键抠除，轮廓是程序化描边）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.processing import background as bg
from .seamless import make_tile_texture
from .tiles import BuildingSheet

logger = logging.getLogger("PixelAnimIDE.tilemap.walls")

#: 16-tile 位约定（槽位 = n | e*2 | s*4 | w*8）
W16 = {"n": 1, "e": 2, "s": 4, "w": 8}
W16_SLOTS: Tuple[str, ...] = (
    "none", "n", "e", "ne", "s", "ns", "es", "nes",
    "w", "wn", "ew", "wne", "sw", "swn", "esw", "nesw",
)
#: 图集在 16 格之后追加的扩展件（同一张图，索引 16..19）
EXTRA_SLOTS: Tuple[str, ...] = ("solid", "door_ew", "door_ns", "pillar")
ATLAS_COLS = 4
ATLAS_ROWS = 5
SOLID_SLOT = 16


@dataclass
class WallArt:
    """墙体艺术参数（纹理 + 实测几何），由 AI 建筑图推导。"""

    size: int
    texture: Image.Image                        # 墙体纹理（无缝、网格对齐）
    top: Optional[Image.Image] = None           # 顶面纹理（AI 顶面组中心格）
    door: Optional[Image.Image] = None          # 门（AI 开口组中心格）
    pillar: Optional[Image.Image] = None        # 装饰立柱（AI 立柱组中心格）
    margin: int = 8                             # 带子外侧让位深度（= 透明边宽度）
    outline: Tuple[int, int, int] = (34, 30, 32)
    corner_radius: int = 0                      # 弯折外角半径（0 = 取 margin）
    top_h: int = 3
    front_h: int = 3
    shade: float = 0.80
    lighten: float = 1.12
    edge_noise: int = 0
    art_meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 掩码与槽位
# --------------------------------------------------------------------------- #
def mask16(mask8: int) -> int:
    """八邻掩码 → 16-tile 掩码（只看四个正交位）。"""
    m = int(mask8) & 255
    out = 0
    if m & 2:
        out |= W16["n"]
    if m & 16:
        out |= W16["e"]
    if m & 64:
        out |= W16["s"]
    if m & 8:
        out |= W16["w"]
    return out


def wall16_index(mask8: int) -> int:
    """八邻掩码 → 16-tile 槽位（0..15）。"""
    return mask16(mask8)


def is_solid(mask8: int) -> bool:
    """八邻全满（含四对角）。注意：**默认不**自动改用 solid 件 —— 实心块与带子
    家族的形状不同，混用会在共享边留下透明/不透明错位；只有调用方显式要
    `SOLID_SLOT` 时才用它（适合整片实心区域）。"""
    return (int(mask8) & 255) == 255


def slot_for_mask(mask8: int) -> int:
    """八邻掩码 → 图集槽位（**始终**落在 16-tile 族内，保证接缝一致）。"""
    return wall16_index(mask8)


def slot_name(slot: int) -> str:
    return W16_SLOTS[slot] if slot < 16 else EXTRA_SLOTS[slot - 16]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _key_white(img: Image.Image, tolerance: int = 42) -> Image.Image:
    return bg.remove_background(
        img.convert("RGBA"), key_color=(255, 255, 255), tolerance=tolerance,
        feather=0, edge_clean=True, mode="hybrid",
    )


def _outline_color(img: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(img.convert("RGBA"))
    mask = arr[..., 3] > 32
    if not mask.any():
        return (34, 30, 32)
    rgb = arr[mask][..., :3].astype(np.float32)
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    k = max(1, int(len(lum) * 0.15))
    return tuple(int(c) for c in np.median(rgb[np.argsort(lum)[:k]], axis=0))


def _disc(mask: np.ndarray, radius: int) -> np.ndarray:
    """圆盘膨胀（等宽边界偏移）。"""
    r = int(radius)
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx == 0 and dy == 0) or dx * dx + dy * dy > r * r:
                continue
            sh = np.zeros_like(mask)
            ys_src = slice(max(0, -dy), mask.shape[0] - max(0, dy))
            ys_dst = slice(max(0, dy), mask.shape[0] - max(0, -dy))
            xs_src = slice(max(0, -dx), mask.shape[1] - max(0, dx))
            xs_dst = slice(max(0, dx), mask.shape[1] - max(0, -dx))
            sh[ys_dst, xs_dst] = mask[ys_src, xs_src]
            out |= sh
    return out


def _disc_fill(mask: np.ndarray, cx: float, cy: float, radius: int) -> np.ndarray:
    ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    return inside


def _dist2(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float) -> np.ndarray:
    return (xs - cx) ** 2 + (ys - cy) ** 2


def _quad(xs: np.ndarray, ys: np.ndarray, ox: int, oy: int, sx: int, sy: int, r: int) -> np.ndarray:
    """以角点 O 为顶点、朝 (sx, sy) 方向、边长 r 的角部方形。"""
    xs_in = (xs <= ox) & (xs > ox + sx * r) if sx < 0 else (xs >= ox) & (xs < ox + sx * r)
    ys_in = (ys <= oy) & (ys > oy + sy * r) if sy < 0 else (ys >= oy) & (ys < oy + sy * r)
    return xs_in & ys_in


def _wobble(n: int, seed: int, amp: int, taper: int, coarse: int = 4) -> np.ndarray:
    """确定性起伏（0..amp，仅向内），两端渐变为 0 → 共享边一致。"""
    if amp <= 0 or n <= 2:
        return np.zeros(max(0, n), dtype=np.int32)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    knots = rng.random(n // max(1, coarse) + 3)
    xs = np.arange(n, dtype=np.float64) / max(1, coarse)
    i0 = np.floor(xs).astype(np.int64)
    t = xs - i0
    t = t * t * (3.0 - 2.0 * t)
    vals = knots[i0] * (1.0 - t) + knots[i0 + 1] * t
    edge = np.minimum(np.arange(n), n - 1 - np.arange(n)).astype(np.float64)
    win = np.clip(edge / max(1.0, float(taper)), 0.0, 1.0)
    return np.rint(vals * win * float(amp)).astype(np.int32)


def _side_seed(bits: int, side: str) -> int:
    return (int(bits) * 2654435761 + (ord(side) << 8) + 17) & 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# 艺术参数
# --------------------------------------------------------------------------- #
def wall_art_from_sheet(
    sheet: BuildingSheet,
    tile_size: int = 32,
    key_tolerance: int = 42,
    thickness_frac: float = 0.56,
    edge_noise_frac: float = 0.04,
) -> WallArt:
    """AI 2×2×3 建筑图 → 墙体艺术（抠白底 + 无缝纹理 + 实测描边/厚度）。"""
    s = int(tile_size)
    core = _key_white(sheet.wall.center, key_tolerance)
    texture = make_tile_texture(core, s, inset_frac=0.05)
    outline = _outline_color(texture)

    # 厚度：AI 中心格若画的是「带子」（不透明比例 <1）就按实测，否则用默认比例
    arr = np.asarray(core.convert("RGBA"))
    opaque_frac = float((arr[..., 3] > 32).mean())
    frac = float(thickness_frac) if opaque_frac > 0.97 else min(0.9, max(0.3, opaque_frac))
    margin = max(2, int(round(s * (1.0 - frac) / 2.0)))
    margin = min(margin, s // 2 - 2)

    def _opt(tile: Image.Image, min_ratio: float, max_ratio: float) -> Optional[Image.Image]:
        try:
            keyed = _key_white(tile, key_tolerance)
            r = float((np.asarray(keyed)[..., 3] > 32).mean())
            if min_ratio < r <= max_ratio:
                return keyed.resize((s, s), Image.Resampling.NEAREST)
        except Exception:  # noqa: BLE001
            return None
        return None

    top = _opt(sheet.top.center, 0.25, 1.01) if sheet.top is not None else None
    door = _opt(sheet.opening.center, 0.04, 0.98) if sheet.opening is not None else None
    pillar = _opt(sheet.pillar.center, 0.04, 0.95) if sheet.pillar is not None else None

    want = 0 if edge_noise_frac <= 0 else max(1, int(round(s * min(0.2, edge_noise_frac))))
    noise = max(0, min(want, margin - 2))
    return WallArt(
        size=s, texture=texture, top=top, door=door, pillar=pillar,
        margin=margin, outline=outline, corner_radius=margin,
        top_h=max(2, s // 10), front_h=max(2, s // 10),
        shade=0.80, lighten=1.12, edge_noise=noise,
        art_meta={
            "tile_size": s, "margin_px": margin,
            "thickness_px": s - 2 * margin,
            "thickness_frac": round((s - 2 * margin) / s, 3),
            "outline": list(outline), "has_top": top is not None,
            "has_door": door is not None, "has_pillar": pillar is not None,
            "edge_noise_px": noise, "family": "wall-16",
        },
    )


# --------------------------------------------------------------------------- #
# 形状合成
# --------------------------------------------------------------------------- #
def footprint(art: WallArt, slot: int) -> Tuple[np.ndarray, np.ndarray]:
    """槽位（0..15 或 SOLID_SLOT）→ (墙体足迹, 透明让位区)。

    **固定截面**是关键：横带的行范围、竖带的列范围都只由 `margin`（+边缘噪声）决定，
    与「另一条轴上的连通情况」无关；某个方向相连只是把对应的带子**伸到那条边**，
    并在中心汇合。这样 A 的 E 位与 B 的 W 位同时为满时，两块在共享边上的墙/透明
    分界完全一致（都是横带的固定行范围），直墙与 T 形、转角之间也不会错位。
    """
    s = art.size
    m = max(1, int(art.margin))
    c = s // 2
    if int(slot) == SOLID_SLOT:
        return np.ones((s, s), dtype=bool), np.zeros((s, s), dtype=bool)
    bits = int(slot) & 0b1111
    amp = max(0, int(art.edge_noise))
    taper = max(2, s // 4)
    ys, xs = np.mgrid[0:s, 0:s]

    def wob(side: str) -> np.ndarray:
        return _wobble(s, _side_seed(bits, side), amp, taper)

    rows_in = (ys >= (m + wob("n"))[None, :]) & (ys < (s - m - wob("s"))[None, :])
    cols_in = (xs >= (m + wob("w"))[:, None]) & (xs < (s - m - wob("e"))[:, None])
    centre = rows_in & cols_in

    if bits == 0:
        return centre, ~centre

    solid = centre.copy()
    if bits & W16["w"]:
        solid |= rows_in & (xs < c + 1)
    if bits & W16["e"]:
        solid |= rows_in & (xs >= c)
    if bits & W16["n"]:
        solid |= cols_in & (ys < c + 1)
    if bits & W16["s"]:
        solid |= cols_in & (ys >= c)

    # 转角（只有两条相邻边相连）：弯折外角按「角部方形减去切圆」倒圆、
    # 内角补一个小圆角；两者都在瓦片内部，不接触共享边。
    pairs = ((("n", "e"), (s - m - 1, m), (m, s - m - 1)),
             (("e", "s"), (s - m - 1, s - m - 1), (m, m)),
             (("s", "w"), (m, s - m - 1), (s - m - 1, m)),
             (("w", "n"), (m, m), (s - m - 1, s - m - 1)))
    if len([k for k in W16 if bits & W16[k]]) == 2:
        r = max(2, int(art.corner_radius or m), m)
        r2 = max(1, m // 2)
        for (a, b), (ox, oy), (ix, iy) in pairs:
            if not (bits & W16[a] and bits & W16[b]):
                continue
            sx = -1 if ox > s // 2 else 1        # 朝瓦片中心的方向
            sy = -1 if oy > s // 2 else 1
            # 外角：角部方形（朝该角）内、切圆之外的像素切掉 → 与两条墙面相切
            cx, cy = ox + sx * r, oy + sy * r
            square = _quad(xs, ys, ox, oy, sx, sy, r)
            solid &= ~(square & (_dist2(xs, ys, cx, cy) > r * r))
            # 内角：凹角处补一个同向的小圆角（沿边长度 ≈ margin）
            c2x, c2y = ix + sx * r2, iy + sy * r2
            square2 = _quad(xs, ys, ix, iy, sx, sy, r2)
            solid |= square2 & (_dist2(xs, ys, c2x, c2y) <= r2 * r2)

    # 倒角可能切出 1px 的孤岛 -> 只保留最大连通块（4 邻域），保证拼件干净
    solid = _keep_largest(solid)
    return solid, ~solid


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    """保留最大 4 邻域连通块（去孤立像素）。"""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: List[Tuple[int, int]] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp: List[Tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def compose_wall_piece(art: WallArt, slot: int, mask8: Optional[int] = None) -> Image.Image:
    """合成一块建筑拼件（RGBA；让位区完全透明、无白边）。

    明暗完全由**足迹边界朝向**决定（而不是由掩码猜）：
    朝上/朝左的边是墙顶（提亮，AI 顶面纹理优先），朝下/朝右的边是立面（压暗），
    最外 1px 统一描边 —— 因此直墙、转角、T 形、十字、端头的外观规则完全一致。
    """
    s = art.size
    if int(slot) == SOLID_SLOT:                 # 只有显式要求时才用实心块
        slot_use = SOLID_SLOT
    else:
        slot_use = int(slot) & 0b1111
    solid, empty = footprint(art, slot_use)
    body = np.asarray(art.texture.convert("RGBA"))
    out = np.zeros((s, s, 4), dtype=np.uint8)
    out[solid] = body[solid]

    def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        pad = np.pad(mask, 1, mode="constant", constant_values=False)
        return pad[1 + dy:1 + dy + s, 1 + dx:1 + dx + s]

    # 相连的边：瓦片外一圈视为「墙继续延伸」——否则每个瓦片边界都会被当成
    # 暴露面，画出多余的阴影/描边，长墙每格就出现一道竖线，接缝也不再一致。
    bits = 0b1111 if slot_use == SOLID_SLOT else slot_use
    pad_full = np.pad(solid, 1, mode="constant", constant_values=False)
    if slot_use != SOLID_SLOT:
        if bits & W16["n"]:
            pad_full[0, :] = True
        if bits & W16["s"]:
            pad_full[-1, :] = True
        if bits & W16["w"]:
            pad_full[:, 0] = True
        if bits & W16["e"]:
            pad_full[:, -1] = True

    def shifted_full(dy: int, dx: int) -> np.ndarray:
        return pad_full[1 + dy:1 + dy + s, 1 + dx:1 + dx + s]

    def tinted(factor: float) -> np.ndarray:
        t = out.copy()
        t[..., :3] = np.clip(t[..., :3].astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return t

    top_h = max(1, min(int(art.top_h), max(1, (s - 2 * art.margin) // 2)))
    front_h = max(1, min(int(art.front_h), max(1, (s - 2 * art.margin) // 2)))
    # 朝上的边沿（含内凹角形成的上沿）→ 顶面
    top_band = np.zeros_like(solid)
    cur = solid & ~shifted_full(-1, 0)
    for _ in range(top_h):
        top_band |= cur
        cur = shifted(cur, 1, 0) & solid
    if art.top is not None:
        top_px = np.asarray(art.top.convert("RGBA"))
        out[top_band] = top_px[top_band]
    else:
        out[top_band] = tinted(art.lighten)[top_band]
    # 朝下/朝右的边沿 → 立面阴影
    shade_band = np.zeros_like(solid)
    cur = solid & (~shifted_full(1, 0) | ~shifted_full(0, 1))
    for _ in range(front_h):
        shade_band |= cur
        cur = (shifted(cur, -1, 0) | shifted(cur, 0, -1)) & solid
    shade_band &= ~top_band
    if shade_band.any():
        out[shade_band] = tinted(art.shade)[shade_band]

    # 描边（仅暴露边界 1px）→ 外部透明、无白边
    exposed = ~(pad_full[1:-1, 1:-1])
    outline_mask = solid & _disc(exposed, 1)
    out[outline_mask] = np.array([*art.outline, 255], dtype=np.uint8)
    out[empty] = (0, 0, 0, 0)
    return Image.fromarray(out, "RGBA")


def compose_door(art: WallArt, horizontal: bool = True) -> Image.Image:
    """直墙 + 门（门件取自 AI 开口组；没有就用描边画一个门洞）。"""
    base = compose_wall_piece(art, W16["e"] | W16["w"] if horizontal else W16["n"] | W16["s"])
    if not horizontal:
        base = base.transpose(Image.Transpose.ROTATE_90)
    s = art.size
    rows = np.nonzero(np.asarray(base)[..., 3].any(axis=1))[0]
    cols = np.nonzero(np.asarray(base)[..., 3].any(axis=0))[0]
    if not rows.size or not cols.size:
        return base
    band_h = int(rows[-1] - rows[0] + 1)
    canvas = base.copy()
    if art.door is not None:
        dw = max(6, int(band_h * 1.25))
        d = art.door.convert("RGBA").resize((dw, max(4, band_h)), Image.Resampling.NEAREST)
    else:
        dw = max(6, int(band_h * 1.25))
        d = Image.new("RGBA", (dw, max(4, band_h)), (0, 0, 0, 0))
        d.paste(Image.new("RGBA", (dw - 4, max(2, band_h - 4)), (*art.outline, 255)), (2, 2))
    canvas.alpha_composite(d, ((s - dw) // 2, int(rows[-1] - d.height + 1)))
    return canvas


def compose_pillar(art: WallArt) -> Image.Image:
    """装饰立柱：优先 AI 立柱件，否则程序化生成柱体（透明外部）。"""
    s = art.size
    if art.pillar is not None:
        arr = np.asarray(art.pillar.convert("RGBA"))
        if (arr[..., 3] > 32).any():
            return art.pillar.convert("RGBA").resize((s, s), Image.Resampling.NEAREST)
    ys, xs = np.mgrid[0:s, 0:s]
    r = max(3, (s - 2 * art.margin) // 2)
    disc = (xs - (s - 1) / 2) ** 2 + (ys - (s - 1) / 2) ** 2 <= r * r
    body = np.asarray(art.texture.convert("RGBA"))
    out = np.zeros((s, s, 4), dtype=np.uint8)
    out[disc] = body[disc]
    out[disc, :3] = np.clip(out[disc, :3].astype(np.float32) * art.lighten, 0, 255).astype(np.uint8)
    out[disc & _disc(~disc, 1)] = np.array([*art.outline, 255], dtype=np.uint8)
    return Image.fromarray(out, "RGBA")


def build_piece_set(art: WallArt, with_door: bool = True, with_pillar: bool = True) -> Dict[str, Image.Image]:
    """整族拼件：16 个墙体件 + solid + 门（横/竖）+ 立柱。"""
    pieces: Dict[str, Image.Image] = {}
    for slot in range(16):
        pieces[W16_SLOTS[slot]] = compose_wall_piece(art, slot)
    pieces["solid"] = compose_wall_piece(art, SOLID_SLOT)
    if with_door:
        pieces["door_ew"] = compose_door(art, horizontal=True)
        pieces["door_ns"] = compose_door(art, horizontal=False)
    if with_pillar:
        pieces["pillar"] = compose_pillar(art)
    return pieces


def build_wall_atlas(art: WallArt, with_door: bool = True, with_pillar: bool = True) -> Tuple[Image.Image, Dict]:
    """导出墙体图集（4×5 = 20 槽：0..15 = 16-tile 族，16=solid，17/18=门，19=立柱）。"""
    s = art.size
    pieces = build_piece_set(art, with_door=with_door, with_pillar=with_pillar)
    names = list(W16_SLOTS) + [n for n in EXTRA_SLOTS if n in pieces]
    sheet = Image.new("RGBA", (ATLAS_COLS * s, ATLAS_ROWS * s), (0, 0, 0, 0))
    slot_of_piece: Dict[str, int] = {}
    for i, name in enumerate(names):
        sheet.paste(pieces[name], ((i % ATLAS_COLS) * s, (i // ATLAS_COLS) * s), pieces[name])
        slot_of_piece[name] = i
    return sheet, {
        "format": "pixel-anim-wall16",
        "family": "wall-16",
        "tile_size": s,
        "sheet_cols": ATLAS_COLS,
        "sheet_rows": ATLAS_ROWS,
        "slot_count": ATLAS_COLS * ATLAS_ROWS,
        "slots": names,
        "slot_of_piece": slot_of_piece,
        "extra_slots": {n: slot_of_piece[n] for n in EXTRA_SLOTS if n in slot_of_piece},
        "bits": dict(W16),
        "mask16_to_slot": {str(b): b for b in range(16)},
        "mask8_to_slot": {str(m): slot_for_mask(m) for m in range(256)},
        "transparent_outside": True,
        **art.art_meta,
    }


def rotate_piece(piece: Image.Image, rot: int) -> Image.Image:
    """旋转拼件（0/90/180/270，逆时针）。"""
    rot = int(rot) % 4
    if rot == 0:
        return piece
    return piece.transpose(
        {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}[rot]
    )
