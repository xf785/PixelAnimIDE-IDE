"""47-tile 瓦片集生成（8 邻域位掩码 + 几何构图）与双网格导出。

算法（Godot「Match Corners and Sides」/ 经典 47-tile 同源）：
- 每张瓦片按 8 邻域位掩码构图：瓦片即「地形区域」——某侧邻居为空则沿该侧
  画边界描边线（只画在有地形处）；四角满足「两侧同时为空（外转角圆盘切）」
  或「两侧同时为满且对角为空（内转角圆盘切）」时，以角点为中心切掉半径
  s/2 的四分之一圆盘（弧线过边中点），沿弧线描边、弦不描边；
- 地形像素一律取中心无缝纹理，边界线用统一颜色 → 任意相邻瓦片共享边/
  角来自同一几何规则，**整张地图无缝由构造保证**；
- 全 256 种掩码按视觉等价归并后恰为 47 种
  （分组枚举：16 + 16 + 8 + 2 + 4 + 1 = 47）；
- 双网格导出：同一构图规则在 2 倍分辨率下按「四分之一块」铺出（填充 /
  上直切 / 左直切 / 转角盘 4 类小块）。

位值约定（与 Godot/常见 47-tile 模板一致）：
TL=1, T=2, TR=4, L=8, R=16, BL=32, B=64, BR=128。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from core.processing.pixelizer import extract_dominant_palette, map_to_palette

logger = logging.getLogger("PixelAnimIDE.tilemap.autotile")

BIT = {"TL": 1, "T": 2, "TR": 4, "L": 8, "R": 16, "BL": 32, "B": 64, "BR": 128}

_CORNERS = (
    # (name, sideA, sideB, diagonal, corner point)
    ("TL", "T", "L", "TL", (0, 0)),
    ("TR", "T", "R", "TR", (1, 0)),
    ("BL", "B", "L", "BL", (0, 1)),
    ("BR", "B", "R", "BR", (1, 1)),
)

SHEET_COLS = 8
SHEET_ROWS = 6  # 48 槽，47 张瓦片 + 1 空槽

_SIDE_BITS = ("T", "B", "L", "R")
_DIAG_NEEDS = {  # 对角位 -> 需要同时为满的两个正交位
    "TL": ("T", "L"),
    "TR": ("T", "R"),
    "BL": ("B", "L"),
    "BR": ("B", "R"),
}


def canonical_mask(mask: int) -> int:
    """规范化 8 邻域掩码：对角位仅在两相邻正交位都为满时生效。

    这是经典 blob / 47-tile 的约定（与 FrameRonin `computeBlobMask`、
    Godot 地形集同源）：对角位表达「转角是否连通」，只有两侧正交位都连通时
    才有意义；例如只有对角邻居而无正交邻居时，等价于孤立瓦片（mask=0）。
    """
    m = int(mask) & 255
    sides = 0
    for name in _SIDE_BITS:
        sides |= m & BIT[name]
    out = sides
    for diag, (a, b) in _DIAG_NEEDS.items():
        if (m & BIT[diag]) and (sides & BIT[a]) and (sides & BIT[b]):
            out |= BIT[diag]
    return out


def canonical_masks() -> List[int]:
    """全部可达掩码（恰好 47 个），按经典分组顺序排列。"""
    out: List[int] = []
    for mask in _group_masks():
        c = canonical_mask(mask)
        if c not in out:
            out.append(c)
    return out


def nearest_mask(mask: int, candidates: Optional[List[int]] = None) -> int:
    """把任意掩码映射到最近的合法掩码（汉明距离，FrameRonin 同款回退）。"""
    m = int(mask) & 255
    pool = candidates or canonical_masks()
    if m in pool:
        return m
    return min(pool, key=lambda k: bin(m ^ k).count("1"))


def mask_from_neighbors(nb) -> int:
    """由 3×3 邻域（中心为当前格）计算规范化位掩码。

    nb: 3×3 的可迭代（bool/int），True/非 0 表示该邻居是同类地形；
    中心格忽略。
    """
    order = [
        (0, 0, "TL"), (0, 1, "T"), (0, 2, "TR"),
        (1, 0, "L"), (2, 0, "BL"), (2, 1, "B"), (2, 2, "BR"),
        (1, 2, "R"),
    ]
    mask = 0
    for r, c, name in order:
        if int(nb[r][c]):
            mask |= BIT[name]
    return canonical_mask(mask)


def compose_tile(
    center: Image.Image,
    mask: int,
    line_color: Tuple[int, int, int] = (0, 0, 0),
    line_width: int = 1,
    radius: Optional[int] = None,
) -> Image.Image:
    """按位掩码由中心纹理合成一张瓦片（RGBA），返回与 center 同尺寸。

    几何规则（经典 47-tile / Godot「Match Corners and Sides」同源）：
    - 某侧邻居为空 -> 沿该侧画边界描边线（仅画在仍有地形的位置）；
    - 四角「圆盘咬合」：两侧邻居同时为空（外转角）或同时为满且对角为空
      （内转角）时，以角点为中心切掉半径 r 的四分之一圆盘（弧线过边中点），
      沿弧线画描边、弦不描边；
    - 地形像素一律取中心无缝纹理 -> 任意相邻瓦片共享边的像素来自同一纹理
      与同一几何规则，整图无缝由构造保证。
    """
    s = center.size[0]
    lw = max(1, min(int(line_width), max(1, s // 4)))
    r = s // 2 if radius is None else max(1, int(radius))
    alpha = np.ones((s, s), dtype=bool)
    line_px: set = set()

    # 1) 角圆盘切（半径 s/2，弧线过边中点）
    for _name, sa, sb, diag, (fx, fy) in _CORNERS:
        a = mask & BIT[sa]
        b = mask & BIT[sb]
        d = mask & BIT[diag]
        outer = not a and not b
        inner = a and b and not d
        if not outer and not inner:
            continue
        cx = 0 if fx == 0 else s - 1
        cy = 0 if fy == 0 else s - 1
        xs = np.arange(s)
        ys = np.arange(s)
        d2 = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2
        disc = d2 < r * r
        ring = (d2 >= r * r) & (d2 < (r + 1.25) ** 2)
        alpha[disc] = False
        line_px.update(
            (int(x), int(y))
            for y, x in zip(*np.nonzero(ring))
            if alpha[y, x]
        )

    # 2) 边描边线：仅画在仍有地形的像素上（圆盘切掉的边段自然跳过）
    if not (mask & BIT["T"]):
        for y in range(lw):
            line_px.update((x, y) for x in range(s) if alpha[y, x])
    if not (mask & BIT["B"]):
        for y in range(s - lw, s):
            line_px.update((x, y) for x in range(s) if alpha[y, x])
    if not (mask & BIT["L"]):
        for x in range(lw):
            line_px.update((x, y) for y in range(s) if alpha[y, x])
    if not (mask & BIT["R"]):
        for x in range(s - lw, s):
            line_px.update((x, y) for y in range(s) if alpha[y, x])

    # 3) 合成：地形取中心纹理，线取统一线色
    out = np.zeros((s, s, 4), dtype=np.uint8)
    src = np.asarray(center.convert("RGBA"))
    out[alpha] = src[alpha]
    lc = np.array([int(line_color[0]), int(line_color[1]), int(line_color[2]), 255], dtype=np.uint8)
    for x, y in line_px:
        if alpha[y, x]:
            out[y, x] = lc
    return Image.fromarray(out, "RGBA")


# --------------------------------------------------------------------------- #
# 转角瓦片推导（与墙面/中心无缝）
# --------------------------------------------------------------------------- #
def _corner_mask(name: str) -> int:
    """转角瓦片对应的位掩码（纯转角：外圆角一处，无对角内咬合）。

    tl 转角：地形向右、下延伸（R/B 为满，T/L 为空），BR 对角置满以
    避免在自身右下角产生内角咬合。
    """
    sides = {
        "tl": ("R", "B", "BR"),
        "tr": ("L", "B", "BL"),
        "bl": ("R", "T", "TR"),
        "br": ("L", "T", "TL"),
    }[name]
    return sum(BIT[s] for s in sides)


def derive_corners(
    center: Image.Image,
    line_color: Tuple[int, int, int],
    line_width: int = 1,
    ai_corners: Optional[Dict[str, Image.Image]] = None,
    detail_keep: float = 0.0,
) -> Dict[str, Image.Image]:
    """由无缝中心纹理推导 4 张转角瓦片（构造上保证与墙面/中心无缝）。

    - 地形区 = 中心纹理，按「转角位掩码」构图（外圆角 + 对侧内圆角 + 边带线）；
    - 可选把 AI 生成转角瓦片的内部细节以 detail_keep 比例混合（仅限远离
      边界线/圆盘 ≥2px 的内部区域），保持转角自带的光影特征，同时边界仍
      与墙面一致；
    - 最终把颜色重新量化到「中心调色板 + 线色」，保证整组瓦片颜色一致。
    """
    corners: Dict[str, Image.Image] = {}
    s = center.size[0]
    for name in ("tl", "tr", "bl", "br"):
        composed = compose_tile(center, _corner_mask(name), line_color, line_width)
        arr = np.asarray(composed).copy()
        if detail_keep and ai_corners and name in ai_corners:
            ai = np.asarray(ai_corners[name].convert("RGBA").resize((s, s), Image.Resampling.NEAREST))
            keep = max(0.0, min(1.0, float(detail_keep)))
            # 只混合远离边界的内部像素（alpha>0 且距 0-alpha 区域 ≥2px）
            a_b = arr[..., 3] > 0
            from PIL import ImageFilter

            eroded = np.asarray(
                Image.fromarray((a_b * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))
            ) > 127
            blend = a_b & (~eroded) & (ai[..., 3] > 0)
            arr[blend, :3] = (arr[blend, :3] * (1 - keep) + ai[blend, :3] * keep).astype(np.uint8)
            arr[blend, 3] = 255
        corners[name] = _requantize(Image.fromarray(arr, "RGBA"), center, line_color)
    return corners


def _requantize(img: Image.Image, center: Image.Image, line_color: Tuple[int, int, int]) -> Image.Image:
    """量化到「中心纹理主色 + 线色」的调色板，保证全组颜色一致（保留 alpha）。"""
    palette = list(extract_dominant_palette(center, 31))
    lc = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
    if lc not in palette:
        palette.append(lc)
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    q = map_to_palette(rgba.convert("RGB"), palette).convert("RGBA")
    q.putalpha(alpha)
    return q


# --------------------------------------------------------------------------- #
# 47-tile 集生成
# --------------------------------------------------------------------------- #
def _group_masks() -> List[int]:
    """按经典分组顺序枚举全部 256 种掩码（每个掩码恰出现一次）。"""
    # 16 种对角线位组合（TL/TR/BL/BR），升序
    all_diags: List[int] = []
    for d in range(16):
        v = 0
        if d & 1:
            v |= BIT["TL"]
        if d & 2:
            v |= BIT["TR"]
        if d & 4:
            v |= BIT["BL"]
        if d & 8:
            v |= BIT["BR"]
        all_diags.append(v)
    all_diags.sort()
    sides = {"T": BIT["T"], "B": BIT["B"], "L": BIT["L"], "R": BIT["R"]}
    groups: List[List[int]] = []
    # k=4：四边全满
    groups.append([(BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]) | d for d in all_diags])
    # k=3：缺一边
    for missing in ("T", "B", "L", "R"):
        base = (BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]) & ~sides[missing]
        groups.append([base | d for d in all_diags])
    # k=2 相邻
    for pair in (("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")):
        base = sides[pair[0]] | sides[pair[1]]
        groups.append([base | d for d in all_diags])
    # k=2 相对
    for pair in (("T", "B"), ("L", "R")):
        base = sides[pair[0]] | sides[pair[1]]
        groups.append([base | d for d in all_diags])
    # k=1
    for side in ("T", "B", "L", "R"):
        groups.append([sides[side] | d for d in all_diags])
    # k=0
    groups.append(list(all_diags))
    masks: List[int] = []
    seen = set()
    for g in groups:
        for m in g:
            if m not in seen:
                seen.add(m)
                masks.append(m)
    assert len(masks) == 256, f"掩码枚举不完整: {len(masks)}"
    return masks


def build_47_sheet(
    center: Image.Image,
    line_color: Tuple[int, int, int] = (0, 0, 0),
    line_width: int = 1,
) -> Tuple[Image.Image, Dict]:
    """由无缝中心纹理生成 47-tile 瓦片集图（8×6，最后一格留空）。

    返回 (sheet RGBA, meta)：meta 含 mask_to_index（256 项）、
    index_to_mask（47 项）、tile_size、line_color、line_width、radius。
    """
    s = center.size[0]
    masks = _group_masks()
    seen: Dict[bytes, int] = {}
    mask_to_index: Dict[int, int] = {}
    index_to_mask: List[int] = []
    tiles: List[Image.Image] = []
    for mask in masks:
        tile = compose_tile(center, mask, line_color, line_width)
        key = tile.tobytes()
        if key in seen:
            mask_to_index[mask] = seen[key]
            continue
        idx = len(tiles)
        seen[key] = idx
        mask_to_index[mask] = idx
        index_to_mask.append(mask)
        tiles.append(tile)
        if len(tiles) >= 47:
            break
    # 经典 47 布局中「孤立瓦片」（mask=0）与「四内角瓦片」（四边满、对角全空）
    # 的图形重合（同为四角圆盘切），但模板保留两个槽位——这里给孤立瓦片
    # 单独槽位（图像复用），保持 47 槽格式兼容。
    if len(tiles) == 46:
        hole_mask = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
        tiles.append(tiles[mask_to_index[hole_mask]])
        mask_to_index[0] = len(tiles) - 1
        index_to_mask.append(0)
    assert len(tiles) == 47, f"47-tile 数量异常: {len(tiles)}"
    sheet = Image.new("RGBA", (SHEET_COLS * s, SHEET_ROWS * s), (0, 0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % SHEET_COLS) * s, (i // SHEET_COLS) * s), tile)
    meta = {
        "format": "pixel-anim-47tile",
        "tile_size": s,
        "sheet_cols": SHEET_COLS,
        "sheet_rows": SHEET_ROWS,
        "tile_count": 47,
        "line_color": list(line_color),
        "line_width": int(line_width),
        "radius": max(2, s // 4),
        "mask_to_index": {str(m): mask_to_index[m] for m in range(256)},
        "index_to_mask": index_to_mask,
    }
    return sheet, meta


# --------------------------------------------------------------------------- #
# 双网格
# --------------------------------------------------------------------------- #
DUAL_PIECE_NAMES = ("fill", "cut_h", "cut_v", "corner")


def build_dual_pieces_sheet(
    center: Image.Image,
    line_color: Tuple[int, int, int] = (0, 0, 0),
    line_width: int = 1,
) -> Tuple[Image.Image, Dict]:
    """生成双网格 16 张四分之一块集图（4 列 × 4 行，块尺寸 = 瓦片/2）。

    每张块由位掩码构图瓦片裁四分之一得到：块与 47 模式同一几何规则，
    因此双网格地图与 47 地图视觉一致。meta 含 pieces 清单（qy/qx/kind）。
    """
    s = center.size[0]
    half = s // 2
    all_sides = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
    kind_masks = {
        0: all_sides,
        1: all_sides & ~BIT["T"],  # 上直切（按块朝向：qy=0 用 T，qy=1 用 B）
        2: all_sides & ~BIT["L"],  # 左直切（qx=0 用 L，qx=1 用 R）
        3: all_sides & ~BIT["T"] & ~BIT["L"],  # 外角盘
    }

    def mask_for(qy, qx, kind):
        m = kind_masks[kind]
        if kind == 1 and qy == 1:
            m = all_sides & ~BIT["B"]
        if kind == 2 and qx == 1:
            m = all_sides & ~BIT["R"]
        if kind == 3:
            m = all_sides
            m &= ~(BIT["T"] if qy == 0 else BIT["B"])
            m &= ~(BIT["L"] if qx == 0 else BIT["R"])
        return m

    box = {
        (0, 0): (0, 0, half, half),
        (0, 1): (half, 0, s, half),
        (1, 0): (0, half, half, s),
        (1, 1): (half, half, s, s),
    }
    sheet = Image.new("RGBA", (half * 4, half * 4), (0, 0, 0, 0))
    pieces: List[Dict] = []
    for qy in range(2):
        for qx in range(2):
            for kind in range(4):
                tile = compose_tile(center, mask_for(qy, qx, kind), line_color, line_width)
                l, t, r, b = box[(qy, qx)]
                piece = tile.crop((l, t, r, b))
                idx = len(pieces)
                sheet.paste(piece, ((idx % 4) * half, (idx // 4) * half), piece)
                pieces.append({"index": idx, "qy": qy, "qx": qx, "kind": kind, "name": DUAL_PIECE_NAMES[kind]})
    meta = {
        "format": "pixel-anim-dual",
        "tile_size": s,
        "piece_size": half,
        "cols": 4,
        "rows": 4,
        "piece_count": 16,
        "line_color": list(line_color),
        "line_width": int(line_width),
        "pieces": pieces,
    }
    return sheet, meta


def dual_grid_map(grid: np.ndarray) -> np.ndarray:
    """把地形网格（1=地形）转成双网格 2 倍分辨率索引图。

    返回 (2h, 2w) 的 uint8 数组，取值 0..3：
    0=填充块, 1=上边直切块, 2=左边直切块, 3=转角圆盘块。
    """
    h, w = grid.shape
    out = np.zeros((h * 2, w * 2), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if not grid[y, x]:
                continue
            mask = mask_from_neighbors(_neighbors(grid, y, x))
            # 四分之一块规则（与 compose_tile 同一几何）
            # TL 块
            out[2 * y, 2 * x] = _quarter(mask, "T", "L", "TL")
            out[2 * y, 2 * x + 1] = _quarter(mask, "T", "R", "TR")
            out[2 * y + 1, 2 * x] = _quarter(mask, "B", "L", "BL")
            out[2 * y + 1, 2 * x + 1] = _quarter(mask, "B", "R", "BR")
    return out


def _quarter(mask: int, sa: str, sb: str, diag: str) -> int:
    a = mask & BIT[sa]
    b = mask & BIT[sb]
    d = mask & BIT[diag]
    if a and b:
        return 3 if not d else 0  # 内转角盘 / 填充
    if a and not b:
        return 2  # 左边直切（a 在上、b 缺 -> 竖切）
    if not a and b:
        return 1  # 上边直切
    return 3  # 外转角盘


def _neighbors(grid: np.ndarray, y: int, x: int) -> List[List[int]]:
    h, w = grid.shape
    return [
        [int(grid[ny, nx]) if 0 <= ny < h and 0 <= nx < w else 0 for nx in range(x - 1, x + 2)]
        for ny in range(y - 1, y + 2)
    ]


# --------------------------------------------------------------------------- #
# 艺术片构图（真实 AI 九宫格艺术拼接，替代程序化描边）
# --------------------------------------------------------------------------- #
def mask_for_terrain(nb, terrain_id: int, base_terrain: Optional[int] = None) -> int:
    """多地形位掩码（规范化）：同地形邻居视为满；基础地形额外把任何非空地形
    视为满（基础地形在生态内部过渡处始终显示填充，过渡艺术由特征地形一侧提供）。"""
    order = [
        (0, 0, "TL"), (0, 1, "T"), (0, 2, "TR"),
        (1, 0, "L"), (2, 0, "BL"), (2, 1, "B"), (2, 2, "BR"),
        (1, 2, "R"),
    ]
    mask = 0
    for r, c, name in order:
        v = int(nb[r][c])
        filled = (v != 0) if (base_terrain is not None and terrain_id == base_terrain) else (v == terrain_id)
        if filled:
            mask |= BIT[name]
    return canonical_mask(mask)


# 每个目标四分之一块：邻居位定义 + 状态 → (源瓦片, 源四分之一, 是否旋转 180°)
_QUARTER_SIDES = {
    "TL": ("T", "L", "TL"),
    "TR": ("T", "R", "TR"),
    "BL": ("B", "L", "BL"),
    "BR": ("B", "R", "BR"),
}

_ART_SRC = {
    # fill=纯地形；t_cut/b_cut/l_cut=直边（取对应边瓦片的同侧四分之一，
    # 其外侧条带即为另一方地形）；outer=外角（本角瓦片的同侧四分之一，含圆角弧）；
    # inner=内角（对角瓦片旋转 180° 的同侧四分之一 = 凹角负形）
    "TL": {
        "fill": ("center", "TL", False), "t_cut": ("top", "TL", False),
        "l_cut": ("left", "TL", False), "outer": ("tl", "TL", False),
        "inner": ("br", "TL", True),
    },
    "TR": {
        "fill": ("center", "TR", False), "t_cut": ("top", "TR", False),
        "r_cut": ("right", "TR", False), "outer": ("tr", "TR", False),
        "inner": ("bl", "TR", True),
    },
    "BL": {
        "fill": ("center", "BL", False), "b_cut": ("bottom", "BL", False),
        "l_cut": ("left", "BL", False), "outer": ("bl", "BL", False),
        "inner": ("tr", "BL", True),
    },
    "BR": {
        "fill": ("center", "BR", False), "b_cut": ("bottom", "BR", False),
        "r_cut": ("right", "BR", False), "outer": ("br", "BR", False),
        "inner": ("tl", "BR", True),
    },
}


def _quarter_box(s: int, quarter: str) -> Tuple[int, int, int, int]:
    half = s // 2
    return {
        "TL": (0, 0, half, half),
        "TR": (half, 0, s, half),
        "BL": (0, half, half, s),
        "BR": (half, half, s, s),
    }[quarter]


def _disc_offsets(radius: int) -> List[Tuple[int, int]]:
    """圆盘结构元的偏移集合（欧氏距离 ≤ radius），用于等宽描边膨胀。"""
    r = max(0, int(radius))
    return [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dx * dx + dy * dy <= r * r
    ]


def _disc_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """圆盘膨胀：任何方向上膨胀宽度都等于 radius（方形膨胀在对角线方向会多出 √2 倍）。"""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for dy, dx in _disc_offsets(radius):
        if dy == 0 and dx == 0:
            continue
        shifted = np.zeros_like(mask)
        ys_src = slice(max(0, -dy), mask.shape[0] - max(0, dy))
        ys_dst = slice(max(0, dy), mask.shape[0] - max(0, -dy))
        xs_src = slice(max(0, -dx), mask.shape[1] - max(0, dx))
        xs_dst = slice(max(0, dx), mask.shape[1] - max(0, -dx))
        shifted[ys_dst, xs_dst] = mask[ys_src, xs_src]
        out |= shifted
    return out


def compose_art_tile(base, mask: int, blend: int = 1) -> Image.Image:
    """按位掩码合成一张瓦片：**AI 纹理 + 程序化几何**，共享边逐像素相等。

    为什么不再逐块抠 AI 的九宫格艺术：AI 画不画得「刚好」取决于它的发挥——
    条带深度、是否连续、有没有格线框都会变，抠四分之一块必然在接缝处错位
    （表现为地图上规律的深色网格/图案跳变）。对齐式构图只从 AI 取**纹理与
    实测参数**，几何完全由算法生成：

    - 特征区 = `base.center`（无缝纹理，网格对齐平铺）；
    - 基础地形区 = `base.base_texture`（同样是网格对齐的无缝纹理）→ 与相邻
      基础地形瓦片在同一像素坐标上取同一纹理，接缝逐像素相等；
    - 暴露侧画 `band` 像素宽的基础地形条带；转角处条带内边界为半径 band 的
      圆弧（外角：条带减去角部圆盘；内角：角上四分之一圆盘的地面凹口）；
    - 描边 = 条带内侧 `line_width` 像素，取实测描边色 → 相邻同侧暴露的瓦片
      描边位置完全相同，连成一条干净的边线。

    `blend` 仅保留签名兼容（对齐式构图不需要中线交叉淡化）。
    """
    s = base.size
    feat = np.asarray(base.center.convert("RGBA"))
    ground_img = base.base_texture if base.base_texture is not None else base.center
    if ground_img.size != (s, s):
        ground_img = ground_img.resize((s, s), Image.Resampling.NEAREST)
    ground = np.asarray(ground_img.convert("RGBA"))
    band = int(base.band or 0) or max(2, s // 4)
    band = max(1, min(s // 2, band))
    radius = int(base.radius or 0) or band
    radius = max(1, min(band, radius))
    rim_w = max(0, min(int(base.line_width or 0), max(1, band - 1)))

    ys, xs = np.mgrid[0:s, 0:s]
    band_mask = np.zeros((s, s), dtype=bool)
    if not (mask & BIT["T"]):
        band_mask |= ys < band
    if not (mask & BIT["B"]):
        band_mask |= ys >= s - band
    if not (mask & BIT["L"]):
        band_mask |= xs < band
    if not (mask & BIT["R"]):
        band_mask |= xs >= s - band

    for _name, sa, sb, diag, (fx, fy) in _CORNERS:
        a = bool(mask & BIT[sa])
        b = bool(mask & BIT[sb])
        d = bool(mask & BIT[diag])
        cx = 0.0 if fx == 0 else float(s - 1)
        cy = 0.0 if fy == 0 else float(s - 1)
        if not a and not b:
            # 外角：把地形的角切掉一块，地面沿弧线补齐——弧与两侧直边相切
            # （切点在 (band+radius, band) 与 (band, band+radius)），因此条带
            # 到圆角是平滑过渡，而不是被挤成一条细缝（旧公式的毛病）。
            ix = float(band + radius) if fx == 0 else float(s - band - radius)
            iy = float(band + radius) if fy == 0 else float(s - band - radius)
            near_x = (xs < band + radius) if fx == 0 else (xs >= s - band - radius)
            near_y = (ys < band + radius) if fy == 0 else (ys >= s - band - radius)
            cut = near_x & near_y & ((xs - ix) ** 2 + (ys - iy) ** 2 > radius * radius)
            band_mask |= cut
        elif a and b and not d:
            # 内角：对角斜插进来一块另一方地形，角上是地面凹口。半径取 band-1，
            # 使凹口在两条边上的覆盖长度正好等于条带宽度（band 像素）——差 1 像素
            # 就会在共享边留下色差（旧公式 radius=band 的毛病）。
            r_pocket = max(1, band - 1)
            band_mask |= (xs - cx) ** 2 + (ys - cy) ** 2 <= r_pocket * r_pocket

    base_px = ground.copy()
    rim_px = np.array([int(base.line_color[0]), int(base.line_color[1]), int(base.line_color[2]), 255], dtype=np.uint8)
    if rim_w > 0:
        # 描边 = 边界偏移带：圆盘膨胀保证任意方向等宽（方形膨胀在对角/圆弧方向会
        # 多出 √2 倍宽度，转角处比相邻瓦片多 1-2 像素）。
        rim_mask = band_mask & _disc_dilate(~band_mask, rim_w)
    else:
        rim_mask = np.zeros((s, s), dtype=bool)

    out = feat.copy()
    out[band_mask] = base_px[band_mask]
    out[rim_mask] = rim_px
    out[..., 3] = 255  # 地块瓦片一律不透明（无透明楔形/空边）
    return Image.fromarray(out, "RGBA")


def build_47_sheet_art(base, blend: int = 1) -> Tuple[Image.Image, Dict]:
    """用艺术片构图生成 47-tile 瓦片集（8×6 图 + 全 256 掩码映射元数据）。

    只枚举 **47 个可达（规范化）掩码**（对角位依赖正交位，FrameRonin 同款），
    每个掩码一个槽位；非可达掩码通过汉明距离回退到最近槽位，
    因此 `mask_to_index` 对全部 256 种邻域都有确定答案，且槽位顺序稳定。
    """
    s = base.size
    masks = canonical_masks()
    index_to_mask: List[int] = []
    tiles: List[Image.Image] = []
    seen: Dict[bytes, int] = {}
    slot_of_mask: Dict[int, int] = {}
    for mask in masks:
        tile = compose_art_tile(base, mask, blend=blend)
        key = tile.tobytes()
        if key in seen:
            slot_of_mask[mask] = seen[key]
            continue
        idx = len(tiles)
        seen[key] = idx
        slot_of_mask[mask] = idx
        index_to_mask.append(mask)
        tiles.append(tile)
    unique = len(tiles)
    # 经典布局中「孤立瓦片」（mask=0）与「四内角瓦片」（四边满、对角全空）
    # 图形重合（同为四角圆盘切），模板惯例保留双槽位。
    hole_mask = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
    if hole_mask in slot_of_mask and slot_of_mask.get(0) == slot_of_mask[hole_mask]:
        tiles.append(tiles[slot_of_mask[hole_mask]])
        slot_of_mask[0] = len(tiles) - 1
        index_to_mask.append(0)
        unique = len(tiles)
    while len(tiles) < 47:
        tiles.append(tiles[0])
    # 全 256 掩码 -> 槽位（不可达掩码按汉明距离回退）
    mask_to_index = {m: slot_of_mask[nearest_mask(m, masks)] for m in range(256)}
    sheet = Image.new("RGBA", (SHEET_COLS * s, SHEET_ROWS * s), (0, 0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % SHEET_COLS) * s, (i // SHEET_COLS) * s), tile)
    meta = {
        "format": "pixel-anim-47tile-art",
        "tile_size": s,
        "sheet_cols": SHEET_COLS,
        "sheet_rows": SHEET_ROWS,
        "tile_count": 47,
        "reachable_masks": masks,
        "mask_convention": "TL=1 T=2 TR=4 L=8 R=16 BL=32 B=64 BR=128; 对角位仅在两正交位为满时有效",
        "radius": s // 2,
        "blend": int(blend),
        "mask_to_index": {str(m): mask_to_index[m] for m in range(256)},
        "index_to_mask": index_to_mask,
    }
    return sheet, meta


def build_dual_pieces_sheet_art(base, blend: int = 1) -> Tuple[Image.Image, Dict]:
    """双网格 16 块集（艺术片版）：由九宫格艺术构图瓦片裁四分之一块。"""
    s = base.size
    half = s // 2
    all_sides = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
    kind_masks = {
        0: all_sides,
        1: all_sides & ~BIT["T"],
        2: all_sides & ~BIT["L"],
        3: all_sides & ~BIT["T"] & ~BIT["L"],
    }

    def mask_for(qy, qx, kind):
        m = kind_masks[kind]
        if kind == 1 and qy == 1:
            m = all_sides & ~BIT["B"]
        if kind == 2 and qx == 1:
            m = all_sides & ~BIT["R"]
        if kind == 3:
            m = all_sides
            m &= ~(BIT["T"] if qy == 0 else BIT["B"])
            m &= ~(BIT["L"] if qx == 0 else BIT["R"])
        return m

    sheet = Image.new("RGBA", (half * 4, half * 4), (0, 0, 0, 0))
    pieces: List[Dict] = []
    for qy in range(2):
        for qx in range(2):
            for kind in range(4):
                tile = compose_art_tile(base, mask_for(qy, qx, kind), blend=blend)
                l, t, r, b = _quarter_box(s, "TL" if qy == 0 and qx == 0 else "TR" if qy == 0 else "BL" if qx == 0 else "BR")
                piece = tile.crop((l, t, r, b))
                idx = len(pieces)
                sheet.paste(piece, ((idx % 4) * half, (idx // 4) * half), piece)
                pieces.append({"index": idx, "qy": qy, "qx": qx, "kind": kind, "name": DUAL_PIECE_NAMES[kind]})
    meta = {
        "format": "pixel-anim-dual-art",
        "tile_size": s,
        "piece_size": half,
        "cols": 4,
        "rows": 4,
        "piece_count": 16,
        "blend": int(blend),
        "pieces": pieces,
    }
    return sheet, meta
