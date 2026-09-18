"""3×3 基础九宫格裁切与规格归一。

AI 生图尺寸通常不是 3 的整倍数；「自适应规格」策略：
1. 以整图短边为基准计算单格尺寸 cell = floor(min(w, h) / 3)，并向下取到偶数
   （后续四分块构图需要 2 的倍数）；
2. 在整图中心取 3*cell × 3*cell 区域（AI 常在四周留白/水印边缘，居中裁切
   更稳），等分裁出 9 张瓦片；
3. 归一化到目标尺寸（默认 32，偶数），最近邻缩放保持像素硬边。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

from core.processing.pixelizer import resize_nearest

logger = logging.getLogger("PixelFoundry.tilemap.tiles")

# 九宫格语义位置（行, 列）：角 / 边 / 中心
GRID_POSITIONS = {
    "tl": (0, 0), "top": (0, 1), "tr": (0, 2),
    "left": (1, 0), "center": (1, 1), "right": (1, 2),
    "bl": (2, 0), "bottom": (2, 1), "br": (2, 2),
}

EDGE_NAMES = ("top", "bottom", "left", "right")
CORNER_NAMES = ("tl", "tr", "bl", "br")


@dataclass
class BaseTileSet:
    """处理后的 9 张基础瓦片（RGBA，同一尺寸）。

    第 8 轮起，地形艺术走「对齐式构图」：`center` 是该地形的无缝特征纹理，
    `base_texture` + `band`/`radius` + `line_color`/`line_width` 是另一方地形的
    纹理与实测几何参数，`edges`/`corners` 仅作保存/编辑参考（构图不再依赖
    AI 把边界画在格内哪个位置，从而保证 47 张瓦片共享边逐像素相等）。
    """

    size: int
    center: Image.Image
    edges: Dict[str, Image.Image] = field(default_factory=dict)    # top/bottom/left/right
    corners: Dict[str, Image.Image] = field(default_factory=dict)  # tl/tr/bl/br
    line_color: Tuple[int, int, int] = (0, 0, 0)                   # 统一边界线色（=描边色）
    line_width: int = 1                                            # 边界线宽（=描边宽度）
    band: int = 0                                                  # 基础地形条带厚度（0=按 1/4 推算）
    radius: int = 0                                                # 转角圆角半径（0=取 band）
    base_texture: Optional[Image.Image] = None                      # 另一方（基础）地形纹理
    art_meta: Dict = field(default_factory=dict)                    # 实测参数（日志/元数据）

    def tile(self, name: str) -> Image.Image:
        """按名字取瓦片：'center' / 'top'… / 'tl'…。"""
        if name == "center":
            return self.center
        if name in self.edges:
            return self.edges[name]
        if name in self.corners:
            return self.corners[name]
        raise KeyError(f"未知瓦片: {name}")

    def all(self) -> List[Image.Image]:
        order = ["tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"]
        return [self.tile(n) for n in order]


def _snap_even(value: int, minimum: int = 8) -> int:
    value = max(minimum, int(value))
    return value - (value % 2)


def compute_cell_size(w: int, h: int, rows: int = 3, cols: int = 3) -> int:
    """按图短边自适应计算单格尺寸（偶数，≥8）。"""
    side = min(w // cols, h // rows)
    return _snap_even(side)


def grid_cell_px(
    sheet_size: int,
    cells: int,
    max_side: int = 2048,
    min_cell: int = 32,
) -> int:
    """由「期望生图边长」推算单格像素：取 32（优先 64）的倍数并保证不超上限。

    生图尺寸必须是格数的整数倍，否则 AI 画的网格与请求边长不一致、裁切必然错位；
    这里统一计算「边长 = cells × 单格」，并把同一个单格像素写进提示词。
    """
    cells = max(1, int(cells))
    side = max(min_cell * cells, min(int(sheet_size or 0) or max_side, max_side))
    side = min(side, max_side)
    cell = max(min_cell, side // cells)
    cell = (cell // 32) * 32 or min_cell
    if cell >= 64:
        cell = max(64, (cell // 64) * 64)
    while cells * cell > max_side and cell > min_cell:
        cell -= 32
    return cell


def crop_base_3x3(img: Image.Image) -> Tuple[List[Image.Image], int]:
    """把整图裁切成 3×3 瓦片列表（行优先，tl..br），返回 (tiles, cell)。

    cell = min(w,h)//3 取偶：**始终使用整张图**（按比例居中铺满），
    不做「只取中心一小块」的窗口裁剪——后者会让 AI 画的 3×3 网格整体错位。
    需要目标尺寸时由调用方 `normalize_tileset` 统一缩放。
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    cell = compute_cell_size(w, h)
    x0 = (w - cell * 3) // 2
    y0 = (h - cell * 3) // 2
    tiles: List[Image.Image] = []
    for r in range(3):
        for c in range(3):
            box = (x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell)
            tiles.append(rgba.crop(box))
    return tiles, cell


def to_base_set(tiles: List[Image.Image]) -> BaseTileSet:
    """9 张行优先瓦片 -> BaseTileSet（命名映射，尺寸取第一张）。"""
    if len(tiles) != 9:
        raise ValueError(f"需要 9 张瓦片，实际 {len(tiles)}")
    size = tiles[4].size
    if size[0] != size[1]:
        raise ValueError(f"瓦片必须为正方形: {size}")
    names = ["tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"]
    named = {n: t.convert("RGBA") for n, t in zip(names, tiles)}
    return BaseTileSet(
        size=size[0],
        center=named["center"],
        edges={n: named[n] for n in EDGE_NAMES},
        corners={n: named[n] for n in CORNER_NAMES},
    )


def _mode_downscale(arr: np.ndarray, th: int, tw: int) -> Optional[np.ndarray]:
    """整数倍块众数降采样：每个目标像素取对应源块中出现最多的颜色。

    AI 通常按「每格 128px」绘制像素风，直接点采样（最近邻取一个像素）会丢格色；
    块众数能精确还原 32×32 的像素画（全向量化，确定性并列取最小值）。
    """
    h, w = arr.shape[:2]
    if h % th or w % tw:
        return None
    kh, kw = h // th, w // tw
    c = arr.shape[2]
    blocks = (
        arr.reshape(th, kh, tw, kw, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(th * tw, kh * kw, c)
        .astype(np.int64)
    )
    key = np.zeros(blocks.shape[:2], dtype=np.int64)
    for i in range(c):
        key = (key << 8) | blocks[:, :, i]
    s = np.sort(key, axis=1)
    idx = np.arange(s.shape[1], dtype=np.int64)[None, :]
    change = np.ones(s.shape, dtype=bool)
    change[:, 1:] = s[:, 1:] != s[:, :-1]
    start = np.where(change, idx, 0)
    np.maximum.accumulate(start, axis=1, out=start)
    run_len = idx - start + 1
    best = np.argmax(run_len, axis=1)
    vals = s[np.arange(s.shape[0]), best]
    out = np.empty((vals.shape[0], c), dtype=np.uint8)
    for i in range(c - 1, -1, -1):
        out[:, i] = (vals & 255).astype(np.uint8)
        vals >>= 8
    return out.reshape(th, tw, c)


def resize_tile(img: Image.Image, target: int) -> Image.Image:
    """瓦片尺寸归一：整数倍缩小用块众数（保留像素格色），否则最近邻。"""
    if img.size == (target, target):
        return img
    src = img.convert("RGBA")
    if target < src.width:
        out = _mode_downscale(np.asarray(src), target, target)
        if out is not None:
            return Image.fromarray(out, "RGBA")
    return resize_nearest(src, (target, target))


def normalize_tileset(base: BaseTileSet, target_size: int = 32) -> BaseTileSet:
    """把 BaseTileSet 全部瓦片归一化到 target_size（偶数；缩小用块众数）。"""
    target = _snap_even(target_size, minimum=8)
    if base.size == target:
        return base
    scale = lambda im: resize_tile(im, target)
    return BaseTileSet(
        size=target,
        center=scale(base.center),
        edges={n: scale(t) for n, t in base.edges.items()},
        corners={n: scale(t) for n, t in base.corners.items()},
        line_color=base.line_color,
        line_width=base.line_width,
    )


# --------------------------------------------------------------------------- #
# 2×2 块图（每块 3×3）：地块生态图 / 建筑图
# --------------------------------------------------------------------------- #
@dataclass
class EcosystemSheet:
    """地块生态图（2×2 块，每块 3×3 瓦片组）。

    基础块位置由 `detect_base_block` 自动识别（AI 常不遵守「左上」的位置要求，
    识别可避免把特征块当成地面纹理这类灾难性错位）；也可显式指定。
    base: 基础地形块（同一生态下所有元素无缝融合的基准）；
    features: {特征名: 其余三块按阅读顺序}（中心=特征纯纹理，周边=与基础的过渡）。
    """

    base: BaseTileSet
    features: Dict[str, BaseTileSet] = field(default_factory=dict)
    base_pos: str = "tl"                # 实际使用的基础块位置 tl/tr/bl/br
    base_pos_detected: bool = False     # True=自动识别所得，False=显式指定/回退默认

    def terrain_sets(self) -> Dict[int, BaseTileSet]:
        """地形 id → 瓦片组（1=基础，2..=特征，顺序即 features 插入序）。"""
        out = {1: self.base}
        for i, (name, base) in enumerate(self.features.items(), start=2):
            out[i] = base
        return out


@dataclass
class BuildingSheet:
    """建筑图（2×2 块，每块 3×3 瓦片组）：wall / top / opening / pillar。"""

    wall: BaseTileSet
    top: BaseTileSet
    opening: BaseTileSet
    pillar: BaseTileSet


def crop_blocks(
    img: Image.Image,
    block_rows: int = 2,
    block_cols: int = 2,
) -> Tuple[List[List[List[Image.Image]]], int]:
    """把整图裁成 block_rows×block_cols 块、每块 3×3 瓦片。

    cell = min(w // (block_cols*3), h // (block_rows*3)) 取偶：**使用整张图**
    （按比例居中铺满），不使用「目标瓦片尺寸」作为窗口大小——否则 AI 画出的
    2×2×3 网格会被从中心截取一小块，导致所有瓦片错位（历史严重 bug）。

    返回 (blocks[br][bc][9 tiles 行优先], cell)。
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    cols = block_cols * 3
    rows = block_rows * 3
    cell = compute_cell_size(w, h, rows=rows, cols=cols)
    x0 = (w - cell * cols) // 2
    y0 = (h - cell * rows) // 2
    blocks: List[List[List[Image.Image]]] = []
    for br in range(block_rows):
        row: List[List[Image.Image]] = []
        for bc in range(block_cols):
            tiles: List[Image.Image] = []
            for r in range(3):
                for c in range(3):
                    box = (
                        x0 + (bc * 3 + c) * cell,
                        y0 + (br * 3 + r) * cell,
                        x0 + (bc * 3 + c + 1) * cell,
                        y0 + (br * 3 + r + 1) * cell,
                    )
                    tiles.append(rgba.crop(box))
            row.append(tiles)
        blocks.append(row)
    return blocks, cell


BLOCK_POSITIONS: Dict[str, Tuple[int, int]] = {
    "tl": (0, 0), "tr": (0, 1), "bl": (1, 0), "br": (1, 1),
}
BASE_POS_CHOICES = ("auto", "tl", "tr", "bl", "br")

# 判定「该块是纯基础地形块」的阈值：外侧带与中心色差超过它就说明该块内部有
# 边界（即它是特征块）；两块得分过近时视为不可判别，退回位置默认（左上）。
_BASE_SCORE_MAX = 30.0
_BASE_TIE_GAP = 6.0


def block_base_score(tiles: List[Image.Image]) -> float:
    """「该块是纯基础地形块」的代价，越低越像（0 附近=纯无缝纹理块）。

    判据是几何契约的逆命题：特征块的中心格是纯特征、外侧约 1/4 条带是基础地形，
    因此**外侧带颜色与中心格颜色必然不同**；基础地形块则处处同一纹理。

    三项：
    1. `ring_gap`：外侧带像素到中心格中位色的距离的中位数（主判据；中位数对邻块
       溢出的碎点、少量装饰稳健）；
    2. `centre_spread`：中心格自身的杂色度（特征纹理比纯地面更花）；
    3. `block_std`：整块亮度标准差（特征团块即使画得偏小、中心格仍以基础色为主，
       整块的强对比也会把它暴露出来——避免「小团块被误判成地面」）。
    """
    arrs = [np.asarray(t.convert("RGB"), dtype=np.float64) for t in tiles]
    if len(arrs) != 9:
        raise ValueError(f"需要 9 张瓦片，实际 {len(arrs)}")
    side = arrs[0].shape[0]
    if side < 4:
        return 0.0
    block = np.stack([np.stack(arrs[r * 3:r * 3 + 3], axis=1) for r in range(3)], axis=0)
    block = block.reshape(side * 3, side * 3, 3)
    centre = arrs[4].reshape(-1, 3)
    med = np.median(centre, axis=0)
    band = max(1, side // 4)
    ring = np.concatenate(
        [
            block[:band].reshape(-1, 3),
            block[-band:].reshape(-1, 3),
            block[:, :band].reshape(-1, 3),
            block[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )
    ring_gap = float(np.median(np.linalg.norm(ring - med, axis=1)))
    centre_spread = float(np.median(np.linalg.norm(centre - med, axis=1)))
    block_std = float(block.mean(axis=2).std())
    return ring_gap + 0.5 * centre_spread + 0.25 * block_std


def detect_base_block(
    blocks,
    max_score: float = _BASE_SCORE_MAX,
    tie_gap: float = _BASE_TIE_GAP,
) -> Tuple[str, Dict[str, float], bool]:
    """自动识别 2×2 块图中哪一块是「纯基础地形块」。

    返回 (位置键, 各位置得分, 是否可信)。
    - 正常：取得分最低者（= 内部没有边界的块）；
    - 四块得分都高（都像特征块，AI 没画出纯基础块）→ 回退位置默认 "tl"、不可信；
    - 最优与次优几乎同分（例如两块都是同一基础地形）→ 仍取最低分，但标记不可信，
      调用方据此提示用户核对。
    """
    if len(blocks) != 2 or len(blocks[0]) != 2 or len(blocks[1]) != 2:
        raise ValueError("需要 2×2 块图")
    scores = {k: block_base_score(blocks[br][bc]) for k, (br, bc) in BLOCK_POSITIONS.items()}
    order = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_score = order[0]
    second, second_score = order[1]
    if best_score > max_score:
        logger.warning("未找到纯基础地形块（最低分 %.1f > %.1f），回退左上块", best_score, max_score)
        return "tl", scores, False
    confident = not (second_score - best_score < tie_gap and second_score < max_score * 1.6)
    if not confident:
        logger.info("基础块识别不唯一（%s=%.1f vs %s=%.1f），按最低分取 %s", best, best_score, second, second_score, best)
    return best, scores, confident


def cell_box(
    img: Image.Image,
    rows: int,
    cols: int,
    r: int,
    c: int,
) -> Tuple[int, int, int, int]:
    """整图第 (r, c) 格的像素框（与 crop_blocks/crop_base_3x3 同一居中几何）。

    用于把「对某一格的修补」写回整图（例如文字修补后同步清理后的底图）。
    """
    w, h = img.size
    cell = compute_cell_size(w, h, rows=rows, cols=cols)
    x0 = (w - cell * cols) // 2 + c * cell
    y0 = (h - cell * rows) // 2 + r * cell
    return (x0, y0, x0 + cell, y0 + cell)


def ecosystem_from_blocks(
    blocks,
    tile_size: int = 32,
    feature_names=None,
    base_pos: str = "auto",
) -> EcosystemSheet:
    """2×2 块图 -> EcosystemSheet。

    base_pos: "auto"（默认识别「最像纯基础地形」的块，见 `detect_base_block`）
    或 "tl"/"tr"/"bl"/"br" 显式指定；其余三块按阅读顺序（左上→右上→左下→右下）
    依次对应用户给出的特征列表。识别不可信时回退位置默认（左上=基础）。
    """
    if len(blocks) != 2 or len(blocks[0]) != 2 or len(blocks[1]) != 2:
        raise ValueError("需要 2×2 块图")
    pos = (base_pos or "auto").lower()
    if pos not in BASE_POS_CHOICES:
        raise ValueError(f"未知基础块位置: {base_pos}（可选 {'/'.join(BASE_POS_CHOICES)}）")
    detected = pos == "auto"
    if detected:
        pos, _scores, _ok = detect_base_block(blocks)
    names = list(feature_names) if feature_names else ["feature_tr", "feature_bl", "feature_br"]
    br0, bc0 = BLOCK_POSITIONS[pos]
    order = [(br, bc) for br, bc in ((0, 0), (0, 1), (1, 0), (1, 1)) if (br, bc) != (br0, bc0)]
    feats: Dict[str, BaseTileSet] = {}
    for (br, bc), name in zip(order, names):
        base = to_base_set(blocks[br][bc])
        feats[name] = normalize_tileset(base, target_size=tile_size)
    return EcosystemSheet(
        base=normalize_tileset(to_base_set(blocks[br0][bc0]), target_size=tile_size),
        features=feats,
        base_pos=pos,
        base_pos_detected=detected,
    )


def building_from_blocks(blocks, tile_size: int = 32) -> BuildingSheet:
    """2×2 块图 -> BuildingSheet（左上 wall / 右上 top / 左下 opening / 右下 pillar）。"""
    if len(blocks) != 2 or len(blocks[0]) != 2 or len(blocks[1]) != 2:
        raise ValueError("需要 2×2 块图")
    norm = lambda tiles: normalize_tileset(to_base_set(tiles), target_size=tile_size)
    return BuildingSheet(
        wall=norm(blocks[0][0]),
        top=norm(blocks[0][1]),
        opening=norm(blocks[1][0]),
        pillar=norm(blocks[1][1]),
    )
