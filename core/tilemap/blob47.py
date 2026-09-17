"""FrameRonin「Blob」16 / 47 图块拼接逻辑（写法与约定搬运自 systemchester/FrameRonin）。

**来源**：<https://github.com/systemchester/FrameRonin>
- `frontend/src/components/infiniteMap/blobTerrain.ts`
- `frontend/public/map/blob/map.html`（原始独立示例）

**搬运范围与许可说明**：该仓库未附开源许可证。因此本项目只搬运其
**邻域掩码约定、47 掩码 → 图集槽位映射表、就近回退规则、图集几何（3×24）、
程序化地形参数**等「算法与数据约定」，Python 代码全部由本项目重写；
若要直接引用其源码或美术资源（`public/map/blob/frame_*.png`），请先与原作者确认授权。

与其实现逐条对应的搬运要点
----------------------------
1. **八邻掩码**（`compute_blob_mask`）：N=2 S=64 W=8 E=16；对角 TL=1 TR=4 BL=32 BR=128，
   且**对角位只在两个相邻正交位同为同类时才置位**（与 Godot / Tiled 的 blob 约定一致）。
2. **47 个可达掩码 → 图集槽位**（`MASK_TO_INDEX`）：槽位 0..70，**71 为占位空底、禁止选用**；
   孤立格（mask=0）→ 槽位 13，四面全满（mask=255）→ 槽位 4。
3. **未知掩码就近回退**（`nearest_tile_index`）：在 47 个合法键上取汉明距离最小者；
   仍不可用则回退槽位 `FALLBACK_TILE_INDEX = 4`。
4. **图集几何**（`atlas_cell_size` / `atlas_cell_box` / `slice_atlas_tile`）：
   3 列 × 24 行 = 72 槽，子块尺寸 = (宽//3, 高//24)，行 0 在上（不做垂直翻转）。
5. **程序化地形**（`ProceduralTerrain`）：Perlin + FBM 陆高 / 山地场、河谷走廊判水、
   按 32×32 世界格分块缓存；`sample_tile_index()` 给出每格的「水 / 山 / 平原 + 图集槽位」，
   与 `is_walkable()` / `is_land()` 可走性判定。
6. **16 图块族**（`build_tile16_atlas`）：FrameRonin 只实现 47 槽；这里按同一套约定补上
   经典 4 位（N/E/S/W）16 槽版本：槽位 = n | e·2 | s·4 | w·8，4×4 图集，
   对角位取「两个相邻正交位都满」，因此不出现内凹角（16 图块族的画法）。
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .tiles import BaseTileSet

logger = logging.getLogger("PixelAnimIDE.tilemap.blob47")

# --------------------------------------------------------------------------- #
# 1) 常量（与 FrameRonin 完全一致）
# --------------------------------------------------------------------------- #
BLOB_TILE_COLS = 3
BLOB_TILE_ROWS = 24
BLOB_SLOT_COUNT = BLOB_TILE_COLS * BLOB_TILE_ROWS          # 72
BLOB_CHUNK_SIZE = 32
MAX_CACHED_CHUNKS = 128

FORBIDDEN_TILE_INDEX = 71                                   # 占位空底，禁止选用
FALLBACK_TILE_INDEX = 4                                     # = mask 255（全满内部块）

BIT = {"TL": 1, "T": 2, "TR": 4, "L": 8, "R": 16, "BL": 32, "B": 64, "BR": 128}

#: 47 个可达掩码 → 图集槽位（数据表逐条对应 FrameRonin `MASK_TO_INDEX`）
MASK_TO_INDEX: Dict[int, int] = {
    0: 13,      # 孤立一格
    208: 0, 248: 1, 104: 2, 214: 3, 255: 4, 107: 5, 22: 6, 31: 7, 11: 8,
    80: 9, 24: 10, 72: 11, 66: 12, 18: 15, 10: 17, 64: 19, 16: 21, 90: 22,
    8: 23, 2: 25, 88: 28, 82: 30, 74: 32, 26: 34, 95: 37, 123: 39, 222: 41,
    250: 43, 127: 45, 223: 46, 251: 48, 254: 49, 86: 51, 75: 52, 210: 54,
    106: 55, 120: 57, 216: 58, 27: 60, 30: 61, 218: 63, 122: 64, 94: 66,
    91: 67, 126: 69, 219: 70,
}
MASK_KEYS: Tuple[int, ...] = tuple(sorted(MASK_TO_INDEX))

# 16 图块族（4 位：N/E/S/W）槽位 = n | e*2 | s*4 | w*8
TILE16_COLS = 4
TILE16_ROWS = 4
TILE16_SLOT_COUNT = TILE16_COLS * TILE16_ROWS
TILE16_BITS = {"N": 1, "E": 2, "S": 4, "W": 8}


# --------------------------------------------------------------------------- #
# 2) 掩码与就近回退
# --------------------------------------------------------------------------- #
def pop8(n: int) -> int:
    """8 位中 1 的个数（汉明距离用）。"""
    n &= 255
    c = 0
    while n:
        c += n & 1
        n >>= 1
    return c


def compute_blob_mask(x: int, y: int, pred: Callable[[int, int], bool]) -> int:
    """八邻 Blob 掩码；`pred(nx, ny)` 为「该格与当前格同类」。

    与 FrameRonin `computeBlobMask` 一致：对角位仅在两个相邻正交位同为同类时置位。
    """
    n = bool(pred(x, y - 1))
    s = bool(pred(x, y + 1))
    w = bool(pred(x - 1, y))
    e = bool(pred(x + 1, y))
    mask = 0
    if n:
        mask += BIT["T"]
    if s:
        mask += BIT["B"]
    if w:
        mask += BIT["L"]
    if e:
        mask += BIT["R"]
    if n and w and pred(x - 1, y - 1):
        mask += BIT["TL"]
    if n and e and pred(x + 1, y - 1):
        mask += BIT["TR"]
    if s and w and pred(x - 1, y + 1):
        mask += BIT["BL"]
    if s and e and pred(x + 1, y + 1):
        mask += BIT["BR"]
    return mask


def nearest_tile_index(mask: int) -> int:
    """mask → 图集槽位；无精确匹配时按汉明距离就近，且永不返回 71。"""
    mask &= 255
    direct = MASK_TO_INDEX.get(mask)
    if direct is not None and direct != FORBIDDEN_TILE_INDEX:
        return direct
    best_key = MASK_KEYS[0]
    best_d = 99
    for k in MASK_KEYS:
        v = MASK_TO_INDEX[k]
        if v == FORBIDDEN_TILE_INDEX:
            continue
        d = pop8(mask ^ k)
        if d < best_d:
            best_d = d
            best_key = k
    out = MASK_TO_INDEX.get(best_key)
    if out is None or out == FORBIDDEN_TILE_INDEX:
        return FALLBACK_TILE_INDEX
    return out


def blob_mask_to_slot_table() -> Dict[int, int]:
    """全 256 种掩码 → 图集槽位（导出元数据用，与 FrameRonin 运行时行为一致）。"""
    return {m: nearest_tile_index(m) for m in range(256)}


def mask_for_slot(slot: int) -> Optional[int]:
    """槽位 → 其代表掩码（占位/空槽返回 None）。"""
    for mask, idx in MASK_TO_INDEX.items():
        if idx == slot:
            return mask
    return None


def tile16_index(mask: int) -> int:
    """八邻掩码 → 16 槽下标（n | e·2 | s·4 | w·8）。"""
    m = mask & 255
    idx = 0
    if m & BIT["T"]:
        idx |= TILE16_BITS["N"]
    if m & BIT["R"]:
        idx |= TILE16_BITS["E"]
    if m & BIT["B"]:
        idx |= TILE16_BITS["S"]
    if m & BIT["L"]:
        idx |= TILE16_BITS["W"]
    return idx


def canonical_tile16_mask(mask: int) -> int:
    """16 图块族构图用的八邻掩码：对角位 = 两个相邻正交位都满（无内凹角）。"""
    m = mask & 255
    out = m & (BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"])
    pairs = (("TL", "T", "L"), ("TR", "T", "R"), ("BL", "B", "L"), ("BR", "B", "R"))
    for diag, a, b in pairs:
        if (out & BIT[a]) and (out & BIT[b]):
            out |= BIT[diag]
    return out


# --------------------------------------------------------------------------- #
# 3) 图集几何（3×24 = 72 槽；行 0 在上，不做垂直翻转）
# --------------------------------------------------------------------------- #
def atlas_cell_size(width: int, height: int, cols: int = BLOB_TILE_COLS, rows: int = BLOB_TILE_ROWS) -> Tuple[int, int]:
    """子块尺寸 = (宽//cols, 高//rows)（整数格，避免非整数缩放导致错缝）。"""
    return (int(width) // cols, int(height) // rows)


def atlas_cell_box(
    index: int,
    width: int,
    height: int,
    cols: int = BLOB_TILE_COLS,
    rows: int = BLOB_TILE_ROWS,
) -> Tuple[int, int, int, int]:
    """槽位 → 图集像素框 (x0, y0, x1, y1)。"""
    cw, ch = atlas_cell_size(width, height, cols, rows)
    col = index % cols
    row = index // cols
    return (col * cw, row * ch, (col + 1) * cw, (row + 1) * ch)


def slice_atlas_tile(
    sheet: Image.Image,
    index: int,
    cols: int = BLOB_TILE_COLS,
    rows: int = BLOB_TILE_ROWS,
) -> Image.Image:
    """按 FrameRonin 的几何切出某个槽位的子块（越界槽位返回空白块）。"""
    w, h = sheet.size
    box = atlas_cell_box(index, w, h, cols, rows)
    return sheet.convert("RGBA").crop(box)


# --------------------------------------------------------------------------- #
# 4) 用本项目「对齐式构图」生成 FrameRonin 布局的图集
# --------------------------------------------------------------------------- #
def build_blob47_atlas(art: BaseTileSet, compose=None) -> Tuple[Image.Image, Dict]:
    """生成 3×24（72 槽）FrameRonin 布局的 47 图块图集。

    每个有掩码对应的槽位放该掩码的瓦片；没有掩码对应的槽位（含 71 号）留空透明，
    与 FrameRonin 图集约定一致（槽位 71 永不使用）。
    """
    from .autotile import compose_art_tile

    compose = compose or compose_art_tile
    s = art.size
    sheet = Image.new("RGBA", (BLOB_TILE_COLS * s, BLOB_TILE_ROWS * s), (0, 0, 0, 0))
    for mask, idx in MASK_TO_INDEX.items():
        if idx == FORBIDDEN_TILE_INDEX:
            continue
        tile = compose(art, mask)
        sheet.paste(tile, ((idx % BLOB_TILE_COLS) * s, (idx // BLOB_TILE_COLS) * s), tile)
    meta = {
        "format": "frameronin-blob47",
        "source": "systemchester/FrameRonin blobTerrain.ts（算法/映射表搬运，代码重写）",
        "tile_size": s,
        "sheet_cols": BLOB_TILE_COLS,
        "sheet_rows": BLOB_TILE_ROWS,
        "slot_count": BLOB_SLOT_COUNT,
        "tile_count": len(MASK_TO_INDEX),
        "forbidden_index": FORBIDDEN_TILE_INDEX,
        "fallback_index": FALLBACK_TILE_INDEX,
        "mask_to_index": {str(m): idx for m, idx in MASK_TO_INDEX.items()},
        "slot_to_mask": {str(idx): m for m, idx in MASK_TO_INDEX.items()},
        "all_masks_to_index": {str(m): v for m, v in blob_mask_to_slot_table().items()},
        "mask_convention": "N=2 S=64 W=8 E=16; 对角 TL=1 TR=4 BL=32 BR=128（仅两正交位为满时有效）",
    }
    return sheet, meta


def build_tile16_atlas(art: BaseTileSet, compose=None) -> Tuple[Image.Image, Dict]:
    """生成 4×4（16 槽）经典 4 位图块图集（同一条搬运约定的 16 图块族）。"""
    from .autotile import compose_art_tile

    compose = compose or compose_art_tile
    s = art.size
    sheet = Image.new("RGBA", (TILE16_COLS * s, TILE16_ROWS * s), (0, 0, 0, 0))
    for mask in MASK_TO_INDEX:
        idx = tile16_index(mask)
        tile = compose(art, canonical_tile16_mask(mask))
        sheet.paste(tile, ((idx % TILE16_COLS) * s, (idx // TILE16_COLS) * s), tile)
    meta = {
        "format": "frameronin-tile16",
        "source": "systemchester/FrameRonin 约定（16 图块族为本项目按同一约定补齐）",
        "tile_size": s,
        "sheet_cols": TILE16_COLS,
        "sheet_rows": TILE16_ROWS,
        "slot_count": TILE16_SLOT_COUNT,
        "tile_count": TILE16_SLOT_COUNT,
        "index_bits": dict(TILE16_BITS),
        "mask_convention": "slot = n | e*2 | s*4 | w*8（对角位取两正交位都为满）",
    }
    return sheet, meta


# --------------------------------------------------------------------------- #
# 5) 程序化地形（Perlin / FBM / 河谷 / 山地，FrameRonin 同款参数）
# --------------------------------------------------------------------------- #
IMG_MTN = 0
IMG_NORM = 1

LAND_SCALE = 26.0
LAND_OCTAVES = 3
LAND_PERSISTENCE = 0.52
LAND_LACUNARITY = 2.05
MOUNTAIN_SCALE = 42.0
MOUNTAIN_OCTAVES = 3
MOUNTAIN_PERSISTENCE = 0.5
MOUNTAIN_LACUNARITY = 2.02
RIVER_SCALE = 12.5
RIVER_OCTAVES = 2
RIVER_PERSISTENCE = 0.48
RIVER_LACUNARITY = 2.02
RIVER_ABS_THRESH = 0.1
RIVER_HEIGHT_BAND = 0.16


def string_seed_to_uint32(text: str) -> int:
    """FNV-1a：任意字符串 → 32 位种子。"""
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def _mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = (a ^ (a >> 15)) * (a | 1) & 0xFFFFFFFF
        t ^= (t + (t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rnd


class Perlin2D:
    """FrameRonin 同款 2D Perlin 噪声（自身 permutation + grad2，可复现）。"""

    def __init__(self, seed: int):
        rnd = _mulberry32((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
        p = list(range(256))
        for i in range(255, 0, -1):
            j = int(rnd() * (i + 1))
            p[i], p[j] = p[j], p[i]
        self.perm = np.array(p + p, dtype=np.int32)

    def __call__(self, x: float, y: float) -> float:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        xf, yf = x - xi, y - yi
        X, Y = xi & 255, yi & 255
        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
        perm = self.perm

        def grad(h: int, gx: float, gy: float) -> float:
            h &= 3
            uu = gx if h < 2 else gy
            vv = gy if h < 2 else gx
            return (uu if (h & 1) == 0 else -uu) + (vv if (h & 2) == 0 else -vv)

        aa = int(perm[X + int(perm[Y])])
        ab = int(perm[X + int(perm[Y + 1])])
        ba = int(perm[X + 1 + int(perm[Y])])
        bb = int(perm[X + 1 + int(perm[Y + 1])])
        x1 = grad(aa, xf, yf) + u * (grad(ba, xf - 1, yf) - grad(aa, xf, yf))
        x2 = grad(ab, xf, yf - 1) + u * (grad(bb, xf - 1, yf - 1) - grad(ab, xf, yf - 1))
        return x1 + v * (x2 - x1)

    def fbm(self, x: float, y: float, octaves: int, persistence: float, lacunarity: float) -> float:
        total = 0.0
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for _ in range(octaves):
            total += amp * self(x * freq, y * freq)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / norm


class ProceduralTerrain:
    """无限程序化地形（陆高 / 河 / 山），按 32×32 分块缓存 —— FrameRonin `BlobWorld` 同款。

    - `land(x, y)`：1 陆 / 0 水（海、湖、河）
    - `biome(x, y)`：陆上 `IMG_MTN` 山 / `IMG_NORM` 平
    - `mask_water` / `mask_land_biome`：八邻掩码
    - `sample_tile_index(x, y)`：`("water"|"mtn"|"norm", 图集槽位)`
    """

    def __init__(self, seed: str = "default", sea_level: float = 0.42, mountain_threshold: float = 0.48):
        self.sea_level = float(sea_level)
        self.mountain_threshold = float(mountain_threshold)
        self._cache: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self._queue: List[Tuple[int, int]] = []
        self.reseed(seed)

    # -- 参数 / 种子 ------------------------------------------------------ #
    def reseed(self, seed: str) -> None:
        base = string_seed_to_uint32((seed or "default").strip() or "default")
        self.n_height = Perlin2D(base)
        self.n_mountain = Perlin2D(base ^ 0x414004)
        self.n_river = Perlin2D(base ^ 0x927B51C1)
        self.clear_cache()

    def set_params(self, sea_level: Optional[float] = None, mountain_threshold: Optional[float] = None) -> None:
        if sea_level is not None:
            self.sea_level = float(sea_level)
        if mountain_threshold is not None:
            self.mountain_threshold = float(mountain_threshold)
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache.clear()
        self._queue.clear()

    # -- 分块生成 --------------------------------------------------------- #
    def _build_chunk(self, cx: int, cy: int) -> Dict[str, np.ndarray]:
        size = BLOB_CHUNK_SIZE
        lx, ly = np.meshgrid(np.arange(size), np.arange(size))
        wx = cx * size + lx
        wy = cy * size + ly

        raw = self._fbm_grid(self.n_height, wx / LAND_SCALE, wy / LAND_SCALE,
                             LAND_OCTAVES, LAND_PERSISTENCE, LAND_LACUNARITY)
        height = np.clip((raw + 1.0) * 0.5, 0.0, 1.0)

        river = self._fbm_grid(self.n_river, wx / RIVER_SCALE + 31.4, wy / RIVER_SCALE + 12.8,
                               RIVER_OCTAVES, RIVER_PERSISTENCE, RIVER_LACUNARITY)
        corridor = np.abs(river) < RIVER_ABS_THRESH
        low = height < self.sea_level + RIVER_HEIGHT_BAND
        lake_or_sea = height <= self.sea_level
        is_river = corridor & low & ~lake_or_sea
        water = lake_or_sea | is_river
        land = np.where(water, 0, 1).astype(np.uint8)

        mtn_raw = self._fbm_grid(self.n_mountain, wx / MOUNTAIN_SCALE + 2.7, wy / MOUNTAIN_SCALE + 1.1,
                                 MOUNTAIN_OCTAVES, MOUNTAIN_PERSISTENCE, MOUNTAIN_LACUNARITY)
        mtn = (mtn_raw + 1.0) * 0.5
        biome = np.where(mtn > self.mountain_threshold, IMG_MTN, IMG_NORM).astype(np.uint8)
        biome[water] = 0
        return {"land": land, "biome": biome}

    def _fbm_grid(self, noise: Perlin2D, xs: np.ndarray, ys: np.ndarray,
                  octaves: int, persistence: float, lacunarity: float) -> np.ndarray:
        """FBM 的 numpy 实现（逐倍频整片求值，避免逐格 Python 循环）。"""
        total = np.zeros(xs.shape, dtype=np.float64)
        amp, freq, norm = 1.0, 1.0, 0.0
        for _ in range(octaves):
            vec = np.vectorize(lambda a, b: noise(a * freq, b * freq))
            total += amp * vec(xs, ys)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / norm

    def _chunk(self, cx: int, cy: int) -> Dict[str, np.ndarray]:
        key = (cx, cy)
        got = self._cache.get(key)
        if got is not None:
            return got
        ch = self._build_chunk(cx, cy)
        self._cache[key] = ch
        self._queue.append(key)
        while len(self._queue) > MAX_CACHED_CHUNKS:
            old = self._queue.pop(0)
            self._cache.pop(old, None)
        return ch

    # -- 世界查询 --------------------------------------------------------- #
    def land(self, wx: int, wy: int) -> int:
        cx, cy = wx // BLOB_CHUNK_SIZE, wy // BLOB_CHUNK_SIZE
        ch = self._chunk(cx, cy)
        lx = wx - cx * BLOB_CHUNK_SIZE
        ly = wy - cy * BLOB_CHUNK_SIZE
        return int(ch["land"][ly, lx])

    def biome(self, wx: int, wy: int) -> int:
        cx, cy = wx // BLOB_CHUNK_SIZE, wy // BLOB_CHUNK_SIZE
        ch = self._chunk(cx, cy)
        lx = wx - cx * BLOB_CHUNK_SIZE
        ly = wy - cy * BLOB_CHUNK_SIZE
        return int(ch["biome"][ly, lx])

    def mask_water(self, wx: int, wy: int) -> int:
        return compute_blob_mask(wx, wy, lambda x, y: self.land(x, y) == 0)

    def mask_land_biome(self, wx: int, wy: int, biome: int) -> int:
        return compute_blob_mask(
            wx, wy, lambda x, y: self.land(x, y) == 1 and self.biome(x, y) == biome
        )

    def sample_tile_index(self, wx: int, wy: int) -> Tuple[str, int]:
        """→ (kind, 图集槽位)；kind ∈ water / mtn / norm。"""
        if self.land(wx, wy) == 0:
            return "water", nearest_tile_index(self.mask_water(wx, wy))
        b = self.biome(wx, wy)
        mask = self.mask_land_biome(wx, wy, b)
        return ("mtn" if b == IMG_MTN else "norm"), nearest_tile_index(mask)

    def is_walkable(self, wx: int, wy: int) -> bool:
        """仅平地可走（FrameRonin `isBlobTileWalkable` 同款）。"""
        return self.land(wx, wy) == 1 and self.biome(wx, wy) == IMG_NORM

    def is_land(self, wx: int, wy: int) -> bool:
        """陆地（含山地），树木等装饰用（`isBlobTileLandNotWater` 同款）。"""
        return self.land(wx, wy) == 1

    def grid(self, width: int, height: int, origin: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """批量取一块世界的 (kind, 槽位)：返回 (grid_kind, grid_slot) 两个数组。

        kind 用整数表达：0=水 1=平 2=山（便于地图模型直接使用）。
        """
        ox, oy = origin
        kinds = np.zeros((height, width), dtype=np.int32)
        slots = np.zeros((height, width), dtype=np.int32)
        for y in range(height):
            for x in range(width):
                kind, slot = self.sample_tile_index(ox + x, oy + y)
                kinds[y, x] = {"water": 0, "norm": 1, "mtn": 2}[kind]
                slots[y, x] = slot
        return kinds, slots
