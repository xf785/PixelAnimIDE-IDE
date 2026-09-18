"""程序化地形（柏林噪声 + FBM + 河谷 + 山地场）—— 供柏林噪声大地图使用。

从原 `blob47` 模块迁出并**去掉第三方图集约定**：这里只保留"世界生成"部分，
输出的是**我们自己的掩码**（8 邻域 canonical 掩码），可直接喂给
`compose_art_tile` / `TileMapModel`，不再绑定任何外部图集布局。

- `land(x, y)`：1 陆 / 0 水（海、湖、河）
- `biome(x, y)`：陆上 `IMG_MTN` 山 / `IMG_NORM` 平
- `mask_water` / `mask_land_biome`：八邻掩码（对角位仅在两侧正交位都为满时有效）
- `grid(w, h, origin)`：批量取一块世界 → (kinds, masks)，kind 0=水 1=平 2=山
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .autotile import BIT, canonical_mask

logger = logging.getLogger("PixelFoundry.tilemap.terrain_noise")

IMG_MTN = 0
IMG_NORM = 1

CHUNK_SIZE = 32
MAX_CACHED_CHUNKS = 128

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
    """2D Perlin 噪声（自身 permutation + grad2，同种子可复现）。"""

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


def neighbour_mask(x: int, y: int, pred: Callable[[int, int], bool]) -> int:
    """按预测函数取 8 邻域掩码（对角位遵循 canonical 规则）。"""
    mask = 0
    for (dx, dy, bit) in ((-1, -1, BIT["TL"]), (0, -1, BIT["T"]), (1, -1, BIT["TR"]),
                          (-1, 0, BIT["L"]), (1, 0, BIT["R"]),
                          (-1, 1, BIT["BL"]), (0, 1, BIT["B"]), (1, 1, BIT["BR"])):
        if pred(x + dx, y + dy):
            mask |= bit
    return canonical_mask(mask)


class ProceduralTerrain:
    """无限程序化地形（陆高 / 河 / 山），按 32×32 分块缓存。"""

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
        size = CHUNK_SIZE
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
        cx, cy = wx // CHUNK_SIZE, wy // CHUNK_SIZE
        ch = self._chunk(cx, cy)
        return int(ch["land"][wy - cy * CHUNK_SIZE, wx - cx * CHUNK_SIZE])

    def biome(self, wx: int, wy: int) -> int:
        cx, cy = wx // CHUNK_SIZE, wy // CHUNK_SIZE
        ch = self._chunk(cx, cy)
        return int(ch["biome"][wy - cy * CHUNK_SIZE, wx - cx * CHUNK_SIZE])

    def mask_water(self, wx: int, wy: int) -> int:
        return neighbour_mask(wx, wy, lambda x, y: self.land(x, y) == 0)

    def mask_land_biome(self, wx: int, wy: int, biome: int) -> int:
        return neighbour_mask(wx, wy, lambda x, y: self.land(x, y) == 1 and self.biome(x, y) == biome)

    def sample_tile(self, wx: int, wy: int) -> Tuple[str, int]:
        """→ (kind, 掩码)；kind ∈ water / mtn / norm。掩码可直接喂给 `compose_art_tile`。"""
        if self.land(wx, wy) == 0:
            return "water", self.mask_water(wx, wy)
        b = self.biome(wx, wy)
        return ("mtn" if b == IMG_MTN else "norm"), self.mask_land_biome(wx, wy, b)

    def is_walkable(self, wx: int, wy: int) -> bool:
        """仅平地可走。"""
        return self.land(wx, wy) == 1 and self.biome(wx, wy) == IMG_NORM

    def is_land(self, wx: int, wy: int) -> bool:
        """陆地（含山地），树木等装饰用。"""
        return self.land(wx, wy) == 1

    def grid(self, width: int, height: int, origin: Tuple[int, int] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """批量取一块世界：返回 (kinds, masks)；kind 0=水 1=平 2=山。"""
        ox, oy = origin
        kinds = np.zeros((height, width), dtype=np.int32)
        masks = np.zeros((height, width), dtype=np.int32)
        for y in range(height):
            for x in range(width):
                kind, mask = self.sample_tile(ox + x, oy + y)
                kinds[y, x] = {"water": 0, "norm": 1, "mtn": 2}[kind]
                masks[y, x] = mask
        return kinds, masks
