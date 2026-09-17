"""第 10 轮回归：演示地图覆盖全部瓦片结构 + 分层描边（更接近手绘 47 图块）。

之前默认演示图是把整张地图填满同一种地形，预览里只有「全填充」一张瓦片，
所以外角/内角/单行/单列这些瓦片在成品里根本看不到；边界也只有一条平色描边。
"""
import numpy as np
import pytest
from PIL import Image

from core.tilemap import BIT, TileMapModel, canonical_mask, compose_art_tile, mask_for_terrain
from core.tilemap.tiles import BaseTileSet, EDGE_NAMES
from core.workflow.tilemap_workflow import TilemapParams, _demo_map

S = 32
T, B, L, R = BIT["T"], BIT["B"], BIT["L"], BIT["R"]
TL, TR, BL, BR = BIT["TL"], BIT["TR"], BIT["BL"], BIT["BR"]


def _art(feat=(38, 104, 172), ground=(236, 240, 246), band=8, radius=8, outline=None):
    tex = Image.new("RGBA", (S, S), feat + (255,))
    gtex = Image.new("RGBA", (S, S), ground + (255,))
    meta = {"outline": outline or [[81, 75, 66], [150, 152, 150]], "bevel": [], "band_px": band}
    return BaseTileSet(
        size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
        corners={n: tex for n in ("tl", "tr", "bl", "br")},
        line_color=(81, 75, 66), line_width=2, band=band, radius=radius,
        base_texture=gtex, art_meta=meta,
    )


def test_showcase_demo_map_covers_every_tile_class():
    """展示地形必须真的用到孤立/单行/单列/内角/外角/端头这些结构。"""
    m = TileMapModel(28, 18, tile_size=S)
    for tid in (1, 2, 3):
        m.set_terrain(tid, _art())
    m.set_base_terrain(1)
    _demo_map(m, TilemapParams(description="t", category="ground", map_source="showcase",
                               map_width=28, map_height=18))
    grid = m.grid
    seen = set()
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            nb = [[int(grid[ny, nx]) if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] else 0
                   for nx in range(x - 1, x + 2)] for ny in range(y - 1, y + 2)]
            seen.add(canonical_mask(mask_for_terrain(nb, int(grid[y, x]), base_terrain=1)))
    assert len(seen) >= 25, f"展示图覆盖的掩码类型太少: {len(seen)}"
    for name, mask in (
        ("孤立格", 0),
        ("单行横条(L|R)", L | R),
        ("单列竖条(T|B)", T | B),
        ("单侧/端头(L)", L),
        ("端头(T)", T),
        ("外角(R|B|BR)", R | B | BR),
        ("内角(255^TL)", 255 & ~TL),
    ):
        assert mask in seen, f"展示图缺少 {name}（mask={mask:#x}）"


def test_edges_are_layered_not_flat():
    """边界必须是分层描边（实测色调 + 接触阴影 + 地形侧倒角），而不是一条平色线。"""
    art = _art()
    tile = np.asarray(compose_art_tile(art, (T | B | L | R | TL | TR | BL | BR) & ~T))
    band = art.band
    outline = [tuple(c) for c in art.art_meta["outline"]]
    # 紧贴地形的两圈 = 实测描边色调（由内向外）
    for k, tone in enumerate(outline, start=1):
        got = tuple(int(v) for v in tile[band - k, S // 2, :3])
        assert max(abs(a - b) for a, b in zip(got, tone)) <= 40, (k, got, tone)
    # 描边外侧还有一圈接触阴影：比纯地面暗、比描边亮
    shadow = tuple(int(v) for v in tile[band - len(outline) - 1, S // 2, :3])
    pure = tuple(int(v) for v in np.asarray(art.base_texture)[band - len(outline) - 1, S // 2, :3])
    assert sum(shadow) < sum(pure), f"描边外侧应有接触阴影: {shadow} vs {pure}"
    # 地形侧倒角：边界内侧第一圈比纹理本体更亮或更暗（不是平贴）
    ring = tuple(int(v) for v in tile[band, S // 2, :3])
    body = tuple(int(v) for v in np.asarray(art.center)[band, S // 2, :3])
    assert ring != body, "地形侧应有倒角/内描边"


def test_thin_strip_keeps_feature_core():
    """条带过宽时会被钳制，保证单行/单列瓦片仍留有地形核心（不会整格变成地面）。"""
    art = _art(band=20)          # 20 > 半格，若不钳制则核心为 0
    tile = np.asarray(compose_art_tile(art, L | R))
    core = tile[art.band if art.band < S // 2 else 4:S - 4, S // 2, :3]
    feat = np.asarray(art.center)[0, 0, :3]
    assert (core == feat).any(), "细条瓦片必须保留地形核心"


def test_edge_noise_makes_boundary_organic_without_breaking_seams():
    """边缘噪声：边界沿边起伏（不再笔直），但共享边仍逐像素相等。"""
    sides = T | B | L | R | TL | TR | BL | BR
    flat_art = _art()
    noisy_art = _art()
    noisy_art.art_meta["edge_noise_px"] = 3

    def boundary_depth(tile, x):
        """第一条不再是「纯地面纹理」的行 = 边界深度（随噪声起伏）。"""
        g = np.asarray(noisy_art.base_texture)[:, x, :3]
        col = tile[:, x, :3]
        same = (col == g).all(axis=-1)
        idx = np.nonzero(~same)[0]
        return int(idx[0]) if idx.size else -1

    top_exposed = sides & ~T
    flat = np.asarray(compose_art_tile(flat_art, top_exposed))
    noisy = np.asarray(compose_art_tile(noisy_art, top_exposed))
    assert not (flat == noisy).all(), "开启边缘噪声后瓦片应有变化"
    depths = [boundary_depth(noisy, x) for x in (0, 8, 16, 24, S - 1)]
    assert len(set(depths)) > 1, f"边界应沿边起伏: {depths}"

    # 相邻两块（掩码不同 -> 噪声相位不同）在共享边上必须完全一致
    left = np.asarray(compose_art_tile(noisy_art, sides & ~T & ~L))
    right = np.asarray(compose_art_tile(noisy_art, sides & ~T & ~R))
    assert (left[:, S - 1] == right[:, 0]).all(), "边缘噪声不得破坏共享边一致性"
