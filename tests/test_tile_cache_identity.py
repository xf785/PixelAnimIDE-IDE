"""地图瓦片缓存的同一性回归：缓存键用了 id(base)，必须校验对象身份。

背景：`compose_art_tile_cached` 以 `id(base)` 为键缓存合成结果。旧地形对象被回收后，
新对象可能复用同一地址，于是取到上一套地形的旧瓦片（地图上偶发「串图」，
测试里表现为 test_tilemap_round15 随机失败）。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from core.tilemap.autotile import _TILE_CACHE, compose_art_tile, compose_art_tile_cached
from core.tilemap.tiles import EDGE_NAMES, BaseTileSet

S = 32


def _terrain(color) -> BaseTileSet:
    tex = Image.new("RGBA", (S, S), color)
    return BaseTileSet(size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
                       corners={n: tex for n in ("tl", "tr", "bl", "br")},
                       line_color=(20, 18, 20), line_width=1, band=8, radius=8,
                       base_texture=tex,
                       art_meta={"outline": [[20, 18, 20]], "outline_px": 1, "bevel": [],
                                 "bevel_px": 0, "edge_noise_px": 2})


def test_cache_reuses_entry_for_same_object():
    art = _terrain((80, 150, 90, 255))
    first = compose_art_tile_cached(art, 5)
    again = compose_art_tile_cached(art, 5)
    assert again is first, "同一对象 + 同一掩码应命中缓存（返回同一张图）"


def test_cache_rejects_entry_from_recycled_address():
    """地址复用（id 相同但对象不同）时必须重算，而不是返回上一套地形的格子。"""
    old = _terrain((200, 40, 40, 255))
    old_tile = compose_art_tile_cached(old, 0)

    fresh = _terrain((40, 80, 220, 255))
    key = (id(fresh), 0, fresh.band, fresh.line_width,
           int((fresh.art_meta or {}).get("edge_noise_px", 0) or 0))
    # 伪造「地址被复用」的坏缓存：键属于 fresh，存的却是 old 的格子
    _TILE_CACHE[key] = (old, old_tile)

    got = compose_art_tile_cached(fresh, 0)
    expected = compose_art_tile(fresh, 0)
    assert np.array_equal(np.asarray(got), np.asarray(expected)), "不应命中旧地形的缓存"
    assert not np.array_equal(np.asarray(got), np.asarray(old_tile))


def test_cache_key_distinguishes_different_art():
    a = _terrain((10, 200, 10, 255))
    b = _terrain((200, 10, 200, 255))
    ta = compose_art_tile_cached(a, 3)
    tb = compose_art_tile_cached(b, 3)
    assert not np.array_equal(np.asarray(ta), np.asarray(tb))
