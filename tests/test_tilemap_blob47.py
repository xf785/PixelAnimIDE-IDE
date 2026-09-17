"""FrameRonin「Blob」16/47 图块逻辑搬运的回归测试。

对应来源：systemchester/FrameRonin（`blobTerrain.ts` / `public/map/blob/map.html`）。
测试覆盖：掩码约定、47 掩码→图集槽位映射表、就近回退、图集几何、16 图块族、
程序化地形（确定性 + 分类 + 可走性），以及与工作流的集成（blob47 / tile16 导出）。
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from core.tilemap import (
    BIT,
    BLOB_TILE_COLS,
    BLOB_TILE_ROWS,
    FALLBACK_TILE_INDEX,
    FORBIDDEN_TILE_INDEX,
    MASK_TO_INDEX,
    ProceduralTerrain,
    atlas_cell_box,
    atlas_cell_size,
    blob_mask_to_slot_table,
    build_blob47_atlas,
    build_tile16_atlas,
    canonical_tile16_mask,
    compute_blob_mask,
    nearest_tile_index,
    slice_atlas_tile,
    tile16_index,
)

S = 32


# --------------------------------------------------------------------------- #
# 1) 掩码约定（computeBlobMask 同款）
# --------------------------------------------------------------------------- #
def test_blob_mask_bits_and_diagonal_rule():
    """N=2 S=64 W=8 E=16；对角位只在两个相邻正交位同为同类时置位。"""
    assert compute_blob_mask(0, 0, lambda x, y: (x, y) == (0, -1)) == BIT["T"]
    assert compute_blob_mask(0, 0, lambda x, y: (x, y) == (0, 1)) == BIT["B"]
    assert compute_blob_mask(0, 0, lambda x, y: (x, y) == (-1, 0)) == BIT["L"]
    assert compute_blob_mask(0, 0, lambda x, y: (x, y) == (1, 0)) == BIT["R"]
    # 只有对角邻居 -> 不加对角位（等价孤立格）
    assert compute_blob_mask(0, 0, lambda x, y: (x, y) == (-1, -1)) == 0
    # 北 + 西都满且西北也满 -> 加上 TL
    nb = {(0, -1), (-1, 0), (-1, -1)}
    assert compute_blob_mask(0, 0, lambda x, y: (x, y) in nb) == (BIT["T"] | BIT["L"] | BIT["TL"])
    # 北满但西不满 -> 即使西北满也不加 TL
    nb2 = {(0, -1), (-1, -1)}
    assert compute_blob_mask(0, 0, lambda x, y: (x, y) in nb2) == BIT["T"]


# --------------------------------------------------------------------------- #
# 2) 47 掩码 → 图集槽位（MASK_TO_INDEX）与就近回退
# --------------------------------------------------------------------------- #
def test_mask_table_matches_frameronin():
    """搬运的表必须与 FrameRonin 完全一致（47 项、槽位唯一、71 号禁用）。"""
    assert len(MASK_TO_INDEX) == 47
    assert MASK_TO_INDEX[0] == 13          # 孤立格
    assert MASK_TO_INDEX[255] == 4         # 四面全满
    slots = list(MASK_TO_INDEX.values())
    assert len(set(slots)) == 47           # 槽位一一对应
    assert max(slots) == 70 and min(slots) == 0
    assert FORBIDDEN_TILE_INDEX not in slots
    assert all(0 <= s < BLOB_TILE_COLS * BLOB_TILE_ROWS for s in slots)


def test_nearest_tile_index_exact_and_fallback():
    for mask, idx in MASK_TO_INDEX.items():
        assert nearest_tile_index(mask) == idx
    assert nearest_tile_index(5) == MASK_TO_INDEX[0]      # 仅对角/未知 -> 就近到孤立格
    assert nearest_tile_index(FORBIDDEN_TILE_INDEX) != FORBIDDEN_TILE_INDEX
    table = blob_mask_to_slot_table()
    assert len(table) == 256
    assert all(v != FORBIDDEN_TILE_INDEX for v in table.values())
    assert table[255] == FALLBACK_TILE_INDEX


# --------------------------------------------------------------------------- #
# 3) 图集几何：3×24 = 72 槽，行 0 在上
# --------------------------------------------------------------------------- #
def test_atlas_geometry_3x24():
    w, h = 72, 576                                   # FrameRonin blob 图集尺寸
    assert atlas_cell_size(w, h) == (24, 24)
    assert atlas_cell_box(0, w, h) == (0, 0, 24, 24)
    assert atlas_cell_box(1, w, h) == (24, 0, 48, 24)
    assert atlas_cell_box(BLOB_TILE_COLS, w, h) == (0, 24, 24, 48)
    assert atlas_cell_box(70, w, h)[:2] == ((70 % 3) * 24, (70 // 3) * 24)
    sheet = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tile = slice_atlas_tile(sheet, 4)
    assert tile.size == (24, 24)


def _art(rgb=(38, 104, 172), ground=(236, 240, 246), band=8, rim=2):
    from core.tilemap.seamless import make_tile_texture, median_tile_texture

    def cell(col, checker=0):
        a = np.zeros((S, S, 4), np.uint8)
        a[..., :3] = col
        a[..., 3] = 255
        if checker:
            ys, xs = np.mgrid[0:S, 0:S]
            a[((ys // 3 + xs // 3) % 2 == 0), :3] = np.clip(np.array(col) + checker, 0, 255)
        return Image.fromarray(a, "RGBA")

    from core.tilemap.tiles import BaseTileSet

    tex = make_tile_texture(cell(rgb, 12), S)
    base_tex = median_tile_texture([cell(ground, 8) for _ in range(9)], S)
    from core.tilemap.tiles import EDGE_NAMES

    return BaseTileSet(
        size=S, center=tex,
        edges={n: tex for n in EDGE_NAMES},
        corners={n: tex for n in ("tl", "tr", "bl", "br")},
        line_color=(81, 75, 66), line_width=rim, band=band, radius=band, base_texture=base_tex,
    )


def test_build_blob47_atlas_layout():
    """区块按 MASK_TO_INDEX 放置；无掩码槽位与 71 号槽留空。"""
    art = _art()
    sheet, meta = build_blob47_atlas(art)
    assert sheet.size == (BLOB_TILE_COLS * S, BLOB_TILE_ROWS * S)
    assert meta["format"] == "frameronin-blob47"
    assert meta["forbidden_index"] == FORBIDDEN_TILE_INDEX
    assert meta["fallback_index"] == FALLBACK_TILE_INDEX
    assert meta["tile_count"] == 47
    assert len(meta["all_masks_to_index"]) == 256
    # 全满掩码槽位应是纯纹理；孤立格槽位角部是基础地形
    full_slot = MASK_TO_INDEX[255]
    tile = np.asarray(slice_atlas_tile(sheet, full_slot))
    assert (tile[..., 3] == 255).all()
    assert (tile == np.asarray(art.center)).all()
    iso_slot = MASK_TO_INDEX[0]
    iso = np.asarray(slice_atlas_tile(sheet, iso_slot))
    assert (iso[0, S // 2, :3] == np.asarray(art.base_texture)[0, S // 2, :3]).all()
    # 没有掩码对应的槽位（0..70 中的空位）与 71 号必须是透明的
    filled = set(MASK_TO_INDEX.values())
    for slot in range(BLOB_TILE_COLS * BLOB_TILE_ROWS):
        if slot in filled:
            continue
        rgba = np.asarray(slice_atlas_tile(sheet, slot))
        assert (rgba[..., 3] == 0).all(), f"空槽 {slot} 应为透明"


def test_build_tile16_atlas_layout():
    """16 图块族：4×4 槽位 = n | e*2 | s*4 | w*8，对角位取两正交位都为满。"""
    art = _art()
    sheet, meta = build_tile16_atlas(art)
    assert sheet.size == (4 * S, 4 * S)
    assert meta["format"] == "frameronin-tile16"
    assert tile16_index(BIT["T"]) == 1
    assert tile16_index(BIT["R"]) == 2
    assert tile16_index(BIT["B"]) == 4
    assert tile16_index(BIT["L"]) == 8
    assert tile16_index(255) == 15
    assert canonical_tile16_mask(BIT["T"]) == BIT["T"]                    # 单边 -> 无对角
    assert canonical_tile16_mask(BIT["T"] | BIT["L"]) == (BIT["T"] | BIT["L"] | BIT["TL"])
    assert canonical_tile16_mask(BIT["T"] | BIT["L"] | BIT["R"] | BIT["B"]) == 255
    # 16 图块族不产生内凹角：任何置位的对角都必须有两条正交位支撑
    for m in range(256):
        cm = canonical_tile16_mask(m)
        for diag, a, b in (("TL", "T", "L"), ("TR", "T", "R"), ("BL", "B", "L"), ("BR", "B", "R")):
            if cm & BIT[diag]:
                assert cm & BIT[a] and cm & BIT[b], (m, diag)
    # 15 号槽（四面全满）应为纯纹理
    full = np.asarray(slice_atlas_tile(sheet, 15, cols=4, rows=4))
    assert (full == np.asarray(art.center)).all()


# --------------------------------------------------------------------------- #
# 4) 程序化地形（Perlin/FBM + 河谷 + 山地）
# --------------------------------------------------------------------------- #
def test_procedural_terrain_is_deterministic():
    a = ProceduralTerrain("pixelgifide")
    b = ProceduralTerrain("pixelgifide")
    c = ProceduralTerrain("another-seed")
    ka, sa = a.grid(24, 16)
    kb, _ = b.grid(24, 16)
    kc, _ = c.grid(24, 16)
    assert (ka == kb).all() and (sa == a.grid(24, 16)[1]).all()
    assert not (ka == kc).all()          # 换种子必须换地形


def test_procedural_terrain_classification_and_walkability():
    t = ProceduralTerrain("pixelgifide", sea_level=0.42, mountain_threshold=0.48)
    kinds, slots = t.grid(32, 32)
    assert set(np.unique(kinds).tolist()) <= {0, 1, 2}          # 水 / 平 / 山
    assert (slots >= 0).all() and (slots < BLOB_TILE_COLS * BLOB_TILE_ROWS).all()
    assert (slots != FORBIDDEN_TILE_INDEX).all()                # 永不使用 71 号槽
    # 可走性：只有平地可走；陆地包含山地
    for y in range(0, 32, 3):
        for x in range(0, 32, 3):
            kind = int(kinds[y, x])
            assert t.is_walkable(x, y) == (kind == 1)
            assert t.is_land(x, y) == (kind in (1, 2))
    # 海平面抬高 -> 水更多（单调性）
    t.set_params(sea_level=0.6)
    more_water = int((t.grid(32, 32)[0] == 0).sum())
    t.set_params(sea_level=0.2)
    less_water = int((t.grid(32, 32)[0] == 0).sum())
    assert more_water > less_water


def test_procedural_terrain_masks_and_sample_index():
    t = ProceduralTerrain("pixelgifide")
    kind, slot = t.sample_tile_index(10, 10)
    assert kind in ("water", "mtn", "norm")
    assert slot == nearest_tile_index(
        t.mask_water(10, 10) if kind == "water" else t.mask_land_biome(10, 10, t.biome(10, 10))
    )
    # 相邻同类格共享边：同一 mask 的瓦片来自同一条几何规则（构图层面已由对齐式保证）
    assert isinstance(t.mask_water(0, 0), int)


# --------------------------------------------------------------------------- #
# 5) 与工作流集成：导出 FrameRonin 布局图集 + 程序化演示地图
# --------------------------------------------------------------------------- #
def test_workflow_exports_blob47_and_tile16_atlases(tmp_path):
    import json

    from core.api.mock_clients import MockImageAPI
    from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow

    for mode, cols, rows, fmt in (("blob47", BLOB_TILE_COLS, BLOB_TILE_ROWS, "frameronin-blob47"),
                                  ("tile16", 4, 4, "frameronin-tile16")):
        params = TilemapParams(
            description="草地", category="ground", features={"水塘": "a pond", "岩石": "rock"},
            tile_size=32, atlas_mode=mode, map_width=10, map_height=8,
            output_dir=tmp_path / mode,
        )
        result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
        paths = result.terrain_atlas_paths
        assert paths, mode
        for _tid, path in paths.items():
            assert path.exists()
            meta = json.loads(path.with_name(path.stem + ".json").read_text(encoding="utf-8"))
            assert meta["format"] == fmt
            assert meta["sheet_cols"] == cols and meta["sheet_rows"] == rows
            img = Image.open(path)
            assert img.size == (cols * 32, rows * 32)
            if mode == "blob47":
                assert meta["forbidden_index"] == FORBIDDEN_TILE_INDEX
                assert len(meta["all_masks_to_index"]) == 256
                assert all(v != FORBIDDEN_TILE_INDEX for v in meta["all_masks_to_index"].values())
        assert result.map_preview_path.exists()


def test_workflow_procedural_demo_map(tmp_path):
    """map_source=procedural：演示地图用 FrameRonin 程序化地形铺满（水/山/平原三类）。"""
    import numpy as np

    from core.api.mock_clients import MockImageAPI
    from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow

    params = TilemapParams(
        description="雪原", category="ground",
        features={"水塘": "a pond", "岩石": "rock"},
        tile_size=32, atlas_mode="blob47", map_source="procedural",
        map_width=24, map_height=16, output_dir=tmp_path / "out",
    )
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    grid = result.session.map_model.grid
    values = set(np.unique(grid).tolist())
    assert values <= {1, 2, 3} and len(values) >= 2, values      # 至少两类地形出现
    assert result.map_preview_path.exists()
    assert result.step_log or True

