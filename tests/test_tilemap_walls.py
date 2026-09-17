"""建筑类（墙体 16-tile 族）生成链路测试。

覆盖用户提出的要点：
- 瓦片类型要丰富：墙内部 / 墙边 / 墙角（内角+外角）/ 门 / 柱 / 各类连接件；
- **非内部填充瓦片外部必须完全透明**（可叠加到地块层上），且不能有白边；
- 接缝：相邻墙块共享边逐像素一致（含透明段），长墙不会每格出现一道竖线。
"""
import numpy as np
import pytest
from PIL import Image

from core.tilemap.autotile import canonical_mask
from core.tilemap.seamless import make_tile_texture
from core.tilemap.walls import (
    ATLAS_COLS,
    ATLAS_ROWS,
    EXTRA_SLOTS,
    SOLID_SLOT,
    W16,
    W16_SLOTS,
    WallArt,
    build_piece_set,
    build_wall_atlas,
    compose_wall_piece,
    footprint,
    mask16,
    rotate_piece,
    slot_for_mask,
    wall16_index,
    wall_art_from_sheet,
)
from core.tilemap.tiles import BuildingSheet, BaseTileSet

S = 32


def _wall_texture(size: int = S) -> Image.Image:
    a = np.zeros((96, 96, 4), np.uint8)
    a[..., :3] = (152, 140, 126)
    a[..., 3] = 255
    for y in range(0, 96, 12):
        a[y:y + 2, :, :3] = (112, 100, 90)
    for x in range(0, 96, 24):
        a[:, x:x + 1, :3] = (118, 108, 98)
    return make_tile_texture(Image.fromarray(a, "RGBA"), size)


def _art(margin: int = 8, noise: int = 0) -> WallArt:
    return WallArt(size=S, texture=_wall_texture(), margin=margin, outline=(34, 30, 32),
                   corner_radius=margin, edge_noise=noise)


def _render_grid(art: WallArt, grid):
    """按墙格布局渲染（与工作流同一套掩码/槽位逻辑）。"""
    H, W = len(grid), len(grid[0])
    tiles = {}
    for y in range(H):
        for x in range(W):
            if not grid[y][x]:
                continue
            m = 0
            for (dy, dx, bit) in ((-1, 0, 2), (1, 0, 64), (0, -1, 8), (0, 1, 16),
                                  (-1, -1, 1), (-1, 1, 4), (1, -1, 32), (1, 1, 128)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny][nx]:
                    m |= bit
            m = canonical_mask(m)
            tiles[(x, y)] = np.asarray(
                compose_wall_piece(art, slot_for_mask(m), mask8=m)
            )
    return tiles


# --------------------------------------------------------------------------- #
# 1) 16-tile 族完整性
# --------------------------------------------------------------------------- #
def test_wall16_family_is_complete_and_named():
    assert len(W16_SLOTS) == 16
    assert W16_SLOTS[0] == "none" and W16_SLOTS[15] == "nesw"
    pieces = build_piece_set(_art())
    assert set(W16_SLOTS) <= set(pieces)
    assert {"solid", "door_ew", "door_ns", "pillar"} <= set(pieces)
    # 四种连接类型都能由掩码得到：直墙/转角/端头/T 形/十字
    expect = {
        8 | 16: "ew", 2 | 64: "ns",
        2 | 16: "ne", 1 | 2 | 8: "wn",
        2 | 16 | 64: "nes", 8 | 16 | 64: "esw",
        2: "n", 8: "w", 255: "nesw",
    }
    for mask, name in expect.items():
        assert W16_SLOTS[wall16_index(mask)] == name, (mask, name)
    assert mask16(255) == 0b1111
    assert slot_for_mask(255) == 15          # 默认沿用 16 族（保证接缝一致）


# --------------------------------------------------------------------------- #
# 2) 透明外部（可叠加地块）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("slot", [0, 1, 3, 5, 11, 13])
def test_non_interior_pieces_are_transparent_outside(slot):
    art = _art()
    arr = np.asarray(compose_wall_piece(art, slot))
    alpha = arr[..., 3]
    assert (alpha == 0).mean() > 0.25, f"槽位 {slot} 应有明显透明区"
    # 半透明像素不允许出现（像素画硬边；也避免叠加时出现灰边）
    assert set(np.unique(alpha).tolist()) <= {0, 255}
    # 透明区不得有近白的不透明像素（白边/白底残留）
    near_white = (alpha > 0) & (arr[..., :3].min(axis=-1) > 235)
    assert near_white.mean() < 0.01


def test_interior_pieces_are_opaque():
    """十字件 = 两条带子的并集（四角是空的，这是 16 族的标准外形）；
    实心件（显式要求）才是整格不透明。"""
    art = _art()
    cross = np.asarray(compose_wall_piece(art, W16["n"] | W16["e"] | W16["s"] | W16["w"]))
    assert (cross[..., 3] == 255).mean() > 0.55
    assert (cross[16, 16, 3] == 255) and (cross[2, 2, 3] == 0)
    solid = np.asarray(compose_wall_piece(art, SOLID_SLOT))
    assert (solid[..., 3] == 255).all()


def test_wall_band_geometry_is_centered_and_no_isolated_pixels():
    art = _art(margin=8)
    solid, empty = footprint(art, W16["e"] | W16["w"])
    rows = np.nonzero(solid.any(axis=1))[0]
    full_rows = [r for r in range(S) if solid[r].all()]
    assert rows[0] == 8 and rows[-1] == S - 9, (rows[0], rows[-1])   # 上下各让位 8px
    assert full_rows == list(range(8, S - 8)), "带子应恰好是中间 [margin, s-margin) 条"
    assert not solid[:8].any() and not solid[S - 8:].any()
    # 转角件：不能有孤立像素（倒角残留），且形状是 L 形
    solid_c, _ = footprint(art, W16["n"] | W16["e"])
    from core.tilemap.walls import _keep_largest
    assert (_keep_largest(solid_c) == solid_c).all(), "转角件不应有孤立像素"
    assert solid_c[2, 16] and solid_c[16, 28]     # 上臂 + 右臂有墙
    assert not solid_c[28, 4] and not solid_c[4, 4]  # 左下/左上为空（L 形）


# --------------------------------------------------------------------------- #
# 3) 接缝一致性（含透明段）
# --------------------------------------------------------------------------- #
def test_wall_map_has_no_seam_mismatch():
    art = _art(noise=1)
    g = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    tiles = _render_grid(art, g)
    bad = []
    for (x, y), t in tiles.items():
        if (x + 1, y) in tiles and not (t[:, S - 1] == tiles[(x + 1, y)][:, 0]).all():
            bad.append(("x", x, y))
        if (x, y + 1) in tiles and not (t[S - 1, :] == tiles[(x, y + 1)][0, :]).all():
            bad.append(("y", x, y))
    assert not bad, f"墙体共享边不一致: {bad[:5]}"


def test_long_wall_has_no_per_tile_vertical_outline():
    """长墙不能每格出现一道描边竖线：相连侧不得被当成暴露面。"""
    art = _art()
    straight = np.asarray(compose_wall_piece(art, W16["e"] | W16["w"]))
    outline = np.array(art.outline)
    for col in (0, S - 1):
        px = straight[:, col]
        opaque = np.nonzero(px[:, 3] > 0)[0]
        assert opaque.size
        rgb = px[opaque][:, :3].astype(int)
        is_outline = np.abs(rgb - outline).max(axis=1) < 8
        # 只有带子的上下两条暴露面允许有描边，中段不允许（否则长墙每格一道竖线）
        middle = (opaque >= art.margin + 3) & (opaque < S - art.margin - 3)
        assert not is_outline[middle].any(), f"第 {col} 列中段不应有描边"


# --------------------------------------------------------------------------- #
# 4) 从 AI 建筑图推导 + 图集导出
# --------------------------------------------------------------------------- #
def _sheet_with_wall() -> BuildingSheet:
    """模拟 AI 2×2×3 建筑图：墙体块中心格填满、其余块白底上有小物件。"""
    def solid_tile(color, size=S):
        return Image.new("RGBA", (size, size), (*color, 255))

    wall = BaseTileSet(size=S, center=solid_tile((150, 140, 128)), edges={}, corners={})
    top = BaseTileSet(size=S, center=solid_tile((186, 176, 160)), edges={}, corners={})
    opening = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    opening.paste(Image.new("RGBA", (10, 20), (90, 70, 50, 255)), (11, 8))
    pillar = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    pillar.paste(Image.new("RGBA", (12, 22), (140, 130, 118, 255)), (10, 5))
    return BuildingSheet(
        wall=wall, top=top,
        opening=BaseTileSet(size=S, center=opening, edges={}, corners={}),
        pillar=BaseTileSet(size=S, center=pillar, edges={}, corners={}),
    )


def test_wall_art_from_sheet_and_atlas():
    art = wall_art_from_sheet(_sheet_with_wall(), tile_size=S, thickness_frac=0.56)
    assert art.art_meta["family"] == "wall-16"
    assert 2 <= art.margin <= S // 2 - 2
    assert art.texture.size == (S, S)
    # 纹理网格对齐（左右/上下边缘逐像素相等）→ 长墙无缝
    tex = np.asarray(art.texture)
    assert (tex[:, 0] == tex[:, -1]).all() and (tex[0, :] == tex[-1, :]).all()

    pieces = build_piece_set(art)
    assert len(pieces) == 20
    sheet, meta = build_wall_atlas(art)
    assert sheet.size == (ATLAS_COLS * S, ATLAS_ROWS * S)
    assert meta["format"] == "pixel-anim-wall16"
    assert meta["slots"][:16] == list(W16_SLOTS)
    assert len(meta["mask8_to_slot"]) == 256
    assert all(0 <= v < ATLAS_COLS * ATLAS_ROWS for v in meta["mask8_to_slot"].values())
    # 槽位与拼件一一对应
    for name, slot in meta["slot_of_piece"].items():
        assert meta["slots"][slot] == name
    for name in EXTRA_SLOTS:
        assert name in meta["slot_of_piece"]


def test_rotation_keeps_transparency_aligned():
    art = _art()
    p = compose_wall_piece(art, W16["n"])
    r = rotate_piece(p, 2)
    assert r.size == p.size
    assert (np.asarray(r)[..., 3] == 0).mean() == (np.asarray(p)[..., 3] == 0).mean()
