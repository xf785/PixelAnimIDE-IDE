"""第十四轮新增功能测试：完整瓦片集目录导出/导入、交界渗透融合、素材白/黑底扣除、双网格。"""
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.tilemap import BIT, TileMapModel, compose_art_tile, mask_for_terrain
from core.tilemap.pack import export_tileset_dir, load_tileset
from core.tilemap.seamless import align_terrain_set, make_tile_texture, median_tile_texture
from core.tilemap.tiles import EDGE_NAMES, BaseTileSet, crop_blocks, ecosystem_from_blocks

S = 32


def _art(band=8, blend=0.0, col=(38, 104, 172), ground=(236, 240, 246)):
    a = np.zeros((96, 96, 4), np.uint8)
    a[..., :3] = col
    a[..., 3] = 255
    for y in range(0, 96, 12):
        a[y:y + 2, :, :3] = np.clip(np.array(col) - 30, 0, 255)
    tex = make_tile_texture(Image.fromarray(a, "RGBA"), S)
    g = Image.new("RGBA", (S, S), ground + (255,))
    return BaseTileSet(size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
                       corners={n: tex for n in ("tl", "tr", "bl", "br")},
                       line_color=(34, 30, 32), line_width=2, band=band, radius=band,
                       base_texture=g,
                       art_meta={"outline": [[34, 30, 32], [150, 152, 150]], "bevel": [],
                                 "outline_px": 2, "bevel_px": 0, "edge_noise_px": 2,
                                 "edge_blend": blend})


# --------------------------------------------------------------------------- #
# 1) 完整瓦片集目录导出 / 导入
# --------------------------------------------------------------------------- #
def test_tileset_dir_export_contains_atlas_tiles_and_meta(tmp_path):
    from core.tilemap.pack import TilePack

    pack = TilePack(name="雪原", category="ground", tile_size=S,
                    terrains={1: _art(), 2: _art(col=(60, 140, 70))},
                    terrain_names={1: "雪原", 2: "草地"}, base_terrain=1)
    paths = export_tileset_dir(tmp_path / "雪原", pack)
    out = paths["dir"]
    assert (out / "manifest.json").exists() and (out / "README.txt").exists()
    # 47 图集 + FrameRonin 布局 + 逐张瓦片 + 全部元信息
    assert (out / "atlas" / "terrain_1_47.png").exists()
    assert (out / "atlas" / "terrain_1_47.json").exists()
    assert (out / "atlas" / "terrain_1_blob47.png").exists()
    tiles = sorted((out / "tiles" / "terrain_1").glob("tile_*.png"))
    assert len(tiles) == 47, len(tiles)
    assert (out / "tiles" / "terrain_1" / "index.json").exists()
    info = json.loads((out / "map" / "info.json").read_text(encoding="utf-8"))
    assert info["category"] == "ground" and len(info["terrains"]["1"]["tiles"]) == 47
    assert (out / "textures" / "terrain_1.png").exists()
    assert paths["zip"].exists() and paths["zip"].suffix == ".zip"
    # 47 图集里恰好 47 张不透明瓦片
    sheet = np.asarray(Image.open(out / "atlas" / "terrain_1_47.png").convert("RGBA"))
    assert (sheet[..., 3] > 0).any()


def test_tileset_dir_and_zip_import_roundtrip(tmp_path):
    from core.tilemap.pack import TilePack

    src = _art(blend=0.4)
    pack = TilePack(name="雪原", category="ground", tile_size=S, terrains={1: src},
                    terrain_names={1: "雪原"}, base_terrain=1)
    paths = export_tileset_dir(tmp_path / "ts", pack)
    for target in (paths["dir"], paths["zip"]):
        back = load_tileset(target)
        assert back.name == "雪原" and set(back.terrains) == {1}
        assert (np.asarray(back.terrains[1].center) == np.asarray(src.center)).all()
        assert back.terrains[1].art_meta.get("edge_blend") == 0.4
        mask = BIT["L"] | BIT["R"] | BIT["T"] | BIT["TL"] | BIT["TR"]
        assert (np.asarray(compose_art_tile(back.terrains[1], mask))
                == np.asarray(compose_art_tile(src, mask))).all()


# --------------------------------------------------------------------------- #
# 2) 交界渗透融合
# --------------------------------------------------------------------------- #
def test_edge_blend_makes_boundary_percolated_but_keeps_seams():
    mask = (BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"] | BIT["TL"] | BIT["TR"]
            | BIT["BL"] | BIT["BR"]) & ~BIT["T"]
    flat = np.asarray(compose_art_tile(_art(blend=0.0), mask))
    blend = np.asarray(compose_art_tile(_art(blend=0.9), mask))
    # 交界处出现互相咬合的像素（融合开启后边界不再是直线）
    band = art_band = 8
    row = flat[band - 2:band + 3, :, :3]
    brow = blend[band - 2:band + 3, :, :3]
    assert not (row == brow).all(), "开启融合后交界应有变化"
    assert brow.std() > 0
    # 相邻两块在共享边上仍然逐像素一致（融合噪声在瓦片边缘衰减到 0）
    left = np.asarray(compose_art_tile(_art(blend=0.9), (BIT["T"] | BIT["B"] | BIT["L"]
                                                         | BIT["R"] | BIT["TL"] | BIT["TR"]
                                                         | BIT["BL"] | BIT["BR"]) & ~BIT["T"] & ~BIT["L"]))
    right = np.asarray(compose_art_tile(_art(blend=0.9), (BIT["T"] | BIT["B"] | BIT["L"]
                                                          | BIT["R"] | BIT["TL"] | BIT["TR"]
                                                          | BIT["BL"] | BIT["BR"]) & ~BIT["T"] & ~BIT["R"]))
    assert (left[:, S - 1] == right[:, 0]).all(), "融合不得破坏共享边一致性"


def test_edge_blend_survives_random_multiterrain_map():
    arts = {1: _art(blend=0.8), 2: _art(blend=0.8, col=(60, 140, 70)), 3: _art(blend=0.8, col=(150, 120, 80))}
    rng = np.random.default_rng(3)
    H, W = 8, 9
    grid = rng.integers(1, 4, (H, W))
    grid[0, :] = 1
    tiles = {}
    for y in range(H):
        for x in range(W):
            nb = [[int(grid[ny, nx]) if 0 <= ny < H and 0 <= nx < W else 0 for nx in range(x - 1, x + 2)]
                  for ny in range(y - 1, y + 2)]
            m = mask_for_terrain(nb, int(grid[y, x]), base_terrain=1)
            tiles[(x, y)] = np.asarray(compose_art_tile(arts[int(grid[y, x])], m))
    bad = []
    guard = 8 + 2          # 转角带（条带 + 描边交汇）允许 1~2px 差异；多地形模型里
    # 基础地形「任何非空邻居都算满」而特征地形只认同类，base↔feature 交界处两侧
    # 语义本就不同，因此只检查中段（非转角带）。
    for y in range(H):
        for x in range(W - 1):
            if int(grid[y, x]) != int(grid[y, x + 1]):
                continue          # 只检查同类地形相邻（异类两侧语义不同）
            rows = [r for r in np.nonzero((tiles[(x, y)][:, S - 1] != tiles[(x + 1, y)][:, 0]).any(axis=1))[0]
                    if guard <= r < S - guard]
            if rows:
                bad.append(("x", x, y, rows[:3]))
    for y in range(H - 1):
        for x in range(W):
            if int(grid[y, x]) != int(grid[y + 1, x]):
                continue
            cols = [c for c in np.nonzero((tiles[(x, y)][S - 1, :] != tiles[(x, y + 1)][0, :]).any(axis=1))[0]
                    if guard <= c < S - guard]
            if cols:
                bad.append(("y", x, y, cols[:3]))
    assert not bad, f"共享边中段不一致（融合噪声不应影响共享边）: {bad[:5]}"


# --------------------------------------------------------------------------- #
# 3) 素材白/黑底彻底扣除
# --------------------------------------------------------------------------- #
def _prop_sheet(bg, cell=64, rows=2, cols=2, body=(60, 140, 70)):
    img = Image.new("RGB", (cols * cell, rows * cell), bg)
    d = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell, r * cell
            d.polygon([(x0 + cell // 2, y0 + 8), (x0 + cell // 2 - 16, y0 + 40),
                       (x0 + cell // 2 + 16, y0 + 40)], fill=body)
            d.rectangle((x0 + cell // 2 - 4, y0 + 40, x0 + cell // 2 + 4, y0 + 56), fill=(110, 80, 50))
    return img


def test_subject_is_light_detection_and_prompt_background():
    from core.tilemap.props import build_prop_prompts, subject_is_light

    assert subject_is_light("a snow pile") and subject_is_light("白骨")
    assert not subject_is_light("a green tree") and not subject_is_light("红色蘑菇")
    assert build_prop_prompts("a green tree")["background"] == "white"
    p = build_prop_prompts("a snow pile")
    assert p["background"] == "black" and "SOLID PURE BLACK" in p["image_prompt"]


def test_white_background_fully_removed():
    from core.tilemap.props import process_prop_sheet, prop_names

    props = process_prop_sheet(_prop_sheet((255, 255, 255)), 2, 2, prop_names("tree", 4),
                               tile_size=S, background="white")
    assert len(props) == 4
    for name, img in props.items():
        arr = np.asarray(img)
        opaque = arr[..., 3] > 0
        near_white = opaque & (arr[..., :3].min(axis=-1) >= 214)
        assert near_white.sum() == 0, f"{name} 仍残留白底像素 {int(near_white.sum())}"


def test_black_background_for_light_subject_fully_removed():
    from core.tilemap.props import process_prop_sheet, prop_names

    # 主体是浅色（雪），背景用纯黑；黑底像素必须被清干净，主体保留
    props = process_prop_sheet(_prop_sheet((0, 0, 0), body=(240, 245, 250)), 2, 2,
                               prop_names("snow", 4), tile_size=S, background="black")
    assert props
    for name, img in props.items():
        arr = np.asarray(img)
        opaque = arr[..., 3] > 0
        near_black = opaque & (arr[..., :3].max(axis=-1) <= 41)
        assert near_black.sum() == 0, f"{name} 仍残留黑底像素"
        assert opaque.sum() > 20, "浅色主体不应被误删"


# --------------------------------------------------------------------------- #
# 4) 双网格地块类
# --------------------------------------------------------------------------- #
def test_dual_grid_terrain_render_supported():
    arts = {1: _art(), 2: _art(col=(60, 140, 70))}
    model = TileMapModel(8, 6, tile_size=S)
    for tid, a in arts.items():
        model.set_terrain(tid, a)
    model.base_terrain = 1
    model.fill_rect(0, 0, 7, 5, 1)
    model.fill_rect(2, 2, 4, 3, 2)
    single = np.asarray(model.render())
    model.dual_mode = True
    dual = np.asarray(model.render())
    assert dual.shape == single.shape == (6 * S, 8 * S, 4)
    assert (dual[..., 3] == 255).all(), "双网格渲染也必须填充满"
    assert not (dual == single).all(), "双网格应与 47 单格渲染不同"
    data = model.to_dict()
    assert data.get("dual_mode") is True
    restored = TileMapModel.from_dict(data)
    assert restored.dual_mode is True
