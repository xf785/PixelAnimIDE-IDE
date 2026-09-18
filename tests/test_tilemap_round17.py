"""两个反馈的回归测试：残留深色边框裁除 + 预览侧 2.5D 可用。"""
import numpy as np
from PIL import Image, ImageDraw

from core.tilemap import TileMapModel
from core.tilemap.cliff import cliff_art_from_terrain
from core.tilemap.fidelity import faithful_tile_texture
from core.tilemap.tiles import EDGE_NAMES, BaseTileSet

S = 32


def _cell_with_frame(size: int = 128) -> Image.Image:
    """中心格艺术 + 一圈深色边框（模拟 AI 画了格框/描边没清干净）。"""
    img = Image.new("RGBA", (size, size), (90, 150, 100, 255))
    d = ImageDraw.Draw(img)
    for i in range(0, size, 16):
        d.rectangle((i, 0, i + 7, size), fill=(74, 132, 88, 255))
    d.rectangle((0, 0, size - 1, size - 1), outline=(28, 24, 26, 255), width=3)
    return img


def test_residual_dark_frame_is_trimmed_so_tiles_have_no_dark_border():
    """残留深色边框必须被裁掉：瓦片最外圈不应比内部明显更暗。"""
    framed = _cell_with_frame()
    tex = np.asarray(faithful_tile_texture(framed, S))
    ring = np.concatenate([tex[0, :, :3].reshape(-1, 3), tex[-1, :, :3].reshape(-1, 3),
                           tex[:, 0, :3].reshape(-1, 3), tex[:, -1, :3].reshape(-1, 3)])
    inner = tex[2:-2, 2:-2, :3].reshape(-1, 3)
    assert ring.mean() > inner.mean() * 0.86, f"边缘仍偏暗（有深色框）: {ring.mean():.1f} vs {inner.mean():.1f}"


def test_clean_cell_is_not_trimmed_away():
    """正常艺术（没有边框）不应被误裁：结果仍与整数倍降采样一致。"""
    from core.tilemap.tiles import _mode_downscale

    img = Image.new("RGBA", (128, 128), (90, 150, 100, 255))
    ImageDraw.Draw(img).rectangle((20, 20, 60, 60), fill=(200, 60, 60, 255))
    tex = np.asarray(faithful_tile_texture(img, S))
    ref = _mode_downscale(np.asarray(img), S, S)
    same = (tex == ref).all(axis=2)
    assert same.mean() > 0.87, f"正常艺术不应被改动: {same.mean():.3f}"


def _art(col=(80, 150, 90)) -> BaseTileSet:
    a = np.zeros((S, S, 4), np.uint8)
    a[..., :3] = col
    a[..., 3] = 255
    tex = Image.fromarray(a, "RGBA")
    return BaseTileSet(size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
                       corners={n: tex for n in ("tl", "tr", "bl", "br")},
                       line_color=(24, 22, 24), line_width=1, band=8, radius=8,
                       base_texture=tex, art_meta={"outline": [[24, 22, 24]], "outline_px": 1,
                                                   "bevel": [], "bevel_px": 0, "edge_noise_px": 0})


def test_preview_model_needs_cliff_arts_rebuilt_from_session():
    """预览从 JSON 重建模型后，必须重新挂上崖壁艺术，2.5D 才有画面。"""
    from ui.widgets.tilemap_view import TilemapView  # noqa: F401  (导入即验证依赖可用)

    land, water = _art(), _art((70, 120, 190))
    model = TileMapModel(6, 6, tile_size=S)
    model.set_terrain(1, land)
    model.set_terrain(2, water)
    model.base_terrain = 1
    model.fill_rect(0, 0, 5, 5, 1)
    model.fill_rect(0, 3, 5, 5, 2)
    model.terrain_heights = {1: 0, 2: -1}
    flat = np.asarray(model.render())

    # 模拟"序列化后再恢复"（预览就是这条路）
    restored = TileMapModel.from_dict(model.to_dict())
    for tid, tset in ((1, land), (2, water)):
        restored.set_terrain(tid, tset)
    assert int(restored.terrain_heights[2]) == -1, "地形高度应随地图 JSON 保留"
    no_cliff = np.asarray(restored.render())
    assert (no_cliff == flat).all(), "没挂崖壁艺术时应与平面渲染一致（这正是之前的 bug）"
    restored.enable_height_layer({1: cliff_art_from_terrain(land), 2: cliff_art_from_terrain(water)})
    with_cliff = np.asarray(restored.render())
    assert not (with_cliff == no_cliff).all(), "挂上崖壁艺术后应出现 2.5D 效果"
    assert set(restored.cliff_cells()) == {(x, 3) for x in range(6)}


def test_pack_meta_carries_terrain_heights(tmp_path):
    """瓦片包 meta 带上地形高度，跨包加载也能恢复 2.5D。"""
    from core.tilemap.pack import TilePack, export_tileset_dir, load_tileset

    pack = TilePack(name="草地", category="ground", tile_size=S, terrains={1: _art(), 2: _art()},
                    terrain_names={1: "草地", 2: "水"}, base_terrain=1,
                    meta={"terrain_heights": {1: 0, 2: -1}})
    paths = export_tileset_dir(tmp_path / "g", pack)
    pack2 = load_tileset(paths["dir"])
    assert {int(k): int(v) for k, v in pack2.meta["terrain_heights"].items()} == {1: 0, 2: -1}
