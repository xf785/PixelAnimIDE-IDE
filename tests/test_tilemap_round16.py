"""画质保真 + 2.5D 自动高度（水岸/岩石）回归测试。"""
import numpy as np
from PIL import Image, ImageDraw

from core.tilemap import TileMapModel
from core.tilemap.fidelity import weld_edges
from core.tilemap.seamless import make_tile_texture, median_tile_texture
from core.tilemap.tiles import EDGE_NAMES, BaseTileSet

S = 32


def _rich_cell(size: int = 128) -> Image.Image:
    """128px 的「AI 像素画」：4 倍像素格 + 可辨识花纹（对角砖 + 斑点 + 亮边）。"""
    img = Image.new("RGBA", (size, size), (72, 132, 84, 255))
    d = ImageDraw.Draw(img)
    step = 4                                     # 4 倍像素格（32 * 4 = 128）
    for by in range(0, size, step * 4):          # 砖块纹理
        for bx in range(0, size, step * 4):
            if (bx // (step * 4) + by // (step * 4)) % 2:
                d.rectangle((bx, by, bx + step * 4 - 1, by + step * 4 - 1), fill=(60, 118, 74, 255))
    rng = np.random.default_rng(5)               # 细节斑点
    arr = np.asarray(img).copy()
    for _ in range(60):
        x, y = int(rng.integers(0, size - step)), int(rng.integers(0, size - step))
        arr[y:y + step, x:x + step, :3] = (188, 206, 120)
    img = Image.fromarray(arr, "RGBA")
    d = ImageDraw.Draw(img)                      # 亮色描边细节
    d.rectangle((10, 10, 40, 40), outline=(240, 236, 200, 255), width=step)
    return img


def _expected_reference(cell: Image.Image) -> np.ndarray:
    """期望结果：对 AI 原图做整数倍块众数降采样（不做任何缝合/平移）。"""
    from core.tilemap.tiles import _mode_downscale

    arr = np.asarray(cell.convert("RGBA"))
    small = _mode_downscale(arr, S, S)
    assert small is not None, "128 应能 4 倍整数降采样到 32"
    return small


# --------------------------------------------------------------------------- #
# 1) 保真：瓦片纹理必须与 AI 原图（整数倍降采样）几乎逐像素一致
# --------------------------------------------------------------------------- #
def test_tile_texture_preserves_ai_art():
    cell = _rich_cell()
    tex = np.asarray(make_tile_texture(cell, S))
    ref = _expected_reference(cell)
    same = (tex == ref).all(axis=2)
    # 只有最外 1 圈行列被"焊合"改动（32 格上最多 2*(32+32)-4 = 124 px）
    changed = int((~same).sum())
    assert changed <= 2 * (S + S) - 4, f"除焊缝外不应改动像素，实际改了 {changed}"
    assert same.mean() > 0.87, f"保真度过低: {same.mean():.3f}"
    # 颜色数基本不丢（不再量化）
    assert len({tuple(px) for px in tex.reshape(-1, 4).tolist()}) >= 3
    # 边缘焊合仍然成立（相邻瓦片共享边逐像素相等的基石）
    assert (tex[:, 0] == tex[:, -1]).all() and (tex[0, :] == tex[-1, :]).all()


def test_new_extraction_does_not_translate_or_ghost_the_art():
    """含唯一标记的图案：新实现必须把标记留在**原来的象限**（旧的偏移缝合会把它挪到对角）。"""
    cell = Image.new("RGBA", (128, 128), (100, 150, 100, 255))
    ImageDraw.Draw(cell).rectangle((0, 0, 63, 63), fill=(24, 24, 28, 255))     # 左上角标记
    new = np.asarray(make_tile_texture(cell, S))
    dark = new[..., :3].max(axis=2) < 90

    def quadrant(arr) -> str:
        d = arr[..., :3].max(axis=2) < 90
        scores = {"TL": d[:16, :16].mean(), "TR": d[:16, 16:].mean(),
                  "BL": d[16:, :16].mean(), "BR": d[16:, 16:].mean()}
        return max(scores, key=scores.get)

    assert quadrant(new) == "TL", f"标记应留在原象限（左上），实际最暗象限={quadrant(new)}"
    assert dark.mean() > 0.15, "标记应被保留（不应被淡化掉）"


def test_non_integer_scale_still_faithful_and_seamless():
    """非整数倍（AI 没按整数倍画）也不能重影：用面积平均且保持边缘相等。"""
    cell = _rich_cell(100)
    tex = np.asarray(make_tile_texture(cell, S))
    assert tex.shape == (S, S, 4)
    assert (tex[:, 0] == tex[:, -1]).all() and (tex[0, :] == tex[-1, :]).all()
    # 不发明新颜色（块众数/面积平均都只取源图的颜色），且与源图整体色调一致
    src = np.asarray(cell.convert("RGBA"))
    src_colours = {tuple(c) for c in src.reshape(-1, 4).tolist()}
    inner = tex[1:-1, 1:-1]
    tex_colours = {tuple(c) for c in inner.reshape(-1, 4).tolist()}
    assert tex_colours <= src_colours, "除焊缝外，降采样不应引入源图没有的颜色"
    # 整体色调保持（不因重采样而偏色/发暗）
    assert abs(float(tex[..., :3].mean()) - float(src[..., :3].mean())) < 12


def test_median_texture_drops_text_but_keeps_art():
    cells = [_rich_cell() for _ in range(9)]
    for c in cells[:1]:                                     # 只在第 1 格写"文字"
        d = ImageDraw.Draw(c)
        d.rectangle((8, 8, 120, 24), fill=(255, 255, 255, 255))
    tex = np.asarray(median_tile_texture(cells, S))
    assert (tex[:, 0] == tex[:, -1]).all()
    assert not (tex[..., :3] > 245).all(axis=2).any(), "文字应被中值投票抹掉"


def test_weld_edges_only_touches_border():
    arr = np.zeros((8, 8, 4), np.uint8)
    arr[..., 3] = 255
    arr[:, :, 0] = np.arange(8)[None, :] * 10
    out = np.asarray(weld_edges(Image.fromarray(arr, "RGBA")))
    assert (out[:, 0] == out[:, -1]).all() and (out[0, :] == out[-1, :]).all()
    assert (out[1:-1, 1:-1] == arr[1:-1, 1:-1]).all()


# --------------------------------------------------------------------------- #
# 2) 2.5D：水岸 / 岩石自动出崖壁
# --------------------------------------------------------------------------- #
def _art(col=(80, 150, 90)) -> BaseTileSet:
    a = np.zeros((S, S, 4), np.uint8)
    a[..., :3] = col
    a[..., 3] = 255
    tex = Image.fromarray(a, "RGBA")
    ground = Image.new("RGBA", (S, S), (60, 120, 70, 255))
    return BaseTileSet(size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
                       corners={n: tex for n in ("tl", "tr", "bl", "br")},
                       line_color=(24, 22, 24), line_width=1, band=8, radius=8,
                       base_texture=ground, art_meta={"outline": [[24, 22, 24]], "outline_px": 1,
                                                      "bevel": [], "bevel_px": 0, "edge_noise_px": 0})


def test_feature_height_hint_mapping():
    from core.workflow.tilemap_workflow import _terrain_height_hint

    assert _terrain_height_hint("热带水塘") == -1 and _terrain_height_hint("lake") == -1
    assert _terrain_height_hint("岩石") == 1 and _terrain_height_hint("rocky outcrop") == 1
    assert _terrain_height_hint("泥潭") == 0 or _terrain_height_hint("泥潭") == -1   # 泥潭可按水处理
    assert _terrain_height_hint("草地") == 0


def test_water_shore_draws_cliff_on_water_side_with_land_material():
    """水（-1）在北侧是陆地时：崖壁画在**水格**上，且材质取较高一侧（陆地）。"""
    from core.tilemap.cliff import cliff_art_from_terrain

    land_art, water_art = _art((90, 150, 90)), _art((70, 120, 190))
    model = TileMapModel(6, 6, tile_size=S)
    model.set_terrain(1, land_art)
    model.set_terrain(2, water_art)
    model.base_terrain = 1
    model.fill_rect(0, 0, 5, 5, 1)
    model.fill_rect(0, 3, 5, 5, 2)                       # 下半是水
    flat = np.asarray(model.render())
    model.enable_height_layer({1: cliff_art_from_terrain(land_art), 2: cliff_art_from_terrain(water_art)})
    model.terrain_heights = {1: 0, 2: -1}                # 水比陆地低一级
    raised = np.asarray(model.render())
    cells = model.cliff_cells()
    assert set(cells) == {(x, 3) for x in range(6)}, f"崖壁应出现在水格(第 3 行): {sorted(cells)}"
    band = raised[3 * S:3 * S + model.cliff_arts[1].face_h, :]
    assert (band[..., 3] == 255).all(), "水格上半部应被崖壁覆盖"
    assert (raised[:3 * S] == flat[:3 * S]).all(), "陆地本身不应被改动"
    assert not (flat == raised).all()
    # 材质来自陆地（更高一侧）：颜色应接近陆地崖壁而非水面
    land_face = np.asarray(cliff_art_from_terrain(land_art).face)[..., :3].reshape(-1, 3).mean(axis=0)
    got = band[..., :3].reshape(-1, 3).mean(axis=0)
    assert np.abs(got - land_face).mean() < 40, f"水岸崖壁应使用陆地材质: {got} vs {land_face}"


def test_rock_plateau_draws_cliff_south_of_it():
    from core.tilemap.cliff import cliff_art_from_terrain

    model = TileMapModel(6, 6, tile_size=S)
    model.set_terrain(1, _art())
    model.set_terrain(3, _art((140, 130, 120)))
    model.base_terrain = 1
    model.fill_rect(0, 0, 5, 5, 1)
    model.fill_rect(2, 2, 4, 2, 3)                        # 岩石高台
    model.enable_height_layer({1: cliff_art_from_terrain(_art()), 3: cliff_art_from_terrain(_art((140, 130, 120)))})
    model.terrain_heights = {1: 0, 3: 1}
    cells = model.cliff_cells()
    assert set(cells) == {(x, 3) for x in (2, 3, 4)}, sorted(cells)
    data = model.to_dict()
    assert data["terrain_heights"] == {"1": 0, "3": 1}
    again = TileMapModel.from_dict(data)
    assert again.terrain_heights == {1: 0, 3: 1}
    assert again.effective_height(3, 2) == 1 and again.effective_height(3, 3) == 0


def test_flat_terrain_has_no_cliffs():
    from core.tilemap.cliff import cliff_art_from_terrain

    model = TileMapModel(4, 4, tile_size=S)
    model.set_terrain(1, _art())
    model.base_terrain = 1
    model.fill_rect(0, 0, 3, 3, 1)
    model.enable_height_layer({1: cliff_art_from_terrain(_art())})
    model.terrain_heights = {1: 0}
    assert model.cliff_cells() == {}
    assert (np.asarray(model.render())[..., 3] == 255).all()
