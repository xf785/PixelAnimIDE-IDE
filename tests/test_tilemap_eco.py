"""瓦片地图重构核心测试：偏移缝合保纹理、生态/建筑分类、艺术片 47 构图、
多地形掩码与 overlay 渲染。"""
import numpy as np
import pytest
from PIL import Image

from core.tilemap import (
    BIT,
    BaseTileSet,
    BuildingSheet,
    EcosystemSheet,
    TileMapModel,
    align_terrain_set,
    building_from_blocks,
    build_47_sheet_art,
    compose_art_tile,
    crop_blocks,
    ecosystem_from_blocks,
    make_texture_seamless,
    mask_for_terrain,
    prepare_terrain_set,
    process_building_sheet,
    rotate_piece,
)
from core.tilemap.prompts import build_building_prompts, build_ecosystem_prompts

S = 32
DIAG = BIT["TL"] | BIT["TR"] | BIT["BL"] | BIT["BR"]
SIDES = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]


# --------------------------------------------------------------------------- #
# 1) 偏移错位缝合：纹理细节保留 + 无缝
# --------------------------------------------------------------------------- #
def test_quilt_preserves_texture_details():
    """接缝窄带之外的像素必须原样保留（不再全图退化）。"""
    rng = np.random.default_rng(7)
    # 少量灰阶（≤256 色）保证重量化为恒等，便于逐像素比较
    arr = (rng.integers(0, 8, (S, S), dtype=np.uint8) * 36)
    rgb = np.stack([arr, arr, arr], axis=-1)
    # 特征块：非接缝带内
    rgb[6:12, 6:12] = (10, 200, 60)
    img = Image.fromarray(np.dstack([rgb, np.full((S, S), 255, np.uint8)]), "RGBA")
    out = np.asarray(make_texture_seamless(img, max_colors=256))
    band = S // 8 + 1
    interior = out[band:-band, band:-band]
    orig = rgb[band:-band, band:-band]
    assert (interior[..., :3] == orig).all()
    assert (out[:, 0] == out[:, -1]).all()
    assert (out[0, :] == out[-1, :]).all()


def test_quilt_deterministic():
    img = Image.new("RGBA", (S, S), (100, 40, 40, 255))
    a = make_texture_seamless(img)
    b = make_texture_seamless(img)
    assert a.tobytes() == b.tobytes()


# --------------------------------------------------------------------------- #
# 2) 生态/建筑分类裁切与提示词
# --------------------------------------------------------------------------- #
def _colored_sheet(cell: int):
    """6×6 网格图，每格唯一色（用于验证块映射）。"""
    img = Image.new("RGB", (cell * 6, cell * 6), (255, 255, 255))
    for r in range(6):
        for c in range(6):
            col = (r * 40 + 10, c * 40 + 10, 200 - r * 30)
            for y in range(r * cell, (r + 1) * cell):
                for x in range(c * cell, (c + 1) * cell):
                    img.putpixel((x, y), col)
    return img


def test_crop_blocks_and_sheet_mapping():
    img = _colored_sheet(16)
    blocks, cell = crop_blocks(img)
    assert cell == 16
    assert len(blocks) == 2 and len(blocks[0]) == 2
    # 左上块（0..2 行、0..2 列）中心格颜色
    center_px = blocks[0][0][4].getpixel((8, 8))[:3]
    assert center_px == (50, 50, 170)  # r=1,c=1: (50, 50, 200-30)
    # 右下块中心格 (r=4,c=4)
    br_center = blocks[1][1][4].getpixel((8, 8))[:3]
    assert br_center == (170, 170, 80)

    eco = ecosystem_from_blocks(blocks, tile_size=16, feature_names=["pond", "sparse", "rock"])
    assert isinstance(eco, EcosystemSheet)
    assert list(eco.features) == ["pond", "sparse", "rock"]
    assert eco.terrain_sets()[1].center.getpixel((8, 8))[:3] == (50, 50, 170)

    bld = building_from_blocks(blocks, tile_size=16)
    assert isinstance(bld, BuildingSheet)
    assert bld.wall.center.getpixel((8, 8))[:3] == (50, 50, 170)
    assert bld.pillar.center.getpixel((8, 8))[:3] == (170, 170, 80)


def test_crop_uses_whole_sheet_not_center_window():
    """回归（严重 bug）：裁切必须使用整张图。

    历史实现把「目标瓦片尺寸」当成裁切窗口（min(tile_size, w//6)），
    对 768×768 的 6×6 生态图只取了中心 192×192，AI 画的网格被整体切错位。
    """
    img = Image.new("RGBA", (768, 768), (30, 30, 30, 255))
    px = img.load()
    for y in range(640, 768):
        for x in range(640, 768):
            px[x, y] = (255, 0, 0, 255)  # 右下角格标记
    blocks, cell = crop_blocks(img)
    assert cell == 128  # 768 / 6
    br_tile = blocks[1][1][8].convert("RGB")
    assert any(p == (255, 0, 0) for p in br_tile.getdata()), "整图裁切后应包含右下角格内容"


# --------------------------------------------------------------------------- #
# 2b) 基础地形块位置自动识别（AI 常不遵守「左上」位置要求）
# --------------------------------------------------------------------------- #
BASE_C = (236, 240, 246)
FEAT_C = (36, 104, 172)
RIM_C = (94, 84, 72)
POS_ORDER = ("tl", "tr", "bl", "br")


def _eco_sheet(base_pos: str = "tl", cell: int = 32, plain_blocks=(), all_features: bool = False):
    """构造 6×6 生态底图：base_pos（及 plain_blocks）为纯基础地形，其余为圆角团块。

    每块 3×3：中心格纯特征；边格外侧 1/4 基础地形 + 描边；角格外侧 1/4 基础地形。
    all_features=True 时四块都画成团块（模拟 AI 没画纯基础块）。
    """
    size = cell * 6
    img = Image.new("RGBA", (size, size), BASE_C + (255,))
    px = img.load()
    plain = set(plain_blocks) if all_features else set(plain_blocks) | {base_pos}
    feat_cols = {"tl": (36, 104, 172), "tr": (120, 168, 96), "bl": (168, 152, 120), "br": (72, 120, 190)}
    band = max(2, cell // 4)
    for key in POS_ORDER:
        if key in plain:
            continue
        br, bc = {"tl": (0, 0), "tr": (0, 1), "bl": (1, 0), "br": (1, 1)}[key]
        col = feat_cols[key]
        for r in range(3):
            for c in range(3):
                y0 = (br * 3 + r) * cell
                x0 = (bc * 3 + c) * cell
                for y in range(cell):
                    for x in range(cell):
                        outer = (
                            (r == 0 and y < band) or (r == 2 and y >= cell - band)
                            or (c == 0 and x < band) or (c == 2 and x >= cell - band)
                        )
                        if outer:
                            px[x0 + x, y0 + y] = BASE_C + (255,)
                            continue
                        rim = (
                            (r == 0 and y < band + 1) or (r == 2 and y >= cell - band - 1)
                            or (c == 0 and x < band + 1) or (c == 2 and x >= cell - band - 1)
                        )
                        px[x0 + x, y0 + y] = (RIM_C if rim else col) + (255,)
    return img


@pytest.mark.parametrize("base_pos", POS_ORDER)
def test_detect_base_block_finds_plain_block_in_any_position(base_pos):
    """基础块无论被 AI 画在哪个位置，都必须被识别出来（否则会把特征当地面）。"""
    from core.tilemap import detect_base_block

    blocks, _cell = crop_blocks(_eco_sheet(base_pos))
    pos, scores, confident = detect_base_block(blocks)
    assert pos == base_pos, f"scores={scores}"
    assert confident
    assert scores[base_pos] < 6.0, "纯基础地形块得分应接近 0"
    # 其余三块都是带边界的团块 -> 得分显著更高
    for other in POS_ORDER:
        if other != base_pos:
            assert scores[other] > 60.0, f"{other} scores={scores}"


def test_ecosystem_blocks_use_detected_base_and_reading_order():
    """base 自动识别 + 其余三块按阅读顺序对应特征名。"""
    img = _eco_sheet("tr")
    blocks, _cell = crop_blocks(img)
    eco = ecosystem_from_blocks(blocks, tile_size=32, feature_names=["water", "grass", "rock"])
    assert eco.base_pos == "tr" and eco.base_pos_detected
    assert np.asarray(eco.base.center.convert("RGB")).reshape(-1, 3).mean(0).round(0).tolist() == list(BASE_C)
    assert list(eco.features) == ["water", "grass", "rock"]
    # 特征块按阅读顺序：tl -> water, bl -> grass, br -> rock
    assert np.asarray(eco.features["water"].center.convert("RGB")).reshape(-1, 3).mean(0).round(0).tolist() == [36, 104, 172]
    assert np.asarray(eco.features["grass"].center.convert("RGB")).reshape(-1, 3).mean(0).round(0).tolist() == [168, 152, 120]
    assert np.asarray(eco.features["rock"].center.convert("RGB")).reshape(-1, 3).mean(0).round(0).tolist() == [72, 120, 190]


def test_base_pos_explicit_override_wins():
    """手动指定必须覆盖自动识别（用户可在确认底图后纠正误判）。"""
    blocks, _cell = crop_blocks(_eco_sheet("tr"))
    eco = ecosystem_from_blocks(blocks, tile_size=32, feature_names=["a", "b", "c"], base_pos="bl")
    assert eco.base_pos == "bl" and not eco.base_pos_detected
    assert np.asarray(eco.base.center.convert("RGB")).reshape(-1, 3).mean(0).round(0).tolist() == [168, 152, 120]
    with pytest.raises(ValueError):
        ecosystem_from_blocks(blocks, tile_size=32, base_pos="center")


def test_detect_base_block_falls_back_when_no_plain_block():
    """四块都是带边界的团块（AI 没画纯基础块）-> 回退左上且不可信，供 UI 提示。"""
    from core.tilemap import detect_base_block

    blocks_no_plain, _cell = crop_blocks(_eco_sheet("tl", all_features=True))
    pos, scores, confident = detect_base_block(blocks_no_plain)
    assert pos == "tl" and not confident
    assert min(scores.values()) > 60.0
    # 正常底图（左上为纯块）则可信
    assert detect_base_block(crop_blocks(_eco_sheet("tl"))[0])[2] is True


def test_plain_sheet_detection_is_safe():
    """四块都是同一基础地形（用户只画了地面）时仍取左上、不报错。"""
    img = Image.new("RGBA", (96, 96), BASE_C + (255,))
    eco = ecosystem_from_blocks(crop_blocks(img)[0], tile_size=32, feature_names=["a"])
    assert eco.base_pos == "tl"
    assert np.asarray(eco.base.center.convert("RGB")).reshape(-1, 3).mean(0).round(0).tolist() == list(BASE_C)


def test_grid_cell_px_is_multiple_and_within_limit():
    from core.tilemap.tiles import grid_cell_px

    assert grid_cell_px(768, 6) == 128          # 默认：768 = 6×128
    assert grid_cell_px(512, 6) == 64           # 取 32/64 倍数
    assert grid_cell_px(256, 6) == 32
    for sheet in (256, 384, 512, 768, 1024, 1536, 2048):
        cell = grid_cell_px(sheet, 6)
        assert cell % 32 == 0 and cell >= 32
        assert 6 * cell <= 2048


def test_mode_downscale_recovers_pixel_art():
    """块众数降采样：4 倍放大的像素画应被精确还原（而非点采样丢格色）。"""
    from core.tilemap.tiles import resize_tile

    rng = np.random.default_rng(3)
    small = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    small[:8, :8] = (200, 30, 30)
    src = Image.fromarray(np.dstack([small, np.full((32, 32), 255, np.uint8)]), "RGBA")
    big = src.resize((128, 128), Image.Resampling.NEAREST)  # 模拟 AI 按 128px/格 出图
    back = np.asarray(resize_tile(big, 32))
    assert (back[..., :3] == small).all()


def test_ecosystem_and_building_prompts():
    p = build_ecosystem_prompts(
        "草地", {"水塘": "a pond", "岩石": "rocks"}, tile_size=32, cell_px=128
    )
    text = p["image_prompt"]
    assert "2x2 grid of blocks" in text
    assert "6 columns x 6 rows" in text
    assert "128x128 pixels" in text and "768x768 pixels" in text  # 与请求尺寸严格一致
    assert "BASE TERRAIN" in text
    # 特征块几何契约：外侧约 1/4 条带 + 角格圆弧（算法按同侧四分之一块拼接的前提）
    assert "OUTER (top) band" in text and "about the outer 1/4" in text
    assert "quarter-circle arc" in text
    # 风格契约：必须有深色描边与明暗（参考示意图的关键特征）
    assert "darker 1-2 px outline" in text and "2-3 tone shading" in text
    assert "pixel-consistent across all 4 blocks" in text
    assert p["features"] == ["水塘", "岩石"]
    assert p["cell_px"] == 128

    b = build_building_prompts("stone wall", tile_size=32, cell_px=128)
    assert "WALL SET" in b["image_prompt"] and "PILLAR SET" in b["image_prompt"]
    assert "PURE WHITE" in b["image_prompt"]
    assert "FILLS THE ENTIRE CELL" in b["image_prompt"]
    assert "768x768 pixels" in b["image_prompt"]


# --------------------------------------------------------------------------- #
# 3) 艺术片 47 构图（真实九宫格艺术拼接）
# --------------------------------------------------------------------------- #
def _solid_base():
    """每张瓦片单一颜色（验证四分之一块映射）。"""
    colors = {
        "tl": (200, 0, 0), "top": (0, 200, 0), "tr": (0, 0, 200),
        "left": (200, 200, 0), "center": (128, 128, 128), "right": (0, 200, 200),
        "bl": (200, 0, 200), "bottom": (60, 60, 200), "br": (40, 160, 160),
    }
    solid = lambda c: Image.new("RGBA", (S, S), c + (255,))
    return BaseTileSet(
        size=S,
        center=solid(colors["center"]),
        edges={n: solid(colors[n]) for n in ("top", "bottom", "left", "right")},
        corners={n: solid(colors[n]) for n in ("tl", "tr", "bl", "br")},
    )


def _lum(rgb) -> float:
    """感知亮度（描边判定用；描边色会被纹理轻微调制，因此不做逐像素等值断言）。"""
    r, g, b = (float(c) for c in rgb[:3])
    return 0.299 * r + 0.587 * g + 0.114 * b

def _quarter_px(tile, quarter):
    off = {"TL": (S // 4, S // 4), "TR": (3 * S // 4, S // 4),
           "BL": (S // 4, 3 * S // 4), "BR": (3 * S // 4, 3 * S // 4)}[quarter]
    return tile.getpixel(off)[:3]


FEAT_RGB = (38, 104, 172)
GROUND_RGB = (236, 240, 246)
RIM_RGB = (81, 75, 66)


def _cell(col, checker=0, size=S):
    """模拟 AI 的一格像素画（带轻微棋盘纹理，便于检查纹理相位是否对齐）。"""
    a = np.zeros((size, size, 4), np.uint8)
    a[..., :3] = col
    a[..., 3] = 255
    if checker:
        ys, xs = np.mgrid[0:size, 0:size]
        m = ((ys // 3 + xs // 3) % 2 == 0)
        a[m, :3] = np.clip(np.array(col) + checker, 0, 255)
    return Image.fromarray(a, "RGBA")


def _art(feat_rgb=FEAT_RGB, ground=GROUND_RGB, band=8, rim=2, radius=8, checker=14, ground_checker=10):
    """按真实管线造一套「对齐式」地形艺术（纹理经偏移缝合 → 边缘逐像素可接）。"""
    from core.tilemap.seamless import make_tile_texture, median_tile_texture

    ground_tex = median_tile_texture([_cell(ground, ground_checker) for _ in range(9)], S)
    tex = make_tile_texture(_cell(feat_rgb, checker), S)
    return BaseTileSet(
        size=S, center=tex,
        edges={n: tex for n in ("top", "bottom", "left", "right")},
        corners={n: tex for n in ("tl", "tr", "bl", "br")},
        line_color=RIM_RGB, line_width=rim, band=band, radius=radius, base_texture=ground_tex,
    )


def test_compose_aligned_full_and_isolated():
    """对齐式构图：全邻瓦片 = 纯特征纹理；孤立足 = 特征岛 + 四周基础地形条带 + 描边。"""
    art = _art()
    full = np.asarray(compose_art_tile(art, SIDES | DIAG))
    assert full.shape == (S, S, 4)
    # 全邻：整格就是纹理本身（逐像素等于 base.center）
    assert (full == np.asarray(art.center)).all()
    assert (full[..., 3] == 255).all()          # 地块瓦片不允许透明

    band = art.band
    rim = art.line_width
    iso = np.asarray(compose_art_tile(art, 0))
    # 四条边：外侧 band 像素是基础地形纹理（内侧 rim 像素是描边）
    ground = np.asarray(art.base_texture)
    assert (iso[: band // 2, S // 2, :3] == ground[: band // 2, S // 2, :3]).all()
    assert (iso[-(band - rim - 3) :, S // 2, :3] == ground[-(band - rim - 3) :, S // 2, :3]).all()
    assert (iso[S // 2, : band // 2, :3] == ground[S // 2, : band // 2, :3]).all()
    # 描边：条带内侧 rim 像素 = 描边色
    assert _lum(iso[band - 1, S // 2, :3]) < 130, "条带内侧应是暗描边"
    assert _lum(iso[S // 2, band - 1, :3]) < 130, "条带内侧应是暗描边"
    # 内部仍是特征纹理
    assert tuple(iso[S // 2, S // 2, :3]) == tuple(np.asarray(art.center)[S // 2, S // 2, :3])


def test_compose_aligned_edges_and_corners():
    """边/角几何：暴露侧是基础地形条带，外角圆角、内角凹口。"""
    art = _art()
    band = art.band
    ground = np.asarray(art.base_texture)
    center = np.asarray(art.center)

    top = np.asarray(compose_art_tile(art, (SIDES & ~BIT["T"]) | DIAG))
    assert (top[: band // 2, :, :3] == ground[: band // 2, :, :3]).all(), "上侧应为基础地形条带"
    assert _lum(top[band - 1, S // 2, :3]) < 130, "条带内侧应有描边"
    assert (top[band + 2 :, :, :3] == center[band + 2 :, :, :3]).all(), "其余仍是特征纹理"

    left = np.asarray(compose_art_tile(art, (SIDES & ~BIT["L"]) | DIAG))
    assert (left[:, : band // 2, :3] == ground[:, : band // 2, :3]).all()
    assert _lum(left[S // 2, band - 1, :3]) < 130

    # 外角（上、左都暴露）：角部是基础地形（圆弧向外凸），圆角内侧才是特征
    outer = np.asarray(compose_art_tile(art, BIT["R"] | BIT["B"] | BIT["BR"]))
    assert (outer[1, 1, :3] == ground[1, 1, :3]).all()
    # 圆弧最靠近角点的位置约在 (band+0.41r, band+0.41r)，再往里才是特征纹理
    diag = band + art.radius - 2
    assert tuple(outer[diag, diag, :3]) == tuple(center[diag, diag, :3])

    # 内角（四边满、TL 对角空）：角上是基础地形凹口（半径 band-1，边缘处正好 band 像素）
    inner = np.asarray(compose_art_tile(art, SIDES | (DIAG & ~BIT["TL"])))
    assert (inner[0, 0, :3] == ground[0, 0, :3]).all(), "内角角点应是另一方地形"
    assert tuple(inner[band, band, :3]) == tuple(center[band, band, :3]), "凹口外仍是特征"
    assert _lum(inner[band - 1, 0, :3]) < 130, "凹口边界是描边"
    assert (inner[band + 3, 0, :3] == center[band + 3, 0, :3]).all(), "凹口沿边长度 = band（往外回到特征）"


def test_measured_band_and_rim_from_ai_block():
    """几何参数实测自 AI 九宫格：条带深度与描边宽度按比例换算到目标瓦片。"""
    from core.tilemap.seamless import measure_terrain_art

    cell = 64
    band_cell = 16          # 1/4
    rim_cell = 4
    tiles = {}
    for name in ("tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"):
        arr = np.zeros((cell, cell, 4), np.uint8)
        arr[..., :3] = FEAT_RGB
        arr[..., 3] = 255
        ys, xs = np.mgrid[0:cell, 0:cell]
        if name != "center":
            band = (
                ((name in ("top", "tl", "tr")) & (ys < band_cell))
                | ((name in ("bottom", "bl", "br")) & (ys >= cell - band_cell))
                | ((name in ("left", "tl", "bl")) & (xs < band_cell))
                | ((name in ("right", "tr", "br")) & (xs >= cell - band_cell))
            )
            rim = (
                ((name in ("top", "tl", "tr")) & (ys < band_cell + rim_cell))
                | ((name in ("bottom", "bl", "br")) & (ys >= cell - band_cell - rim_cell))
                | ((name in ("left", "tl", "bl")) & (xs < band_cell + rim_cell))
                | ((name in ("right", "tr", "br")) & (xs >= cell - band_cell - rim_cell))
            )
            arr[band] = GROUND_RGB + (255,)
            arr[rim & ~band] = RIM_RGB + (255,)
        tiles[name] = Image.fromarray(arr, "RGBA")
    raw = BaseTileSet(
        size=cell, center=tiles["center"],
        edges={n: tiles[n] for n in ("top", "bottom", "left", "right")},
        corners={n: tiles[n] for n in ("tl", "tr", "bl", "br")},
    )
    ground = np.array(GROUND_RGB, dtype=np.float32)
    measured = measure_terrain_art(raw, ground_rgb=ground)
    # 条带 = 基础地形像素 + 描边像素（描边画在条带内侧），因此 16+4 像素 = 0.3125
    assert abs(measured["band_frac"] - (band_cell + rim_cell) / cell) < 0.03, measured
    assert abs(measured["rim_frac"] - rim_cell / cell) < 0.02, measured
    assert max(abs(a - b) for a, b in zip(measured["rim_rgb"], RIM_RGB)) < 40, measured

    art = align_terrain_set(raw, base_texture=_cell(GROUND_RGB), tile_size=S, ground_rgb=ground)
    assert art.band == 10 and art.line_width == 2, art.art_meta
    # 拼接后：外侧 8 像素是基础地形、第 9-10 像素是描边（与 AI 底图比例一致）
    tile = np.asarray(compose_art_tile(art, (SIDES & ~BIT["T"]) | DIAG))
    assert (tile[:5, :, :3] == np.asarray(art.base_texture)[:5, :, :3]).all()
    assert _lum(tile[9, S // 2, :3]) < 130


def test_aligned_output_is_seamless_regardless_of_ai_layout():
    """核心不变量：随机多地形地图里，相邻瓦片共享边逐像素相等（转角带除外）。

    这是「47-tile 衔接」的构造性保证：纹理网格对齐 + 几何只依赖位掩码，
    因此 AI 把边界画在哪个深度、格子是否连续、有没有格线框都不影响接缝。
    """
    art = {
        1: _art(GROUND_RGB, GROUND_RGB, checker=10, ground_checker=10),
        2: _art(FEAT_RGB, GROUND_RGB, checker=14),
        3: _art((168, 152, 120), GROUND_RGB, checker=16),
    }
    band = 8
    rng = np.random.default_rng(7)
    H, W = 10, 12
    grid = rng.integers(1, 4, (H, W))
    grid[0, :] = 1
    tiles = {}
    for y in range(H):
        for x in range(W):
            nb = [
                [int(grid[ny, nx]) if 0 <= ny < H and 0 <= nx < W else 0 for nx in range(x - 1, x + 2)]
                for ny in range(y - 1, y + 2)
            ]
            mask = mask_for_terrain(nb, int(grid[y, x]), base_terrain=1)
            tiles[(x, y)] = np.asarray(compose_art_tile(art[int(grid[y, x])], mask))
    corner = band + 2   # 转角带：两条描边线在此交汇，允许 ±1 像素
    bad = []
    for y in range(H):
        for x in range(W - 1):
            left, right = tiles[(x, y)][:, S - 1], tiles[(x + 1, y)][:, 0]
            rows = np.nonzero((left != right).any(axis=1))[0]
            rows = [r for r in rows if corner <= r < S - corner]
            if rows:
                bad.append((x, y, rows[:4]))
    for y in range(H - 1):
        for x in range(W):
            top, bot = tiles[(x, y)][S - 1, :], tiles[(x, y + 1)][0, :]
            cols = np.nonzero((top != bot).any(axis=1))[0]
            cols = [c for c in cols if corner <= c < S - corner]
            if cols:
                bad.append((x, y, cols[:4]))
    assert not bad, f"共享边（非转角带）出现像素差: {bad[:5]}"


def _reference_like_block(strip: float = 0.25, size: int = S):
    """构造「参考示意图」风格的九宫格：圆角团块 + 外侧定宽基础地形条带。"""
    base_c = (235, 238, 245)
    feat_c = (32, 96, 168)
    rim_c = (96, 86, 74)
    b = int(size * strip)
    tiles = {}
    tiles["center"] = Image.new("RGBA", (size, size), feat_c + (255,))

    def edge(where: str):
        img = Image.new("RGBA", (size, size), feat_c + (255,))
        px = img.load()
        for y in range(size):
            for x in range(size):
                outer = (
                    (where == "top" and y < b) or (where == "bottom" and y >= size - b)
                    or (where == "left" and x < b) or (where == "right" and x >= size - b)
                )
                if outer:
                    px[x, y] = base_c + (255,)
                # 交界处画一圈深色描边（模拟参考图的岩边）
                elif (
                    (where == "top" and y == b) or (where == "bottom" and y == size - b - 1)
                    or (where == "left" and x == b) or (where == "right" and x == size - b - 1)
                ):
                    px[x, y] = rim_c + (255,)
        return img

    def corner(where: str):
        img = Image.new("RGBA", (size, size), feat_c + (255,))
        px = img.load()
        for y in range(size):
            for x in range(size):
                outer = (
                    (where in ("tl",) and (y < b or x < b))
                    or (where == "tr" and (y < b or x >= size - b))
                    or (where == "bl" and (y >= size - b or x < b))
                    or (where == "br" and (y >= size - b or x >= size - b))
                )
                if outer:
                    px[x, y] = base_c + (255,)
        return img

    for name in ("top", "bottom", "left", "right"):
        tiles[name] = edge(name)
    for name in ("tl", "tr", "bl", "br"):
        tiles[name] = corner(name)
    return BaseTileSet(
        size=size,
        center=tiles["center"],
        edges={n: tiles[n] for n in ("top", "bottom", "left", "right")},
        corners={n: tiles[n] for n in ("tl", "tr", "bl", "br")},
    )


def test_art_strip_keeps_ai_boundary_depth():
    """回归（第 6~8 轮）：AI 把基础地形条带画在格外侧 1/4 时也必须正确。

    旧实现逐块抠 AI 的九宫格艺术，条带深度/连续性/格线框都会影响成品；
    第 8 轮改为「AI 纹理 + 程序化几何」：只实测条带深度与描边，几何由算法
    生成，因此这里断言的是实测结果与拼装后条带位置一致。
    """
    art = _art()
    tile = np.asarray(compose_art_tile(art, (SIDES & ~BIT["T"]) | DIAG))
    ground = np.asarray(art.base_texture)
    inner = art.band - art.line_width - 2
    assert (tile[:inner, :, :3] == ground[:inner, :, :3]).all(), "条带外侧应为基础地形（内侧是描边/接触阴影）"
    assert tuple(tile[S // 2, S // 2, :3]) == tuple(np.asarray(art.center)[S // 2, S // 2, :3])
    assert align_terrain_set(_reference_like_block(strip=0.25), base_texture=_cell(ground[0, 0, :3], 0),
                             tile_size=S, ground_rgb=np.array(GROUND_RGB, dtype=np.float32)).band == 8


def test_build_47_sheet_art_complete():
    base = _solid_base()
    sheet, meta = build_47_sheet_art(base)
    assert meta["tile_count"] == 47
    assert len(meta["mask_to_index"]) == 256
    # 全邻瓦片 = 纯中心
    idx = meta["mask_to_index"]["255"]
    tile = sheet.crop(((idx % 8) * S, (idx // 8) * S, (idx % 8 + 1) * S, (idx // 8 + 1) * S))
    assert np.asarray(tile)[:, :, :3].min() == 128 and np.asarray(tile)[:, :, :3].max() == 128


# --------------------------------------------------------------------------- #
# 4) 多地形掩码与渲染、建筑 overlay
# --------------------------------------------------------------------------- #
def test_mask_for_terrain_base_counts_features():
    nb = [
        [1, 1, 2],
        [2, 0, 2],  # 中心是地形 1（基础）；周围混合 1/2
        [1, 2, 1],
    ]
    m_base = mask_for_terrain(nb, 1, base_terrain=1)
    assert m_base == 255  # 基础地形：任何非空都算满
    m_feat = mask_for_terrain(nb, 2, base_terrain=1)
    # 仅同地形 2 的邻居；TR 对角位因「上方 T 不是同类」被规范化剔除（经典 blob 规则）
    assert m_feat == (BIT["L"] | BIT["R"] | BIT["B"])


def test_multiterrain_render_and_overlay():
    """多地形对齐式渲染：特征地形显示自己的纹理 + 描边，基础地形为纯纹理。"""
    base_art = _art(GROUND_RGB, GROUND_RGB, checker=10, ground_checker=10)
    pond_art = _art(FEAT_RGB, GROUND_RGB, checker=14)
    model = TileMapModel(5, 5, tile_size=S)
    model.set_terrain(1, base_art)
    model.set_terrain(2, pond_art)
    model.set_base_terrain(1)
    model.fill_rect(0, 0, 4, 4, value=1)
    model.set_cell(2, 2, 2)  # 中心一格水塘
    img = np.asarray(model.render())
    assert img.shape == (5 * S, 5 * S, 4)
    band = pond_art.band
    x0, y0 = 2 * S, 2 * S
    # 孤立的特征格：四条边外侧是基础地形，条带内侧是描边
    assert (img[y0, x0 + S // 2, :3] == np.asarray(pond_art.base_texture)[0, S // 2, :3]).all()
    assert _lum(img[y0 + band - 1, x0 + S // 2, :3]) < 130
    # 中心是特征纹理本身（不再是四分之一块拼接）：取中列/中行避开条带与转角
    mid = S // 2
    assert (img[y0 + mid, x0 + band + 3 : x0 + S - band - 3, :3]
            == np.asarray(pond_art.center)[mid, band + 3 : S - band - 3, :3]).all()
    assert (img[y0 + band + 3 : y0 + S - band - 3, x0 + mid, :3]
            == np.asarray(pond_art.center)[band + 3 : S - band - 3, mid, :3]).all()
    # 基础地形格子（内部 (1,1)：八邻全满）：整格纯纹理
    assert (img[S : 2 * S, S : 2 * S, :3] == np.asarray(base_art.center)[..., :3]).all()

    # 建筑 overlay：直墙段（不透明红块）叠放
    wall = Image.new("RGBA", (S, S), (255, 0, 0, 255))
    model.set_overlay(1, 1, wall, rot=0)
    img2 = np.asarray(model.render())
    assert img2[S + S // 2, S + S // 2, :3].tolist() == [255, 0, 0]  # overlay 覆盖地块
    model.remove_overlay(1, 1)
    img3 = np.asarray(model.render())
    assert (img3[S : 2 * S, S : 2 * S, :3] == np.asarray(base_art.center)[..., :3]).all()  # 移除后恢复


# --------------------------------------------------------------------------- #
# 5) 地块组预处理（保艺术）与建筑拼件
# --------------------------------------------------------------------------- #
def test_prepare_terrain_set_preserves_art_and_wraps():
    base = _solid_base()
    # 边瓦片左右接缝不同色 -> 轴向缝合应让 top 瓦片左右相等
    proc = prepare_terrain_set(base)
    c = np.asarray(proc.center)
    assert (c[:, 0] == c[:, -1]).all() and (c[0, :] == c[-1, :]).all()
    # 角瓦片艺术完全保留（唯一色直接量化原样）
    for name in ("tl", "tr", "bl", "br"):
        assert np.asarray(proc.corners[name]).shape == (S, S, 4)


def test_building_pieces_and_overlay_transparency():
    wall_tile = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    # 中间一个石墙色方块，四周白底
    arr = np.asarray(wall_tile).copy()
    arr[6:26, 6:26] = (90, 90, 90, 255)
    wall = Image.fromarray(arr, "RGBA")
    sheet = BuildingSheet(
        wall=BaseTileSet(size=S, center=wall, edges={}, corners={}),
        top=BaseTileSet(size=S, center=wall, edges={}, corners={}),
        opening=BaseTileSet(size=S, center=wall, edges={}, corners={}),
        pillar=BaseTileSet(size=S, center=wall, edges={}, corners={}),
    )
    out = process_building_sheet(sheet)
    pieces = out["pieces"]
    for name in ("straight", "end", "corner", "pillar"):
        assert pieces[name].mode == "RGBA"
    # 白底已抠除：直段角落透明、中心不透明（墙体全格）
    straight = np.asarray(pieces["straight"])
    assert straight[S // 2, S // 2, 3] == 255
    # 端头右端圆盘切 -> 右下角透明
    end = np.asarray(pieces["end"])
    assert end[-1, -1, 3] == 0
    assert end[S // 2, S // 2, 3] == 255
    # 旋转：转角件旋转 90° 后尺寸不变、内容转置
    rot = rotate_piece(pieces["corner"], 1)
    assert rot.size == (S, S)
    assert np.asarray(pieces["corner"])[0, -1, 3] == np.asarray(rot)[-1, -1, 3]


# --------------------------------------------------------------------------- #
# 6) 规范化掩码（FrameRonin blob 规则）与修复回归
# --------------------------------------------------------------------------- #
def test_canonical_mask_rule():
    """对角位仅在两相邻正交位为满时才生效（经典 blob 规则）。"""
    from core.tilemap import canonical_mask, canonical_masks, mask_from_neighbors

    # 只有对角邻居 -> 等价孤立瓦片
    nb = [[1, 0, 1], [0, 0, 0], [0, 0, 0]]
    assert mask_from_neighbors(nb) == 0
    # 正交位为满时对角位才保留
    nb2 = [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert mask_from_neighbors(nb2) == (BIT["TL"] | BIT["T"] | BIT["L"])
    # 可达掩码恰好 47 个
    masks = canonical_masks()
    assert len(masks) == 47
    assert all(canonical_mask(m) == m for m in masks)
    assert 255 in masks and 0 in masks


def test_nearest_mask_fallback_and_full_mapping():
    from core.tilemap import canonical_masks, nearest_mask

    masks = canonical_masks()
    assert nearest_mask(255) == 255          # 合法掩码原样返回
    assert nearest_mask(5) == 0              # 仅对角位 -> 最近的孤立掩码
    assert nearest_mask(7) in masks
    sheet_meta_masks = None
    # 导出映射覆盖全部 256 掩码且指向合法槽位
    from core.tilemap import build_47_sheet_art

    _, meta = build_47_sheet_art(_solid_base())
    sheet_meta_masks = meta["mask_to_index"]
    assert len(sheet_meta_masks) == 256
    assert set(meta["reachable_masks"]) == set(masks)
    assert all(0 <= v < 47 for v in sheet_meta_masks.values())


def test_compose_aligned_textures_are_wrap_equal():
    """纹理不变量：对齐式构图依赖「左右/上下边缘逐像素相等」（周期平铺零接缝）。"""
    art = _art(checker=14)
    tex = np.asarray(art.center)
    ground = np.asarray(art.base_texture)
    assert (tex[:, 0] == tex[:, -1]).all() and (tex[0, :] == tex[-1, :]).all()
    assert (ground[:, 0] == ground[:, -1]).all() and (ground[0, :] == ground[-1, :]).all()
    # 全邻瓦片的左右边缘因此逐像素相等（相邻同地形格严丝合缝）
    full = np.asarray(compose_art_tile(art, SIDES | DIAG))
    assert (full[:, 0] == full[:, -1]).all()


def test_overlay_serialization_and_overlay_only_render():
    """建筑 overlay：带名称序列化/恢复；center=None 时也能渲染出拼件。"""
    model = TileMapModel(4, 3, tile_size=S)
    piece = Image.new("RGBA", (S, S), (255, 0, 0, 255))
    model.set_overlay(1, 1, piece, rot=1, name="straight")
    data = model.to_dict()
    assert data["overlay"] == [{"x": 1, "y": 1, "piece": "straight", "rot": 1, "scale": 1.0}]
    # 无 center、无地形：仍应渲染出 overlay（修复前 overlay 被丢弃 → 全透明）
    img = np.asarray(model.render())
    assert img[S + S // 2, S + S // 2, 3] == 255
    # 按 pieces 注册表恢复
    restored = TileMapModel.from_dict(data, pieces={"straight": piece})
    assert (1, 1) in restored.overlay
    assert np.asarray(restored.render())[S + S // 2, S + S // 2, 3] == 255


def test_overlay_scale_spans_multiple_tiles():
    """素材不都占一格：放置缩放以格子底部中心为锚点，并能序列化恢复。"""
    model = TileMapModel(4, 3, tile_size=S)
    model.set_cell(1, 1, 1)
    piece = Image.new("RGBA", (S, S), (255, 0, 0, 255))
    model.set_overlay(1, 1, piece, 0, name="tree", scale=2.0)
    img = np.asarray(model.render())
    # 2 倍大小 => 覆盖 2x2 格，且底部与锚点格底边对齐
    assert img[2 * S - 1, S, :3].tolist() == [255, 0, 0]
    assert img[S + 1, S - 1, :3].tolist() == [255, 0, 0]      # 向左上扩张
    assert img[0, 0, 3] == 0                                   # 2×2 覆盖范围之外仍透明
    assert img[S - 1, S - 1, 3] == 255                         # 向左上扩张到锚点格左上
    data = model.to_dict()
    assert data["overlay"][0]["scale"] == 2.0
    restored = TileMapModel.from_dict(data, pieces={"tree": piece})
    assert (np.asarray(restored.render()) == img).all()
    # 缩小的素材同样以底部中心为锚点
    small = TileMapModel(4, 3, tile_size=S)
    small.set_cell(1, 1, 1)
    small.set_overlay(1, 1, piece, 0, name="tree", scale=0.5)
    arr = np.asarray(small.render())
    ys, xs = np.nonzero(arr[..., 0] > 200)
    assert ys.max() == 2 * S - 1, "缩小后仍应底部对齐锚点格底边"
    center = (int(xs.min()) + int(xs.max())) / 2
    assert abs(center - (S + S // 2)) <= 1.5, f"应水平居中于锚点格：{center}"
