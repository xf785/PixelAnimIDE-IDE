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


def _quarter_px(tile, quarter):
    off = {"TL": (S // 4, S // 4), "TR": (3 * S // 4, S // 4),
           "BL": (S // 4, 3 * S // 4), "BR": (3 * S // 4, 3 * S // 4)}[quarter]
    return tile.getpixel(off)[:3]


def test_compose_art_full_and_isolated():
    base = _solid_base()
    full = compose_art_tile(base, SIDES | DIAG)
    assert np.asarray(full).shape == (S, S, 4)
    for q in ("TL", "TR", "BL", "BR"):
        assert _quarter_px(full, q) == (128, 128, 128)  # 全部取自中心

    iso = compose_art_tile(base, 0)
    assert _quarter_px(iso, "TL") == (200, 0, 0)      # tl 角艺术片
    assert _quarter_px(iso, "TR") == (0, 0, 200)
    assert _quarter_px(iso, "BL") == (200, 0, 200)
    assert _quarter_px(iso, "BR") == (40, 160, 160)


def test_compose_art_edge_mapping():
    base = _solid_base()
    # 上边界（T 空，其余满，对角满）：上半取 top 瓦片同侧四分之一，下半取中心
    top = compose_art_tile(base, (SIDES & ~BIT["T"]) | DIAG)
    assert _quarter_px(top, "TL") == (0, 200, 0)      # top 艺术片
    assert _quarter_px(top, "TR") == (0, 200, 0)
    assert _quarter_px(top, "BL") == (128, 128, 128)
    assert _quarter_px(top, "BR") == (128, 128, 128)
    # 左边界：左半取 left 艺术、右半取中心
    left = compose_art_tile(base, (SIDES & ~BIT["L"]) | DIAG)
    assert _quarter_px(left, "TL") == (200, 200, 0)
    assert _quarter_px(left, "BL") == (200, 200, 0)
    assert _quarter_px(left, "TR") == (128, 128, 128)
    # 外角：T/L 皆空 -> TL 取本角瓦片（tl）同侧四分之一
    outer = compose_art_tile(base, (BIT["R"] | BIT["B"] | BIT["BR"]))
    assert _quarter_px(outer, "TL") == (200, 0, 0)
    # 内角：四边满 + 仅 TL 对角空 -> TL 取对角瓦片（br）旋转 180° 的负形（凹角）
    inner = compose_art_tile(base, SIDES | (DIAG & ~BIT["TL"]))
    assert _quarter_px(inner, "TL") == (40, 160, 160)
    assert _quarter_px(inner, "BR") == (128, 128, 128)


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
    """回归：AI 把基础地形条带画在格外侧 1/4 时，拼装后条带必须仍在 1/4 处。

    旧实现取「源瓦片的下半/右半四分之一」，会把条带挪到目标瓦片中线（或直接丢失），
    导致相邻瓦片交界错位 —— 这是「拼接很烂」的核心原因之一。
    """
    base = _reference_like_block(strip=0.25)
    feat_c = (32, 96, 168)
    base_c = (235, 238, 245)
    # 仅上方为异类地形：整条顶部条带应保留在 1/4 处
    tile = np.asarray(compose_art_tile(base, (SIDES & ~BIT["T"]) | DIAG, blend=0))
    top_rows = tile[: S // 4, S // 2, :3]
    assert (top_rows == np.array(base_c)).all(), "顶部 1/4 应为基础地形条带"
    assert tuple(tile[S // 2, S // 2, :3]) == feat_c, "中线处仍应为特征纹理（条带未挪到中线）"
    # 外角：TL 四分之一块的外侧条带来自角瓦片，中心仍为特征
    corner = np.asarray(compose_art_tile(base, BIT["R"] | BIT["B"] | BIT["BR"], blend=0))
    assert tuple(corner[2, 2, :3]) == base_c
    assert tuple(corner[S // 2, S // 2, :3]) == feat_c
    # 内角：凹角处的负形来自对角瓦片旋转 180°（外侧仍为基础地形）
    inner = np.asarray(compose_art_tile(base, SIDES | (DIAG & ~BIT["TL"]), blend=0))
    assert tuple(inner[2, 2, :3]) == base_c
    assert tuple(inner[S // 2, S // 2, :3]) == feat_c
    assert tuple(inner[S - 3, S - 3, :3]) == feat_c


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
    base_terrain = _solid_base()
    # 特征地形：中心=水（蓝），周边=过渡（青）
    pond_colors = {
        "tl": (0, 128, 128), "top": (0, 160, 160), "tr": (0, 128, 128),
        "left": (0, 160, 160), "center": (0, 0, 255), "right": (0, 160, 160),
        "bl": (0, 128, 128), "bottom": (0, 160, 160), "br": (0, 128, 128),
    }
    solid = lambda c: Image.new("RGBA", (S, S), c + (255,))
    pond = BaseTileSet(
        size=S, center=solid(pond_colors["center"]),
        edges={n: solid(pond_colors[n]) for n in ("top", "bottom", "left", "right")},
        corners={n: solid(pond_colors[n]) for n in ("tl", "tr", "bl", "br")},
    )
    model = TileMapModel(5, 5, tile_size=S)
    model.set_terrain(1, base_terrain)
    model.set_terrain(2, pond)
    model.set_base_terrain(1)
    model.fill_rect(0, 0, 4, 4, value=1)
    model.set_cell(2, 2, 2)  # 中心一格水塘
    img = np.asarray(model.render())
    assert img.shape == (5 * S, 5 * S, 4)
    # 中心格（水塘孤立）：四角显示 pond 角艺术（青）；中心为四角艺术的内侧（同为角艺术色）
    assert img[2 * S + S // 4, 2 * S + S // 4, :3].tolist() == [0, 128, 128]  # TL 角盘
    assert img[2 * S + S // 2, 2 * S + S // 2, :3].tolist() == [0, 128, 128]  # 孤立足=四角艺术拼接
    # 基础地形格子（内部 (1,1)：八邻全满）：纯中心色（内部过渡显示填充）
    assert img[S + S // 2, S + S // 2, :3].tolist() == [128, 128, 128]

    # 建筑 overlay：直墙段（不透明红块）叠放
    wall = Image.new("RGBA", (S, S), (255, 0, 0, 255))
    model.set_overlay(1, 1, wall, rot=0)
    img2 = np.asarray(model.render())
    assert img2[S + S // 2, S + S // 2, :3].tolist() == [255, 0, 0]  # overlay 覆盖地块
    model.remove_overlay(1, 1)
    img3 = np.asarray(model.render())
    assert img3[S + S // 2, S + S // 2, :3].tolist() == [128, 128, 128]  # 移除后恢复


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


def test_compose_art_blend_is_symmetric():
    """接触带融合必须左右对称（修复写回顺序 bug）。"""
    left_color = (255, 0, 0)
    right_color = (0, 0, 255)
    top_tile = Image.new("RGBA", (S, S), left_color + (255,))
    center_tile = Image.new("RGBA", (S, S), right_color + (255,))
    base = BaseTileSet(
        size=S,
        center=center_tile,
        edges={n: top_tile for n in ("top", "bottom", "left", "right")},
        corners={n: top_tile for n in ("tl", "tr", "bl", "br")},
    )
    # 上边界瓦片：上半取 top 艺术（红），下半取中心（蓝）→ 中线两侧应严格对称
    tile = np.asarray(compose_art_tile(base, (SIDES & ~BIT["T"]) | DIAG, blend=1)).astype(np.int16)
    mid_left = tile[S // 2, S // 2 - 1, :3]
    mid_right = tile[S // 2, S // 2, :3]
    assert abs(int(mid_left[0]) - int(mid_right[0])) <= 2
    assert abs(int(mid_left[2]) - int(mid_right[2])) <= 2
    # 中线上方保留红色、下方保留蓝色（未整块糊掉）
    assert tile[S // 4, S // 2, 0] > tile[S // 4, S // 2, 2]
    assert tile[3 * S // 4, S // 2, 2] > tile[3 * S // 4, S // 2, 0]


def test_overlay_serialization_and_overlay_only_render():
    """建筑 overlay：带名称序列化/恢复；center=None 时也能渲染出拼件。"""
    model = TileMapModel(4, 3, tile_size=S)
    piece = Image.new("RGBA", (S, S), (255, 0, 0, 255))
    model.set_overlay(1, 1, piece, rot=1, name="straight")
    data = model.to_dict()
    assert data["overlay"] == [{"x": 1, "y": 1, "piece": "straight", "rot": 1}]
    # 无 center、无地形：仍应渲染出 overlay（修复前 overlay 被丢弃 → 全透明）
    img = np.asarray(model.render())
    assert img[S + S // 2, S + S // 2, 3] == 255
    # 按 pieces 注册表恢复
    restored = TileMapModel.from_dict(data, pieces={"straight": piece})
    assert (1, 1) in restored.overlay
    assert np.asarray(restored.render())[S + S // 2, S + S // 2, 3] == 255
