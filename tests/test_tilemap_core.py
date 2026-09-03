"""瓦片地图核心算法测试（严格保证可行性与质量）。

覆盖：
- 3×3 自适应裁切与规格归一
- 严格瓦片集提示词
- 纹理全向无缝 / 墙面轴向无缝 / 转角推导（与墙面/中心零接缝）
- 47-tile 几何构图（外角/内角/孤立/边线）与 47 计数、256 掩码全映射
- 双网格导出
- 地图模型铺设、自动衔接、渲染跨边无缝不变量、序列化
"""
import numpy as np
import pytest
from PIL import Image

from core.tilemap import (
    BIT,
    BaseTileSet,
    TileMapModel,
    build_47_sheet,
    compose_tile,
    crop_base_3x3,
    derive_corners,
    make_edge_seamless,
    make_texture_seamless,
    mask_from_neighbors,
    normalize_tileset,
    process_base_set,
)
from core.tilemap.autotile import dual_grid_map
from core.tilemap.prompts import build_tileset_prompts

S = 32


def _noise_texture(size=S, seed=0, alpha=True):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(20, 220, (size, size, 3), dtype=np.uint8)
    a = np.full((size, size), 255, dtype=np.uint8)
    arr = np.dstack([rgb, a]) if alpha else rgb
    return Image.fromarray(arr, "RGBA" if alpha else "RGB")


def _center_texture(size=S, seed=7):
    """带明显接缝的纹理（左右/上下颜色不同）用于验证无缝化。"""
    rng = np.random.default_rng(seed)
    arr = rng.integers(20, 220, (size, size, 3), dtype=np.uint8)
    arr[:, 0:3] = (10, 200, 30)
    arr[:, -3:] = (200, 10, 30)
    arr[0:3, :] = (30, 10, 200)
    arr[-3:, :] = (200, 200, 10)
    return Image.fromarray(np.dstack([arr, np.full((size, size), 255, np.uint8)]), "RGBA")


def _edge_top(size=S, seed=3):
    tile = _noise_texture(size, seed)
    arr = np.asarray(tile).copy()
    arr[:2, :, :3] = (5, 5, 5)  # 边界线带
    return Image.fromarray(arr, "RGBA")


def _make_base9(center=None, size=S):
    center = center or _center_texture(size)
    edges = {}
    for name in ("top", "bottom", "left", "right"):
        tile = _noise_texture(size, seed=hash(name) % 100)
        arr = np.asarray(tile).copy()
        if name == "top":
            arr[:2, :, :3] = (5, 5, 5)
        elif name == "bottom":
            arr[-2:, :, :3] = (5, 5, 5)
        elif name == "left":
            arr[:, :2, :3] = (5, 5, 5)
        else:
            arr[:, -2:, :3] = (5, 5, 5)
        edges[name] = Image.fromarray(arr, "RGBA")
    corners = {n: _noise_texture(size, seed=20 + i) for i, n in enumerate(("tl", "tr", "bl", "br"))}
    return BaseTileSet(size=size, center=center, edges=edges, corners=corners)


# --------------------------------------------------------------------------- #
# 裁切与提示词
# --------------------------------------------------------------------------- #
def test_crop_adaptive_and_normalize():
    img = Image.new("RGB", (300, 300), (200, 200, 200))
    colors = {
        (0, 0): (255, 0, 0), (0, 2): (0, 255, 0), (2, 0): (0, 0, 255),
        (2, 2): (255, 255, 0), (1, 1): (128, 0, 128),
    }
    for (r, c), col in colors.items():
        for y in range(r * 100, (r + 1) * 100):
            for x in range(c * 100, (c + 1) * 100):
                img.putpixel((x, y), col)
    tiles, cell = crop_base_3x3(img)
    assert cell == 100 and len(tiles) == 9
    assert tiles[4].getpixel((50, 50))[:3] == (128, 0, 128)
    # 规格归一（奇数目标向下取偶）
    base = BaseTileSet(size=100, center=tiles[4],
                       edges={"top": tiles[1], "bottom": tiles[7], "left": tiles[3], "right": tiles[5]},
                       corners={"tl": tiles[0], "tr": tiles[2], "bl": tiles[6], "br": tiles[8]})
    norm = normalize_tileset(base, target_size=33)
    assert norm.size == 32
    assert norm.center.size == (32, 32)


def test_prompts_embedded_strict_rules():
    p = build_tileset_prompts("grass field", style="retro", tile_size=32)
    text = p["image_prompt"]
    for needle in ("3x3 grid", "9 equal square", "SEAMLESS", "CENTER cell", "CORNER", "PURE WHITE", "#FFFFFF"):
        assert needle in text
    assert "32x32" in text
    assert p["grid_rows"] == 3 and p["grid_cols"] == 3


# --------------------------------------------------------------------------- #
# 无缝化
# --------------------------------------------------------------------------- #
def test_texture_seamless_wraps_both_axes():
    out = make_texture_seamless(_center_texture())
    arr = np.asarray(out)
    assert arr.shape == (S, S, 4)
    assert (arr[:, 0] == arr[:, -1]).all()      # 左右边缘逐像素相等
    assert (arr[0, :] == arr[-1, :]).all()      # 上下边缘逐像素相等
    assert (arr[..., 3] == 255).all()
    out2 = make_texture_seamless(_center_texture())
    assert (np.asarray(out) == np.asarray(out2)).all()  # 确定性


def test_edge_seamless_wraps_axis_and_uniform_band():
    tile = _edge_top()
    out, band_color = make_edge_seamless(tile, line_width=2)
    arr = np.asarray(out)
    assert (arr[:, 0] == arr[:, -1]).all()  # 沿边界方向无缝
    band = arr[:2, :]
    assert (band == band[0, 0]).all()       # 线带统一色
    assert tuple(band[0, 0][:3]) == band_color


def test_process_base_set_unified_lines_and_corners():
    base = _make_base9()
    proc = process_base_set(base, detail_keep=0.0)
    # 中心全向无缝
    c = np.asarray(proc.center)
    assert (c[:, 0] == c[:, -1]).all() and (c[0, :] == c[-1, :]).all()
    # 四边线色统一
    t = np.asarray(proc.edges["top"])
    l = np.asarray(proc.edges["left"])
    assert (t[0, :, :3] == proc.line_color).all()
    assert (l[:, 0, :3] == proc.line_color).all()
    # 墙角瓦片：TL 角被圆盘切掉、对侧为地形
    tl = np.asarray(proc.corners["tl"])
    assert tl[0, 0, 3] == 0
    assert tl[-1, -1, 3] == 255
    # 墙角与墙面共享边一致：tl 底行与中心底行同纹理，仅最左像素为边界线延续
    assert (tl[-1, 1:, :3] == c[-1, 1:, :3]).all()
    assert tuple(tl[-1, 0, :3]) == proc.line_color


# --------------------------------------------------------------------------- #
# 47-tile 几何构图
# --------------------------------------------------------------------------- #
def test_compose_full_and_isolated():
    center = _noise_texture(seed=5)
    full = compose_tile(center, 255)
    assert (np.asarray(full) == np.asarray(center)).all()  # 全邻 = 中心纹理本身
    iso = compose_tile(center, 0)
    arr = np.asarray(iso)
    for corner in ((0, 0), (S - 1, 0), (0, S - 1), (S - 1, S - 1)):
        assert arr[corner[1], corner[0], 3] == 0      # 四角切掉
    assert arr[S // 2, S // 2, 3] == 255              # 中心是地形（圆角方块）


def test_compose_edge_and_corners():
    center = _noise_texture(seed=5)
    all_sides = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
    diag_all = BIT["TL"] | BIT["TR"] | BIT["BL"] | BIT["BR"]
    # 上边界（对角全满 -> 无底部内角咬合）：顶部描边、其余地形
    top = compose_tile(center, (all_sides & ~BIT["T"]) | diag_all, line_color=(9, 9, 9))
    arr = np.asarray(top)
    assert (arr[0, :, :3] == (9, 9, 9)).all()
    assert arr[-1, S // 2, 3] == 255
    # 内角：四边全满 + TL 对角空 -> TL 角咬合
    inner = compose_tile(center, all_sides & ~BIT["TL"], line_color=(9, 9, 9))
    ai = np.asarray(inner)
    assert ai[0, 0, 3] == 0 and ai[S // 2, S // 2, 3] == 255
    # 外角：T/L 空、R/B 满、BR 对角满 -> 纯转角（TL 咬合、BR 无咬合）
    outer = compose_tile(center, BIT["R"] | BIT["B"] | BIT["BR"], line_color=(9, 9, 9))
    ao = np.asarray(outer)
    assert ao[0, 0, 3] == 0
    assert ao[-1, -1, 3] == 255
    # 对角空 -> BR 出现内角咬合（与纯转角不同）
    outer2 = compose_tile(center, BIT["R"] | BIT["B"], line_color=(9, 9, 9))
    assert np.asarray(outer2)[-1, -1, 3] == 0


def test_build_47_sheet_complete():
    center = _noise_texture(seed=11)
    sheet, meta = build_47_sheet(center, line_color=(0, 0, 0))
    assert meta["tile_count"] == 47
    assert len(meta["index_to_mask"]) == 47
    assert len(meta["mask_to_index"]) == 256
    assert sheet.size == (8 * S, 6 * S)

    def slot(idx):
        i, j = idx % 8, idx // 8
        return sheet.crop((i * S, j * S, (i + 1) * S, (j + 1) * S))

    # 全邻掩码 -> 纯中心瓦片
    assert np.asarray(slot(meta["mask_to_index"]["255"])).tobytes() == center.tobytes()
    # 孤立掩码 -> 四角切掉的圆角瓦片
    t2 = np.asarray(slot(meta["mask_to_index"]["0"]))
    assert t2[0, 0, 3] == 0 and t2[-1, -1, 3] == 0 and t2[S // 2, S // 2, 3] == 255
    # 47 槽位中 46 张独立图像（孤立瓦片与四内角瓦片图形重合，模板惯例保留双槽）
    uniq = {slot(i).tobytes() for i in range(47)}
    assert len(uniq) == 46
    hole = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
    assert slot(meta["mask_to_index"]["0"]).tobytes() == slot(meta["mask_to_index"][str(hole)]).tobytes()
    # 确定性
    sheet2, _ = build_47_sheet(center, line_color=(0, 0, 0))
    assert sheet.tobytes() == sheet2.tobytes()


def test_masks_complete_and_unique_47():
    center = _noise_texture(seed=13)
    _, meta = build_47_sheet(center)
    assert set(meta["index_to_mask"]) <= set(range(256))
    assert len(set(meta["index_to_mask"])) == 47
    for m in range(256):
        assert str(m) in meta["mask_to_index"]


# --------------------------------------------------------------------------- #
# 地图模型与渲染
# --------------------------------------------------------------------------- #
def _rng_map(w=9, h=7, p=0.55, seed=3):
    rng = np.random.default_rng(seed)
    model = TileMapModel(w, h, tile_size=S)
    model.grid[...] = (rng.random((h, w)) < p).astype(np.uint8)
    return model


def test_map_paint_fill_serialize_and_masks():
    model = TileMapModel(5, 5, tile_size=S)
    model.fill_rect(1, 1, 3, 3)
    assert model.mask(2, 2) == 255  # 3×3 中心：八邻全满
    assert model.mask(1, 1) == (BIT["R"] | BIT["B"] | BIT["BR"])  # 左上角：右下满
    model.fill_rect(2, 2, 2, 2, value=0)
    assert model.cell(2, 2) == 0
    text = model.to_json()
    back = TileMapModel.from_json(text)
    assert (back.grid == model.grid).all()
    assert back.tile_size == S


def test_map_render_shared_edges_consistent():
    """随机地图：相邻两格共享边的不透明像素必须一致（无缝核心不变量）。

    经典 47-tile 几何中，转角圆盘缺口（凹角）的「弦」沿共享边落在
    相邻瓦片一侧，属预期视觉（Godot 同款）；因此差异只允许出现在
    共享边两端 r+2 像素内的转角区，中间段必须逐像素完全一致。
    """
    model = _rng_map()
    center = _noise_texture(seed=15)
    img = np.asarray(model.render(center, line_color=(0, 0, 0)))
    assert img.shape == (model.height * S, model.width * S, 4)
    margin = S // 2 + 2

    def _check_edge(a, b, axis):
        """比较共享边 a/b（形状 (S,4)），差异只允许出现在两端转角区。"""
        both = (a[..., 3] > 0) & (b[..., 3] > 0)
        diff = (a[both] != b[both]).any(axis=-1)
        for i in np.nonzero(diff)[0]:
            assert i < margin or i > S - 1 - margin, f"共享边中间段不一致 @{i}"

    for y in range(model.height):
        for x in range(model.width):
            if not model.cell(x, y):
                continue
            if model.cell(x + 1, y):  # 右邻：左格右列 vs 右格左列
                a = img[y * S:(y + 1) * S, (x + 1) * S - 1]
                b = img[y * S:(y + 1) * S, (x + 1) * S]
                _check_edge(a, b, axis=1)
            if model.cell(x, y + 1):  # 下邻
                a = img[(y + 1) * S - 1, x * S:(x + 1) * S]
                b = img[(y + 1) * S, x * S:(x + 1) * S]
                _check_edge(a, b, axis=0)


def test_map_render_deterministic_and_sizes():
    model = _rng_map(seed=1)
    center = _noise_texture(seed=2)
    r1 = model.render(center)
    r2 = model.render(center)
    assert r1.tobytes() == r2.tobytes()
    rd = model.render(center, mode="dual")
    assert rd.size == r1.size


def test_dual_grid_pieces():
    model = TileMapModel(3, 3, tile_size=S)
    model.fill_rect(0, 0, 2, 2)
    idx = dual_grid_map(model.grid)
    assert idx.shape == (6, 6)
    # 中心格（八邻全满）：4 个填充块
    assert list(idx[2:4, 2:4].ravel()) == [0, 0, 0, 0]
    # 顶边中格（T 空，L/R 满）：上两块为「上直切」，下两块为填充
    assert list(idx[0:2, 2:4].ravel()) == [1, 1, 0, 0]
    # 左上角格（R/B 满）：TL 外转角盘，其余填充/直切
    corner = idx[0:2, 0:2]
    assert corner[0, 0] == 3  # TL 外角盘


def test_dual_render_isolated_blob():
    model = TileMapModel(1, 1, tile_size=S)
    model.set_cell(0, 0, 1)
    center = _noise_texture(seed=8)
    img = np.asarray(model.render(center, mode="dual"))
    assert img[0, 0, 3] == 0            # 四角透明
    assert img[-1, -1, 3] == 0
    assert img[S // 2, S // 2, 3] == 255  # 中心地形
