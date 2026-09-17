"""底图清理与对齐式拼接的回归测试（第 8 轮）。

覆盖用户反馈的三个问题：
1. 底图上每格的深色线框 → `strip_grid_frames` 自动抹除；
2. 底图上出现文字/水印 → 强提示词 + `detect_text_marks` 检测 + 纹理修补；
3. 47 拼接衔接错乱 → 对齐式构图（AI 纹理 + 程序化几何），共享边逐像素相等。
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from core.tilemap import BIT, align_terrain_set, compose_art_tile, mask_for_terrain
from core.tilemap.cleanup import detect_text_marks, patch_marks, strip_grid_frames
from core.tilemap.seamless import make_tile_texture, median_tile_texture
from core.tilemap.tiles import BaseTileSet
from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow

S = 32
CELL = 96
BASE_C = (236, 240, 246)
FEAT_C = (38, 104, 172)
RIM_C = (81, 75, 66)
FRAME_C = (54, 50, 60)


# --------------------------------------------------------------------------- #
# 1) 格线框抹除
# --------------------------------------------------------------------------- #
def _framed_sheet(cell: int = CELL, frame_px: int = 2, cells: int = 6, frame: bool = True):
    """6×6 格底图；frame=True 时每格边缘画深色线框（AI 常见毛病）。"""
    size = cell * cells
    arr = np.zeros((size, size, 4), np.uint8)
    arr[..., :3] = BASE_C
    arr[..., 3] = 255
    ys, xs = np.mgrid[0:cell, 0:cell]
    for r in range(cells):
        for c in range(cells):
            patch_col = FEAT_C if (r < 3 and c < 3) else BASE_C
            arr[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell, :3] = patch_col
    if frame:
        line = np.array(FRAME_C + (255,), np.uint8)
        for k in range(cells + 1):
            p = min(size - 1, k * cell)
            for w in range(frame_px):
                q = min(size - 1, p + w)
                arr[q, :] = line
                arr[:, q] = line
    return Image.fromarray(arr, "RGBA")


def test_strip_grid_frames_removes_only_the_lines():
    framed = _framed_sheet(frame=True)
    cleaned, report = strip_grid_frames(framed, cell=CELL, rows=6, cols=6)
    assert report["count"] >= 10, report          # 5 条竖线 + 5 条横线（含外缘）
    assert all(1 <= d["width"] <= 4 for d in report["vertical"] + report["horizontal"])
    a = np.asarray(framed)[..., :3].astype(int)
    b = np.asarray(cleaned)[..., :3].astype(int)
    ref = np.asarray(_framed_sheet(frame=False))[..., :3].astype(int)
    assert (a != b).any(axis=2).sum() > 0
    # 1) 线框颜色必须彻底消失（不再有规律性深色网格）
    assert not (np.abs(b - np.array(FRAME_C)).max(axis=2) < 12).any()
    # 2) 抹除只改动线框所在的行/列，其余像素逐像素不变
    lines = set()
    for d in report["vertical"]:
        lines |= {(y, x) for y in range(a.shape[0]) for x in range(d["x0"], d["x0"] + d["width"])}
    for d in report["horizontal"]:
        lines |= {(y, x) for x in range(a.shape[1]) for y in range(d["x0"], d["x0"] + d["width"])}
    diff = (a != b).any(axis=2)
    outside = diff.copy()
    for (y, x) in lines:
        outside[y, x] = False
    assert outside.sum() == 0, "线框以外的像素不允许被改动"
    # 3) 平坦区域必须与「本来就没有线框」的底图一致
    flat = np.zeros_like(diff)
    flat[10:80, 300:370] = True      # 基础地形块内部
    flat[300:370, 10:80] = True      # 特征块内部
    assert np.abs(b[flat] - ref[flat]).max() <= 1


def test_strip_grid_frames_keeps_clean_sheet_untouched():
    clean = _framed_sheet(frame=False)
    cleaned, report = strip_grid_frames(clean, cell=CELL, rows=6, cols=6)
    assert report["count"] == 0
    assert (np.asarray(clean) == np.asarray(cleaned)).all()


# --------------------------------------------------------------------------- #
# 2) 文字 / 水印检测与修补
# --------------------------------------------------------------------------- #
def _text_cell(text: str, size: int = 22, col=(60, 60, 60)) -> Image.Image:
    img = Image.new("RGB", (CELL, CELL), BASE_C)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
    except OSError:  # 无中文字体时退回默认字体（拉丁字母仍可测）
        font = ImageFont.load_default()
    d.text((14, 34), text, fill=col, font=font)
    return img


@pytest.mark.parametrize("text", ["草地", "GRASS", "water", "12"])
def test_detect_text_marks_finds_text(text):
    assert detect_text_marks(_text_cell(text)), f"应检测到文字: {text}"


def test_detect_text_marks_ignores_art():
    """岩石、草丛、贴花、碎石、纹理噪点都不能被误判成文字。"""
    rock = Image.new("RGB", (CELL, CELL), BASE_C)
    d = ImageDraw.Draw(rock)
    d.ellipse((24, 24, 72, 72), fill=(120, 110, 96))
    d.ellipse((10, 12, 22, 22), fill=(150, 140, 120))
    assert detect_text_marks(rock) == []

    tufts = Image.new("RGB", (CELL, CELL), BASE_C)
    d = ImageDraw.Draw(tufts)
    for i in range(14):
        x = 6 + (i * 7) % 84
        y = 8 + (i * 13) % 80
        d.line((x, y, x, y + 5), fill=(110, 150, 90))
    assert detect_text_marks(tufts) == []

    decals = Image.new("RGB", (CELL, CELL), BASE_C)
    d = ImageDraw.Draw(decals)
    for cx, cy in ((30, 40), (48, 42), (66, 39)):
        d.polygon([(cx, cy - 7), (cx + 6, cy), (cx, cy + 7), (cx - 6, cy)], fill=(90, 130, 180))
    assert detect_text_marks(decals) == []

    rng = np.random.default_rng(5)
    speckle = np.clip(np.array(BASE_C) + rng.integers(-8, 9, (CELL, CELL, 1)), 0, 255).astype(np.uint8)
    assert detect_text_marks(Image.fromarray(speckle, "RGB")) == []


def test_patch_marks_removes_text_pixels():
    img = _text_cell("草地")
    boxes = detect_text_marks(img)
    patched = np.asarray(patch_marks(img, boxes))[..., :3]
    original = np.asarray(img)
    assert (patched != original).any()
    # 文字色（深灰）应不再出现在被覆盖区域
    (x0, y0, x1, y1) = boxes[0]
    region = patched[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
    assert (region < 100).sum() == 0


def test_median_texture_erases_text_drawn_in_one_cell():
    """九格逐像素中位数：只画在某一格里的文字会被投票掉（基础地形纹理的兜底）。"""
    plain = Image.new("RGBA", (CELL, CELL), BASE_C + (255,))
    cells = [plain] * 9
    cells[4] = _text_cell("草地").convert("RGBA")
    tex = np.asarray(median_tile_texture(cells, S, inset_frac=0.0).convert("RGB"))
    assert (tex < 120).sum() == 0, "文字必须在中位数纹理里消失"
    ref = np.asarray(median_tile_texture([plain] * 9, S, inset_frac=0.0).convert("RGB"))
    assert np.abs(tex.astype(int) - ref.astype(int)).max() <= 6


# --------------------------------------------------------------------------- #
# 3) 对齐式拼接：AI 纹理 + 程序化几何 → 共享边逐像素相等
# --------------------------------------------------------------------------- #
def _cell(col, checker=0, size=CELL):
    a = np.zeros((size, size, 4), np.uint8)
    a[..., :3] = col
    a[..., 3] = 255
    if checker:
        ys, xs = np.mgrid[0:size, 0:size]
        a[((ys // 6 + xs // 6) % 2 == 0), :3] = np.clip(np.array(col) + checker, 0, 255)
    return Image.fromarray(a, "RGBA")


def _aligned_art(col, checker, band=8, rim=2):
    ground = median_tile_texture([_cell(BASE_C, 10) for _ in range(9)], S)
    tex = make_tile_texture(_cell(col, checker), S)
    return BaseTileSet(
        size=S, center=tex,
        edges={n: tex for n in ("top", "bottom", "left", "right")},
        corners={n: tex for n in ("tl", "tr", "bl", "br")},
        line_color=RIM_C, line_width=rim, band=band, radius=band, base_texture=ground,
    )


def test_aligned_composition_has_wrap_equal_textures():
    art = _aligned_art(FEAT_C, 14)
    tex = np.asarray(art.center)
    ground = np.asarray(art.base_texture)
    for img in (tex, ground):
        assert (img[:, 0] == img[:, -1]).all() and (img[0, :] == img[-1, :]).all()


def test_aligned_seams_only_differ_in_corner_zones():
    """随机多地形地图：所有共享边在「非转角带」内必须逐像素相等。"""
    arts = {
        1: _aligned_art(BASE_C, 10),
        2: _aligned_art(FEAT_C, 14),
        3: _aligned_art((168, 152, 120), 16),
    }
    band = arts[1].band
    rng = np.random.default_rng(11)
    H, W = 9, 11
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
            tiles[(x, y)] = np.asarray(compose_art_tile(arts[int(grid[y, x])], mask))
    corner = band + 2
    problems = []
    for y in range(H):
        for x in range(W - 1):
            left, right = tiles[(x, y)][:, S - 1], tiles[(x + 1, y)][:, 0]
            rows = [r for r in np.nonzero((left != right).any(axis=1))[0] if corner <= r < S - corner]
            if rows:
                problems.append((x, y, rows[:4]))
    for y in range(H - 1):
        for x in range(W):
            top, bot = tiles[(x, y)][S - 1, :], tiles[(x, y + 1)][0, :]
            cols = [c for c in np.nonzero((top != bot).any(axis=1))[0] if corner <= c < S - corner]
            if cols:
                problems.append((x, y, cols[:4]))
    assert not problems, f"共享边（非转角带）存在像素差: {problems[:5]}"


def test_aligned_band_comes_from_measured_ai_layout():
    """条带深度实测自 AI 底图：1/4 深 + 1/16 描边 → 32px 瓦片为 band=10、rim=2。"""
    cell = 64
    band_cell, rim_cell = 16, 4
    tiles = {}
    for name in ("tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"):
        arr = np.zeros((cell, cell, 4), np.uint8)
        arr[..., :3] = FEAT_C
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
            arr[band] = BASE_C + (255,)
            arr[rim & ~band] = RIM_C + (255,)
        tiles[name] = Image.fromarray(arr, "RGBA")
    raw = BaseTileSet(
        size=cell, center=tiles["center"],
        edges={n: tiles[n] for n in ("top", "bottom", "left", "right")},
        corners={n: tiles[n] for n in ("tl", "tr", "bl", "br")},
    )
    ground = np.array(BASE_C, dtype=np.float32)
    art = align_terrain_set(raw, base_texture=median_tile_texture([_cell(BASE_C)] * 9, S),
                            tile_size=S, ground_rgb=ground)
    assert art.band == 10 and art.line_width == 2, art.art_meta
    diag = BIT["TL"] | BIT["TR"] | BIT["BL"] | BIT["BR"]
    sides = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
    tile = np.asarray(compose_art_tile(art, (sides & ~BIT["T"]) | diag))
    assert (tile[:8, :, :3] == np.asarray(art.base_texture)[:8, :, :3]).all()
    assert tuple(tile[9, S // 2, :3]) == RIM_C


# --------------------------------------------------------------------------- #
# 4) 工作流级：底图清理 + 文字告警
# --------------------------------------------------------------------------- #
def _ground_params(tmp_path):
    return TilemapParams(
        description="雪原", category="ground", features={"水塘": "a pond"},
        tile_size=32, map_width=8, map_height=6, output_dir=tmp_path / "out",
    )


def test_workflow_strips_frames_and_saves_clean_sheet(tmp_path):
    params = _ground_params(tmp_path)
    wf = TilemapWorkflow(image_api=None)
    session = wf.new_session(params)
    session.sheet_image = _framed_sheet()
    wf.finish_from_base(session)
    assert session.frame_report["count"] >= 10
    assert session.sheet_clean_path is not None and session.sheet_clean_path.exists()
    assert any("格线框" in line for line in wf.step_log)
    # 抹除动作确实发生在裁切之前：成品瓦片里不应出现线框颜色
    sheet, meta = session.terrain_sheets[1]
    arr = np.asarray(sheet.convert("RGB")).reshape(-1, 3)
    assert not (np.abs(arr.astype(int) - np.array(FRAME_C)).max(axis=1) < 6).any()


def test_workflow_warns_and_patches_text_on_sheet(tmp_path):
    """底图上写了字：记录报告 + warning + 纹理修补（不再整张平铺文字）。"""
    params = _ground_params(tmp_path)
    sheet = _framed_sheet(frame=False)
    # 把文字画进基础块（左上块）的中心格
    text = _text_cell("草地").convert("RGBA")
    sheet.paste(text, (CELL, CELL))
    wf = TilemapWorkflow(image_api=None)
    session = wf.new_session(params)
    session.sheet_image = sheet
    wf.finish_from_base(session)
    assert session.text_report, "应记录疑似文字"
    assert any(line.startswith("[warning]") for line in wf.step_log)
    tex = np.asarray(session.terrain_sets[1].center.convert("RGB"))
    assert (tex < 120).sum() == 0, "修补后纹理里不应残留文字像素"
