"""素材（道具）生成链路测试：抠底、像素完美 alpha、底部对齐、并入瓦片包。"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.tilemap.pack import TilePack, load_tileset, load_tilepack, save_tilepack
from core.tilemap.props import (
    build_prop_prompts,
    fit_to_tile,
    key_background,
    process_prop_sheet,
    prop_names,
    trim_to_content,
)

S = 32


def _prop_sheet(rows=2, cols=2, cell=64, bg=(255, 255, 255)):
    """合成素材底图：每格一个「树」（三角 + 树干），底部居中，纯色底。"""
    img = Image.new("RGB", (cols * cell, rows * cell), bg)
    d = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell, r * cell
            cx = x0 + cell // 2
            top = y0 + 8
            d.polygon([(cx, top), (cx - 16, y0 + 40), (cx + 16, y0 + 40)], fill=(60, 140, 70))
            d.rectangle((cx - 4, y0 + 40, cx + 4, y0 + 56), fill=(110, 80, 50))
    return img


def test_prop_prompts_describe_grid_and_background():
    p = build_prop_prompts("a tree", style="16-bit", variants=4, cell_px=128)
    text = p["image_prompt"]
    assert "2x2 grid" in text and "128x128 pixels" in text and "256x256 pixels" in text
    assert "SOLID PURE WHITE" in text and "255,255,255" in text
    assert "NO TEXT" in text and "NO anti-aliasing" in text
    assert p["variants"] == 4 and p["grid_cols"] == 2 and p["grid_rows"] == 2
    assert "grid lines" in p["negative_prompt"]


def test_key_background_removes_border_only_and_hardens_alpha():
    img = _prop_sheet()
    keyed = key_background(img)
    arr = np.asarray(keyed)
    assert arr[..., 3][0, 0] == 0                      # 边角背景透明
    assert arr[..., 3][32, 32] == 255                  # 物件本体不透明
    assert set(np.unique(arr[..., 3]).tolist()) <= {0, 255}, "alpha 必须只有 0/255（无灰边）"
    # 内部与背景同色的区域**不能**被挖空：树干旁画一块与背景同色的矩形
    img2 = _prop_sheet()
    d = ImageDraw.Draw(img2)
    d.rectangle((28, 30, 36, 38), fill=(255, 255, 255))
    keyed2 = np.asarray(key_background(img2))
    assert keyed2[34, 32, 3] == 255, "被物件包围的同色区域不应被抠掉"


def test_trim_and_fit_bottom_aligned():
    img = key_background(_prop_sheet(rows=1, cols=1))
    trimmed = trim_to_content(img)
    assert trimmed is not None and trimmed.height < 64
    tile = fit_to_tile(img, S)
    assert tile.size == (S, S)
    arr = np.asarray(tile)
    ys, xs = np.nonzero(arr[..., 3] > 0)
    assert ys.max() == S - 1, "素材应底部对齐（站在格子上）"
    assert abs((int(xs.min()) + int(xs.max())) - (S - 1)) <= 2, "水平居中"


def test_process_prop_sheet_names_and_shapes():
    props = process_prop_sheet(_prop_sheet(), 2, 2, prop_names("tree", 4), tile_size=S)
    assert sorted(props) == ["tree_1", "tree_2", "tree_3", "tree_4"]
    for name, img in props.items():
        assert img.size == (S, S) and img.mode == "RGBA"
        alpha = np.asarray(img)[..., 3]
        assert alpha.max() == 255 and (alpha == 0).any()
    # 空格子会被跳过而不是产出空白素材
    blank = Image.new("RGB", (128, 128), (255, 255, 255))
    only = process_prop_sheet(blank, 2, 2, prop_names("tree", 4), tile_size=S)
    assert only == {}


def test_prop_pack_roundtrip(tmp_path):
    props = process_prop_sheet(_prop_sheet(), 2, 2, prop_names("tree", 4), tile_size=S)
    pack = TilePack(name="树木包", category="prop", tile_size=S, pieces=props)
    path = save_tilepack(tmp_path / "trees", pack)
    loaded = load_tilepack(path)
    assert loaded.category == "prop" and set(loaded.pieces) == set(props)
    for name, img in props.items():
        assert (np.asarray(loaded.pieces[name]) == np.asarray(img)).all()


def test_prop_workflow_end_to_end(tmp_path):
    from core.api.mock_clients import MockImageAPI
    from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow

    params = TilemapParams(
        description="a pixel tree", category="prop", prop_name="tree", prop_variants=4,
        tile_size=S, sheet_size=256, output_dir=tmp_path / "out",
    )
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    session = result.session
    assert session.props, "应产出素材"
    assert result.pieces_dir.exists() and list(result.pieces_dir.glob("*.png"))
    # 现在导出的是**完整瓦片集目录**（含 47 图集/逐张瓦片/元信息），可再次导入
    assert result.atlas_path is not None and result.atlas_path.is_dir()
    assert (result.atlas_path / "manifest.json").exists()
    assert (result.atlas_path / "atlas" / "walls_16.png").exists() or (result.atlas_path / "atlas").exists()
    pack = load_tileset(result.atlas_path)
    assert pack.category == "prop" and set(pack.pieces) == set(session.props)
    for img in session.props.values():
        alpha = np.asarray(img)[..., 3]
        assert set(np.unique(alpha).tolist()) <= {0, 255}


def _prop_sheet_custom(bg, noise=0, touch_top=False, cell=64, rows=2, cols=2):
    img = Image.new("RGB", (cols * cell, rows * cell), bg)
    d = ImageDraw.Draw(img)
    if noise:
        rng = np.random.default_rng(7)
        a = np.asarray(img).astype(int)
        a = np.clip(a + rng.integers(-noise, noise + 1, a.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(a, "RGB")
        d = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell, r * cell
            top = y0 + (0 if touch_top else 8)
            d.polygon([(x0 + cell // 2, top), (x0 + cell // 2 - 16, y0 + 40),
                       (x0 + cell // 2 + 16, y0 + 40)], fill=(60, 140, 70))
            d.rectangle((x0 + cell // 2 - 4, y0 + 40, x0 + cell // 2 + 4, y0 + 56), fill=(110, 80, 50))
    return img


@pytest.mark.parametrize("kw", [
    dict(bg=(255, 255, 255)),
    dict(bg=(255, 0, 255)),
    dict(bg=(255, 255, 255), noise=12),           # AI 常见的「接近纯色但有噪点」
    dict(bg=(255, 255, 255), touch_top=True),     # 物件压到格子边缘
])
def test_key_background_handles_noisy_and_stubborn_backgrounds(kw):
    """抠底后每格只应剩下物件本体（不透明占比明显小于半格）。"""
    props = process_prop_sheet(_prop_sheet_custom(**kw), 2, 2, prop_names("tree", 4), tile_size=S)
    assert len(props) == 4
    for name, img in props.items():
        alpha = np.asarray(img)[..., 3]
        ratio = float((alpha > 0).mean())
        assert ratio < 0.6, f"{name} 抠底不干净：不透明占比 {ratio:.2f}"
        assert set(np.unique(alpha).tolist()) <= {0, 255}


def test_key_background_global_fallback_when_object_matches_background():
    """背景与物件几乎同色时也不能留下一整块底色（全局兜底）。"""
    img = Image.new("RGB", (64, 64), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle((8, 8, 56, 56), fill=(248, 248, 248))     # 只差 2 的「物件」
    keyed = key_background(img)
    ratio = float((np.asarray(keyed)[..., 3] > 0).mean())
    assert ratio < 0.95, f"仍未抠净：{ratio:.2f}"
