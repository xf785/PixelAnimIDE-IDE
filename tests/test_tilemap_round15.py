"""本轮新功能测试：参考图 i2i、2.5D 崖壁高度层、瓦片包保存文生图底图。"""
import json

import numpy as np
import pytest
from PIL import Image

from core.tilemap import TileMapModel
from core.tilemap.cliff import (
    build_cliff_set,
    cliff_art_from_terrain,
    cliff_bits_for,
    compose_cliff_piece,
)
from core.tilemap.seamless import make_tile_texture
from core.tilemap.tiles import EDGE_NAMES, BaseTileSet

S = 32


def _terrain(col=(80, 150, 90)):
    a = np.zeros((96, 96, 4), np.uint8)
    a[..., :3] = col
    a[..., 3] = 255
    for y in range(0, 96, 12):
        a[y:y + 2, :, :3] = np.clip(np.array(col) - 28, 0, 255)
    tex = make_tile_texture(Image.fromarray(a, "RGBA"), S)
    ground = Image.new("RGBA", (S, S), (60, 120, 70, 255))
    return BaseTileSet(size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
                       corners={n: tex for n in ("tl", "tr", "bl", "br")},
                       line_color=(24, 22, 24), line_width=2, band=8, radius=8,
                       base_texture=ground,
                       art_meta={"outline": [[24, 22, 24], [150, 152, 150]], "bevel": [],
                                 "outline_px": 2, "bevel_px": 0, "edge_noise_px": 2})


# --------------------------------------------------------------------------- #
# 1) 2.5D 崖壁
# --------------------------------------------------------------------------- #
def test_cliff_art_is_derived_from_terrain_and_family_complete():
    art = cliff_art_from_terrain(_terrain(), edge_noise=2)
    assert art.art_meta["family"] == "cliff-16"
    assert 4 <= art.face_h <= S
    face = np.asarray(art.face)
    top = np.asarray(art.texture)
    # 立面比顶面暗（同源配色推导）
    assert face[..., :3].mean() < top[..., :3].mean() * 0.9
    pieces = build_cliff_set(art)
    assert len(pieces) == 16
    for name, img in pieces.items():
        assert img.size == (S, S) and img.mode == "RGBA"


def test_cliff_piece_occupies_top_part_and_is_transparent_below():
    art = cliff_art_from_terrain(_terrain())
    arr = np.asarray(compose_cliff_piece(art, 0))
    alpha = arr[..., 3]
    rows = np.nonzero(alpha.any(axis=1))[0]
    assert rows.size and rows[-1] < S - 2, "崖壁只占上半部，下半部应透明（露出低地）"
    assert rows[0] <= 2
    # 下半部分完全透明
    assert (alpha[art.face_h + 2:, :] == 0).all()
    assert set(np.unique(alpha).tolist()) <= {0, 255}
    # 左右端头收口：无相邻崖壁时内侧留出侧壁
    end = np.asarray(compose_cliff_piece(art, 0))                    # 孤立：左右都收口
    run = np.asarray(compose_cliff_piece(art, W_E | W_W))            # 左右相连：贯通
    assert (end[..., 3] > 0).sum() < (run[..., 3] > 0).sum()


W_E, W_W = 2, 8


def test_height_layer_paints_and_renders_cliff_below_plateau():
    art = cliff_art_from_terrain(_terrain())
    model = TileMapModel(8, 6, tile_size=S)
    model.set_terrain(1, _terrain())
    model.base_terrain = 1
    model.fill_rect(0, 0, 7, 5, 1)
    flat = np.asarray(model.render())
    model.enable_height_layer({1: art})
    for x in range(2, 6):
        for y in range(2, 4):
            model.paint_height(x, y, 1)
    raised = np.asarray(model.render())
    assert not (flat == raised).all(), "抬高后画面应有变化"
    # 崖壁画在**低地格**（y=4）的上半部
    band = raised[4 * S:4 * S + art.face_h, 2 * S:6 * S]
    assert band[..., 3].min() == 255, "高台南侧的低地格上半部应被崖壁覆盖"
    assert (raised[5 * S:, :, :] == flat[5 * S:, :, :]).all(), "更南侧的行不应受影响"
    # 崖壁按相邻崖壁自动选型（左右相连 -> 无内端头）
    cells = model.cliff_cells()
    assert set(cells) == {(x, 4) for x in range(2, 6)}
    assert cliff_bits_for(cells, 2, 4) & 2 and cliff_bits_for(cells, 2, 4) & 8 == 0   # 有东邻、无西邻
    # 序列化往返
    data = model.to_dict()
    assert "height_grid" in data
    again = TileMapModel.from_dict(data)
    assert (again.height_grid == model.height_grid).all()
    again.set_terrain(1, _terrain())          # 地形艺术是运行时对象，需重新绑定
    again.enable_height_layer({1: art})
    assert (np.asarray(again.render()) == raised).all()


def test_height_painting_clamps_and_clears():
    art = cliff_art_from_terrain(_terrain())
    model = TileMapModel(4, 4, tile_size=S)
    model.set_terrain(1, _terrain())
    model.base_terrain = 1
    model.enable_height_layer({1: art})
    for _ in range(9):
        model.paint_height(1, 1, 1)
    assert int(model.height_grid[1, 1]) == 4          # 上限 4
    for _ in range(9):
        model.paint_height(1, 1, -1)
    assert int(model.height_grid[1, 1]) == 0          # 下限 0


def test_workflow_builds_cliff_arts_and_demo_plateau(tmp_path):
    from core.api.mock_clients import MockImageAPI
    from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow

    params = TilemapParams(description="grass", category="ground", features={"水": "pond"},
                           tile_size=S, sheet_size=384, output_dir=tmp_path / "out",
                           map_width=14, map_height=10, terrain_25d=True)
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    session = result.session
    assert session.cliff_arts, "应推导出崖壁艺术"
    assert session.map_model.height_grid is not None and session.map_model.height_grid.any(), "演示地图应有高台"
    assert result.map_preview_path.exists()
    # 导出目录里含崖壁图集与文生图底图
    ts = tmp_path / "out" / "export" / "tileset"
    assert (ts / "source" / "ai_sheet.png").exists(), "瓦片包应保存文生图 2×2 底图"
    assert (ts / "source" / "ai_sheet_clean.png").exists()


# --------------------------------------------------------------------------- #
# 2) 参考图（图生图）
# --------------------------------------------------------------------------- #
def test_reference_image_is_forwarded_to_image_api(tmp_path):
    from core.api.mock_clients import MockImageAPI
    from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow

    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (200, 120, 60)).save(ref)
    seen = {}

    class SpyAPI(MockImageAPI):
        def call(self, prompt, **kwargs):          # type: ignore[override]
            seen["image"] = kwargs.get("image")
            return super().call(prompt, **kwargs)

    params = TilemapParams(description="grass", category="ground", tile_size=S,
                           sheet_size=384, output_dir=tmp_path / "out",
                           reference_image=str(ref))
    TilemapWorkflow(image_api=SpyAPI()).run_to_base(params)
    assert seen.get("image"), "参考图字节应传给生图接口（i2i）"
    assert Image.open(__import__("io").BytesIO(seen["image"])).size == (64, 64)


def test_missing_reference_image_reports_error(tmp_path):
    from core.api.mock_clients import MockImageAPI
    from core.workflow.tilemap_workflow import (
        TilemapParams,
        TilemapWorkflow,
        WorkflowError,
    )

    params = TilemapParams(description="grass", category="ground", tile_size=S,
                           sheet_size=384, output_dir=tmp_path / "out",
                           reference_image=str(tmp_path / "nope.png"))
    with pytest.raises(WorkflowError):
        TilemapWorkflow(image_api=MockImageAPI()).run_to_base(params)


# --------------------------------------------------------------------------- #
# 3) 瓦片包保存文生图底图
# --------------------------------------------------------------------------- #
def test_pack_and_export_include_ai_sheet(tmp_path):
    from core.tilemap.pack import TilePack, export_tileset_dir, load_tileset

    sheet = Image.new("RGBA", (192, 192), (255, 255, 255, 255))
    clean = Image.new("RGBA", (192, 192), (250, 250, 250, 255))
    pack = TilePack(name="草地", category="ground", tile_size=S, terrains={1: _terrain()},
                    terrain_names={1: "草地"}, base_terrain=1,
                    sheets={"ai_sheet": sheet, "ai_sheet_clean": clean})
    paths = export_tileset_dir(tmp_path / "grass", pack)
    src = paths["dir"] / "source"
    assert (src / "ai_sheet.png").exists() and (src / "ai_sheet_clean.png").exists()
    info = json.loads((paths["dir"] / "map" / "info.json").read_text(encoding="utf-8"))
    assert set(info["source_sheets"]) == {"ai_sheet", "ai_sheet_clean"}
    man = json.loads((paths["dir"] / "manifest.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in man["sheets"]] == ["ai_sheet", "ai_sheet_clean"]
    # 从目录 / zip 都能读回底图
    for target in (paths["dir"], paths["zip"]):
        back = load_tileset(target)
        assert set(back.sheets) == {"ai_sheet", "ai_sheet_clean"}
        assert (np.asarray(back.sheets["ai_sheet"]) == np.asarray(sheet)).all()
