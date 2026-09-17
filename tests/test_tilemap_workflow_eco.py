"""瓦片地图工作流分类测试：地块生态（多地形 47 艺术集）/ 建筑（拼件）/ 经典回归。"""
import json

from PIL import Image

from core.api.mock_clients import MockImageAPI
from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow


def test_ground_ecosystem_end_to_end(tmp_path):
    params = TilemapParams(
        description="草地",
        style="retro",
        category="ground",
        features={"水塘": "a clear pond", "岩石": "a rocky outcrop"},
        tile_size=32,
        map_width=14,
        map_height=10,
        output_dir=tmp_path / "out",
    )
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    session = result.session
    # 1 基础 + 2 特征 = 3 套地形
    assert len(session.terrain_sets) == 3
    assert len(session.terrain_sheets) == 3
    assert len(result.terrain_atlas_paths) == 3
    for tid, path in result.terrain_atlas_paths.items():
        assert path.exists()
        meta = json.loads(path.with_name(path.stem + ".json").read_text(encoding="utf-8"))
        assert len(meta["mask_to_index"]) == 256
    # 演示地图：多地形预览 + 项目文件
    assert result.map_preview_path.exists()
    img = Image.open(result.map_preview_path)
    assert img.size == (14 * 32, 10 * 32)
    proj = json.loads(result.project_file.read_text(encoding="utf-8"))
    assert proj["category"] == "ground"
    assert proj["features"] == ["水塘", "岩石"]
    # 提示词含生态规范
    assert "2x2 grid of blocks" in session.prompts["image_prompt"]
    # 地图模型注册了多地形
    model = session.map_model
    assert set(model.terrain_sets) == {1, 2, 3}
    assert model.base_terrain == 1


def test_ground_request_size_matches_prompt(tmp_path):
    """请求边长必须 = 6 × 单格像素，且提示词里写的像素与请求一致（避免网格错位）。"""
    params = TilemapParams(
        description="草地", category="ground", features={"水塘": "pond"},
        tile_size=32, sheet_size=768, map_width=8, map_height=6,
        output_dir=tmp_path / "out",
    )
    wf = TilemapWorkflow(image_api=MockImageAPI())
    session = wf.run_to_base(params)
    assert session.cell_px == 128
    assert session.sheet_image.size == (768, 768)  # 6 × 128
    text = session.prompts["image_prompt"]
    assert "128x128 pixels" in text and "768x768 pixels" in text
    # 继续到裁切：单格应是整图 1/6（128px），随后归一化到 tile_size
    result = TilemapWorkflow(image_api=None).finish_from_base(session)
    assert result.session.ecosystem is not None
    assert result.session.ecosystem.base.size == 32
    assert result.map_preview_path.exists()


def test_classic_request_size_matches_cells(tmp_path):
    params = TilemapParams(description="grass", tile_size=32, sheet_size=768, output_dir=tmp_path / "out")
    session = TilemapWorkflow(image_api=MockImageAPI()).run_to_base(params)
    assert session.cell_px == 256           # 768 / 3
    assert session.sheet_image.size == (768, 768)


def test_building_request_size_matches_cells(tmp_path):
    params = TilemapParams(
        description="stone wall", category="building", tile_size=32, sheet_size=384,
        map_width=8, map_height=6, output_dir=tmp_path / "out",
    )
    session = TilemapWorkflow(image_api=MockImageAPI()).run_to_base(params)
    assert session.cell_px == 64
    assert session.sheet_image.size == (384, 384)


def test_building_end_to_end(tmp_path):
    params = TilemapParams(
        description="stone wall",
        category="building",
        tile_size=32,
        map_width=10,
        map_height=8,
        output_dir=tmp_path / "out",
    )
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    session = result.session
    assert session.pieces is not None
    assert set(session.pieces["pieces"]) == {"straight", "end", "corner", "pillar"}
    assert result.pieces_dir.exists()
    assert len(list(result.pieces_dir.glob("*.png"))) == 4
    assert result.map_preview_path.exists()
    proj = json.loads(result.project_file.read_text(encoding="utf-8"))
    assert proj["category"] == "building"
    assert "WALL SET" in session.prompts["image_prompt"]
    # 回归：建筑预览必须真的画出拼件（修复前 overlay 被丢弃 → 整图透明）
    import numpy as np

    preview = np.asarray(Image.open(result.map_preview_path).convert("RGBA"))
    assert (preview[..., 3] > 0).any()
    # overlay 已按名称写入地图 JSON，可恢复
    map_data = json.loads((result.output_dir / "export" / "map_demo.json").read_text(encoding="utf-8"))
    assert map_data.get("overlay"), "建筑 overlay 应被序列化"
    assert all(item.get("piece") for item in map_data["overlay"])


def test_building_sheet_self_check_warns_on_white_wall_block(tmp_path):
    """建筑底图自检：左上块中心格不是「填满整格的墙体」时给出警告（避免静默坏拼件）。"""
    from core.tilemap import opaque_ratio

    params = TilemapParams(
        description="stone wall", category="building", tile_size=32,
        map_width=8, map_height=6, output_dir=tmp_path / "out",
    )
    wf = TilemapWorkflow(image_api=None)
    session = wf.new_session(params)
    # 错误底图：左上块中心格是白底上的一根细柱（AI 把立柱组画到了左上）
    sheet = Image.new("RGB", (192, 192), (255, 255, 255))
    for y in range(70, 122):
        for x in range(88, 104):
            sheet.putpixel((x, y), (120, 120, 130))
    session.sheet_image = sheet
    wf.finish_from_base(session)
    assert any(line.startswith("[warning]") for line in wf.step_log), wf.step_log
    assert opaque_ratio(Image.new("RGB", (32, 32), (255, 255, 255))) == 0.0
    assert opaque_ratio(Image.new("RGB", (32, 32), (128, 96, 64))) == 1.0


def test_classic_path_regression(tmp_path):
    """旧链路（category 缺省 = classic）行为保持不变。"""
    params = TilemapParams(description="grass", tile_size=32, output_dir=tmp_path / "out")
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    assert result.session.processed is not None
    assert result.atlas_path.exists()
    assert len(list(result.tiles_dir.glob("*.png"))) == 9


def test_ground_edit_rerun_local(tmp_path):
    """生态模式：编辑某套地形后可在无 API 下重跑无缝/瓦片集/导出。"""
    params = TilemapParams(
        description="草地", category="ground", features={"水塘": "pond"},
        tile_size=32, map_width=10, map_height=8, output_dir=tmp_path / "out",
    )
    wf = TilemapWorkflow(image_api=MockImageAPI())
    result = wf.run(params)
    session = result.session
    # 编辑基础地形中心瓦片
    from core.tilemap import BaseTileSet

    edited_center = Image.new("RGBA", (32, 32), (10, 40, 90, 255))
    base = session.ecosystem.base
    session.ecosystem.base = BaseTileSet(
        size=base.size, center=edited_center,
        edges=base.edges, corners=base.corners,
        line_color=base.line_color, line_width=base.line_width,
    )
    wf2 = TilemapWorkflow(image_api=None)
    for name in ("seamless", "atlas", "export"):
        wf2.step(name, params, session)
    assert session.result.map_preview_path.exists()
    assert len(session.terrain_sheets) == 2


def test_two_stage_base_then_accept(tmp_path):
    """保留中间底图：run_to_base 停下（底图已存盘）→ finish_from_base 继续。"""
    params = TilemapParams(
        description="草地", category="ground", features={"水塘": "pond"},
        tile_size=32, map_width=10, map_height=8, output_dir=tmp_path / "out",
    )
    wf = TilemapWorkflow(image_api=MockImageAPI())
    session = wf.run_to_base(params)
    # 停在 base：底图已生成并保存，尚未裁切
    assert session.sheet_image is not None
    assert session.sheet_path is not None and session.sheet_path.exists()
    assert session.ecosystem is None
    assert session.result is None
    # 用户确认后继续（本地，无需 API）
    result = TilemapWorkflow(image_api=None).finish_from_base(session)
    assert result.session.ecosystem is not None
    assert result.map_preview_path.exists()
    assert len(result.terrain_atlas_paths) == 2


def test_classic_map_preview_fully_opaque(tmp_path):
    """地块类瓦片必须填充满：经典 3×3 流程的地图预览不允许出现透明像素。"""
    import numpy as np

    params = TilemapParams(description="grass", tile_size=32, output_dir=tmp_path / "out")
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    img = np.asarray(Image.open(result.map_preview_path).convert("RGBA"))
    assert (img[..., 3] == 255).all()  # 全填充、无透明楔形
    sheet = np.asarray(Image.open(result.atlas_path).convert("RGBA"))
    # 8×6=48 槽，47 张瓦片全填充，仅第 48 个空槽透明
    last_slot = sheet[5 * 32:, 7 * 32:]
    assert (last_slot[..., 3] == 0).all()
    rest = sheet.copy()
    rest[5 * 32:, 7 * 32:] = 255
    assert (rest[..., 3] == 255).all()


def _eco_sheet_with_base_at(base_pos: str, cell: int = 32):
    """6×6 生态底图：base_pos 块为纯基础地形，其余三块为「外侧 1/4 条带 + 描边」团块。"""
    import numpy as np

    base_c = (236, 240, 246)
    feat_cols = {"tl": (36, 104, 172), "tr": (120, 168, 96), "bl": (168, 152, 120), "br": (72, 120, 190)}
    pos = {"tl": (0, 0), "tr": (0, 1), "bl": (1, 0), "br": (1, 1)}
    size = cell * 6
    arr = np.zeros((size, size, 4), np.uint8)
    arr[...] = base_c + (255,)
    band = max(2, cell // 4)
    ys, xs = np.mgrid[0:cell, 0:cell]
    for key, (br, bc) in pos.items():
        if key == base_pos:
            continue
        for r in range(3):
            for c in range(3):
                outer = (
                    (r == 0) & (ys < band) | (r == 2) & (ys >= cell - band)
                    | (c == 0) & (xs < band) | (c == 2) & (xs >= cell - band)
                )
                rim = (
                    (r == 0) & (ys < band + 1) | (r == 2) & (ys >= cell - band - 1)
                    | (c == 0) & (xs < band + 1) | (c == 2) & (xs >= cell - band - 1)
                )
                col = np.where(outer[..., None], np.array(base_c + (255,), np.uint8),
                               np.where(rim[..., None], np.array((94, 84, 72, 255), np.uint8),
                                        np.array(feat_cols[key] + (255,), np.uint8)))
                y0 = (br * 3 + r) * cell
                x0 = (bc * 3 + c) * cell
                arr[y0:y0 + cell, x0:x0 + cell] = col
    return Image.fromarray(arr, "RGBA")


def test_ground_workflow_detects_base_block_position(tmp_path):
    """底图不按位置要求画（基础块在右上）时，工作流必须自动识别、日志留痕。"""
    params = TilemapParams(
        description="雪原", category="ground", features={"水塘": "a pond"},
        tile_size=32, map_width=8, map_height=6, output_dir=tmp_path / "out",
    )
    wf = TilemapWorkflow(image_api=None)
    session = wf.new_session(params)
    session.sheet_image = _eco_sheet_with_base_at("tr")
    result = wf.finish_from_base(session)
    eco = session.ecosystem
    assert eco.base_pos == "tr" and eco.base_pos_detected
    assert any("基础地形块位置" in line and "右上" in line for line in wf.step_log)
    # 基础地形必须是那块纯纹理（而不是特征团块）；地图仍能导出
    import numpy as np

    centre = np.asarray(eco.base.center.convert("RGB")).reshape(-1, 3)
    assert centre.min() > 200.0, "基础地形块应是纯基础纹理"
    assert result.map_preview_path.exists()


def test_ground_workflow_manual_base_block(tmp_path):
    """手动指定基础块位置时按指定处理，并在日志中标注「手动指定」。"""
    params = TilemapParams(
        description="雪原", category="ground", features={"水塘": "a pond"},
        tile_size=32, map_width=8, map_height=6, base_block="bl",
        output_dir=tmp_path / "out",
    )
    wf = TilemapWorkflow(image_api=None)
    session = wf.new_session(params)
    session.sheet_image = _eco_sheet_with_base_at("tr")
    wf.finish_from_base(session)
    assert session.ecosystem.base_pos == "bl"
    assert not session.ecosystem.base_pos_detected
    assert any("手动指定" in line for line in wf.step_log)
