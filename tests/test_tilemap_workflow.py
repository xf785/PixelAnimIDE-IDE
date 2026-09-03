"""瓦片地图工作流测试（mock API 端到端 + 手动步骤 + 编辑后本地重跑）。"""
import json

import numpy as np
import pytest
from PIL import Image

from core.api.mock_clients import MockImageAPI
from core.tilemap import BaseTileSet, TileMapModel
from core.workflow.solo_workflow import WorkflowError
from core.workflow.tilemap_workflow import TILEMAP_STEPS, TilemapParams, TilemapWorkflow


def make_params(tmp_path, mode="47", tile_size=32, map_w=12, map_h=8):
    return TilemapParams(
        description="grass field",
        style="retro",
        tile_size=tile_size,
        sheet_size=768,
        atlas_mode=mode,
        map_width=map_w,
        map_height=map_h,
        output_dir=tmp_path / "out",
    )


def test_run_end_to_end_47(tmp_path):
    wf = TilemapWorkflow(image_api=MockImageAPI())
    result = wf.run(make_params(tmp_path, "47"))
    assert result.session is not None
    assert result.session.processed is not None
    assert result.session.atlas_meta["tile_count"] == 47
    assert result.atlas_path.exists()
    assert result.atlas_meta_path.exists()
    assert len(list(result.tiles_dir.glob("*.png"))) == 9
    assert result.map_preview_path.exists()
    assert result.project_file.exists()

    meta = json.loads(result.atlas_meta_path.read_text(encoding="utf-8"))
    assert len(meta["mask_to_index"]) == 256
    proj = json.loads(result.project_file.read_text(encoding="utf-8"))
    assert proj["atlas_mode"] == "47"
    assert proj["tile_size"] == 32

    img = Image.open(result.map_preview_path)
    assert img.size == (12 * 32, 8 * 32)
    # 提示词已嵌入严格规范
    assert "3x3 grid" in result.session.prompts["image_prompt"]
    assert "SEAMLESS" in result.session.prompts["image_prompt"]


def test_run_end_to_end_dual(tmp_path):
    wf = TilemapWorkflow(image_api=MockImageAPI())
    result = wf.run(make_params(tmp_path, "dual"))
    assert result.session.atlas_meta["piece_count"] == 16
    assert result.session.atlas_meta["piece_size"] == 16
    assert result.atlas_path.exists()
    img = Image.open(result.map_preview_path)
    assert img.size == (12 * 32, 8 * 32)


def test_manual_steps_and_edit_rerun(tmp_path):
    params = make_params(tmp_path)
    wf = TilemapWorkflow(image_api=MockImageAPI())
    session = wf.new_session(params)
    for name in TILEMAP_STEPS:
        wf.step(name, params, session)
    assert session.max_done == len(TILEMAP_STEPS) - 1
    assert session.result is not None

    # 编辑中心瓦片后本地重跑（无需 API）：新的中心纹理应全向无缝
    edited_center = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    session.base = BaseTileSet(
        size=session.base.size,
        center=edited_center,
        edges=session.base.edges,
        corners=session.base.corners,
        line_color=session.base.line_color,
        line_width=session.base.line_width,
    )
    wf2 = TilemapWorkflow(image_api=None)
    for name in ("seamless", "atlas", "export"):
        wf2.step(name, params, session)
    c = np.asarray(session.processed.center)
    assert (c[:, 0] == c[:, -1]).all() and (c[0, :] == c[-1, :]).all()
    assert session.result.atlas_path.exists()


def test_step_order_guards(tmp_path):
    wf = TilemapWorkflow(image_api=None)
    session = wf.new_session(make_params(tmp_path))
    with pytest.raises(WorkflowError):
        wf.step("crop", session.params, session)  # 未生成底图
    with pytest.raises(WorkflowError):
        wf.step("base", session.params, session)  # 未提供 image_api


def test_run_without_image_api_fails(tmp_path):
    wf = TilemapWorkflow(image_api=None)
    with pytest.raises(WorkflowError):
        wf.run(make_params(tmp_path))


def test_map_model_in_export_roundtrip(tmp_path):
    wf = TilemapWorkflow(image_api=MockImageAPI())
    result = wf.run(make_params(tmp_path))
    model = result.session.map_model
    assert isinstance(model, TileMapModel)
    back = TileMapModel.from_json((result.output_dir / "export" / "map_demo.json").read_text(encoding="utf-8"))
    assert (back.grid == model.grid).all()
