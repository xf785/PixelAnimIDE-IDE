"""瓦片地图工作流（第 5 模式）：文生瓦片集 → 裁切 → 无缝化 → 47/双网格 → 导出。

步骤（手动模式执行顺序）：
prompts  → 内置严格瓦片集提示词（嵌入纹理/风格描述，无需 LLM）
base     → 文生 3×3 瓦片集底图
crop     → 自适应裁切为 9 张瓦片（3×3 九宫格）
seamless → 无缝化处理（中心全向 / 墙面轴向 + 统一线色 / 转角推导）
atlas    → 47-tile 瓦片集 或 双网格四分之一块集
export   → 瓦片 PNG、瓦片集图 + 掩码映射 JSON、演示地图预览、项目 JSON

编辑瓦片（UI 步骤）：在 crop 之后对 base 的任意瓦片重绘，然后重跑
seamless/atlas/export 即可（这几步纯本地算法，不需要 API）。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from core.api.base import BaseAPI
from core.tilemap import (
    BaseTileSet,
    TileMapModel,
    build_47_sheet,
    crop_base_3x3,
    normalize_tileset,
    process_base_set,
)
from core.tilemap.autotile import build_dual_pieces_sheet
from core.tilemap.prompts import build_tileset_prompts
from core.tilemap.tiles import to_base_set
from core.workflow.solo_workflow import WorkflowError
from ui.i18n import tr

logger = logging.getLogger("PixelAnimIDE.workflow.tilemap")

TILEMAP_STEPS = ("prompts", "base", "crop", "seamless", "atlas", "export")

TILEMAP_STEP_LABELS = {
    "prompts": "瓦片提示词",
    "base": "生成瓦片底图",
    "crop": "裁切瓦片",
    "seamless": "无缝化处理",
    "atlas": "生成瓦片集",
    "export": "导出",
}


@dataclass
class TilemapParams:
    """瓦片地图模式输入参数。"""

    description: str
    style: str = "game sprite"       # 风格描述（嵌入严格提示词）
    tile_size: int = 32              # 目标单格像素（偶数）
    sheet_size: int = 768            # 生图请求边长（3 格总边长）
    atlas_mode: str = "47"           # "47" | "dual"
    line_width: int = 1              # 边界线宽（像素）
    detail_keep: float = 0.3         # AI 转角内部细节混合比例（0~1）
    map_width: int = 14              # 演示地图宽度（格）
    map_height: int = 10             # 演示地图高度（格）
    output_dir: Path = field(default_factory=lambda: Path("output"))


@dataclass
class TilemapSession:
    """瓦片地图逐步执行的中间状态（手动模式会话）。"""

    params: TilemapParams
    prompts: Optional[dict] = None                 # 严格提示词
    sheet_image: Optional[Image.Image] = None      # 生图底图（3×3 整图）
    sheet_path: Optional[Path] = None
    base: Optional[BaseTileSet] = None             # 裁切后的原始 9 片（可编辑）
    processed: Optional[BaseTileSet] = None        # 无缝化处理后的 9 片
    atlas_sheet: Optional[Image.Image] = None      # 47 集图 或 双网格块集图
    atlas_meta: Optional[dict] = None
    map_model: Optional[TileMapModel] = None       # 演示地图
    result: Optional["TilemapResult"] = None
    max_done: int = -1                             # 已完成步骤号（TILEMAP_STEPS 下标）


@dataclass
class TilemapResult:
    """瓦片地图工作流输出。"""

    output_dir: Path
    session: Optional[TilemapSession] = None
    sheet_path: Optional[Path] = None
    tiles_dir: Optional[Path] = None
    atlas_path: Optional[Path] = None
    atlas_meta_path: Optional[Path] = None
    map_preview_path: Optional[Path] = None
    project_file: Optional[Path] = None
    tile_size: int = 0
    atlas_mode: str = "47"
    step_log: List[str] = field(default_factory=list)


def _demo_map(model: TileMapModel) -> None:
    """默认演示地形：一块带缺口的区域（能展示内外角/边界自动衔接）。"""
    w, h = model.width, model.height
    x0, y0 = max(1, w // 4), max(1, h // 5)
    x1, y1 = w - w // 4 - 1, h - h // 5 - 1
    model.fill_rect(x0, y0, x1, y1, 1)
    # 挖出右下一角（L 形缺口，展示内角咬合）
    model.fill_rect(x1 - 1, y1 - 1, x1, y1, 0)
    # 一块孤立地形（展示孤立 blob）
    if w > 8 and h > 8:
        model.set_cell(w // 2, 1, 1)
        model.set_cell(w // 2, 2, 1)


class TilemapWorkflow:
    """瓦片地图流程执行器（纯同步，可在 QThread 中运行）。"""

    def __init__(
        self,
        image_api: Optional[BaseAPI] = None,
        log: Optional[Callable[[str, str], None]] = None,
        cancel: Optional[threading.Event] = None,
    ):
        self.image_api = image_api
        self._log = log
        self._cancel = cancel or threading.Event()
        self.step_log: List[str] = []

    # ------------------------------------------------------------------ #
    def _log_msg(self, level: str, message: str) -> None:
        self.step_log.append(f"[{level}] {message}")
        logger.log(getattr(logging, level.upper(), logging.INFO), "%s", message)
        if self._log:
            try:
                self._log(level, message)
            except Exception:  # noqa: BLE001
                pass

    def _check_cancel(self) -> None:
        from core.workflow.solo_workflow import WorkflowCancelled

        if self._cancel.is_set():
            raise WorkflowCancelled()

    def new_session(self, params: TilemapParams) -> TilemapSession:
        return TilemapSession(params=params)

    def run(self, params: TilemapParams) -> TilemapResult:
        """自动模式：无干涉按顺序执行全部步骤。"""
        session = self.new_session(params)
        for name in TILEMAP_STEPS:
            self._check_cancel()
            self.step(name, params, session)
        if session.result is None:
            raise WorkflowError("瓦片地图流程未产出结果", step="导出")
        return session.result

    def step(self, name: str, params: TilemapParams, session: TilemapSession) -> None:
        """执行单个步骤（手动模式逐步调用；run() 内部依次调用）。"""
        if name not in TILEMAP_STEPS:
            raise WorkflowError(f"未知瓦片地图步骤: {name}", step=name)
        fn = getattr(self, f"_do_{name}")
        self._check_cancel()
        fn(params, session)
        session.max_done = max(session.max_done, TILEMAP_STEPS.index(name))

    # ------------------------------------------------------------------ #
    def _do_prompts(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 1/6：内置严格瓦片集提示词（嵌入用户纹理/风格描述）。"""
        session.prompts = build_tileset_prompts(
            params.description, style=params.style, tile_size=params.tile_size
        )
        self._log_msg("info", tr("瓦片集提示词已生成（内置严格 3×3 规范）"))

    def _do_base(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 2/6：文生 3×3 瓦片集底图。"""
        if self.image_api is None:
            raise WorkflowError("未提供图片 API，无法生成底图", step="base")
        size = max(3 * params.tile_size, min(int(params.sheet_size), 2048))
        result = self.image_api.call(
            prompt=session.prompts["image_prompt"],
            size=f"{size}x{size}",
            n=1,
            negative_prompt=session.prompts.get("negative_prompt"),
        )
        if not result.ok:
            raise WorkflowError(tr("瓦片底图生成失败: {0}").format(result.message), step="base")
        images = (result.data or {}).get("images") or []
        urls = (result.data or {}).get("urls") or []
        if images:
            data = images[0]
        elif urls:
            from core.processing import frame_utils as fu

            data = fu.download_bytes(urls[0])
        else:
            raise WorkflowError("生图接口未返回任何图片", step="base")
        session.sheet_image = _bytes_to_image(data)
        out = Path(params.output_dir) / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        session.sheet_path = out / "tileset_sheet.png"
        session.sheet_image.save(session.sheet_path)
        self._log_msg("info", tr("瓦片底图已生成: {0}").format(session.sheet_path))

    def _do_crop(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 3/6：自适应裁切 3×3 并归一化到目标瓦片尺寸。"""
        if session.sheet_image is None:
            raise WorkflowError("尚未生成底图，请先执行上一步", step="crop")
        tiles, cell = crop_base_3x3(session.sheet_image)
        base = to_base_set(tiles)
        base = normalize_tileset(base, target_size=params.tile_size)
        session.base = base
        self._log_msg(
            "info",
            tr("已裁切 {0} 张瓦片（底图单格 {1}px → 目标 {2}px）").format(9, cell, base.size),
        )

    def _do_seamless(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 4/6：无缝化处理（中心全向 / 墙面轴向 + 统一线色 / 转角推导）。"""
        if session.base is None:
            raise WorkflowError("尚未裁切瓦片，请先执行上一步", step="seamless")
        session.processed = process_base_set(
            session.base,
            line_width=params.line_width,
            detail_keep=params.detail_keep,
        )
        self._log_msg(
            "info",
            tr("无缝化完成：中心全向无缝、四边轴向无缝、转角与墙面零接缝（线色 {0}）").format(
                session.processed.line_color
            ),
        )

    def _do_atlas(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 5/6：生成 47-tile 瓦片集 或 双网格四分之一块集。"""
        if session.processed is None:
            raise WorkflowError("尚未完成无缝化，请先执行上一步", step="atlas")
        p = session.processed
        if params.atlas_mode == "47":
            sheet, meta = build_47_sheet(p.center, p.line_color, p.line_width)
            session.atlas_meta = meta
            self._log_msg("info", tr("47-tile 瓦片集已生成（47 槽 / 46 独立图 / 256 掩码映射）"))
        else:
            sheet, meta = build_dual_pieces_sheet(p.center, p.line_color, p.line_width)
            session.atlas_meta = meta
            self._log_msg("info", tr("双网格四分之一块集已生成（16 块：填充/直切/转角盘）"))
        session.atlas_sheet = sheet

    def _do_export(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 6/6：导出瓦片 PNG、瓦片集图 + 映射 JSON、演示地图预览、项目 JSON。"""
        if session.atlas_sheet is None:
            raise WorkflowError("尚未生成瓦片集，请先执行上一步", step="export")
        out = Path(params.output_dir)
        export_dir = out / "export"
        export_dir.mkdir(parents=True, exist_ok=True)

        # 9 张处理后的瓦片
        tiles_dir = export_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for name, tile in _named_tiles(session.processed):
            tile.save(tiles_dir / f"{name}.png")

        # 瓦片集图 + 掩码映射 JSON
        atlas_path = export_dir / f"tileset_{params.atlas_mode}.png"
        session.atlas_sheet.save(atlas_path)
        meta_path = export_dir / f"tileset_{params.atlas_mode}.json"
        meta_path.write_text(
            json.dumps(session.atlas_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 演示地图 + 预览渲染
        model = TileMapModel(params.map_width, params.map_height, tile_size=params.tile_size)
        _demo_map(model)
        session.map_model = model
        preview = model.render(
            session.processed.center,
            line_color=session.processed.line_color,
            line_width=session.processed.line_width,
            mode="47" if params.atlas_mode == "47" else "dual",
        )
        preview_path = export_dir / "map_preview.png"
        preview.save(preview_path)
        map_json = export_dir / "map_demo.json"
        map_json.write_text(model.to_json(), encoding="utf-8")

        # 项目文件
        project = {
            "format": "pixel-anim-tilemap",
            "tile_size": params.tile_size,
            "atlas_mode": params.atlas_mode,
            "line_color": list(session.processed.line_color),
            "line_width": session.processed.line_width,
            "prompts": session.prompts,
            "map_width": params.map_width,
            "map_height": params.map_height,
        }
        project_file = export_dir / "tilemap_project.json"
        project_file.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")

        result = TilemapResult(
            output_dir=out,
            session=session,
            sheet_path=session.sheet_path,
            tiles_dir=tiles_dir,
            atlas_path=atlas_path,
            atlas_meta_path=meta_path,
            map_preview_path=preview_path,
            project_file=project_file,
            tile_size=params.tile_size,
            atlas_mode=params.atlas_mode,
        )
        result.step_log = list(self.step_log)
        session.result = result
        self._log_msg("info", tr("瓦片地图已导出: {0}").format(export_dir))


def _bytes_to_image(data: bytes) -> Image.Image:
    from core.processing import frame_utils as fu

    return fu.bytes_to_image(data)


def _named_tiles(base: BaseTileSet):
    """按九宫格语义顺序产出 (名字, 瓦片)。"""
    order = ["tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"]
    return [(n, base.tile(n)) for n in order]
