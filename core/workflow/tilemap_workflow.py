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
import numpy as np

from core.api.base import BaseAPI
from core.tilemap import (
    BaseTileSet,
    TileMapModel,
    align_terrain_set,
    build_47_sheet,
    build_47_sheet_art,
    build_blob47_atlas,
    build_tile16_atlas,
    build_dual_pieces_sheet_art,
    building_from_blocks,
    crop_base_3x3,
    crop_blocks,
    detect_text_marks,
    ecosystem_from_blocks,
    median_tile_texture,
    normalize_tileset,
    opaque_ratio,
    patch_marks,
    prepare_terrain_set,
    process_base_set,
    process_building_sheet,
    strip_grid_frames,
)
from core.tilemap.prompts import (
    build_building_prompts,
    build_ecosystem_prompts,
    build_tileset_prompts,
)
from core.tilemap.tiles import (
    BLOCK_POSITIONS,
    BuildingSheet,
    EcosystemSheet,
    cell_box,
    grid_cell_px,
    to_base_set,
)
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


BLOCK_POS_LABELS = {"tl": "左上", "tr": "右上", "bl": "左下", "br": "右下"}


def _measure_border_rgb(base: BaseTileSet) -> tuple:
    """经典 3×3：从四张边瓦片的**外侧条带**实测「格外的另一方地形」颜色（底图背景）。"""
    band = max(2, base.size // 4)
    pixels = []
    arr = {n: np.asarray(base.tile(n).convert("RGB"), dtype=np.float32) for n in ("top", "bottom", "left", "right")}
    pixels.append(arr["top"][:band, :, :].reshape(-1, 3))
    pixels.append(arr["bottom"][-band:, :, :].reshape(-1, 3))
    pixels.append(arr["left"][:, :band, :].reshape(-1, 3))
    pixels.append(arr["right"][:, -band:, :].reshape(-1, 3))
    return tuple(int(c) for c in np.median(np.concatenate(pixels, axis=0), axis=0))


@dataclass
class TilemapParams:
    """瓦片地图模式输入参数。"""

    description: str
    style: str = "game sprite"       # 风格描述（嵌入严格提示词）
    category: str = "classic"        # "ground" 地块生态 | "building" 建筑 | "classic" 经典 3×3
    features: dict = field(default_factory=dict)  # 地块生态：{特征名: 特征描述}（≤3 个）
    base_block: str = "auto"         # 基础地形块位置：auto/tl/tr/bl/br（AI 常不守位置要求）
    tile_size: int = 32              # 目标单格像素（偶数）
    sheet_size: int = 768            # 生图请求边长（3 格总边长）
    atlas_mode: str = "47"           # "47" | "dual" | "blob47"（FrameRonin 3×24 布局）| "tile16"
    map_source: str = "hand"         # 演示地图来源："hand" 手绘示例 | "procedural"（FrameRonin 程序化地形）
    sea_level: float = 0.38          # 程序化地形：海平面（FrameRonin 原值 0.42，演示取更均衡的 0.38）
    mountain_threshold: float = 0.55  # 程序化地形：山地阈值（FrameRonin 原值 0.48，演示取 0.55）
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
    cell_px: int = 0                               # 本次请求的单格像素（边长 = 格数 × cell_px）
    sheet_image: Optional[Image.Image] = None      # 生图底图
    sheet_path: Optional[Path] = None
    sheet_clean: Optional[Image.Image] = None      # 抹除格线框后的底图（后续全部步骤用这张）
    sheet_clean_path: Optional[Path] = None
    frame_report: Dict = field(default_factory=dict)   # 格线框抹除报告
    text_report: List[Dict] = field(default_factory=list)  # 疑似文字/水印检出记录
    base: Optional[BaseTileSet] = None             # 经典 3×3：裁切后的原始 9 片（可编辑）
    processed: Optional[BaseTileSet] = None        # 经典：无缝化处理后的 9 片
    ecosystem: Optional[EcosystemSheet] = None     # 地块生态：1 基础 + 3 特征（可编辑）
    building: Optional[BuildingSheet] = None       # 建筑：墙体/顶面/开口/立柱（可编辑）
    terrain_sets: Dict[int, BaseTileSet] = field(default_factory=dict)  # 生态处理后各地形瓦片组
    pieces: Optional[dict] = None                  # 建筑：处理后的拼件 {"pieces":..., "core":..., ...}
    atlas_sheet: Optional[Image.Image] = None      # 经典：47 集图 或 双网格块集图
    atlas_meta: Optional[dict] = None
    terrain_sheets: Dict[int, tuple] = field(default_factory=dict)  # 生态：地形 id -> (sheet, meta)
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
    terrain_atlas_paths: Dict[int, Path] = field(default_factory=dict)  # 生态：地形 id -> 瓦片集图
    pieces_dir: Optional[Path] = None              # 建筑：拼件 PNG 目录
    map_preview_path: Optional[Path] = None
    project_file: Optional[Path] = None
    tile_size: int = 0
    atlas_mode: str = "47"
    category: str = "classic"
    step_log: List[str] = field(default_factory=list)


def _demo_map(model: TileMapModel, params: Optional[TilemapParams] = None) -> None:
    """默认演示地形：整图铺满（地块类瓦片必须填充满，无透明边）。

    `map_source="procedural"` 时改用 FrameRonin 同款程序化地形（Perlin/FBM + 河谷 +
    山地场），把「水 / 山 / 平原」三类映射到 特征1 / 特征2 / 基础地形。
    """
    w, h = model.width, model.height
    if params is not None and params.map_source == "procedural":
        from core.tilemap.blob47 import ProceduralTerrain

        terrain = ProceduralTerrain(
            seed=params.description or "default",
            sea_level=params.sea_level,
            mountain_threshold=params.mountain_threshold,
        )
        kinds, _slots = terrain.grid(w, h, origin=(0, 0))
        feature_ids = sorted(t for t in model.terrain_sets if t != (model.base_terrain or 1))
        mapping = {0: feature_ids[0] if feature_ids else 1,
                   1: model.base_terrain or 1,
                   2: feature_ids[1] if len(feature_ids) > 1 else (feature_ids[0] if feature_ids else 1)}
        for y in range(h):
            for x in range(w):
                model.set_cell(x, y, mapping[int(kinds[y, x])])
        return
    model.fill_rect(0, 0, w - 1, h - 1, 1)


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

    def run_to_base(self, params: TilemapParams) -> TilemapSession:
        """执行 提示词 + 生图底图 后停下：保留中间结果（底图已存盘到 artifacts），
        等待用户确认后再继续（finish_from_base）。"""
        session = self.new_session(params)
        for name in ("prompts", "base"):
            self._check_cancel()
            self.step(name, params, session)
        return session

    def finish_from_base(self, session: TilemapSession) -> TilemapResult:
        """底图确认后继续执行 裁切→无缝化→瓦片集→导出（纯本地，无需 API）。"""
        params = session.params
        for name in TILEMAP_STEPS[2:]:
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
        """步骤 1/6：按类别生成严格提示词（地块生态 / 建筑 / 经典 3×3）。"""
        cells = 6 if params.category in ("ground", "building") else 3
        cell_px = grid_cell_px(params.sheet_size, cells)
        session.cell_px = cell_px
        if params.category == "ground":
            session.prompts = build_ecosystem_prompts(
                params.description, params.features, style=params.style,
                tile_size=params.tile_size, cell_px=cell_px,
            )
            self._log_msg(
                "info",
                tr("地块生态提示词已生成（2×2 块 × 3×3 = 6×6 格，单格 {0}px）").format(cell_px),
            )
        elif params.category == "building":
            session.prompts = build_building_prompts(
                params.description, style=params.style,
                tile_size=params.tile_size, cell_px=cell_px,
            )
            self._log_msg("info", tr("建筑瓦片提示词已生成（墙体/顶面/开口/立柱，6×6 格）"))
        else:
            session.prompts = build_tileset_prompts(
                params.description, style=params.style,
                tile_size=params.tile_size, cell_px=cell_px,
            )
            self._log_msg(
                "info",
                tr("瓦片集提示词已生成（3×3 格，单格 {0}px）").format(cell_px),
            )

    def _do_base(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 2/6：文生底图（请求边长 = 格数 × 单格像素，与提示词严格一致）。"""
        if self.image_api is None:
            raise WorkflowError("未提供图片 API，无法生成底图", step="base")
        cells = 6 if params.category in ("ground", "building") else 3
        cell_px = session.cell_px or grid_cell_px(params.sheet_size, cells)
        size = cells * cell_px
        session.cell_px = cell_px
        self._log_msg("info", tr("请求生图尺寸 {0}x{0}（{1}×{1} 格，单格 {2}px）").format(size, cells, cell_px))
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
        # 立刻清理（抹除格线框）：用户确认底图时看到的就是「将会被使用的图」
        self._clean_sheet(params, session)

    def _clean_sheet(self, params: TilemapParams, session: TilemapSession) -> None:
        """统一清理底图：抹除每格深色线框 + 检测文字/水印（第 8 轮）。"""
        cells = 6 if params.category in ("ground", "building") else 3
        cleaned, report = strip_grid_frames(session.sheet_image, rows=cells, cols=cells)
        session.sheet_clean = cleaned
        session.frame_report = report
        if report.get("count"):
            widths = sorted({d["width"] for d in report["vertical"] + report["horizontal"]})
            self._log_msg(
                "info",
                tr("已抹除底图上的格线框 {0} 条（宽度 {1} px）").format(
                    report["count"], "/".join(str(w) for w in widths)
                ),
            )
        out = Path(params.output_dir) / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        session.sheet_clean_path = out / "tileset_sheet_clean.png"
        cleaned.save(session.sheet_clean_path)

    def _check_text(self, params: TilemapParams, session: TilemapSession, items, label: str) -> None:
        """文字/水印兜底：检测将作为纹理来源的格子，命中即告警并做纹理修补。

        items: [(名字, 取瓦片, 写回瓦片, 整图格坐标 None|(r, c))]；修补用「自身纹理
        平移半格」的副本覆盖，并把同一修补写回清理后的底图（预览/产物与成品一致）。
        """
        found = []
        cells = 6 if params.category in ("ground", "building") else 3
        for name, getter, setter, pos in items:
            tile = getter()
            boxes = detect_text_marks(tile)
            if not boxes:
                continue
            found.append(name)
            session.text_report.append({"where": f"{label}/{name}", "boxes": [list(b) for b in boxes]})
            patched = patch_marks(tile, boxes)
            setter(patched)
            if pos is not None and session.sheet_clean is not None:
                box = cell_box(session.sheet_clean, cells, cells, pos[0], pos[1])
                session.sheet_clean.paste(patched.convert("RGBA"), box[:2])
                if session.sheet_clean_path is not None:
                    session.sheet_clean.save(session.sheet_clean_path)
        if found:
            self._log_msg(
                "warning",
                tr("底图检测到疑似文字/水印（{0}），已用纹理修补覆盖；建议重新生成以免误伤艺术纹理").format(
                    "、".join(found)
                ),
            )

    def _do_crop(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 3/6：清理底图（抹格线框）→ 按类别裁切（生态/建筑 = 2×2 块 × 3×3；经典 = 3×3）。"""
        if session.sheet_image is None:
            raise WorkflowError("尚未生成底图，请先执行上一步", step="crop")
        if session.sheet_clean is None:      # 手动/子集执行时补做清理
            self._clean_sheet(params, session)
        sheet = session.sheet_clean or session.sheet_image
        if params.category == "ground":
            blocks, cell = crop_blocks(sheet)
            # 文字/水印检测必须在**源分辨率**的原始格上做（归一化后笔画会被压碎），
            # 且四块的中心格都会成为纹理来源，因此四块都要查。
            text_items = []
            for key, (br, bc) in BLOCK_POSITIONS.items():
                blk = blocks[br][bc]
                text_items.append((
                    key,
                    (lambda b=blk: b[4]),
                    (lambda t, b=blk: b.__setitem__(4, t)),
                    (br * 3 + 1, bc * 3 + 1),
                ))
            self._check_text(params, session, text_items, tr("生态底图"))
            session.ecosystem = ecosystem_from_blocks(
                blocks,
                tile_size=params.tile_size,
                feature_names=list(params.features),
                base_pos=params.base_block,
            )
            eco = session.ecosystem
            self._log_msg(
                "info",
                tr("已裁切 4 块 × 9 瓦片（生态：基础 + {0} 特征；底图单格 {1}px → 目标 {2}px）").format(
                    len(eco.features), cell, params.tile_size
                ),
            )
            pos_zh = BLOCK_POS_LABELS.get(eco.base_pos, eco.base_pos)
            if params.base_block != "auto":
                self._log_msg("info", tr("基础地形块位置：{0}（手动指定）").format(pos_zh))
            elif eco.base_pos_detected:
                self._log_msg("info", tr("基础地形块位置：{0}（自动识别）").format(pos_zh))
            else:
                self._log_msg(
                    "warning",
                    tr("未能可靠识别基础地形块，已按左上块处理：底图里应有一块四周无边界/描边的纯基础地形纹理，建议重新生成或手动指定位置"),
                )
            # 文字/水印兜底已在裁切时（源分辨率四块中心格）完成
            eco_items = None
        elif params.category == "building":
            blocks, cell = crop_blocks(sheet)
            bld_items = []
            for idx, (br, bc) in enumerate(BLOCK_POSITIONS.values()):
                blk = blocks[br][bc]
                bld_items.append((
                    ("墙体块", "顶面块", "开口块", "立柱块")[idx],
                    (lambda b=blk: b[4]),
                    (lambda t, b=blk: b.__setitem__(4, t)),
                    (br * 3 + 1, bc * 3 + 1),
                ))
            self._check_text(params, session, bld_items, tr("建筑底图"))
            session.building = building_from_blocks(blocks, tile_size=params.tile_size)
            self._log_msg("info", tr("已裁切 4 块 × 9 瓦片（建筑：墙体/顶面/开口/立柱）"))
            # 自检：墙体组中心格必须填满整格（AI 没按位置画时会出现整块白底）
            fill = opaque_ratio(blocks[0][0][4])
            if fill < 0.9:
                self._log_msg(
                    "warning",
                    tr("建筑底图左上块的中心格只有 {0}% 被填充：该块应是填满整格的墙体（其余块为白底拼件）。请重新生成，并确认墙体组画在左上、立柱组画在右下").format(
                        round(fill * 100)
                    ),
                )
        else:
            tiles, cell = crop_base_3x3(sheet)
            base = normalize_tileset(to_base_set(tiles), target_size=params.tile_size)
            self._check_text(
                params, session,
                [("中心格", lambda: tiles[4].convert("RGBA"),
                  lambda t: tiles.__setitem__(4, t.convert("RGBA")), (1, 1))],
                tr("经典底图"),
            )
            # 修补可能改动了中心格：用（可能已修补的）九格重建
            base = normalize_tileset(to_base_set(tiles), target_size=params.tile_size)
            session.base = base
            self._log_msg(
                "info",
                tr("已裁切 {0} 张瓦片（底图单格 {1}px → 目标 {2}px）").format(9, cell, base.size),
            )

    def _do_seamless(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 4/6：对齐化处理（生态=各特征纹理 + 实测条带/描边；建筑=墙体拼件；经典=单地形）。"""
        if params.category == "ground":
            if session.ecosystem is None:
                raise WorkflowError("尚未裁切瓦片，请先执行上一步", step="seamless")
            raw = session.ecosystem.terrain_sets()
            base_raw = raw[1]
            base_tex = median_tile_texture(base_raw.all(), params.tile_size)
            ground_rgb = np.median(np.asarray(base_tex.convert("RGB")).reshape(-1, 3), axis=0)
            arts: Dict[int, BaseTileSet] = {}
            for tid, tset in raw.items():
                arts[tid] = align_terrain_set(
                    tset, base_texture=base_tex, tile_size=params.tile_size,
                    ground_rgb=ground_rgb, plain=(tid == 1),
                )
            session.terrain_sets = arts
            meta = arts[1].art_meta
            self._log_msg(
                "info",
                tr("生态无缝化完成：{0} 套地形纹理（条带 {1}px、描边 {2}px，实测自底图）").format(
                    len(arts), meta.get("band_px", "?"), meta.get("rim_px", "?")
                ),
            )
        elif params.category == "building":
            if session.building is None:
                raise WorkflowError("尚未裁切瓦片，请先执行上一步", step="seamless")
            session.pieces = process_building_sheet(session.building)
            self._log_msg(
                "info",
                tr("建筑拼件完成：直段/端头/转角/立柱（白底已抠除，可叠放地块）"),
            )
        else:
            if session.base is None:
                raise WorkflowError("尚未裁切瓦片，请先执行上一步", step="seamless")
            # 经典 3×3：单地形。中心纹理 = 特征纹理；「另一方地形」= 边格外侧条带的
            # 实测色（底图背景），因此暴露侧画的就是底图本来的背景色带
            border_rgb = _measure_border_rgb(session.base)
            border_tex = Image.new(
                "RGBA", (params.tile_size, params.tile_size),
                (int(border_rgb[0]), int(border_rgb[1]), int(border_rgb[2]), 255),
            )
            session.processed = align_terrain_set(
                session.base, base_texture=border_tex, tile_size=params.tile_size,
                ground_rgb=np.array(border_rgb, dtype=np.float32),
            )
            meta = session.processed.art_meta
            self._log_msg(
                "info",
                tr("无缝化完成：对齐式纹理（条带 {0}px、描边 {1}px，实测自底图）").format(
                    meta.get("band_px", "?"), meta.get("rim_px", "?")
                ),
            )

    def _do_atlas(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 5/6：生成瓦片集（生态=每地形艺术片 47 集；建筑=无需；经典=程序化）。"""
        if params.category == "ground":
            if not session.terrain_sets:
                raise WorkflowError("尚未完成无缝化，请先执行上一步", step="atlas")
            for tid, tset in session.terrain_sets.items():
                if params.atlas_mode == "dual":
                    sheet, meta = build_dual_pieces_sheet_art(tset)
                elif params.atlas_mode == "blob47":
                    sheet, meta = build_blob47_atlas(tset)
                elif params.atlas_mode == "tile16":
                    sheet, meta = build_tile16_atlas(tset)
                else:
                    sheet, meta = build_47_sheet_art(tset)
                session.terrain_sheets[tid] = (sheet, meta)
            mode_zh = {"blob47": "FrameRonin 3×24 布局", "tile16": "16 图块族 4×4"}.get(params.atlas_mode, "9×6")
            self._log_msg(
                "info",
                tr("生态瓦片集已生成：{0} 套（对齐式构图；布局 {1}）").format(len(session.terrain_sheets), mode_zh),
            )
        elif params.category == "building":
            if session.pieces is None:
                raise WorkflowError("尚未完成建筑拼件，请先执行上一步", step="atlas")
        else:
            if session.processed is None:
                raise WorkflowError("尚未完成无缝化，请先执行上一步", step="atlas")
            p = session.processed
            # 地块类瓦片必须填充满：统一走九宫格艺术片构图（无程序化描边/透明楔形）
            if params.atlas_mode == "dual":
                sheet, meta = build_dual_pieces_sheet_art(p)
            else:
                sheet, meta = build_47_sheet_art(p)
            session.atlas_meta = meta
            session.atlas_sheet = sheet
            self._log_msg("info", tr("47-tile 瓦片集已生成（九宫格艺术片构图，全填充）"))

    def _do_export(self, params: TilemapParams, session: TilemapSession) -> None:
        """步骤 6/6：导出瓦片/瓦片集/演示地图预览/项目 JSON（按类别分支）。"""
        out = Path(params.output_dir)
        export_dir = out / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        if params.category == "ground":
            self._export_ground(params, session, export_dir)
        elif params.category == "building":
            self._export_building(params, session, export_dir)
        else:
            self._export_classic(params, session, export_dir)
        session.result.step_log = list(self.step_log)
        self._log_msg("info", tr("瓦片地图已导出: {0}").format(export_dir))

    # ------------------------------------------------------------------ #
    def _export_classic(self, params: TilemapParams, session: TilemapSession, export_dir: Path) -> None:
        if session.atlas_sheet is None:
            raise WorkflowError("尚未生成瓦片集，请先执行上一步", step="export")
        tiles_dir = export_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for name, tile in _named_tiles(session.processed):
            tile.save(tiles_dir / f"{name}.png")
        atlas_path = export_dir / f"tileset_{params.atlas_mode}.png"
        session.atlas_sheet.save(atlas_path)
        meta_path = export_dir / f"tileset_{params.atlas_mode}.json"
        meta_path.write_text(json.dumps(session.atlas_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        model = TileMapModel(params.map_width, params.map_height, tile_size=params.tile_size)
        _demo_map(model, params)
        # 地块类：注册地形瓦片组走艺术构图渲染（瓦片填充满、无透明楔形）
        model.set_terrain(1, session.processed)
        model.set_base_terrain(1)
        session.map_model = model
        preview = model.render()
        preview_path = export_dir / "map_preview.png"
        preview.save(preview_path)
        (export_dir / "map_demo.json").write_text(model.to_json(), encoding="utf-8")
        project = {
            "format": "pixel-anim-tilemap",
            "category": "classic",
            "tile_size": params.tile_size,
            "atlas_mode": params.atlas_mode,
            "prompts": session.prompts,
        }
        project_file = export_dir / "tilemap_project.json"
        project_file.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        session.result = TilemapResult(
            output_dir=session.params.output_dir, session=session,
            sheet_path=session.sheet_path, tiles_dir=tiles_dir,
            atlas_path=atlas_path, atlas_meta_path=meta_path,
            map_preview_path=preview_path, project_file=project_file,
            tile_size=params.tile_size, atlas_mode=params.atlas_mode, category="classic",
        )

    def _export_ground(self, params: TilemapParams, session: TilemapSession, export_dir: Path) -> None:
        if not session.terrain_sheets:
            raise WorkflowError("尚未生成瓦片集，请先执行上一步", step="export")
        terrain_paths: Dict[int, Path] = {}
        for tid, (sheet, meta) in session.terrain_sheets.items():
            path = export_dir / f"terrain_{tid}_47.png"
            sheet.save(path)
            (export_dir / f"terrain_{tid}_47.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            terrain_paths[tid] = path
        # 演示地图：基础地形铺底 + 特征水塘/岩石区域（多地形自动衔接）
        model = TileMapModel(params.map_width, params.map_height, tile_size=params.tile_size)
        for tid, tset in session.terrain_sets.items():
            model.set_terrain(tid, tset)
        model.set_base_terrain(1)
        model.fill_rect(0, 0, params.map_width - 1, params.map_height - 1, 1)
        feats = sorted(session.terrain_sets.keys())
        if len(feats) > 1:
            cx, cy = params.map_width // 2, params.map_height // 2
            model.fill_rect(cx - 2, cy - 2, cx + 2, cy + 2, feats[1])       # 水塘
            model.fill_rect(cx - 1, cy - 1, cx + 1, cy + 1, feats[1])
            if len(feats) > 2:
                model.fill_rect(cx + 3, cy + 3, cx + 4, cy + 4, feats[2])  # 稀疏
            if len(feats) > 3:
                model.set_cell(cx - 5, cy - 5, feats[3])                    # 岩石
                model.set_cell(cx - 5, cy - 4, feats[3])
        session.map_model = model
        preview = model.render()
        preview_path = export_dir / "map_preview.png"
        preview.save(preview_path)
        (export_dir / "map_demo.json").write_text(model.to_json(), encoding="utf-8")
        project = {
            "format": "pixel-anim-tilemap",
            "category": "ground",
            "tile_size": params.tile_size,
            "features": list(params.features),
            "terrain_ids": sorted(session.terrain_sets.keys()),
            "base_terrain": 1,
            "prompts": session.prompts,
        }
        project_file = export_dir / "tilemap_project.json"
        project_file.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        session.result = TilemapResult(
            output_dir=session.params.output_dir, session=session,
            sheet_path=session.sheet_path, terrain_atlas_paths=terrain_paths,
            map_preview_path=preview_path, project_file=project_file,
            tile_size=params.tile_size, atlas_mode=params.atlas_mode, category="ground",
        )

    def _export_building(self, params: TilemapParams, session: TilemapSession, export_dir: Path) -> None:
        if session.pieces is None:
            raise WorkflowError("尚未生成建筑拼件，请先执行上一步", step="export")
        pieces_dir = export_dir / "pieces"
        pieces_dir.mkdir(parents=True, exist_ok=True)
        for name, piece in session.pieces["pieces"].items():
            piece.save(pieces_dir / f"{name}.png")
        # 演示预览：拼件在透明画布上的摆样（直墙 + 转角 + 端头 + 立柱）
        model = TileMapModel(params.map_width, params.map_height, tile_size=params.tile_size)
        p = session.pieces["pieces"]
        model.set_overlay(2, 2, p["straight"], 0, name="straight")
        model.set_overlay(3, 2, p["straight"], 0, name="straight")
        model.set_overlay(4, 2, p["corner"], 0, name="corner")
        model.set_overlay(4, 3, p["straight"], 1, name="straight")
        model.set_overlay(2, 4, p["end"], 1, name="end")
        model.set_overlay(2, 3, p["pillar"], 0, name="pillar")
        session.map_model = model
        preview = model.render()
        preview_path = export_dir / "map_preview.png"
        preview.save(preview_path)
        # 地图 JSON（含 overlay 拼件名称，可在预览中恢复）
        map_json = export_dir / "map_demo.json"
        map_json.write_text(model.to_json(), encoding="utf-8")
        project = {
            "format": "pixel-anim-tilemap",
            "category": "building",
            "tile_size": params.tile_size,
            "pieces": list(p.keys()),
            "prompts": session.prompts,
        }
        project_file = export_dir / "tilemap_project.json"
        project_file.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        session.result = TilemapResult(
            output_dir=session.params.output_dir, session=session,
            sheet_path=session.sheet_path, pieces_dir=pieces_dir,
            map_preview_path=preview_path, project_file=project_file,
            tile_size=params.tile_size, atlas_mode=params.atlas_mode, category="building",
        )


def _bytes_to_image(data: bytes) -> Image.Image:
    from core.processing import frame_utils as fu

    return fu.bytes_to_image(data)


def _named_tiles(base: BaseTileSet):
    """按九宫格语义顺序产出 (名字, 瓦片)。"""
    order = ["tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"]
    return [(n, base.tile(n)) for n in order]
