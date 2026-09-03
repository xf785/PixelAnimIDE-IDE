"""精灵图工作流：仅用文生图生成网格精灵图（不涉及视频抽帧）。

链路：文生对象底图（默认 1024×1024 原始分辨率）
      → 以底图原始图做参考（图生图）生成 i×j 网格精灵图
      → 算法裁切为帧序列 → 完美像素双分辨率（原生网格分辨率 + 用户设定分辨率）
      → 扣除纯色背景 → 导出 PNG 序列 / 机器严格拼接网格图(+索引 JSON) / GIF（双套）。

内置强提示词：底图（像素风、纯白背景、主体完整居中、严格分辨率/颜色数）
与精灵图（等大网格、帧间角色一致、无文字/边框/格线、每格纯白背景）。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from config.settings import DEFAULT_FPS, DEFAULT_MAX_COLORS, EXPORT_PREFIX
from core.api.base import APIResult, BaseAPI
from core.processing import background as bg_mod
from core.processing import frame_utils as fu
from core.processing import pixelizer as px
from core.processing.prompt_utils import (
    build_fallback_prompts,
    normalize_prompts,
)
from core.workflow.shared import finalize_prompts, generate_prompt_data
from core.workflow.solo_workflow import WorkflowError
from ui.i18n import tr

logger = logging.getLogger("PixelAnimIDE.workflow.sprite")

# 精灵图强提示词模板：{rows}/{cols}/{total}/{action} 占位。
# 核心约束：一整张网格图（不是 i×j 张独立图）、首尾格姿势完全一致（循环无缝）、
# 角色形象逐格绝对一致（不突变），仅动作平滑变化。
SPRITE_SHEET_PROMPT = (
    "Create ONE single pixel-art sprite sheet image: a {rows}×{cols} grid containing exactly "
    "{total} frames of the SAME character performing '{action}' as a seamless looping animation. "
    "This must be a SINGLE image with a uniform {rows}×{cols} grid layout — never generate "
    "{total} separate images, never scatter the frames outside the grid. Every cell must be an "
    "equal square, perfectly aligned in the grid, with the character fully visible and centered "
    "inside each cell (clear margin, never cropped, never touching cell edges). The FIRST cell "
    "and the LAST cell must show the character in the IDENTICAL pose, because the animation "
    "loops back to the start; the cells between them form a smooth progression of the "
    "'{action}' motion. The character design — body shape, outline, colors, proportions — must "
    "be EXACTLY identical in every cell with absolutely no mutation or redesign between frames; "
    "only the pose changes smoothly. A SOLID PURE WHITE background (#FFFFFF) in every cell; "
    "NO text, NO numbers, NO labels, NO borders or grid lines between cells."
)

SPRITE_NEGATIVE_PROMPT = (
    "text, numbers, labels, watermark, grid lines, borders between cells, inconsistent "
    "character, character morphing, changing design, different proportions between frames, "
    "mutating character, first and last frame different, cropped character, cut-off character, "
    "gray background, colored background, gradients, anti-aliasing, blurry, extra cells, "
    "empty cells, separate images, scattered frames"
)

# 精灵图最大边长（图片 API 常见上限）
MAX_SHEET_SIDE = 1024

# 精灵图流程步骤（手动模式的执行顺序）
STEP_ORDER = ("prompts", "base", "sheet", "crop", "pixelize", "key", "export")


@dataclass
class SpriteParams:
    """精灵图生成参数。"""

    description: str
    action: str = ""
    frame_count: int = 16          # 帧数（≤ grid_rows × grid_cols）
    grid_rows: int = 4             # 网格行数 i
    grid_cols: int = 4             # 网格列数 j
    cell_size: int = 64            # 单格像素尺寸（长边）
    cell_inset: int = 0            # 裁切时每格内缩像素（去格子黑框；0 = 自动 4%）
    max_colors: int = DEFAULT_MAX_COLORS
    force_pure_bg: bool = True     # 背景强制纯色
    remove_bg: bool = True         # 一键抠图
    loop_close: bool = True        # 首尾帧一致（末帧=首帧，循环无缝）
    tight_crop: bool = False       # 抠图后按内容包围盒统一裁剪（去四周死边）
    base_size: int = 1024          # 对象底图分辨率（正方形；直接以该原始分辨率做 i2i 参考）
    preserve_native: bool = True   # 双分辨率导出：完美像素原生 + 用户设定分辨率
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def sheet_size(self) -> tuple:
        """精灵图请求尺寸（网格 × 单格，长边上限 1024）。"""
        side = self.cell_size
        w, h = self.grid_cols * side, self.grid_rows * side
        mx = max(w, h)
        if mx > MAX_SHEET_SIDE:
            k = MAX_SHEET_SIDE / mx
            w, h = max(8, int(w * k)), max(8, int(h * k))
        return w, h

    def cell_target_size(self) -> tuple:
        """单格目标像素尺寸（裁切后缩放用）。"""
        rw, rh = self.grid_cols, self.grid_rows
        side = self.cell_size
        if rw >= rh:
            return side, max(1, round(side * rh / rw))
        return max(1, round(side * rw / rh)), side


@dataclass
class SpriteSession:
    """精灵图逐步执行的中间状态（手动模式会话）。

    每个步骤的产物写入对应字段；重跑某步会覆盖旧值，后续步骤在
    「继续」时重新计算，无需手动清空下游产物。
    """

    params: SpriteParams
    prompts: Optional[Dict[str, str]] = None          # LLM 生成的提示词
    base: Optional[Image.Image] = None                # 对象底图
    base_path: Optional[Path] = None
    sheet: Optional[Image.Image] = None               # i×j 网格精灵图
    sheet_path: Optional[Path] = None
    frames: Optional[List[Image.Image]] = None        # 裁切后的原始帧
    preset_frames: Optional[List[Image.Image]] = None  # 用户设定分辨率帧
    native_frames: Optional[List[Image.Image]] = None  # 完美像素原生分辨率帧
    grid: Optional[Tuple[int, int]] = None            # 检测出的像素网格
    result: Optional[SpriteResult] = None
    max_done: int = -1                                # 已成功完成的步骤号（STEP_ORDER 下标）


@dataclass
class SpriteResult:
    """精灵图工作流输出。"""

    output_dir: Path
    base_image: Optional[Path] = None      # 文生对象底图
    sheet_image: Optional[Path] = None     # 原始 i×j 精灵图（未抠图）
    frames_dir: Optional[Path] = None
    png_dir: Optional[Path] = None
    gif_path: Optional[Path] = None
    sheet_path: Optional[Path] = None      # 抠图后机器严格拼接的精灵图（用户分辨率）
    sheet_index: Optional[Path] = None     # 精灵图索引 JSON（FrameRonin 格式）
    native_frames_dir: Optional[Path] = None   # 完美像素原生分辨率帧
    native_png_dir: Optional[Path] = None
    native_gif_path: Optional[Path] = None
    native_sheet_path: Optional[Path] = None
    native_sheet_index: Optional[Path] = None
    grid: Optional[Tuple[int, int]] = None     # 完美像素检测出的网格
    project_file: Optional[Path] = None
    frame_count: int = 0
    width: int = 0
    height: int = 0
    native_width: int = 0
    native_height: int = 0
    step_log: List[str] = field(default_factory=list)


class SpriteWorkflow:
    """精灵图全流程执行器。"""

    def __init__(
        self,
        llm_api: BaseAPI,
        image_api: BaseAPI,
        log: Optional[Callable[[str, str], None]] = None,
        cancel: Optional[threading.Event] = None,
        on_base: Optional[Callable[[Path], None]] = None,
        on_sheet: Optional[Callable[[Path], None]] = None,
    ):
        self.llm_api = llm_api
        self.image_api = image_api
        self._log = log
        self._cancel = cancel or threading.Event()
        self._on_base = on_base
        self._on_sheet = on_sheet
        self.step_log: List[str] = []

    # ------------------------------------------------------------------ #
    def _log_msg(self, level: str, message: str) -> None:
        entry = f"[{level}] {message}"
        self.step_log.append(entry)
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

    # ------------------------------------------------------------------ #
    def run(self, params: SpriteParams) -> SpriteResult:
        """自动模式：无干涉按顺序执行全部步骤。"""
        session = self.new_session(params)
        for name in STEP_ORDER:
            self._check_cancel()
            self.step(name, params, session)
        if session.result is None:
            raise WorkflowError("精灵图流程未产出结果", step="导出")
        return session.result

    def new_session(self, params: SpriteParams) -> "SpriteSession":
        """创建逐步执行会话（手动模式用）。"""
        return SpriteSession(params=params)

    def step(self, name: str, params: SpriteParams, session: "SpriteSession") -> None:
        """执行单个步骤（手动模式逐步调用；run() 内部依次调用）。

        重跑某步时，该步产物会覆盖 session 中的旧值；继续时后续步骤会
        重新计算，因此无需手动清空下游产物。
        """
        if name not in STEP_ORDER:
            raise WorkflowError(f"未知精灵图步骤: {name}", step=name)
        fn = getattr(self, f"_do_{name}")
        self._check_cancel()
        fn(params, session)
        session.max_done = max(session.max_done, STEP_ORDER.index(name))

    # ------------------------------------------------------------------ #
    def _do_prompts(self, params: SpriteParams, session: "SpriteSession") -> None:
        """步骤 1/7：生成提示词。"""
        session.prompts = self._step_prompts(params)

    def _do_base(self, params: SpriteParams, session: "SpriteSession") -> None:
        """步骤 2/7：文生对象底图（默认 1024×1024 原始分辨率）。"""
        path = Path(params.output_dir) / "artifacts" / "base.png"
        session.base = self._step_base(params, session.prompts["image_prompt"], path)
        session.base_path = path

    def _do_sheet(self, params: SpriteParams, session: "SpriteSession") -> None:
        """步骤 3/7：以底图原始图为参考（i2i）生成 i×j 网格精灵图。"""
        if session.base is None:
            raise WorkflowError("尚未生成对象底图，请先执行上一步", step="精灵图生成")
        path = Path(params.output_dir) / "artifacts" / "sprite_sheet.png"
        session.sheet = self._step_sheet(params, session.prompts, session.base, path)
        session.sheet_path = path

    def _do_crop(self, params: SpriteParams, session: "SpriteSession") -> None:
        """步骤 4/7：算法裁切为帧序列（自动内缩去掉 AI 画的格子黑框）。"""
        if session.sheet is None:
            raise WorkflowError("尚未生成精灵图，请先执行上一步", step="裁切")
        sheet = session.sheet
        inset = int(params.cell_inset or 0)
        if not inset:
            cw = sheet.width / params.grid_cols
            ch = sheet.height / params.grid_rows
            inset = max(2, int(min(cw, ch) * 0.04))
        frames = fu.crop_sprite_sheet(
            sheet, params.grid_rows, params.grid_cols, params.frame_count, inset=inset
        )
        if not frames:
            raise WorkflowError("裁切精灵图失败（0 帧）", step="裁切")
        session.frames = frames
        self._log_msg(
            "info",
            f"已裁切 {len(frames)} 帧（{params.grid_rows}×{params.grid_cols} 网格，取前 {params.frame_count}）"
            + (f"，每格内缩 {inset}px 去边框" if inset else ""),
        )

    def _do_pixelize(self, params: SpriteParams, session: "SpriteSession") -> None:
        """步骤 5/7：完美像素双分辨率（原生网格分辨率 + 用户设定分辨率）+ 首尾帧对齐。"""
        if not session.frames:
            raise WorkflowError("尚未裁切帧序列，请先执行上一步", step="像素化")
        frames = session.frames
        target = params.cell_target_size()
        native_frames: Optional[List[Image.Image]] = None
        if params.preserve_native:
            seq = px.perfect_pixelize_sequence(frames)
            if seq is not None:
                native_frames, grid = seq
                session.grid = grid
                self._log_msg("info", tr("完美像素检测到网格 {0}×{1}：全部帧单元采样").format(grid[0], grid[1]))
                eff_colors = max(
                    2, min(len(set(native_frames[0].convert("RGB").getdata())), params.max_colors)
                )
                palette = px.extract_dominant_palette(native_frames[0], eff_colors)
                native_frames = px.pixelize_frames(
                    native_frames,
                    px.PixelizeParams(max_colors=eff_colors, edge_clean=True, palette=palette),
                )
            else:
                session.grid = None
                self._log_msg("info", tr("未检测到像素网格（非像素风）：按用户分辨率单套导出"))
        if native_frames is not None:
            preset_frames = [px.resize_nearest(f, target) for f in native_frames]
            self._log_msg(
                "info",
                f"原生分辨率 {native_frames[0].width}x{native_frames[0].height}"
                f" → 用户分辨率 {target[0]}x{target[1]}（NEAREST 放大）",
            )
        else:
            preset_frames = [px.resize_nearest(f, target) for f in frames]
            self._log_msg("info", tr("帧统一缩放到 {0}x{1}").format(target[0], target[1]))
        # 首尾帧一致：末帧强制等于首帧，保证循环无缝（模型未对齐时由算法兜底）
        if params.loop_close and len(preset_frames) >= 2:
            preset_frames[-1] = preset_frames[0].copy()
            if native_frames is not None and len(native_frames) >= 2:
                native_frames[-1] = native_frames[0].copy()
            self._log_msg("info", tr("首尾帧已对齐：末帧=首帧（循环无缝）"))
        session.preset_frames = preset_frames
        session.native_frames = native_frames

    def _do_key(self, params: SpriteParams, session: "SpriteSession") -> None:
        """步骤 6/7：扣除纯色背景（一键抠图，两套分辨率各自处理）。"""
        if not session.preset_frames:
            raise WorkflowError("尚未完成像素化，请先执行上一步", step="抠图")
        session.preset_frames = self._step_key(params, session.preset_frames)
        if session.native_frames is not None:
            session.native_frames = self._step_key(params, session.native_frames)

    def _do_export(self, params: SpriteParams, session: "SpriteSession") -> None:
        """步骤 7/7：导出（两套分辨率 × PNG 序列 / 严格拼接网格图(+索引) / GIF）。"""
        if not session.preset_frames:
            raise WorkflowError("没有可导出的帧，请先执行上一步", step="导出")
        out = Path(params.output_dir)
        export_dir = out / "export"
        frames = list(session.preset_frames)
        native_frames = session.native_frames
        result = SpriteResult(output_dir=out)
        result.grid = session.grid
        result.base_image = session.base_path
        result.sheet_image = session.sheet_path

        # 可选：按内容包围盒统一裁剪（去掉各帧四周的死边，保持画布一致）
        if (
            params.tight_crop
            and len(frames) >= 2
            and frames[0].mode == "RGBA"
            and frames[0].getchannel("A").getextrema()[0] < 255
        ):
            union: Optional[Tuple[int, int, int, int]] = None
            for f in frames:
                box = fu.content_bbox(f)
                if box is None:
                    continue
                union = box if union is None else (
                    min(union[0], box[0]), min(union[1], box[1]),
                    max(union[2], box[2]), max(union[3], box[3]),
                )
            if union is not None:
                l, t, r, b = union
                if (r - l) < frames[0].width * 0.98 or (b - t) < frames[0].height * 0.98:
                    frames = [f.crop((l, t, r, b)) for f in frames]
                    target = params.cell_target_size()
                    if frames[0].size != target:
                        frames = [f.resize(target, Image.Resampling.NEAREST) for f in frames]
                    self._log_msg("info", tr("内容包围盒裁剪：({0},{1})-({2},{3})，已统一回 {4}x{5}").format(l, t, r, b, target[0], target[1]))

        # 用户分辨率套：PNG 序列 + 严格拼接网格图 + GIF，含索引 JSON
        frames_dir = out / "frames"
        fu.save_png_sequence(frames, frames_dir, prefix="frame")
        result.frames_dir = frames_dir
        result.frame_count = len(frames)
        if frames:
            result.width, result.height = frames[0].size

        gif_path = fu.frames_to_gif(frames, export_dir / f"{EXPORT_PREFIX}.gif", fps=DEFAULT_FPS)
        result.gif_path = gif_path
        png_dir = export_dir / "png"
        fu.save_png_sequence(frames, png_dir, prefix="frame")
        result.png_dir = png_dir
        timestamps = [i / max(1, DEFAULT_FPS) for i in range(len(frames))]
        keyed_sheet, sheet_index = fu.compose_sprite_sheet(
            frames, columns=params.grid_cols, timestamps=timestamps
        )
        result.sheet_path = fu.save_image(keyed_sheet, export_dir / "sprite_sheet.png")
        index_file = export_dir / "sprite_sheet.json"
        index_file.write_text(json.dumps(sheet_index, ensure_ascii=False, indent=2), encoding="utf-8")
        result.sheet_index = index_file
        self._log_msg("info", tr("精灵图索引已导出: {0}（{1} 帧）").format(index_file, len(sheet_index["frames"])))

        # 原生分辨率套：同样导出 PNG 序列 / 拼接网格图 / GIF
        if native_frames is not None:
            native_frames_dir = out / "frames_native"
            fu.save_png_sequence(native_frames, native_frames_dir, prefix="frame")
            result.native_frames_dir = native_frames_dir
            result.native_width, result.native_height = native_frames[0].size
            result.native_gif_path = fu.frames_to_gif(
                native_frames, export_dir / f"{EXPORT_PREFIX}_native.gif", fps=DEFAULT_FPS
            )
            native_png_dir = export_dir / "png_native"
            fu.save_png_sequence(native_frames, native_png_dir, prefix="frame")
            result.native_png_dir = native_png_dir
            native_sheet, native_index = fu.compose_sprite_sheet(
                native_frames, columns=params.grid_cols, timestamps=timestamps
            )
            result.native_sheet_path = fu.save_image(native_sheet, export_dir / "sprite_sheet_native.png")
            native_index_file = export_dir / "sprite_sheet_native.json"
            native_index_file.write_text(
                json.dumps(native_index, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result.native_sheet_index = native_index_file
            self._log_msg(
                "info",
                f"原生分辨率套已导出：{result.native_width}x{result.native_height}"
                "（PNG 序列 / 拼接网格图 / GIF + 索引）",
            )

        # 项目元数据
        meta = {
            "format": "pixel-sprite-sheet",
            "frame_count": len(frames),
            "grid": [params.grid_rows, params.grid_cols],
            "detected_grid": list(result.grid) if result.grid else None,
            "cell_size": list(frames[0].size) if frames else None,
            "base_size": params.base_size,
            "native_resolution": (
                [result.native_width, result.native_height] if native_frames is not None else None
            ),
            "fps": DEFAULT_FPS,
            "action": params.action,
            "loop_close": params.loop_close,
            "preserve_native": params.preserve_native,
        }
        project_file = export_dir / "sprite_project.json"
        project_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        result.project_file = project_file

        result.step_log = list(self.step_log)
        session.result = result
        self._log_msg("info", tr("精灵图完成：{0} 帧 @ {1}x{2}").format(result.frame_count, result.width, result.height))

    # ------------------------------------------------------------------ #
    def _step_prompts(self, params: SpriteParams) -> Dict[str, str]:
        """LLM 生成提示词（失败降级本地模板），图片提示词经 STRICT 强化。

        调用/重试/严格纠正逻辑统一在 core.workflow.shared.generate_prompt_data。
        """
        desc, act = params.description, params.action
        data, last = generate_prompt_data(self.llm_api, desc, act, log=self._log_msg)
        if data is not None:
            prompts = normalize_prompts(data, desc, act)
            return finalize_prompts(
                prompts,
                params.cell_target_size(),
                "1:1",
                params.max_colors,
            )
        if last.ok:
            self._log_msg("warn", tr("LLM 返回无法解析，使用本地模板"))
        else:
            self._log_msg("warn", tr("LLM 调用失败（{0}），使用本地模板").format(last.message))
        return finalize_prompts(
            build_fallback_prompts(desc, act),
            params.cell_target_size(),
            "1:1",
            params.max_colors,
        )

    # ------------------------------------------------------------------ #
    def _step_base(self, params: SpriteParams, image_prompt: str, path: Path) -> Image.Image:
        """文生对象底图：单只对象、纯白背景、主体完整。

        直接以高分辨率（默认 1024×1024）的**原始图片**作为下一步网格精灵图的
        图生图参考（i2i）：参考图细节越丰富，网格里每格的角色细节越清晰。
        """
        w = h = params.base_size
        self._log_msg("info", tr("生成对象底图：{0}x{1}（原始分辨率，直接作 i2i 参考）").format(w, h))
        result = self.image_api.call(prompt=image_prompt, size=f"{w}x{h}", n=1)
        if not result.ok:
            raise WorkflowError(f"底图生成失败: {result.message}", step="文生底图")
        img = self._first_image(result)
        fu.save_image(img, path)
        self._log_msg("info", tr("底图已保存: {0}").format(path))
        if self._on_base:
            try:
                self._on_base(path)
            except Exception:  # noqa: BLE001
                pass
        return img

    def _step_sheet(self, params: SpriteParams, prompts: Dict[str, str], base: Image.Image, path: Path) -> Image.Image:
        """以底图为参考图（图生图）生成 i×j 网格精灵图。"""
        total = min(params.frame_count, params.grid_rows * params.grid_cols)
        sheet_prompt = SPRITE_SHEET_PROMPT.format(
            rows=params.grid_rows,
            cols=params.grid_cols,
            total=total,
            action=params.action or "idle",
        )
        w, h = params.sheet_size()
        self._log_msg("info", tr("生成精灵图：{0}×{1} 网格 / {2} 帧，{3}x{4}").format(params.grid_rows, params.grid_cols, total, w, h))
        base_bytes = fu.image_to_bytes(base, "PNG")
        result = self.image_api.call(
            prompt=sheet_prompt,
            size=f"{w}x{h}",
            n=1,
            image=base_bytes,           # 以底图为参考（图生图）
            negative_prompt=SPRITE_NEGATIVE_PROMPT,
        )
        if not result.ok:
            raise WorkflowError(f"精灵图生成失败: {result.message}", step="精灵图生成")
        img = self._first_image(result)
        fu.save_image(img, path)
        self._log_msg("info", tr("精灵图已保存: {0}（{1}x{2}）").format(path, img.width, img.height))
        if self._on_sheet:
            try:
                self._on_sheet(path)
            except Exception:  # noqa: BLE001
                pass
        return img

    @staticmethod
    def _first_image(result: APIResult) -> Image.Image:
        images = (result.data or {}).get("images") or []
        urls = (result.data or {}).get("urls") or []
        if images:
            return fu.bytes_to_image(images[0])
        if urls:
            return fu.bytes_to_image(fu.download_bytes(urls[0]))
        raise WorkflowError("生图接口未返回任何图片", step="生图")

    def _step_key(self, params: SpriteParams, frames: List[Image.Image]) -> List[Image.Image]:
        """扣除纯色背景（一键抠图）：自适应背景归一化 + 颜色键抠图。

        与 solo / ide 共用 core.processing.background.process_background：
        有精确掩膜走掩膜抠图；无掩膜回退颜色键抠图——键色取 params 指定
        或图像边缘主色自动推断，hybrid 模式（连通背景大容差、主体内部同色
        小容差）避免把主体内部的白色/浅色像素误删。
        """
        if not params.remove_bg and not params.force_pure_bg:
            return frames
        out: List[Image.Image] = []
        for f in frames:
            self._check_cancel()
            img, _normalized = bg_mod.process_background(
                f,
                force_pure_bg=params.force_pure_bg,
                remove_bg=params.remove_bg,
                tolerance=32,
                feather=4,
            )
            out.append(img)
        self._log_msg("info", tr("已扣除背景：{0} 帧").format(len(out)))
        return out
