"""精灵图工作流：仅用文生图生成网格精灵图（不涉及视频抽帧）。

链路：文生对象底图 → 以底图为参考图（图生图）生成 i×j 网格精灵图
      → 算法裁切为帧序列 → 扣除纯色背景 → 导出 GIF / PNG 序列 / 扣背景精灵图。

内置强提示词：底图（像素风、纯白背景、主体完整居中、严格分辨率/颜色数）
与精灵图（等大网格、帧间角色一致、无文字/边框/格线、每格纯白背景）。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from config.settings import DEFAULT_FPS, DEFAULT_MAX_COLORS, EXPORT_PREFIX
from core.api.base import APIResult, BaseAPI
from core.processing import background as bg_mod
from core.processing import frame_utils as fu
from core.processing.prompt_utils import (
    SYSTEM_PROMPT,
    build_fallback_prompts,
    build_user_prompt,
    normalize_prompts,
    parse_json_response,
)
from core.workflow.shared import finalize_prompts, resolve_api_image_size
from core.workflow.solo_workflow import WorkflowError

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
class SpriteResult:
    """精灵图工作流输出。"""

    output_dir: Path
    base_image: Optional[Path] = None      # 文生对象底图
    sheet_image: Optional[Path] = None     # 原始 i×j 精灵图（未抠图）
    frames_dir: Optional[Path] = None
    png_dir: Optional[Path] = None
    gif_path: Optional[Path] = None
    sheet_path: Optional[Path] = None      # 抠图后的精灵图
    project_file: Optional[Path] = None
    frame_count: int = 0
    width: int = 0
    height: int = 0
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
        """执行完整精灵图流程。"""
        out = Path(params.output_dir)
        artifacts = out / "artifacts"
        export_dir = out / "export"
        frames_dir = out / "frames"
        artifacts.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        result = SpriteResult(output_dir=out)

        # 1) 文生对象底图
        prompts = self._step_prompts(params)
        self._check_cancel()
        base = self._step_base(params, prompts["image_prompt"], artifacts / "base.png")
        result.base_image = artifacts / "base.png"

        # 2) 底图为参考图 -> 生成 i×j 网格精灵图
        self._check_cancel()
        sheet = self._step_sheet(params, prompts, base, artifacts / "sprite_sheet.png")
        result.sheet_image = artifacts / "sprite_sheet.png"

        # 3) 算法裁切为帧序列
        self._check_cancel()
        frames = fu.crop_sprite_sheet(sheet, params.grid_rows, params.grid_cols, params.frame_count)
        if not frames:
            raise WorkflowError("裁切精灵图失败（0 帧）", step="裁切")
        self._log_msg("info", f"已裁切 {len(frames)} 帧（{params.grid_rows}×{params.grid_cols} 网格，取前 {params.frame_count}）")
        # 去掉格子黑框：AI 常在格子间画黑线/边框，每格向内收缩 inset px（默认自动 4%）
        inset = int(params.cell_inset or 0)
        if not inset:
            cw = sheet.width / params.grid_cols
            ch = sheet.height / params.grid_rows
            inset = max(2, int(min(cw, ch) * 0.04))
        if inset > 0:
            frames = [f.crop((inset, inset, f.width - inset, f.height - inset)) for f in frames]
            self._log_msg("info", f"已去除格子边框（每格内缩 {inset}px）")
        # 统一帧尺寸到「单格尺寸」（服务商可能返回与请求不同的精灵图大小，NEAREST 保持像素色）
        target = params.cell_target_size()
        if frames[0].size != target:
            frames = [f.resize(target, Image.Resampling.NEAREST) for f in frames]
            self._log_msg("info", f"帧统一缩放到 {target[0]}x{target[1]}")
        # 首尾帧一致：末帧强制等于首帧，保证循环无缝（模型未对齐时由算法兜底）
        if params.loop_close and len(frames) >= 2:
            frames[-1] = frames[0].copy()
            self._log_msg("info", "首尾帧已对齐：末帧=首帧（循环无缝）")

        # 4) 扣除纯色背景（一键抠图）
        frames = self._step_key(params, frames)

        # 5) 导出
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
        keyed_sheet = fu.frames_to_sprite_sheet(frames, columns=params.grid_cols)
        result.sheet_path = fu.save_image(keyed_sheet, export_dir / "sprite_sheet.png")

        # 项目元数据
        meta = {
            "format": "pixel-sprite-sheet",
            "frame_count": len(frames),
            "grid": [params.grid_rows, params.grid_cols],
            "cell_size": list(frames[0].size) if frames else None,
            "fps": DEFAULT_FPS,
            "action": params.action,
            "loop_close": params.loop_close,
        }
        project_file = export_dir / "sprite_project.json"
        project_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        result.project_file = project_file

        result.step_log = list(self.step_log)
        self._log_msg("info", f"精灵图完成：{result.frame_count} 帧 @ {result.width}x{result.height}")
        return result

    # ------------------------------------------------------------------ #
    def _step_prompts(self, params: SpriteParams) -> Dict[str, str]:
        """LLM 生成提示词（失败降级本地模板），图片提示词经 STRICT 强化。"""
        desc, act = params.description, params.action

        def _call(max_tokens: int, strict: bool = False) -> APIResult:
            from core.processing.prompt_utils import STRICT_ANIMATION_CORRECTION

            system = SYSTEM_PROMPT + (STRICT_ANIMATION_CORRECTION if strict else "")
            return self.llm_api.call(
                prompt=build_user_prompt(desc, act),
                system=system,
                action=act,
                max_tokens=max_tokens,
            )

        result = _call(1600)
        data = self._extract_prompt_data(result)
        if data is None:
            self._log_msg("warn", "LLM 输出为空或不可解析，提高 max_tokens 重试一次")
            result = _call(4096)
            data = self._extract_prompt_data(result)
        if data is not None:
            anim = str(data.get("animation_prompt") or "")
            if len([w for w in anim.split() if w.strip()]) > 40:
                result = _call(1600, strict=True)
                data = self._extract_prompt_data(result)
            if data is not None:
                prompts = normalize_prompts(data, desc, act)
                return finalize_prompts(
                    prompts,
                    params.cell_target_size(),
                    "1:1",
                    params.max_colors,
                )
        self._log_msg("warn", f"LLM 调用失败（{result.message}），使用本地模板")
        return finalize_prompts(
            build_fallback_prompts(desc, act),
            params.cell_target_size(),
            "1:1",
            params.max_colors,
        )

    @staticmethod
    def _extract_prompt_data(result: APIResult) -> Optional[dict]:
        if not result.ok:
            return None
        if isinstance(result.data, dict):
            return result.data
        data = parse_json_response(result.data)
        return data or None

    # ------------------------------------------------------------------ #
    def _step_base(self, params: SpriteParams, image_prompt: str, path: Path) -> Image.Image:
        """文生对象底图：单只对象、纯白背景、主体完整。"""
        prompt_text = f"{params.description} {params.action} {image_prompt}"
        w, h = resolve_api_image_size(
            self.image_api.params.get("size"), "1:1", max(params.cell_size, 256), prompt_text
        )
        self._log_msg("info", f"生成对象底图：{w}x{h}")
        result = self.image_api.call(prompt=image_prompt, size=f"{w}x{h}", n=1)
        if not result.ok:
            raise WorkflowError(f"底图生成失败: {result.message}", step="文生底图")
        img = self._first_image(result)
        fu.save_image(img, path)
        self._log_msg("info", f"底图已保存: {path}")
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
        self._log_msg("info", f"生成精灵图：{params.grid_rows}×{params.grid_cols} 网格 / {total} 帧，{w}x{h}")
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
        self._log_msg("info", f"精灵图已保存: {path}（{img.width}x{img.height}）")
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
        """扣除纯色背景（一键抠图）：自适应背景归一化 + 颜色键抠图。"""
        if not params.remove_bg and not params.force_pure_bg:
            return frames
        out: List[Image.Image] = []
        for f in frames:
            self._check_cancel()
            img, mask = f, None
            if params.force_pure_bg:
                img, _fill, mask = bg_mod.normalize_background(img)
            if params.remove_bg:
                if mask is not None:
                    img = bg_mod.apply_background_mask(img, mask, feather=4)
                else:
                    img = bg_mod.remove_background(img, key_color=(255, 255, 255), tolerance=32, feather=4)
            out.append(img)
        self._log_msg("info", f"已扣除背景：{len(out)} 帧")
        return out
