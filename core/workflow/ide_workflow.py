"""IDE 模式分步工作流：每一步可独立执行、编辑、重跑。

与 Solo 的区别：IDE 各步骤解耦，中间结果（提示词 / 首帧 / 帧序列）保存在
IdeSession 中，可随时修改或从任一步重新执行；支持项目保存/加载与多种导出。

步骤：文本生成 → 图片生成 → 视频动画生成 → 像素化 → 背景去除 → 导出。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from config.settings import (
    ASPECT_RATIOS,
    DEFAULT_ASPECT,
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNT,
    DEFAULT_MAX_COLORS,
    DEFAULT_SPEED,
    EXPORT_PREFIX,
)
from core.api.base import APIResult, BaseAPI
from core.processing import background as bg_mod
from core.processing import frame_utils as fu
from core.processing import pixelizer as px
from core.processing.prompt_utils import (
    BACKGROUND_STABILITY_RULE,
    STRICT_ANIMATION_CORRECTION,
    SUBJECT_MARGIN_RULE,
    SYSTEM_PROMPT,
    build_fallback_prompts,
    build_user_prompt,
    normalize_prompts,
    parse_json_response,
)
from core.workflow.shared import finalize_prompts, resolve_api_image_size
from core.workflow.solo_workflow import WorkflowError
from ui.i18n import tr

logger = logging.getLogger("PixelAnimIDE.workflow.ide")

IDE_STEPS = ["文本生成", "图片生成", "视频动画生成", "像素化处理", "背景去除", "导出"]


@dataclass
class IdeSession:
    """IDE 工作区状态：输入参数 + 中间产物 + 可编辑帧序列。"""

    description: str = ""
    action: str = ""
    aspect_ratio: str = DEFAULT_ASPECT
    pixel_size: int = 128
    max_colors: int = DEFAULT_MAX_COLORS
    frame_count: int = DEFAULT_FRAME_COUNT
    fps: int = DEFAULT_FPS
    speed: float = DEFAULT_SPEED
    loop_close: bool = True
    force_pure_bg: bool = True
    remove_bg: bool = True
    pixelate: bool = True
    bg_tolerance: int = 30
    bg_feather: int = 8
    bg_erode: int = 0
    video_image_max_side: int = 512
    video_image_min_side: int = 256
    output_dir: Path = field(default_factory=lambda: Path("output"))
    name: str = "untitled"

    # 中间产物
    prompts: Dict[str, str] = field(default_factory=dict)
    first_frame: Optional[Image.Image] = None
    reference_image: Optional[Image.Image] = None  # 图生图参考图（用户自备）
    frames: List[Image.Image] = field(default_factory=list)
    video_path: Optional[str] = None
    video_duration: Optional[float] = None

    def target_size(self) -> Tuple[int, int]:
        """按宽高比与像素边长计算目标画布尺寸。"""
        rw, rh = ASPECT_RATIOS.get(self.aspect_ratio, ASPECT_RATIOS[DEFAULT_ASPECT])
        if rw >= rh:
            return self.pixel_size, max(1, round(self.pixel_size * rh / rw))
        return max(1, round(self.pixel_size * rw / rh)), self.pixel_size

    # ------------------------------------------------------------------ #
    # 帧编辑（序列操作，UI 时间轴直接调用）
    # ------------------------------------------------------------------ #
    def insert_frame(self, index: int, frame: Image.Image) -> None:
        self.frames.insert(max(0, min(index, len(self.frames))), frame.convert("RGBA"))

    def delete_frame(self, index: int) -> Optional[Image.Image]:
        if not self.frames or not (0 <= index < len(self.frames)):
            return None
        return self.frames.pop(index)

    def duplicate_frame(self, index: int) -> Optional[Image.Image]:
        if not self.frames or not (0 <= index < len(self.frames)):
            return None
        dup = self.frames[index].copy()
        self.frames.insert(index + 1, dup)
        return dup

    def move_frame(self, src: int, dst: int) -> None:
        if not self.frames or not (0 <= src < len(self.frames)):
            return
        dst = max(0, min(dst, len(self.frames) - 1))
        if src == dst:
            return
        frame = self.frames.pop(src)
        self.frames.insert(dst, frame)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "aspect_ratio": self.aspect_ratio,
            "pixel_size": self.pixel_size,
            "max_colors": self.max_colors,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "speed": self.speed,
            "loop_close": self.loop_close,
            "force_pure_bg": self.force_pure_bg,
            "remove_bg": self.remove_bg,
            "pixelate": self.pixelate,
            "bg_tolerance": self.bg_tolerance,
            "bg_feather": self.bg_feather,
            "bg_erode": self.bg_erode,
            "video_image_max_side": self.video_image_max_side,
            "video_image_min_side": self.video_image_min_side,
            "output_dir": str(self.output_dir),
            "prompts": self.prompts,
            "video_path": self.video_path,
            "video_duration": self.video_duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdeSession":
        known = {
            "description", "action", "aspect_ratio", "pixel_size", "max_colors",
            "frame_count", "fps", "speed", "loop_close", "force_pure_bg",
            "remove_bg", "pixelate", "bg_tolerance", "bg_feather", "bg_erode",
            "video_image_max_side", "video_image_min_side", "output_dir",
            "name", "prompts", "video_path", "video_duration",
        }
        cleaned = {k: v for k, v in data.items() if k in known}
        if isinstance(cleaned.get("output_dir"), str):
            cleaned["output_dir"] = Path(cleaned["output_dir"])
        return cls(**cleaned)


class IdeWorkflow:
    """IDE 分步流程执行器：每步独立，写入 session 并返回结果。"""

    def __init__(
        self,
        llm_api: BaseAPI,
        image_api: BaseAPI,
        video_api: BaseAPI,
        log: Optional[Callable[[str, str], None]] = None,
        cancel: Optional[threading.Event] = None,
    ):
        self.llm_api = llm_api
        self.image_api = image_api
        self.video_api = video_api
        self._log = log
        self._cancel = cancel or threading.Event()

    # ------------------------------------------------------------------ #
    def _log_msg(self, level: str, message: str) -> None:
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

    @staticmethod
    def _extract_prompt_data(result: APIResult) -> Optional[dict]:
        if not result.ok:
            return None
        if isinstance(result.data, dict):
            return result.data
        data = parse_json_response(result.data)
        return data or None

    # ------------------------------------------------------------------ #
    # 步骤 1：文本生成
    # ------------------------------------------------------------------ #
    def step_prompts(self, session: IdeSession, description: Optional[str] = None, action: Optional[str] = None) -> dict:
        """生成提示词，写入 session.prompts 并返回。"""
        if description is not None and str(description).strip():
            session.description = str(description).strip()
        if action is not None:
            session.action = str(action or "").strip()
        desc, act = session.description, session.action

        def _call(max_tokens: int, strict: bool = False) -> APIResult:
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
                self._log_msg("warn", "动画提示词过于冗长，按「简洁且忠实于动作」重试一次")
                result = _call(1600, strict=True)
                data = self._extract_prompt_data(result)
            if data is not None:
                prompts = normalize_prompts(data, desc, act)
                session.prompts = finalize_prompts(
                    prompts, session.target_size(), session.aspect_ratio, session.max_colors
                )
                self._log_msg("info", tr("提示词生成成功"))
                return dict(session.prompts)

        self._log_msg("warn", f"LLM 调用失败（{result.message}），使用本地模板")
        session.prompts = finalize_prompts(
            build_fallback_prompts(desc, act),
            session.target_size(),
            session.aspect_ratio,
            session.max_colors,
        )
        return dict(session.prompts)

    # ------------------------------------------------------------------ #
    # 步骤 2：图片生成
    # ------------------------------------------------------------------ #
    def step_image(self, session: IdeSession, prompt: Optional[str] = None, reference: Optional[Image.Image] = None) -> Image.Image:
        """生成首帧图片（可选参考图/图生图），写入 session.first_frame 并返回。

        reference 为参考图 PIL 图像；未显式传入时使用 session.reference_image。
        """
        img_prompt = (prompt or session.prompts.get("image_prompt") or "").strip()
        if not img_prompt:
            raise WorkflowError("请先生成或填写图片提示词", step="图片生成")
        ref = reference if reference is not None else session.reference_image
        prompt_text = f"{session.description} {session.action} {img_prompt}"
        w, h = resolve_api_image_size(
            self.image_api.params.get("size"),
            session.aspect_ratio,
            session.pixel_size,
            prompt_text,
        )
        self._log_msg("info", f"请求生图尺寸 {w}x{h}" + ("（含参考图，图生图）" if ref is not None else ""))
        ref_bytes = fu.image_to_bytes(ref, "PNG") if ref is not None else None
        result = self.image_api.call(
            prompt=img_prompt, size=f"{w}x{h}", n=1, image=ref_bytes
        )
        if not result.ok:
            raise WorkflowError(f"首帧图片生成失败: {result.message}", step="图片生成")
        images = (result.data or {}).get("images") or []
        urls = (result.data or {}).get("urls") or []
        if images:
            data = images[0]
        elif urls:
            self._log_msg("info", f"下载生图结果: {urls[0]}")
            data = fu.download_bytes(urls[0])
        else:
            raise WorkflowError("生图接口未返回任何图片", step="图片生成")
        img = fu.bytes_to_image(data)
        if session.force_pure_bg:
            whitened, _fill, mask = bg_mod.normalize_background(img)
            if mask is not None:
                img = whitened
                self._log_msg("info", "首帧背景已归一化（纯色）")
        session.first_frame = img
        self._log_msg("info", f"首帧图片已生成（{img.width}x{img.height}）")
        return img

    # ------------------------------------------------------------------ #
    # 步骤 3：视频动画生成
    # ------------------------------------------------------------------ #
    def step_animation(self, session: IdeSession, prompt: Optional[str] = None) -> List[Image.Image]:
        """图转视频 → 拆帧/采样，写入 session.frames 并返回。"""
        if session.first_frame is None:
            raise WorkflowError("请先生成或导入首帧图片", step="动画生成")
        anim_prompt = (prompt or session.prompts.get("animation_prompt") or "smooth looping animation").strip()
        parts = [anim_prompt, SUBJECT_MARGIN_RULE]
        if session.force_pure_bg:
            parts.append(BACKGROUND_STABILITY_RULE)
        anim_prompt = " ".join(parts)
        first_bytes = fu.image_to_bytes(session.first_frame, "PNG")
        if session.video_image_min_side:
            # 首帧过小时最近邻放大到 API 最低要求（像素画不模糊）
            first_bytes = fu.upscale_to_min_side_bytes(first_bytes, min_side=session.video_image_min_side)
        if 0 < session.video_image_max_side < 4096:
            first_bytes = fu.downscale_bytes(first_bytes, max_side=session.video_image_max_side)

        result = self.video_api.call(
            image_bytes=first_bytes,
            prompt=anim_prompt,
            frames=session.frame_count,
            fps=session.fps,
        )
        if not result.ok:
            raise WorkflowError(f"动画生成失败: {result.message}", step="动画生成")

        data = result.data or {}
        frames_bytes: List[bytes] = data.get("frames") or []
        video_url: Optional[str] = data.get("video_url")

        if frames_bytes:
            frames = [fu.bytes_to_image(b) for b in frames_bytes]
        elif video_url:
            self._log_msg("info", f"下载视频: {video_url}")
            artifacts = session.output_dir / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            raw_video = artifacts / "video.mp4"
            raw_video.write_bytes(fu.download_bytes(video_url))
            # 生成的视频应无声：remux 去除音轨
            video_path = fu.strip_audio(raw_video, artifacts / "video_silent.mp4")
            session.video_path = str(video_path)
            frames, meta = fu.extract_video_frames_meta(
                video_path, max_frames=session.frame_count * 3
            )
            session.video_duration = meta.get("duration")
        else:
            raise WorkflowError("图转视频接口未返回帧序列或视频 URL", step="动画生成")

        frames = fu.sample_loop_frames(frames, session.frame_count, loop=session.loop_close)
        if not frames:
            raise WorkflowError("动画结果为空（0 帧）", step="动画生成")

        # 按实际时长校准输出帧率（1x 原速 × 用户倍速）
        if session.video_duration and session.video_duration > 0:
            base_fps = max(1, min(30, round(len(frames) / session.video_duration)))
            session.fps = max(1, min(30, round(base_fps * session.speed)))

        session.frames = frames
        self._log_msg("info", f"动画生成：{len(frames)} 帧 @ {session.fps}fps")
        return list(frames)

    # ------------------------------------------------------------------ #
    # 步骤 4：像素化
    # ------------------------------------------------------------------ #
    def step_pixelize(self, session: IdeSession) -> List[Image.Image]:
        """像素化 session.frames（就地替换），返回结果。"""
        if not session.frames:
            raise WorkflowError("没有可像素化的帧", step="像素化处理")
        if not session.pixelate:
            self._log_msg("info", "已跳过像素化（选项关闭）")
            return list(session.frames)

        target = session.target_size()
        self._log_msg("info", f"像素化：目标 {target}，颜色上限 {session.max_colors}")
        seq = px.perfect_pixelize_sequence(session.frames)
        if seq is not None:
            sampled, grid = seq
            self._log_msg("info", f"像素风（网格 {grid[0]}×{grid[1]}）：首帧定网格，全部帧硬缩放")
            base = sampled
        else:
            self._log_msg("info", "非像素风格：跳过完美像素，目标尺寸缩放")
            base = [px.resize_nearest(f, target) for f in session.frames]

        base = [px.resize_nearest(f, target) for f in base]
        eff_colors = session.max_colors
        if base:
            unique = len(set(base[0].convert("RGB").getdata()))
            eff_colors = max(2, min(unique, session.max_colors))
        palette = px.extract_dominant_palette(base[0], eff_colors) if base else None
        session.frames = px.pixelize_frames(
            base, px.PixelizeParams(max_colors=eff_colors, edge_clean=True, palette=palette)
        )
        return list(session.frames)

    # ------------------------------------------------------------------ #
    # 步骤 5：背景去除
    # ------------------------------------------------------------------ #
    def step_background(self, session: IdeSession) -> List[Image.Image]:
        """背景去除/归一化 session.frames（就地替换），返回结果。"""
        if not session.remove_bg and not session.force_pure_bg:
            return list(session.frames)
        out: List[Image.Image] = []
        for f in session.frames:
            self._check_cancel()
            img, mask = f, None
            if session.force_pure_bg:
                img, _fill, mask = bg_mod.normalize_background(img)
            if session.remove_bg:
                if mask is not None:
                    img = bg_mod.apply_background_mask(
                        img, mask, feather=session.bg_feather, erode=session.bg_erode
                    )
                else:
                    img = bg_mod.remove_background(
                        img, key_color=(255, 255, 255), tolerance=session.bg_tolerance,
                        feather=session.bg_feather, erode=session.bg_erode,
                    )
            out.append(img)
        session.frames = out
        return list(out)

    # ------------------------------------------------------------------ #
    # 步骤 6：导出
    # ------------------------------------------------------------------ #
    def export(
        self,
        session: IdeSession,
        export_dir: Optional[Path | str] = None,
        fps: Optional[int] = None,
        formats: Tuple[str, ...] = ("gif", "png", "json"),
    ) -> dict:
        """导出帧序列，返回路径 dict（keys: gif/png_dir/sprite/metadata）。"""
        if not session.frames:
            raise WorkflowError("没有可导出的帧", step="导出")
        out = Path(export_dir or (session.output_dir / "export"))
        out.mkdir(parents=True, exist_ok=True)
        fps = int(fps or session.fps or DEFAULT_FPS)
        paths: Dict[str, str] = {}
        if "png" in formats:
            png_dir = out / "png"
            fu.save_png_sequence(session.frames, png_dir, prefix=EXPORT_PREFIX)
            paths["png_dir"] = str(png_dir)
        if "gif" in formats:
            paths["gif"] = str(fu.frames_to_gif(session.frames, out / f"{EXPORT_PREFIX}.gif", fps=fps))
        if "apng" in formats:
            paths["apng"] = str(fu.frames_to_apng(session.frames, out / f"{EXPORT_PREFIX}.apng", fps=fps))
        if "sprite" in formats:
            paths["sprite"] = str(fu.save_image(fu.frames_to_sprite_sheet(session.frames), out / "sprite_sheet.png"))
        if "json" in formats:
            meta = fu.animation_meta(session.frames, fps)
            meta_file = out / "metadata.json"
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            paths["metadata"] = str(meta_file)
        return paths


# --------------------------------------------------------------------------- #
# 项目保存 / 加载
# --------------------------------------------------------------------------- #
def save_ide_project(session: IdeSession, project_dir: Path | str) -> Path:
    """保存 IDE 项目：帧序列 PNG + first_frame + ide_project.json。"""
    project_dir = Path(project_dir)
    frames_dir = project_dir / "frames"
    fu.save_png_sequence(session.frames, frames_dir, prefix="frame")

    data = session.to_dict()
    data["frames_dir"] = "frames"
    data["frame_count"] = len(session.frames)
    if session.frames:
        data["width"], data["height"] = session.frames[0].size
    else:
        data["width"], data["height"] = 0, 0
    if session.first_frame is not None:
        fu.save_image(session.first_frame, project_dir / "first_frame.png")
        data["first_frame"] = "first_frame.png"
    else:
        data["first_frame"] = None

    project_file = project_dir / "ide_project.json"
    project_file.parent.mkdir(parents=True, exist_ok=True)
    project_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("IDE 项目已保存: %s", project_file)
    return project_file


def load_ide_project(project_dir: Path | str) -> IdeSession:
    """加载 IDE 项目（帧序列 + first_frame + 元数据）。"""
    project_dir = Path(project_dir)
    data = json.loads((project_dir / "ide_project.json").read_text(encoding="utf-8"))
    session = IdeSession.from_dict(data)
    frames_dir = project_dir / str(data.get("frames_dir", "frames"))
    if frames_dir.exists():
        session.frames = [fu.load_image(p) for p in sorted(frames_dir.glob("*.png"))]
    first = data.get("first_frame")
    if first:
        p = project_dir / str(first)
        if p.exists():
            session.first_frame = fu.load_image(p)
    return session
