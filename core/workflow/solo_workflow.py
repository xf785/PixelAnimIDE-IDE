"""Solo 模式全自动工作流（阶段1 MVP）。

链路：文本描述 -> [LLM] 提示词 -> [生图] 首帧 -> [图转视频] 动画
     -> [像素化] 严格像素化 -> [背景去除] 透明背景 -> [导出] GIF / PNG 序列帧。

设计要点：
- 纯同步实现，可在线程（QThread）中运行，通过回调上报进度/日志。
- cancel 事件（threading.Event）支持取消，取消点分布在各步骤与帧循环中。
- 每一步的中间产物都保留在输出目录，失败后可断点重试（阶段1记录，阶段2完善）。
- LLM 失败时可降级为本地模板提示词，保证流程不中断。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from config.settings import (
    ASPECT_RATIOS,
    DEFAULT_ASPECT,
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNT,
    EXPORT_PREFIX,
)
from core.api.base import BaseAPI
from core.processing import background as bg_mod
from core.processing import frame_utils as fu
from core.processing import pixelizer as px
from ui.i18n import tr
from core.processing.prompt_utils import (
    BACKGROUND_STABILITY_RULE,
    BACKGROUND_STABILITY_RULE_DARK,
    SUBJECT_MARGIN_RULE,
    build_fallback_prompts,
    is_pixel_prompt,
    normalize_prompts,
    recommended_frames,
)
from core.storage.project import Project, save_project
from core.workflow.shared import finalize_prompts, generate_prompt_data, resolve_api_image_size

logger = logging.getLogger("PixelAnimIDE.workflow.solo")

RGB = Tuple[int, int, int]

STEPS = ["提示词生成", "首帧图片生成", "动画生成", "像素化处理", "背景去除", "导出"]


def _parse_rec(data: dict) -> Dict[str, Optional[int]]:
    """从 LLM 返回中解析建议的动画参数（frame_count/fps/duration_seconds），非法值置 None。

    支持两种写法：
    - 直接给 frame_count / fps；
    - 给 duration_seconds（+可选 fps），换算 frame_count = round(duration * fps)。
    """
    rec: Dict[str, Optional[int]] = {"frame_count": None, "fps": None}

    def _int(value, lo, hi):
        try:
            v = int(value)
            return v if lo <= v <= hi else None
        except (TypeError, ValueError):
            return None

    fps = _int(data.get("fps"), 4, 30)
    frames = _int(data.get("frame_count"), 4, 120)
    duration = data.get("duration_seconds")
    if duration is not None and frames is None:
        try:
            secs = float(duration)
            if 0.3 <= secs <= 6.0:
                frames = max(4, min(120, int(round(secs * (fps or 8)))))
        except (TypeError, ValueError):
            pass
    rec["fps"] = fps
    rec["frame_count"] = frames
    return rec


class WorkflowError(Exception):
    """工作流执行失败（带步骤上下文）。"""

    def __init__(self, message: str, step: str = ""):
        super().__init__(message)
        self.message = message
        self.step = step


class WorkflowCancelled(Exception):
    """用户取消。"""


@dataclass
class SoloParams:
    """Solo 模式输入参数。"""

    description: str
    action: str = ""                       # 动作类型/动画提示词（可选，预设名或自由文本）
    aspect_ratio: str = DEFAULT_ASPECT     # 如 "1:1"、"16:9"
    pixel_size: int = 128                  # 像素化目标边长（长边）
    frame_count: int = DEFAULT_FRAME_COUNT       # 默认 1s（8 帧 @ 8fps），LLM 按动作自动调整
    fps: int = DEFAULT_FPS
    speed: float = 1.0                     # 播放倍速：0.5 / 1 / 1.5 / 2（AI 动作偏慢，可提速播放）
    loop_close: bool = True                # 首尾帧一致（循环闭合）
    force_pure_bg: bool = True             # 强制纯色背景（主体浅色时自动黑底，否则白底）
    video_image_max_side: int = 512        # 发给视频 API 的首帧图长边上限（图片按像素计费，缩小可省 token）
    video_image_min_side: int = 256        # 首帧图长边低于该值时最近邻放大到该值（满足 API 最低分辨率，不模糊像素）
    max_colors: int = 16
    pixelate: bool = True
    remove_bg: bool = True
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    bg_tolerance: int = 30
    bg_feather: int = 8
    reference_image: Optional[Path] = None   # 图生图参考图（用户自备）
    output_dir: Path = field(default_factory=lambda: Path("output"))
    export_gif: bool = True
    export_png: bool = True
    export_apng: bool = False
    export_sprite: bool = False
    save_intermediate: bool = True

    def target_size(self) -> Tuple[int, int]:
        """按宽高比与像素边长计算目标画布尺寸。"""
        rw, rh = ASPECT_RATIOS.get(self.aspect_ratio, ASPECT_RATIOS[DEFAULT_ASPECT])
        if rw >= rh:
            return self.pixel_size, max(1, round(self.pixel_size * rh / rw))
        return max(1, round(self.pixel_size * rw / rh)), self.pixel_size


@dataclass
class SoloResult:
    """Solo 工作流输出。"""

    output_dir: Path
    fps: int = 8
    video_duration: Optional[float] = None   # 源视频实际时长（秒），未知为 None
    prompts: Dict[str, str] = field(default_factory=dict)
    first_frame: Optional[Path] = None
    video_path: Optional[Path] = None
    frames_dir: Optional[Path] = None
    gif_path: Optional[Path] = None
    png_dir: Optional[Path] = None
    apng_path: Optional[Path] = None
    sprite_path: Optional[Path] = None
    project_file: Optional[Path] = None
    frame_count: int = 0
    width: int = 0
    height: int = 0
    # 完美像素原生（网格）分辨率版本：与用户预设分辨率并存
    native_width: int = 0
    native_height: int = 0
    native_gif_path: Optional[Path] = None
    native_png_dir: Optional[Path] = None
    step_log: List[str] = field(default_factory=list)


class SoloWorkflow:
    """Solo 全自动流程执行器。"""

    def __init__(
        self,
        llm_api: BaseAPI,
        image_api: BaseAPI,
        video_api: BaseAPI,
        params: SoloParams,
        progress: Optional[Callable[[int, int, str, float, str], None]] = None,
        log: Optional[Callable[[str, str], None]] = None,
        cancel: Optional[threading.Event] = None,
        on_prompts: Optional[Callable[[Dict[str, str]], None]] = None,
        on_first_frame: Optional[Callable[[Path], None]] = None,
    ):
        """on_prompts / on_first_frame：中间产物就绪时回调（可用于 UI 实时展示）。"""
        self.llm_api = llm_api
        self.image_api = image_api
        self.video_api = video_api
        self.params = params
        self._progress = progress
        self._log = log
        self._cancel = cancel or threading.Event()
        self._on_prompts_cb = on_prompts
        self._on_first_frame_cb = on_first_frame
        self.step_log: List[str] = []

    # ------------------------------------------------------------------ #
    def run(self) -> SoloResult:
        """执行完整 Solo 流程，返回结果。失败抛 WorkflowError / WorkflowCancelled。"""
        p = self.params
        self._log_msg("info", tr("开始 Solo 流程：{desc}").format(desc=p.description[:80]))
        out = Path(p.output_dir)
        artifacts = out / "artifacts"
        export_dir = out / "export"
        frames_dir = out / "frames"
        artifacts.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        result = SoloResult(output_dir=out, fps=p.fps)
        total = len(STEPS)

        try:
            # ---------- 1. 提示词生成 ----------
            self._begin(0, total, "提示词生成")
            prompts, rec = self._step_prompts()
            result.prompts = prompts
            self._apply_llm_params(rec)  # LLM 按动作建议调整帧数/帧率（用户改过则不覆盖）
            self._save_json(artifacts / "prompts.json", prompts)
            self._report(0, total, "提示词生成", 1.0, "完成")

            # ---------- 2. 首帧图片生成 ----------
            self._begin(1, total, "首帧图片生成")
            self._check_cancel()
            first_frame_bytes, first_frame_path, bg_fill = self._step_first_frame(
                prompts["image_prompt"], artifacts
            )
            result.first_frame = first_frame_path
            self._report(1, total, "首帧图片生成", 1.0, str(first_frame_path))

            # ---------- 3. 动画生成（图转视频/帧序列） ----------
            self._begin(2, total, "动画生成")
            self._check_cancel()
            raw_frames, video_path, eff_fps, video_duration = self._step_animation(
                first_frame_bytes, prompts["animation_prompt"], artifacts, bg_fill=bg_fill
            )
            result.video_path = video_path
            result.video_duration = video_duration
            result.fps = eff_fps
            if p.save_intermediate:
                fu.save_png_sequence(raw_frames, artifacts / "frames_raw", prefix="raw")
            self._report(2, total, "动画生成", 1.0, f"{len(raw_frames)} 帧（{eff_fps}fps）")

            # ---------- 4. 像素化 ----------
            self._begin(3, total, "像素化处理")
            self._check_cancel()
            pix_params = px.PixelizeParams(
                target_size=p.target_size(),
                max_colors=p.max_colors,
                edge_clean=True,
            )
            frames, native_frames = self._step_pixelize(raw_frames, pix_params)
            result.frame_count = len(frames)
            if frames:
                result.width, result.height = frames[0].size
            if native_frames:
                result.native_width, result.native_height = native_frames[0].size
            if p.save_intermediate:
                fu.save_png_sequence(frames, artifacts / "frames_pixel", prefix="pix")
                if native_frames:
                    fu.save_png_sequence(native_frames, artifacts / "frames_pixel_native", prefix="pix")
            self._report(3, total, "像素化处理", 1.0, f"{result.width}x{result.height}")

            # ---------- 5. 背景去除 ----------
            self._begin(4, total, "背景去除")
            self._check_cancel()
            frames = self._step_background(frames)
            if native_frames:
                native_frames = self._step_background(native_frames)
            fu.save_png_sequence(frames, frames_dir, prefix="frame")
            result.frames_dir = frames_dir
            self._report(4, total, "背景去除", 1.0, "完成")

            # ---------- 6. 导出 ----------
            self._begin(5, total, "导出")
            self._check_cancel()
            self._step_export(frames, native_frames, export_dir, result)
            self._save_project(export_dir, result)
            self._report(5, total, "导出", 1.0, "完成")

        except WorkflowCancelled:
            self._log_msg("warn", tr("流程已取消"))
            raise
        except WorkflowError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Solo 流程异常")
            raise WorkflowError(str(exc)) from exc

        result.step_log = list(self.step_log)
        self._log_msg("info", tr("Solo 流程完成"))
        return result

    # ------------------------------------------------------------------ #
    # 步骤实现
    # ------------------------------------------------------------------ #
    def _finalize_prompts(self, prompts: Dict[str, str]) -> Dict[str, str]:
        """内置强制项：像素尺寸 + 颜色数量 + 纯白背景，严格写入图片提示词。"""
        p = self.params
        return finalize_prompts(prompts, p.target_size(), p.aspect_ratio, p.max_colors)

    def _step_prompts(self) -> Tuple[Dict[str, str], Dict[str, Optional[int]]]:
        """生成提示词并解析 LLM 建议的动画参数。

        返回 (prompts, rec)，rec 含 LLM 按动作推荐的 frame_count/fps（可能为 None）。
        调用/重试/严格纠正逻辑统一在 core.workflow.shared.generate_prompt_data；
        此处负责解析动画参数、归一化 + 内置强制项，以及失败时降级本地模板。
        """
        p = self.params
        data, last = generate_prompt_data(
            self.llm_api, p.description, p.action, log=self._log_msg
        )
        if data is not None:
            rec = _parse_rec(data)
            prompts = self._finalize_prompts(normalize_prompts(data, p.description, p.action))
            self._log_msg("info", tr("提示词生成成功"))
            self._notify_prompts(prompts)
            return prompts, rec
        if last.ok:
            self._log_msg("warn", tr("LLM 返回无法解析，使用本地模板"))
        else:
            self._log_msg("warn", tr("LLM 调用失败（{0}），使用本地模板").format(last.message))
        prompts = self._finalize_prompts(build_fallback_prompts(p.description, p.action))
        self._notify_prompts(prompts)
        # 本地模板按动作类别给出建议（帧率保持用户/默认）
        frames = recommended_frames(p.action, p.fps)
        rec = {"frame_count": frames, "fps": None}
        return prompts, rec

    def _apply_llm_params(self, rec: Dict[str, Optional[int]]) -> None:
        """把 LLM 建议的动画参数应用到本次运行（用户显式改过的值不被覆盖）。"""
        p = self.params
        changed = []
        if rec.get("frame_count") and p.frame_count == DEFAULT_FRAME_COUNT:
            p.frame_count = int(rec["frame_count"])
            changed.append(f"帧数 {DEFAULT_FRAME_COUNT}→{p.frame_count}")
        if rec.get("fps") and p.fps == DEFAULT_FPS:
            p.fps = int(rec["fps"])
            changed.append(f"帧率 {DEFAULT_FPS}→{p.fps}")
        if changed:
            self._log_msg(
                "info",
                tr("LLM 已按动作建议调整动画参数：{0}（可下次生成前手动修改）").format(
                    "、".join(changed)
                ),
            )

    def _notify_prompts(self, prompts: Dict[str, str]) -> None:
        if self._on_prompts_cb:
            try:
                self._on_prompts_cb(dict(prompts))
            except Exception:  # noqa: BLE001
                pass

    def _step_first_frame(self, image_prompt: str, artifacts: Path) -> Tuple[bytes, Path, Optional[RGB]]:
        """生首帧图。返回 (发送字节, 保存路径, 背景填充色或 None)。

        bg_fill 为 (0,0,0)（浅色主体归一化成黑底）或 (255,255,255)（白底）或 None
        （未归一化）——后续动画提示词的「背景稳定」约束必须与它一致，
        否则提示词说白底而首帧是黑底会让模型困惑、背景漂移。
        """
        p = self.params
        w, h = self._api_image_size(image_prompt)
        ref_bytes = None
        if p.reference_image:
            try:
                ref_bytes = fu.image_to_bytes(fu.load_image(p.reference_image), "PNG")
                self._log_msg("info", tr("已附加参考图（图生图）: {0}").format(p.reference_image))
            except Exception as exc:  # noqa: BLE001
                self._log_msg("warn", tr("参考图读取失败，忽略: {0}").format(exc))
        result = self.image_api.call(prompt=image_prompt, size=f"{w}x{h}", n=1, image=ref_bytes)
        if not result.ok:
            raise WorkflowError(tr("首帧图片生成失败: {0}").format(result.message), step="首帧图片生成")
        images = (result.data or {}).get("images") or []
        urls = (result.data or {}).get("urls") or []
        if images:
            data = images[0]
        elif urls:
            self._log_msg("info", tr("下载生图结果: {0}").format(urls[0]))
            data = fu.download_bytes(urls[0])
        else:
            raise WorkflowError(tr("生图接口未返回任何图片"), step="首帧图片生成")
        img = fu.bytes_to_image(data)
        bg_fill: Optional[RGB] = None
        if p.force_pure_bg:
            whitened, fill, mask = bg_mod.normalize_background(img)
            if mask is not None:
                img = whitened
                data = fu.image_to_bytes(img, "PNG")  # 重新编码，发送给视频 API 的也是纯色背景
                bg_fill = fill
                self._log_msg(
                    "info",
                    tr("首帧背景已归一化：{0}").format(
                        tr("主体浅色 → 黑底") if fill == (0, 0, 0) else tr("主体正常 → 白底")
                    ),
                )
        path = fu.save_image(img, artifacts / "first_frame.png")
        self._log_msg("info", tr("首帧已保存: {0}").format(path))
        if self._on_first_frame_cb:
            try:
                self._on_first_frame_cb(path)
            except Exception:  # noqa: BLE001
                pass
        return data, path, bg_fill

    def _api_image_size(self, image_prompt: str = "") -> Tuple[int, int]:
        """生图 API 的请求尺寸（复用共享实现，仅补像素风日志）。"""
        p = self.params
        prompt_text = f"{p.description} {p.action} {image_prompt}"
        cfg_size = self.image_api.params.get("size")
        if not cfg_size and is_pixel_prompt(prompt_text):
            size = resolve_api_image_size(None, p.aspect_ratio, p.pixel_size, prompt_text)
            self._log_msg("info", tr("检测到像素风格意图，生图尺寸强制为 {0}x{1}").format(size[0], size[1]))
            return size
        return resolve_api_image_size(cfg_size, p.aspect_ratio, p.pixel_size, prompt_text)

    def _animation_prompt(self, prompt: str, bg_fill: Optional[RGB] = None) -> str:
        """动画提示词（图转视频）：附加主体完整性约束；
        背景归一化为纯色时附加与「实际背景色」一致的稳定性约束。"""
        parts = [prompt, SUBJECT_MARGIN_RULE]
        if self.params.force_pure_bg:
            # 首帧被归一化成黑底时用黑底规则，否则默认白底规则——
            # 规则必须与实际首帧背景一致，否则模型会在中间帧把背景漂移回去
            rule = BACKGROUND_STABILITY_RULE_DARK if bg_fill == (0, 0, 0) else BACKGROUND_STABILITY_RULE
            parts.append(rule)
        return " ".join(parts)

    def _step_animation(
        self,
        first_frame_bytes: bytes,
        animation_prompt: str,
        artifacts: Path,
        bg_fill: Optional[RGB] = None,
    ) -> Tuple[List[Image.Image], Optional[Path], int, Optional[float]]:
        """图转视频 -> 拆帧/采样。

        返回 (frames, video_path, effective_fps, video_duration)：
        - 抽取完整视频（不截断到请求时长），按「保留首帧与尾帧、中间均匀采样」
          策略抽到目标帧数，保证完整动作都体现；
        - 输出帧率按视频实际时长校准，保证动画与原视频 1x 速度一致
          （例如视频只有 1s、用户却要 2s 的动画时，会自动提速播放）；
        - 服务商直接返回帧序列时无法得知真实时长，使用请求帧率。
        """
        p = self.params
        # 首帧过小时最近邻放大到 API 最低要求（像素画不模糊）
        if p.video_image_min_side:
            up = fu.upscale_to_min_side_bytes(first_frame_bytes, min_side=p.video_image_min_side)
            if len(up) != len(first_frame_bytes):
                self._log_msg("info", tr("首帧过小，已最近邻放大至长边 ≥{0}px（像素保持锐利）").format(p.video_image_min_side))
            first_frame_bytes = up
        # 首帧缩放到长边 ≤ video_image_max_side 再发送：图片按像素计费，缩小可省大量 token
        if p.video_image_max_side and p.video_image_max_side < 4096:
            sent_frame = fu.downscale_bytes(first_frame_bytes, max_side=p.video_image_max_side)
            if len(sent_frame) != len(first_frame_bytes):
                self._log_msg("info", tr("首帧已缩放至长边 ≤{0}px 再发送（节省图片 token）").format(p.video_image_max_side))
            first_frame_bytes = sent_frame
        result = self.video_api.call(
            image_bytes=first_frame_bytes,
            prompt=self._animation_prompt(animation_prompt, bg_fill),
            frames=p.frame_count,
            fps=p.fps,
        )
        if not result.ok:
            raise WorkflowError(tr("动画生成失败: {0}").format(result.message), step="动画生成")

        data = result.data or {}
        frames_bytes: List[bytes] = data.get("frames") or []
        video_url: Optional[str] = data.get("video_url")
        video_path: Optional[Path] = None
        video_duration: Optional[float] = None
        requested_secs = p.frame_count / max(1, p.fps)  # 用户期望的动画时长

        if frames_bytes:
            self._log_msg("info", tr("图转视频 API 直接返回 {0} 帧").format(len(frames_bytes)))
            frames = [fu.bytes_to_image(b) for b in frames_bytes]
        elif video_url:
            self._log_msg("info", tr("下载视频: {0}").format(video_url))
            raw_video = artifacts / "video.mp4"
            raw_video.write_bytes(fu.download_bytes(video_url))
            # 生成的视频应无声：remux 去除音轨（最佳努力，失败保留原视频）
            video_path = fu.strip_audio(raw_video, artifacts / "video_silent.mp4")
            self._log_msg("info", tr("视频已静音: {0}").format(video_path))
            frames, meta = fu.extract_video_frames_meta(
                video_path,
                max_frames=p.frame_count * 3,
            )
            video_duration = meta.get("duration")
            duration_txt = f"{video_duration:.2f}s" if video_duration else tr("未知")
            self._log_msg(
                "info",
                tr("视频拆帧 {0} 帧（实际时长约 {1}，请求片段 {2:.2f}s）").format(
                    len(frames), duration_txt, requested_secs
                ),
            )
        else:
            raise WorkflowError(tr("图转视频接口未返回帧序列或视频 URL"), step="动画生成")

        # 去除完全相同的连续帧（AI 视频常以静态起/尾帧收尾）：动画更紧凑、循环更顺滑
        deduped = fu.dedupe_frames(frames)
        if len(deduped) < len(frames):
            self._log_msg("info", tr("已去除 {0} 帧近似重复的连续帧（静态停留）").format(len(frames) - len(deduped)))
        frames = deduped

        frames = fu.sample_loop_frames(frames, p.frame_count, loop=p.loop_close)
        if not frames:
            raise WorkflowError(tr("动画结果为空（0 帧）"), step="动画生成")
        if len(frames) < p.frame_count:
            self._log_msg("warn", tr("帧数不足（{0}/{1}），按实际帧数继续").format(len(frames), p.frame_count))
        if p.loop_close and len(frames) >= 2:
            self._log_msg("info", tr("已做循环闭合：首尾帧保持一致"))

        # 按实际时长校准输出帧率：先保证与原视频 1x 速度一致，再乘用户倍速
        if video_duration and video_duration > 0:
            base_fps = round(len(frames) / video_duration)
            base_fps = max(1, min(30, base_fps))
            effective_fps = max(1, min(30, round(base_fps * p.speed)))
            if effective_fps != p.fps:
                speed_note = tr("（保持 1x 原速）") if p.speed == 1.0 else tr("（提速播放）")
                self._log_msg(
                    "warn",
                    tr("视频实际时长 {0:.2f}s：原速 {1}fps × {2:g}x = 输出 {3}fps{4}").format(
                        video_duration, base_fps, p.speed, effective_fps, speed_note
                    ),
                )
        else:
            effective_fps = max(1, min(30, round(p.fps * p.speed)))

        if requested_secs < 1.5:
            self._log_msg(
                "warn",
                tr(
                    "提示：AI 视频动作通常较慢，1.5s 以内的片段可能无法完整呈现动作；"
                    "建议增大帧数或使用更高播放倍速"
                ),
            )
        return frames, video_path, effective_fps, video_duration

    def _step_pixelize(
        self, frames: List[Image.Image], pix_params: px.PixelizeParams
    ) -> Tuple[List[Image.Image], Optional[List[Image.Image]]]:
        """完美像素化管线（高效版）。

        返回 (frames, native_frames)：
        - frames：用户预设分辨率的最终帧；
        - native_frames：完美像素算法检测出的网格原生分辨率版本（与 frames 并存，
          共享同一调色板；非像素风 / 未开启像素化时为 None）。

        像素风判定以第一帧为准：
        - 像素风：首帧跑完整 Perfect Pixel（检测网格+对齐），
          全部帧用同一套网格坐标采样（检测只做一次，帧间网格一致、效率高）；
        - 非像素风：跳过 Perfect Pixel，直接目标尺寸缩放 + 量化。
        最后统一尺寸 + 共享调色板量化（帧间颜色一致、与生成图颜色数一致）。
        """
        if not self.params.pixelate:
            return frames, None
        self._log_msg("info", tr("像素化：目标 {0}，颜色上限 {1}").format(pix_params.target_size, pix_params.max_colors))

        native: Optional[List[Image.Image]] = None
        result = px.perfect_pixelize_sequence(frames)
        if result is not None:
            sampled, grid = result
            self._log_msg(
                "info",
                tr("像素风（网格 {0}×{1}）：首帧定网格，全部帧单元采样").format(grid[0], grid[1]),
            )
            native = sampled
            refined: List[Image.Image] = sampled
        else:
            self._log_msg("info", tr("非像素风格：跳过完美像素，目标尺寸缩放 + 色彩量化"))
            refined = [px.resize_nearest(f, pix_params.target_size) for f in frames]

        # 统一尺寸（Perfect Pixel 输出为检测到的网格分辨率 -> 用户预设分辨率）
        refined = [px.resize_nearest(f, pix_params.target_size) for f in refined]
        # 调色板与生成图一致：优先用首帧实际颜色数，上限为用户设置的 max_colors
        eff_colors = pix_params.max_colors
        if refined:
            unique = len(set(refined[0].convert("RGB").getdata()))
            eff_colors = max(2, min(unique, pix_params.max_colors))
            self._log_msg(
                "info",
                tr("生成帧实际 {0} 色 → 调色板取 {1} 色（上限 {2}）").format(
                    unique, eff_colors, pix_params.max_colors
                ),
            )
        # 共享调色板量化 + 去杂点（保证帧间颜色一致、不闪烁）
        # 用频率主导调色板：离散格色按出现频率取前 N，精确保留主色（MEDIANCUT 会混色）
        shared_palette = px.extract_dominant_palette(refined[0], eff_colors) if refined else None
        quantized = px.pixelize_frames(
            refined,
            px.PixelizeParams(max_colors=eff_colors, edge_clean=True, palette=shared_palette),
        )

        if native is not None:
            if native[0].size == quantized[0].size:
                self._log_msg("info", tr("完美像素网格与用户预设分辨率一致，仅导出预设分辨率"))
                native = None
            else:
                # 网格原生分辨率版本：与预设版本共享同一调色板（颜色完全一致）
                native = px.pixelize_frames(
                    native,
                    px.PixelizeParams(max_colors=eff_colors, edge_clean=True, palette=shared_palette),
                )
                self._log_msg(
                    "info",
                    tr("保留两种分辨率：完美像素原生 {0}×{1}，用户预设 {2}×{3}").format(
                        native[0].size[0], native[0].size[1],
                        quantized[0].size[0], quantized[0].size[1],
                    ),
                )
        return quantized, native

    def _step_background(self, frames: List[Image.Image]) -> List[Image.Image]:
        p = self.params
        if not p.remove_bg and not p.force_pure_bg:
            return frames
        if p.remove_bg:
            key_note = tr("，键色 {0}，容差 {1}").format(p.bg_color, p.bg_tolerance)
        else:
            key_note = ""
        self._log_msg(
            "info",
            tr("背景处理：纯色背景={0}，抠图={1}{2}").format(
                p.force_pure_bg, p.remove_bg, key_note
            ),
        )
        normalized_count = 0
        out = []
        for f in frames:
            self._check_cancel()
            img, normalized = bg_mod.process_background(
                f,
                force_pure_bg=p.force_pure_bg,
                remove_bg=p.remove_bg,
                key_color=p.bg_color,
                tolerance=p.bg_tolerance,
                feather=p.bg_feather,
            )
            if normalized:
                normalized_count += 1
            out.append(img)
        if normalized_count:
            self._log_msg("info", tr("背景归一化：{0}/{1} 帧").format(normalized_count, len(frames)))
        return out

    def _step_export(
        self,
        frames: List[Image.Image],
        native_frames: Optional[List[Image.Image]],
        export_dir: Path,
        result: SoloResult,
    ) -> None:
        """导出最终产物：用户预设分辨率 + 完美像素原生分辨率（存在时）。"""
        p = self.params
        if p.export_png:
            png_dir = export_dir / "png"
            fu.save_png_sequence(frames, png_dir, prefix=EXPORT_PREFIX)
            result.png_dir = png_dir
            self._log_msg("info", tr("PNG 序列帧已导出: {0}（{1} 张）").format(png_dir, len(frames)))
            if native_frames:
                native_png_dir = export_dir / "png_native"
                fu.save_png_sequence(native_frames, native_png_dir, prefix=EXPORT_PREFIX)
                result.native_png_dir = native_png_dir
                self._log_msg("info", tr("PNG 序列帧（完美像素原生分辨率）已导出: {0}").format(native_png_dir))
        if p.export_gif:
            gif_path = fu.frames_to_gif(frames, export_dir / f"{EXPORT_PREFIX}.gif", fps=result.fps)
            result.gif_path = gif_path
            self._log_msg("info", tr("GIF 已导出: {0}（{1}fps）").format(gif_path, result.fps))
            if native_frames:
                native_gif_path = export_dir / f"{EXPORT_PREFIX}_native.gif"
                fu.frames_to_gif(native_frames, native_gif_path, fps=result.fps)
                result.native_gif_path = native_gif_path
                self._log_msg("info", tr("GIF（完美像素原生分辨率）已导出: {0}").format(native_gif_path))
        if p.export_apng:
            apng_path = fu.frames_to_apng(frames, export_dir / f"{EXPORT_PREFIX}.apng", fps=result.fps)
            result.apng_path = apng_path
            self._log_msg("info", tr("APNG 已导出: {0}").format(apng_path))
        if p.export_sprite:
            # 雪碧图 + 索引 JSON（FrameRonin 风格：每帧坐标 + 时间戳）
            # 自动方形布局（列数≈√N），避免帧数多时单行超长
            timestamps = [i / max(1, result.fps) for i in range(len(frames))]
            sheet, sheet_index = fu.compose_sprite_sheet(
                frames, timestamps=timestamps, auto_square=True
            )
            sprite_path = fu.save_image(sheet, export_dir / "sprite_sheet.png")
            result.sprite_path = sprite_path
            self._save_json(export_dir / "sprite_sheet.json", sheet_index)
            self._log_msg("info", tr("雪碧图已导出: {0}（含索引 JSON）").format(sprite_path))
        meta = fu.animation_meta(frames, result.fps)
        if native_frames:
            meta["native_resolution"] = {
                "width": native_frames[0].width,
                "height": native_frames[0].height,
                "gif": result.native_gif_path.name if result.native_gif_path else None,
                "png_dir": result.native_png_dir.name if result.native_png_dir else None,
            }
        self._save_json(export_dir / "metadata.json", meta)

    def _save_project(self, export_dir: Path, result: SoloResult) -> None:
        p = self.params
        params = asdict(p)
        # JSON 序列化兼容：Path -> str，tuple -> list
        params["output_dir"] = str(params["output_dir"])
        params["reference_image"] = str(params["reference_image"]) if params.get("reference_image") else None
        if isinstance(params.get("bg_color"), tuple):
            params["bg_color"] = list(params["bg_color"])
        project = Project(
            name=f"solo_{p.description.strip()[:24] or 'untitled'}",
            fps=result.fps,
            frame_count=result.frame_count,
            width=result.width,
            height=result.height,
            frames_dir=str(result.frames_dir) if result.frames_dir else None,
            gif_path=str(result.gif_path) if result.gif_path else None,
            apng_path=str(result.apng_path) if result.apng_path else None,
            sprite_path=str(result.sprite_path) if result.sprite_path else None,
            source_video=str(result.video_path) if result.video_path else None,
            first_frame=str(result.first_frame) if result.first_frame else None,
            video_duration=result.video_duration,
            native_width=result.native_width,
            native_height=result.native_height,
            native_gif_path=str(result.native_gif_path) if result.native_gif_path else None,
            native_png_dir=str(result.native_png_dir) if result.native_png_dir else None,
            prompts=result.prompts,
            params=params,
        )
        result.project_file = save_project(project, export_dir / "project.json")

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    def _save_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _begin(self, step: int, total: int, name: str) -> None:
        name = tr(name)
        self._report(step, total, name, 0.0, tr("开始"))
        self._log_msg("info", tr("—— 步骤 {step}/{total}：{name} ——").format(step=step + 1, total=total, name=name))

    def _report(self, step: int, total: int, name: str, pct: float, message: str) -> None:
        name = tr(name)
        # 进度文案的通用状态词走翻译；路径/计数等 message 原样透传
        if message in ("完成", "开始"):
            message = tr(message)
        if self._progress:
            try:
                self._progress(step, total, name, float(pct), message)
            except Exception:  # noqa: BLE001
                pass

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
        if self._cancel and self._cancel.is_set():
            raise WorkflowCancelled()
