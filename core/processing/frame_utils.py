"""帧提取、保存、合成工具：PNG 序列帧、GIF、雪碧图、视频拆帧。"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image, ImageSequence

logger = logging.getLogger("PixelAnimIDE.processing.frame_utils")


# --------------------------------------------------------------------------- #
# 图像 IO
# --------------------------------------------------------------------------- #
def load_image(path: Path | str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def save_image(img: Image.Image, path: Path | str, fmt: Optional[str] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt)
    return path


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGBA")


def download_bytes(url: str, timeout: float = 180.0) -> bytes:
    """下载远程资源（视频/图片）到内存。"""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


# --------------------------------------------------------------------------- #
# 视频拆帧
# --------------------------------------------------------------------------- #
def extract_video_frames(
    video_path: Path | str,
    max_frames: Optional[int] = None,
    max_duration: Optional[float] = None,
) -> List[Image.Image]:
    """用 imageio-ffmpeg 拆帧，返回 RGBA 图像列表（保持原签名兼容）。

    max_frames 给定时间均匀采样；max_duration 只保留前 N 秒的帧
    （用于“只要目标时长内的运动”）。
    """
    frames, _ = extract_video_frames_meta(video_path, max_frames=max_frames, max_duration=max_duration)
    return frames


def extract_video_frames_meta(
    video_path: Path | str,
    max_frames: Optional[int] = None,
    max_duration: Optional[float] = None,
) -> Tuple[List[Image.Image], dict]:
    """拆帧并返回 (frames, meta)。

    meta: {"fps": float|None, "duration": float|None, "source_frame_count": int}
    duration 为源视频（或截取片段）的估计时长，来自 meta.fps 与帧数。
    """
    import imageio.v2 as imageio

    video_path = str(video_path)
    frames: List[Image.Image] = []
    fps: float = 0.0
    total: int = 0
    with imageio.get_reader(video_path, format="ffmpeg") as reader:
        meta_raw = reader.get_meta_data()
        try:
            fps = float(meta_raw.get("fps") or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        try:
            nframes_raw = meta_raw.get("nframes", 0) or 0
            total = int(nframes_raw) if nframes_raw != float("inf") else 0
        except (TypeError, ValueError, OverflowError):
            total = 0
        for i, frame in enumerate(reader.iter_data()):
            # 只取目标时长内的帧（时间 < max_duration）
            if max_duration and fps and fps > 0 and i > 0 and (i / fps) >= max_duration:
                break
            frames.append(Image.fromarray(frame).convert("RGBA"))
            # 元数据没有帧数时，读满两倍即停（用于采样）；已知总帧数则读全量
            if total == 0 and max_frames and len(frames) >= max_frames * 2:
                break

    source_count = len(frames)
    duration = (source_count / fps) if fps and fps > 0 else None

    if max_frames and len(frames) > max_frames:
        indices = _evenly_sample_indices(len(frames), max_frames)
        frames = [frames[i] for i in indices]

    meta = {
        "fps": fps or None,
        "duration": duration,
        "source_frame_count": source_count,
    }
    return frames, meta


def _evenly_sample_indices(total: int, count: int) -> List[int]:
    """均匀采样 total 个索引到 count 个（含首尾）；count==1 时取首索引。"""
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def _sample_frames(frames: List[Image.Image], max_frames: Optional[int]) -> List[Image.Image]:
    if not max_frames or len(frames) <= max_frames:
        return frames
    return [frames[i] for i in _evenly_sample_indices(len(frames), max_frames)]


sample_frames = _sample_frames  # 公开别名


# --------------------------------------------------------------------------- #
# 循环动画抽帧（首尾帧一致）
# --------------------------------------------------------------------------- #
def sample_frames_preserve_ends(frames: List[Image.Image], count: int) -> List[Image.Image]:
    """保留首帧与尾帧、中间均匀采样的抽帧策略。

    保证视频完整动作都体现：首帧（起始姿态）与尾帧（结束姿态）必被保留，
    中间帧均匀铺开（避免循环段搜索丢弃首尾、或只截取部分片段导致动作不完整）。
    """
    n = len(frames)
    if not frames or count <= 0:
        return []
    if n <= count:
        return list(frames)
    if count == 1:
        return [frames[0]]
    if count == 2:
        return [frames[0], frames[-1]]
    # 中间 count-2 帧：在索引 1..n-2 之间均匀取
    middle = _evenly_sample_indices(n - 2, count - 2)
    indices = [0] + [i + 1 for i in middle] + [n - 1]
    return [frames[i] for i in indices]


def sample_loop_frames(
    frames: List[Image.Image],
    target_count: int,
    loop: bool = True,
    segment_ratio: float = 3.0,
) -> List[Image.Image]:
    """抽帧 + 可选循环闭合。

    抽帧策略：保留首帧与尾帧，中间均匀采样（完整动作都体现）。
    loop=True 时强制最后一帧 = 第一帧（循环播放无跳变）。
    """
    frames = sample_frames_preserve_ends(list(frames), target_count)
    if loop and len(frames) >= 2:
        frames = list(frames)
        frames[-1] = frames[0]  # 首尾帧一致
    return frames


def strip_audio(src: Path | str, dst: Path | str) -> Path:
    """去除视频音轨（ffmpeg `-c copy` 快速 remux，不重编码视频）。

    生成的视频应无声；此操作为最佳努力——remux 失败时保留原视频
    （拆帧只读视频流，不受音轨影响）。返回最终可用的视频路径。
    """
    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe

    src = Path(src)
    dst = Path(dst)
    if src == dst:
        dst = src.with_name(f"{src.stem}_silent{src.suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [get_ffmpeg_exe(), "-y", "-i", str(src), "-an", "-c", "copy", str(dst)],
            check=True,
            capture_output=True,
        )
        return dst
    except Exception as exc:  # noqa: BLE001
        logger.warning("去除音轨失败（%s），保留原视频", exc)
        return src


def downscale_bytes(data: bytes, max_side: int = 512) -> bytes:
    """缩放图片字节，长边不超过 max_side；已达标时原样返回。

    用于把首帧图缩小后再发给视频 API —— 图片按像素计费（token），
    缩小 4 倍面积可大幅降低视频生成的图片 token 成本。
    """
    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_side:
        return data
    rgb = img.convert("RGB")
    scale = max_side / max(rgb.size)
    rgb = rgb.resize(
        (max(8, int(rgb.width * scale)), max(8, int(rgb.height * scale))),
        Image.Resampling.LANCZOS,
    )
    return image_to_bytes(rgb, "PNG")


def upscale_to_min_side_bytes(data: bytes, min_side: int = 256) -> bytes:
    """把过小的图片放大到长边 ≥ min_side（视频 API 最低分辨率要求）。

    使用**最近邻**（NEAREST）放大：像素画放大后仍是硬边方块，
    不会像双线性/双三次那样产生模糊插值（关键：缩放不能使像素点模糊）。
    已达标时原样返回。
    """
    img = Image.open(io.BytesIO(data))
    if max(img.size) >= min_side:
        return data
    rgba = img.convert("RGBA")
    scale = min_side / max(rgba.size)
    rgba = rgba.resize(
        (max(8, int(rgba.width * scale)), max(8, int(rgba.height * scale))),
        Image.Resampling.NEAREST,
    )
    return image_to_bytes(rgba, "PNG")


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def save_png_sequence(frames: Iterable[Image.Image], out_dir: Path | str, prefix: str = "frame") -> List[Path]:
    """保存 PNG 序列帧，返回文件路径列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for index, frame in enumerate(frames):
        path = out_dir / f"{prefix}_{index:04d}.png"
        frame.save(path, format="PNG")
        paths.append(path)
    return paths


def frames_to_gif(
    frames: List[Image.Image],
    out_path: Path | str,
    fps: int = 8,
    loop: int = 0,
    optimize: bool = False,
) -> Path:
    """合成 GIF 动画（支持透明）。

    注意：PIL 的 GIF 编码器会把「内容完全相同」的连续帧合并为一个
    （无论 optimize 如何设置），这是 GIF 格式的固有行为。需要严格保留
    每一帧（含重复帧）时请使用 PNG 序列帧导出。
    loop=0 表示无限循环。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = [make_gif_frame(f) for f in frames]
    if not gif_frames:
        raise ValueError("没有可用帧，无法生成 GIF")
    duration_ms = max(20, int(1000 / max(1, fps)))
    kwargs: dict = {
        "save_all": True,
        "append_images": gif_frames[1:],
        "duration": duration_ms,
        "loop": int(loop),
        "optimize": optimize,
    }
    # 只要任一帧含透明，就统一用索引 255 作为透明槽（量化到 255 色，
    # 索引 0-254 为颜色、255 恒为空闲，不透明帧标记它也不会误伤像素）
    if any("transparency" in f.info for f in gif_frames):
        for f in gif_frames:
            if "transparency" not in f.info:
                f.info["transparency"] = 255
        kwargs["transparency"] = 255
        kwargs["disposal"] = 2  # 每帧恢复背景，避免透明残留
    gif_frames[0].save(out_path, format="GIF", **kwargs)
    return out_path


def make_gif_frame(img: Image.Image) -> Image.Image:
    """把任意图像转为带透明索引的 P 模式 GIF 帧（索引 255 为透明）。"""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    has_transparency = alpha.getextrema()[0] < 255
    rgb = rgba.convert("RGB")
    p = rgb.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    if has_transparency:
        mask = alpha.point(lambda a: 255 if a == 0 else 0)
        px = p.load()
        mx = mask.load()
        for y in range(p.height):
            for x in range(p.width):
                if mx[x, y]:
                    px[x, y] = 255
        p.info["transparency"] = 255
    return p


def frames_to_apng(
    frames: List[Image.Image],
    out_path: Path | str,
    fps: int = 8,
    loop: int = 0,
) -> Path:
    """合成 APNG 动画（Pillow PNG save_all，支持透明与逐帧独立时长）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("没有可用帧，无法生成 APNG")
    duration_ms = max(20, int(1000 / max(1, fps)))
    rgba = [f.convert("RGBA") for f in frames]
    rgba[0].save(
        out_path,
        format="PNG",
        save_all=True,
        append_images=rgba[1:],
        duration=duration_ms,
        loop=int(loop),
    )
    return out_path


def apng_frame_count(path: Path | str) -> int:
    """读取 APNG 的帧数。"""
    with Image.open(path) as img:
        return int(getattr(img, "n_frames", 1))


def frames_to_sprite_sheet(frames: List[Image.Image], columns: Optional[int] = None, spacing: int = 0) -> Image.Image:
    """合成雪碧图（横向排列，可指定列数换行）。"""
    if not frames:
        raise ValueError("没有可用帧")
    fw, fh = frames[0].size
    cols = columns or len(frames)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * fw + (cols - 1) * spacing, rows * fh + (rows - 1) * spacing), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        x = (i % cols) * (fw + spacing)
        y = (i // cols) * (fh + spacing)
        sheet.paste(frame.convert("RGBA"), (x, y), frame.convert("RGBA"))
    return sheet


def crop_sprite_sheet(
    sheet: Image.Image,
    rows: int,
    cols: int,
    count: Optional[int] = None,
) -> List[Image.Image]:
    """按 i×j 网格把精灵图裁切为帧序列（行优先）。

    count 指定要取的帧数（≤ rows×cols，行优先取前 count 个）；默认取全部。
    """
    if sheet is None or rows < 1 or cols < 1:
        raise ValueError("精灵图与网格参数无效")
    w, h = sheet.size
    cw, ch = w / cols, h / rows
    frames: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = round(c * cw), round(r * ch)
            x1, y1 = round((c + 1) * cw), round((r + 1) * ch)
            frames.append(sheet.crop((x0, y0, x1, y1)))
    if count is not None:
        frames = frames[: max(0, int(count))]
    return frames


def gif_frame_count(path: Path | str) -> int:
    """读取 GIF 的帧数。"""
    with Image.open(path) as img:
        return sum(1 for _ in ImageSequence.Iterator(img))


# --------------------------------------------------------------------------- #
# 元数据
# --------------------------------------------------------------------------- #
def animation_meta(frames: List[Image.Image], fps: int) -> dict:
    """生成 JSON 元数据（帧率、尺寸、帧数等）。"""
    if not frames:
        raise ValueError("没有可用帧，无法生成元数据")
    first = frames[0].convert("RGBA")
    return {
        "format": "pixel-animation",
        "frame_count": len(frames),
        "fps": int(fps),
        "width": first.width,
        "height": first.height,
        "loop": True,
        "duration_ms_per_frame": int(1000 / max(1, int(fps))),
    }
