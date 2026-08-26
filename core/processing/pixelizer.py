"""严格像素化算法（确定性，不依赖 AI）。

阶段1 实现（对应文档 10.1 的简化版）：
1. 像素网格对齐：居中裁剪到目标宽高比后，最近邻缩放到目标尺寸。
2. 色彩量化：优先使用固定调色板（欧氏距离最近色映射）；
   未指定调色板时自动提取（中位切分）。
3. 去伪影与边缘清理：ModeFilter 去孤立杂点（保留硬边缘）。
4. 帧间一致性：多帧处理时对所有帧使用同一调色板。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("PixelAnimIDE.processing.pixelizer")

RGB = Tuple[int, int, int]


@dataclass
class PixelizeParams:
    """像素化参数。"""

    target_size: Optional[Tuple[int, int]] = None  # (w, h)；None 表示不缩放
    scale_factor: Optional[float] = None           # 备选：按比例缩小（NEAREST）
    max_colors: int = 16                           # 自动调色板的最大颜色数
    palette: Optional[List[RGB]] = None            # 固定调色板（帧间一致性的关键）
    edge_clean: bool = True                        # 去孤立杂点
    dither: bool = False                           # 量化时是否抖动（一般关闭）

    def resolve_target_size(self, src: Image.Image) -> Tuple[int, int]:
        """根据源图与参数计算最终目标尺寸。"""
        if self.target_size:
            return tuple(int(v) for v in self.target_size)
        if self.scale_factor:
            f = float(self.scale_factor)
            return (max(1, int(src.width * f)), max(1, int(src.height * f)))
        return src.size


# --------------------------------------------------------------------------- #
# 像素网格对齐
# --------------------------------------------------------------------------- #
def center_crop_to_ratio(img: Image.Image, ratio: Tuple[float, float]) -> Image.Image:
    """按目标宽高比居中裁剪，保证像素为正方形（不拉伸）。"""
    target_ratio = ratio[0] / ratio[1]
    src_ratio = img.width / max(1, img.height)
    if abs(src_ratio - target_ratio) < 1e-6:
        return img
    if src_ratio > target_ratio:  # 太宽 -> 裁宽
        new_w = int(img.height * target_ratio)
        x0 = (img.width - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, img.height))
    new_h = int(img.width / target_ratio)
    y0 = (img.height - new_h) // 2
    return img.crop((0, y0, img.width, y0 + new_h))


def resize_nearest(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """最近邻缩放，保持硬像素边缘。

    注意：PIL 自带的 NEAREST 缩小采用“中心采样”，会把整幅图偏移半像素
    （例如 64->32 时源图 (0,0) 会被采样掉）。这里用 numpy 实现原点对齐
    （dest(x) -> src(floor(x * sw/dw))），与放大行为一致且完全确定。
    """
    tw, th = int(size[0]), int(size[1])
    if (img.width, img.height) == (tw, th):
        return img
    arr = np.asarray(img)
    src_y = np.floor(np.arange(th) * img.height / th).astype(np.int64)
    src_x = np.floor(np.arange(tw) * img.width / tw).astype(np.int64)
    out = arr[src_y][:, src_x]
    return Image.fromarray(out, img.mode)


# --------------------------------------------------------------------------- #
# 调色板
# --------------------------------------------------------------------------- #
def extract_palette(img: Image.Image, max_colors: int) -> List[RGB]:
    """自动提取调色板（中位切分），返回 RGB 颜色列表。"""
    n = max(2, min(256, int(max_colors)))
    q = img.convert("RGB").quantize(colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = q.getpalette() or []
    colors = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)][:n]
    # 去重（保留顺序）
    seen = set()
    unique: List[RGB] = []
    for c in colors:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def extract_dominant_palette(img: Image.Image, max_colors: int) -> List[RGB]:
    """按颜色出现频率取前 N 个颜色（保留主色，避免 MEDIANCUT 混色）。

    适合「单元采样后的离散格色」：格色本身就是真实颜色，
    按频率取前 N 个最常出现的颜色即可精确保留主色。
    """
    n = max(1, min(256, int(max_colors)))
    arr = np.asarray(img.convert("RGB")).reshape(-1, 3)
    packed = (arr[:, 0].astype(np.int64) << 16) | (arr[:, 1].astype(np.int64) << 8) | arr[:, 2].astype(np.int64)
    uniq, counts = np.unique(packed, return_counts=True)
    top = np.argsort(counts)[::-1][:n]
    return [(((int(uniq[i]) >> 16) & 255), ((int(uniq[i]) >> 8) & 255), (int(uniq[i]) & 255)) for i in top]


def build_palette_image(colors: List[RGB]) -> Image.Image:
    """把 RGB 颜色列表转成 P 模式调色板图像（供 quantize(palette=...) 使用）。

    用第一个颜色填充剩余调色板槽位（而非黑色），避免产生虚假的黑色吸引点。
    """
    if not colors:
        colors = [(0, 0, 0)]
    flat: List[int] = []
    for c in colors:
        flat.extend([int(c[0]), int(c[1]), int(c[2])])
    pad_color = colors[0]
    while len(flat) < 768:
        flat.extend(pad_color)
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat)
    return pal


def map_to_palette(img: Image.Image, colors: List[RGB], dither: bool = False) -> Image.Image:
    """把每个像素映射到调色板中最近的颜色，返回 P 模式图像。"""
    pal_img = build_palette_image(colors)
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return img.convert("RGB").quantize(palette=pal_img, dither=dither_mode)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def pixelize_image(img: Image.Image, params: PixelizeParams) -> Image.Image:
    """对单张图像执行严格像素化，返回 RGBA 图像（保留原 alpha）。"""
    # 1) 保留 alpha
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    alpha = None
    rgb_src = img.convert("RGB")
    if has_alpha:
        alpha = img.convert("RGBA").getchannel("A")

    # 2) 像素网格对齐
    size = params.resolve_target_size(rgb_src)
    if params.target_size or params.scale_factor:
        ratio = (size[0] / max(1, size[1]), 1.0)
        rgb_src = center_crop_to_ratio(rgb_src, (size[0], size[1]))
        rgb_src = resize_nearest(rgb_src, size)

    # 3) 色彩量化
    colors = params.palette
    if not colors:
        colors = extract_palette(rgb_src, params.max_colors)
    quantized = map_to_palette(rgb_src, colors, dither=params.dither)

    # 4) 去孤立杂点（保留硬边缘）
    if params.edge_clean:
        quantized = quantized.filter(ImageFilter.ModeFilter(size=3))

    result = quantized.convert("RGBA")
    if alpha is not None:
        # 用原点对齐的最近邻缩放 alpha，避免 PIL 中心采样丢失边缘像素
        result.putalpha(resize_nearest(alpha, result.size))
    return result


def pixelize_frames(frames: List[Image.Image], params: PixelizeParams) -> List[Image.Image]:
    """多帧像素化：所有帧使用同一调色板，保证帧间颜色一致（不闪烁）。

    调色板取自第一帧（按目标尺寸处理后提取）。
    """
    if not frames:
        return []
    # 用第一帧确定目标尺寸与调色板
    probe_params = PixelizeParams(
        target_size=params.target_size,
        scale_factor=params.scale_factor,
        max_colors=params.max_colors,
        palette=params.palette,
        edge_clean=False,
        dither=params.dither,
    )
    probe = pixelize_image(frames[0], probe_params)
    if not params.palette:
        params = PixelizeParams(
            target_size=params.target_size,
            scale_factor=params.scale_factor,
            max_colors=params.max_colors,
            palette=extract_palette(probe, params.max_colors),
            edge_clean=params.edge_clean,
            dither=params.dither,
        )
    return [pixelize_image(f, params) for f in frames]


# --------------------------------------------------------------------------- #
# Perfect Pixel 封装（内嵌 perfectPixel-main 算法）
# --------------------------------------------------------------------------- #
def perfect_pixelize(
    img: Image.Image,
    sample_method: str = "majority",
    grid_size: Optional[Tuple[int, int]] = None,
    max_side: int = 768,
) -> Optional[Image.Image]:
    """Perfect Pixel 完美像素化：自动检测网格 -> 网格对齐 -> 单元采样。

    适用于 AI 生成的像素风图片/视频帧（自动消除网格偏移、非正方格等瑕疵）。
    输入过大时先等比缩小（网格比例不变）；检测失败或异常返回 None（由调用方回退）。
    """
    from core.processing import perfect_pixel as pp

    rgb = img.convert("RGB")
    if max(rgb.size) > max_side:
        scale = max_side / max(rgb.size)
        rgb = rgb.resize(
            (max(8, int(rgb.width * scale)), max(8, int(rgb.height * scale))),
            # NEAREST：像素格/网格结构原样保留（BILINEAR 会模糊格子导致检测失败）
            Image.Resampling.NEAREST,
        )
    try:
        w, h, out = pp.get_perfect_pixel(
            np.asarray(rgb),
            sample_method=sample_method,
            grid_size=grid_size,
            refine_intensity=0.25,
            fix_square=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Perfect Pixel 处理失败，回退默认像素化: %s", exc)
        return None
    if out is None or out.size == 0:
        return None
    # 未找到真实像素格（检测失败，或检测结果退化为≈原图分辨率的网格，
    # 即单元≈1px）时按非像素风处理：返回 None 由调用方走常规缩放路径，
    # 避免把平滑/真实感图片强行“像素化”或导出无意义的全尺寸原生版。
    if w is None or h is None or (w >= rgb.width * 0.95 and h >= rgb.height * 0.95):
        logger.info("未检测到有效像素网格（网格≈原图分辨率），按非像素风格处理")
        return None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")


def perfect_pixelize_sequence(
    frames: List[Image.Image],
    sample_method: str = "majority",
    max_side: int = 768,
) -> Optional[Tuple[List[Image.Image], Tuple[int, int]]]:
    """像素风帧序列：首帧定网格，全部帧（含首帧）按同一网格做「单元采样」。

    相比 NEAREST 硬缩放（每格只取 1 个像素，颜色损失严重），单元采样对每个
    网格单元取「众数/均值」的真实格色：
    - 首帧跑完整网格检测（perfect_pixel_plan，只做一次）得到网格坐标；
    - 其余帧复用同一网格坐标逐格采样（向量化，快且帧间一致）；
    - 内容相同的帧（循环闭合首尾帧）输出完全相同。

    返回 (采样后的帧列表, (grid_w, grid_h))；非像素风（检测失败）或空帧返回 None。
    """
    if not frames:
        return None
    from core.processing import perfect_pixel as pp

    # 1) 首帧：统一工作尺寸（过大先等比缩小，网格比例不变）并检测网格计划
    first_rgb = frames[0].convert("RGB")
    scale = 1.0
    if max(first_rgb.size) > max_side:
        scale = max_side / max(first_rgb.size)
        work_size = (max(8, int(first_rgb.width * scale)), max(8, int(first_rgb.height * scale)))
        first_rgb = first_rgb.resize(work_size, Image.Resampling.NEAREST)
    else:
        work_size = first_rgb.size
    try:
        plan = pp.perfect_pixel_plan(
            np.asarray(first_rgb),
            sample_method=sample_method,
            refine_intensity=0.25,
            fix_square=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Perfect Pixel 网格检测失败，回退默认像素化: %s", exc)
        return None
    if plan is None:
        return None
    grid = (plan[0], plan[1])
    # 退化网格（≈原图分辨率，单元≈1px）视为非像素风
    if grid[0] >= first_rgb.width * 0.95 and grid[1] >= first_rgb.height * 0.95:
        logger.info("未检测到有效像素网格（网格≈原图分辨率），按非像素风格处理")
        return None
    x_coords, y_coords = plan[2], plan[3]

    # 2) 全部帧（含首帧）按同一网格单元采样
    out: List[Image.Image] = []
    for f in frames:
        rgb = f.convert("RGB")
        if rgb.size != work_size:
            rgb = rgb.resize(work_size, Image.Resampling.NEAREST)
        arr = np.asarray(rgb)
        if sample_method == "median":
            sampled = pp.sample_median(arr, x_coords, y_coords)
        elif sample_method == "center":
            sampled = pp.sample_center(arr, x_coords, y_coords)
        else:
            sampled = pp.sample_mode_exact(arr, x_coords, y_coords)
        out.append(Image.fromarray(np.clip(sampled, 0, 255).astype(np.uint8)).convert("RGBA"))
    return out, grid
