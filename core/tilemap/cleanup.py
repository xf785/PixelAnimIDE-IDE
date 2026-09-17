"""AI 底图清理：抹除每格深色线框（格线）、检测并修补文字/水印等异物。

生图模型常见两类「脏」底图：
1. **格线框**：沿每格边缘画一圈深色线（AI 把「九宫格」理解成带边框的表格）。
   若不清理，线框会被当成艺术内容拼进每张瓦片，整张地图出现规律性深色网格——
   这是「地块瓦片衔接明显有问题」的首要原因。
2. **文字/水印**：格子里写上「草地」之类的词，或签名、编号。绝对不可接受：
   算法无法可靠识别语义，只能在提示词层面严禁，并在检测到之后**报告 + 修补**。

两个算法的共同原则：只改动被判定为「异物」的像素，其余像素逐像素保持原样
（像素画硬边不被模糊），且全部确定性（无随机）。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("PixelAnimIDE.tilemap.cleanup")

Box = Tuple[int, int, int, int]  # (x0, y0, x1, y1) 半开区间


# --------------------------------------------------------------------------- #
# 1) 格线框抹除
# --------------------------------------------------------------------------- #
def _luma(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _line_run_at(profile: np.ndarray, center: int, thresh: float, max_width: int) -> Tuple[int, int]:
    """在亮度剖面上量出以 center 为中心、连续低于 thresh 的暗带 [lo, hi]（闭区间）。"""
    n = len(profile)
    if not (0 <= center < n) or profile[center] >= thresh:
        return (-1, -1)
    lo = center
    while lo - 1 >= 0 and center - (lo - 1) < max_width and profile[lo - 1] < thresh:
        lo -= 1
    hi = center
    while hi + 1 < n and (hi + 1) - center < max_width and profile[hi + 1] < thresh:
        hi += 1
    return (lo, hi)


def strip_grid_frames(
    img: Image.Image,
    cell: Optional[int] = None,
    rows: int = 6,
    cols: int = 6,
    max_width: int = 4,
    darkness: float = 0.80,
) -> Tuple[Image.Image, Dict]:
    """抹除沿格线画的深色线框，返回 (清理后图像, 报告)。

    cell: 单格像素（None 时按 rows/cols 由短边推算，取偶，与裁切一致）。
    判定：候选格线处「线带亮度中位数」显著低于紧邻两侧参考带（< darkness 倍）
    才判为线框；逐条线独立判定，因此只影响真的画了线框的底图。
    抹除：线带像素用它左右（上下）相邻像素的平均值填回，其余像素不动。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    h, w = arr.shape[:2]
    if cell is None:
        cell = max(4, int(min(w // max(1, cols), h // max(1, rows))) & ~1)
    if cell < 8:
        return rgba, {"vertical": [], "horizontal": [], "cell": cell}
    lum = _luma(arr)
    out = arr.copy()
    report: Dict = {"cell": int(cell), "vertical": [], "horizontal": []}

    def process(axis: int, positions: Sequence[int]) -> List[Dict]:
        found: List[Dict] = []
        n = w if axis == 1 else h
        for pos in positions:
            window = range(max(0, pos - max_width - 2), min(n, pos + max_width + 3))
            prof = np.median(lum[:, list(window)], axis=0) if axis == 1 else np.median(lum[list(window), :], axis=1)
            lo = max(0, pos - max_width - 2)
            center = pos - lo
            left_idx = [i for i in range(len(prof)) if (i + lo) <= pos - max_width - 1]
            right_idx = [i for i in range(len(prof)) if (i + lo) >= pos + max_width + 1]
            if not left_idx and not right_idx:
                continue
            refs = [
                float(np.median(prof[idx]))
                for idx in (left_idx, right_idx) if idx
            ]
            # 参考亮度取两侧中较暗的一侧：线框必须比它所分隔的两侧都暗，
            # 否则「深色地形贴着一个线的另一侧」会被误判成更宽的线（实测 bug）。
            ref = min(refs)
            if ref <= 1.0:
                continue
            thresh = ref * darkness
            run_lo, run_hi = _line_run_at(prof, center, thresh, max_width)
            if run_lo < 0:
                continue
            x0 = run_lo + lo
            x1 = run_hi + 1 + lo
            if x1 - x0 < 1 or x1 > n:
                continue
            line_vals = prof[run_lo:run_hi + 1]
            if float(np.median(line_vals)) >= thresh:
                continue
            # 抹除：用带外最近两像素（共 4 个样本）的**下中位数**填回——取真实像素
            # 而不做平均，避免在「地形色边界正好落在格线上」时留下一条中间色涂抹带；
            # 图像外缘线只有一侧参考，直接复制该侧。
            if axis == 1:
                if x0 == 0:
                    out[:, 0:x1, :] = out[:, x1:x1 + 1, :]
                elif x1 >= n:
                    out[:, x0:n, :] = out[:, x0 - 1:x0, :]
                else:
                    samples = np.stack([out[:, x0 - 1, :], out[:, max(0, x0 - 2), :],
                                        out[:, x1, :], out[:, min(n - 1, x1 + 1), :]], axis=0)
                    samples.sort(axis=0)
                    out[:, x0:x1, :] = samples[1][:, None, :]
            else:
                if x0 == 0:
                    out[0:x1, :, :] = out[x1:x1 + 1, :, :]
                elif x1 >= n:
                    out[x0:n, :, :] = out[x0 - 1:x0, :, :]
                else:
                    samples = np.stack([out[x0 - 1, :, :], out[max(0, x0 - 2), :, :],
                                        out[x1, :, :], out[min(n - 1, x1 + 1), :, :]], axis=0)
                    samples.sort(axis=0)
                    out[x0:x1, :, :] = samples[1][None, :, :]
            found.append({
                "pos": int(pos), "x0": int(x0), "width": int(x1 - x0),
                "contrast": round(1.0 - float(np.median(line_vals)) / ref, 3),
            })
        return found

    # 图像外缘（0 与末位）也可能是线框，一并检测：AI 画「表格」时四边都会加框
    vert_positions = [0] + [k * cell for k in range(1, max(1, w // cell))] + [w - 1]
    horz_positions = [0] + [k * cell for k in range(1, max(1, h // cell))] + [h - 1]
    report["vertical"] = process(1, vert_positions)
    # 抹除竖直线的结果会影响水平线的参考带，故重新取亮度
    lum = _luma(out)
    report["horizontal"] = process(0, horz_positions)
    report["count"] = len(report["vertical"]) + len(report["horizontal"])
    cleaned = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")
    if report["count"]:
        widths = sorted({d["width"] for d in report["vertical"] + report["horizontal"]})
        logger.info("抹除格线框 %d 条（宽度 %s px）", report["count"], widths)
    return cleaned, report


# --------------------------------------------------------------------------- #
# 2) 文字 / 水印检测与修补
# --------------------------------------------------------------------------- #
def _components(mask: np.ndarray, max_components: int = 400) -> List[Dict]:
    """4 邻域连通域（纯 numpy + 栈式洪泛，规模受格尺寸限制）。"""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: List[Dict] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        area = 0
        y0 = y1 = sy
        x0 = x1 = sx
        while stack:
            y, x = stack.pop()
            area += 1
            y0, y1 = min(y0, y), max(y1, y)
            x0, x1 = min(x0, x), max(x1, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append({"area": area, "box": (x0, y0, x1 + 1, y1 + 1)})
        if len(out) >= max_components:
            break
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """方形结构元膨胀（radius=1 时等价 3×3）。"""
    out = mask.copy()
    for _ in range(max(0, int(radius))):
        m = out
        grown = m.copy()
        grown[1:, :] |= m[:-1, :]
        grown[:-1, :] |= m[1:, :]
        grown[:, 1:] |= m[:, :-1]
        grown[:, :-1] |= m[:, 1:]
        out = grown
    return out


def _row_runs(flags: np.ndarray) -> int:
    """一维布尔序列中 True 连续段的个数。"""
    if flags.size == 0 or not flags.any():
        return 0
    return int(np.count_nonzero(flags[1:] & ~flags[:-1]) + (1 if flags[0] else 0))


def _stroke_signature(within: np.ndarray, box: Box) -> Tuple[float, int]:
    """文字的「等宽细笔画」特征：返回 (笔画宽度众数占比, 细笔画条数)。

    文字笔画宽度基本恒定 -> 同宽度反复出现（众数占比高）；石头/菱形/草丛这类自然
    装饰的横切长度是渐变的，众数占比低，据此把「字」与「装饰」区分开。
    """
    x0, y0, x1, y1 = box
    bw = x1 - x0
    thin_max = max(2, int(round(0.25 * bw)))
    lengths: List[int] = []
    for y in range(y0, y1):
        cur = 0
        for v in within[y, x0:x1].tolist():
            if v:
                cur += 1
            elif cur:
                if cur <= thin_max:
                    lengths.append(cur)
                cur = 0
        if cur and cur <= thin_max:
            lengths.append(cur)
    if len(lengths) < 6:
        return 0.0, len(lengths)
    _vals, counts = np.unique(np.array(lengths), return_counts=True)
    return float(counts.max()) / float(len(lengths)), len(lengths)


def _expand_text_line(comps: List[Dict], box: Box) -> Box:
    """把与主标记同一行（或同一列）、水平（垂直）相邻的笔画一并纳入修补范围。

    汉字/单词常被 3px 膨胀拆成多个连通域（「雪」「原」各自一块），只修补一块会
    留下半行字。这里按「中心线对齐 + 间距不超过主块尺寸」合并。
    """
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ax0, ay0, ax1, ay1 = x0, y0, x1, y1
    for c in comps:
        cx0, cy0, cx1, cy1 = c["box"]
        cw, ch = cx1 - cx0, cy1 - cy0
        if cw > 2.0 * bw or ch > 2.0 * bh:
            continue
        cy = (cy0 + cy1) / 2
        cx = (cx0 + cx1) / 2
        same_row = ay0 - 0.3 * bh <= cy <= ay1 + 0.3 * bh and abs(cx0 - ax1) <= 1.2 * bw + cw
        same_col = ax0 - 0.3 * bw <= cx <= ax1 + 0.3 * bw and abs(cy0 - ay1) <= 1.2 * bh + ch
        if same_row or same_col:
            ax0, ay0 = min(ax0, cx0), min(ay0, cy0)
            ax1, ay1 = max(ax1, cx1), max(ay1, cy1)
    return (ax0, ay0, ax1, ay1)


def detect_text_marks(
    tile: Image.Image,
    distance: float = 55.0,
    max_area_frac: float = 0.25,
    max_side_frac: float = 0.8,
) -> List[Box]:
    """启发式检测「像一行字/一个水印」的异物区域，返回其外接框（可为空）。

    判据（保守，宁可漏检不误判）——把与格中位色差异明显的像素聚成一个「标记区」
    （膨胀后的最大连通域）：
    1. 外接框 ≤ 格子的 80%、面积占比 ≤ 25%（整格纹理/大团块直接排除）；
    2. **等宽细笔画签名**：笔画横切长度基本恒定（众数占比 ≥ 0.35 且细笔画 ≥ 6 条）
       ——文字的核心特征，汉字/拉丁字母都满足，而石头、菱形贴花、草丛、噪点纹理
       的横切长度是渐变的，不满足；
    3. 或 ≥3 个等高小块整齐排成一行/一列（拉丁字母/数字型）。
    """
    rgb = np.asarray(tile.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    med = np.median(rgb.reshape(-1, 3), axis=0)
    foreign = np.linalg.norm(rgb - med, axis=2) > float(distance)
    if foreign.sum() < 8:
        return []
    comps = _components(_dilate(foreign, 1), max_components=120)
    if not comps:
        return []
    main = max(comps, key=lambda c: c["area"])
    x0, y0, x1, y1 = main["box"]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return []
    if bw > max_side_frac * w or bh > max_side_frac * h:
        return []
    within = np.zeros_like(foreign)
    within[y0:y1, x0:x1] = foreign[y0:y1, x0:x1]
    if within.sum() / float(w * h) > max_area_frac:
        return []
    # 一行字至少要有一定「体量」：单根细线（如一根草）不算文字
    content_cols = int(within.any(axis=0).sum())
    content_rows = int(within.any(axis=1).sum())
    if content_cols < max(6, 0.25 * bw) or content_rows < max(4, 0.25 * bh):
        return []
    share, thin_runs = _stroke_signature(within, (x0, y0, x1, y1))
    if thin_runs >= 6 and share >= 0.35:
        return [_expand_text_line(comps, (x0, y0, x1, y1))]
    # 拉丁字母型：多个等高、笔画状的小块排成一行/一列（实心菱形/方块不算）
    small = [c for c in comps if 2 <= c["area"] <= 0.06 * w * h]
    if len(small) >= 3:
        boxes = [c["box"] for c in small]
        heights = [b[3] - b[1] for b in boxes]
        widths = [b[2] - b[0] for b in boxes]
        fills = [c["area"] / max(1, (b[2] - b[0]) * (b[3] - b[1])) for c, b in zip(small, boxes)]
        if max(fills) <= 0.45 and max(heights) <= 2.5 * min(heights):
            cy = [(b[1] + b[3]) / 2 for b in boxes]
            cx = [(b[0] + b[2]) / 2 for b in boxes]
            row_like = (max(cy) - min(cy)) <= 0.35 * h and (max(cx) - min(cx)) >= 0.35 * w
            col_like = (max(cx) - min(cx)) <= 0.35 * w and (max(cy) - min(cy)) >= 0.35 * h
            if (row_like or col_like) and max(widths) <= 0.6 * w and max(heights) <= 0.6 * h:
                bx0 = min(b[0] for b in boxes)
                by0 = min(b[1] for b in boxes)
                bx1 = max(b[2] for b in boxes)
                by1 = max(b[3] for b in boxes)
                return [(bx0, by0, bx1, by1)]
    return []


def patch_marks(tile: Image.Image, boxes: Sequence[Box], pad: int = 2) -> Image.Image:
    """用「自身纹理平移半格」的副本覆盖异物区域（保留原调色板与纹理统计）。"""
    if not boxes:
        return tile.convert("RGBA")
    arr = np.asarray(tile.convert("RGBA")).astype(np.uint8)
    h, w = arr.shape[:2]
    src = np.roll(arr, (h // 2, w // 2), axis=(0, 1))
    out = arr.copy()
    for (x0, y0, x1, y1) in boxes:
        ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
        ax1, ay1 = min(w, x1 + pad), min(h, y1 + pad)
        out[ay0:ay1, ax0:ax1] = src[ay0:ay1, ax0:ax1]
    return Image.fromarray(out, "RGBA")


def clean_centre_tile(tile: Image.Image) -> Tuple[Image.Image, List[Box]]:
    """检测并修补单张纹理瓦片上的文字/水印，返回 (处理后瓦片, 检出框)。"""
    boxes = detect_text_marks(tile)
    if not boxes:
        return tile.convert("RGBA"), []
    logger.warning("检测到疑似文字/水印 %d 处，已用纹理修补", len(boxes))
    return patch_marks(tile, boxes), boxes
