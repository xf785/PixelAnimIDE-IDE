"""Perfect Pixel 算法（纯 NumPy 实现，内嵌自 perfectPixel-main）。

原始项目：https://github.com/theamusing/perfectPixel （MIT License）
实现来源：perfect_pixel_noCV2.py（无 OpenCV 依赖版）

作用：自动检测 AI 生成像素图的网格尺寸 -> 网格线与边缘对齐 -> 逐网格单元采样，
输出网格对齐、像素完美的结果。适用于“AI 生成的像素风图片/视频帧”的精修。

本文件为忠实移植，仅做如下调整：
- print() 改为 logging（默认 debug 级别，不干扰 UI 日志）；
- 移除 matplotlib 调试绘图（grid_layout），避免引入依赖。
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("PixelAnimIDE.processing.perfect_pixel")


# ----------------------------
# Small utilities
# ----------------------------
def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8/float -> gray float32"""
    img = image_rgb.astype(np.float32)
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def normalize_minmax(x: np.ndarray, a=0.0, b=1.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32) + a
    y = (x - mn) / (mx - mn)
    return (a + (b - a) * y).astype(np.float32)


def conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution (same) for grayscale float32, vectorized over shifts."""
    img = image.astype(np.float32, copy=False)
    k = kernel.astype(np.float32, copy=False)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for dy in range(kh):
        for dx in range(kw):
            w = k[dy, dx]
            if w == 0:
                continue
            out += w * pad[dy : dy + img.shape[0], dx : dx + img.shape[1]]
    return out


def sobel_xy(gray: np.ndarray, ksize: int = 3):
    """Return (gx, gy) similar to cv2.Sobel for ksize 3 or 5."""
    if ksize == 3:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    elif ksize == 5:
        kx = np.array(
            [
                [-5, -4, 0, 4, 5],
                [-8, -10, 0, 10, 8],
                [-10, -20, 0, 20, 10],
                [-8, -10, 0, 10, 8],
                [-5, -4, 0, 4, 5],
            ],
            dtype=np.float32,
        )
        ky = kx.T
    else:
        raise ValueError("ksize must be 3 or 5")
    gx = conv2d_same(gray, kx)
    gy = conv2d_same(gray, ky)
    return gx, gy


def magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


# ----------------------------
# Grid detection
# ----------------------------
def compute_fft_magnitude(gray_image):
    f = np.fft.fft2(gray_image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = 1 - np.log1p(mag)
    return normalize_minmax(mag, 0.0, 1.0)


def smooth_1d(v, k=17):
    k = int(k)
    if k < 3:
        return v
    if k % 2 == 0:
        k += 1
    sigma = k / 6.0
    x = np.arange(k) - k // 2
    ker = np.exp(-(x * x) / (2 * sigma * sigma))
    ker = ker / (ker.sum() + 1e-8)
    return np.convolve(v, ker, mode="same")


def detect_peak(proj, peak_width=6, rel_thr=0.35, min_dist=6):
    center = len(proj) // 2
    mx = float(proj.max())
    if mx < 1e-6:
        return None
    thr = mx * float(rel_thr)
    candidates = []
    for i in range(1, len(proj) - 1):
        is_peak = True
        for j in range(1, peak_width):
            if i - j < 0 or i + j >= len(proj):
                continue
            if proj[i - j + 1] < proj[i - j] or proj[i + j - 1] < proj[i + j]:
                is_peak = False
                break
        if is_peak and proj[i] >= thr:
            left_climb = 0
            for k in range(i, 0, -1):
                if proj[k] > proj[k - 1]:
                    left_climb = abs(proj[i] - proj[k - 1])
                else:
                    break
            right_fall = 0
            for k in range(i, len(proj) - 1):
                if proj[k] > proj[k + 1]:
                    right_fall = abs(proj[i] - proj[k + 1])
                else:
                    break
            candidates.append({"index": i, "climb": left_climb, "fall": right_fall, "score": max(left_climb, right_fall)})
    if not candidates:
        return None
    left = [c for c in candidates if c["index"] < center - min_dist and c["index"] > center * 0.25]
    right = [c for c in candidates if c["index"] > center + min_dist and c["index"] < center * 1.75]
    left.sort(key=lambda x: x["score"], reverse=True)
    right.sort(key=lambda x: x["score"], reverse=True)
    if not left or not right:
        return None
    return abs(left[0]["index"] - right[0]["index"]) / 2


def find_best_grid(origin, range_val_min, range_val_max, grad_mag, thr=0):
    best = round(origin)
    peaks = []
    mx = np.max(grad_mag)
    if mx < 1e-6:
        return best
    rel_thr = mx * thr
    for i in range(-round(range_val_min), round(range_val_max) + 1):
        candidate = round(origin + i)
        if candidate <= 0 or candidate >= len(grad_mag) - 1:
            continue
        if grad_mag[candidate] > grad_mag[candidate - 1] and grad_mag[candidate] > grad_mag[candidate + 1] and grad_mag[candidate] >= rel_thr:
            peaks.append((grad_mag[candidate], candidate))
    if len(peaks) == 0:
        return best
    peaks.sort(key=lambda x: x[0], reverse=True)
    return peaks[0][1]


def estimate_grid_fft(gray, peak_width=6):
    """Return (grid_w, grid_h) or None."""
    H, W = gray.shape
    mag = compute_fft_magnitude(gray)
    band_row = W // 2
    band_col = H // 2
    row_sum = np.sum(mag[:, W // 2 - band_row : W // 2 + band_row], axis=1)
    col_sum = np.sum(mag[H // 2 - band_col : H // 2 + band_col, :], axis=0)
    row_sum = normalize_minmax(row_sum, 0.0, 1.0).flatten()
    col_sum = normalize_minmax(col_sum, 0.0, 1.0).flatten()
    row_sum = smooth_1d(row_sum, k=17)
    col_sum = smooth_1d(col_sum, k=17)
    scale_row = detect_peak(row_sum, peak_width=peak_width)
    scale_col = detect_peak(col_sum, peak_width=peak_width)
    if scale_row is None or scale_col is None or scale_col <= 0:
        return None
    return scale_col, scale_row


def estimate_grid_gradient(gray, rel_thr=0.2):
    H, W = gray.shape
    grad_x, grad_y = sobel_xy(gray, ksize=3)
    grad_x_sum = np.sum(np.abs(grad_x), axis=0).reshape(-1)
    grad_y_sum = np.sum(np.abs(grad_y), axis=1).reshape(-1)
    peak_x = []
    peak_y = []
    thr_x = float(rel_thr) * float(grad_x_sum.max())
    thr_y = float(rel_thr) * float(grad_y_sum.max())
    min_interval = 4
    for i in range(1, len(grad_x_sum) - 1):
        if grad_x_sum[i] > grad_x_sum[i - 1] and grad_x_sum[i] > grad_x_sum[i + 1] and grad_x_sum[i] >= thr_x:
            if len(peak_x) == 0 or i - peak_x[-1] >= min_interval:
                peak_x.append(i)
    for i in range(1, len(grad_y_sum) - 1):
        if grad_y_sum[i] > grad_y_sum[i - 1] and grad_y_sum[i] > grad_y_sum[i + 1] and grad_y_sum[i] >= thr_y:
            if len(peak_y) == 0 or i - peak_y[-1] >= min_interval:
                peak_y.append(i)
    if len(peak_x) < 4 or len(peak_y) < 4:
        return None, None
    intervals_x = [peak_x[i] - peak_x[i - 1] for i in range(1, len(peak_x))]
    intervals_y = [peak_y[i] - peak_y[i - 1] for i in range(1, len(peak_y))]
    scale_x = W / np.median(intervals_x)
    scale_y = H / np.median(intervals_y)
    logger.debug("Detected grid size from gradient: (%.2f, %.2f)", scale_x, scale_y)
    return int(round(scale_x)), int(round(scale_y))


def detect_grid_scale(image, peak_width=6, max_ratio=1.5, min_size=4.0):
    gray = rgb_to_gray(image)
    H, W = gray.shape
    # estimate_grid_fft 失败时返回 None（非二元组），需先判空再解包
    grid_fft = estimate_grid_fft(gray, peak_width=peak_width)
    grid_w = grid_h = None
    if grid_fft is not None:
        grid_w, grid_h = grid_fft
        pixel_size_x = W / grid_w
        pixel_size_y = H / grid_h
        max_pixel_size = 20.0
        if (
            min(pixel_size_x, pixel_size_y) < min_size
            or max(pixel_size_x, pixel_size_y) > max_pixel_size
            or pixel_size_x / pixel_size_y > max_ratio
            or pixel_size_y / pixel_size_x > max_ratio
        ):
            logger.debug("Inconsistent grid size detected (FFT-based), fallback to gradient-based method.")
            grid_w = grid_h = None
    if grid_w is None or grid_h is None:
        logger.debug("FFT-based grid estimation failed, fallback to gradient-based method.")
        grid_w, grid_h = estimate_grid_gradient(gray)
    if grid_w is None or grid_h is None:
        logger.debug("Gradient-based grid estimation failed.")
        return None, None
    pixel_size_x = W / grid_w
    pixel_size_y = H / grid_h
    if pixel_size_x / pixel_size_y > max_ratio or pixel_size_y / pixel_size_x > max_ratio:
        pixel_size = min(pixel_size_x, pixel_size_y)
    else:
        pixel_size = (pixel_size_x + pixel_size_y) / 2.0
    logger.debug("Detected pixel size: %.2f", pixel_size)
    grid_w = int(round(W / pixel_size))
    grid_h = int(round(H / pixel_size))
    return grid_w, grid_h


def _cell_purity(rgb, grid_w, grid_h):
    """候选网格的『格内纯度』：平均占优色占比（0~1）。

    网格越准 → 每个格内像素越接近单一纯色 → 占优色占比越高。
    全向量化：pack 颜色 + 按 (cell, color) 全局直方图 + reduceat 分组取众数。
    """
    H, W = rgb.shape[:2]
    gx = np.linspace(0, W, grid_w + 1).astype(np.int64)
    gy = np.linspace(0, H, grid_h + 1).astype(np.int64)
    cell_x = np.clip(np.searchsorted(gx, np.arange(W), side="right") - 1, 0, grid_w - 1)
    cell_y = np.clip(np.searchsorted(gy, np.arange(H), side="right") - 1, 0, grid_h - 1)
    cell = (cell_y[:, None] * grid_w + cell_x[None, :]).ravel()
    packed = (
        (rgb[..., 0].astype(np.int64) << 16)
        | (rgb[..., 1].astype(np.int64) << 8)
        | rgb[..., 2].astype(np.int64)
    ).ravel()
    key = cell * (1 << 24) + packed
    uniq, counts = np.unique(key, return_counts=True)
    ucell = uniq >> 24
    order = np.argsort(ucell, kind="stable")
    ucs, cnts = ucell[order], counts[order]
    bounds = np.flatnonzero(np.diff(ucs)) + 1
    starts = np.concatenate([[0], bounds])
    group_max = np.maximum.reduceat(cnts, starts)
    cell_cnt = np.bincount(cell, minlength=grid_w * grid_h)
    frac = group_max / np.maximum(cell_cnt, 1)
    return float(frac.mean())


def detect_grid_size(
    image,
    peak_width=6,
    min_size=4.0,
    max_size=24.0,
    search_radius=8,
    max_search_side=224,
):
    """检测像素网格大小：FFT 估计 + 格内纯度候选精修。

    仅靠 FFT 频率估计存在分辨率/压缩/谐波导致的偏移（实测：干净图 24→22、
    32→24，模糊图 16→14、10→32，非整数格距甚至返回 None）。
    这里以 FFT 估计为锚点，在 ±search_radius 及若干谐波变体附近，用
    『格内纯度』评分选出最准确的网格；估计失败时做全范围搜索。

    返回 (grid_w, grid_h)；检测失败（无有效网格 / 非像素风）返回 (None, None)。
    """
    H, W = image.shape[:2]
    # 快速评分图：双线性缩小（网格格数不随缩放改变；块均值保留格内主色）
    scale = min(1.0, max_search_side / max(H, W))
    if scale < 1.0:
        from PIL import Image as PILImage

        sw, sh = max(8, int(W * scale)), max(8, int(H * scale))
        small = np.asarray(
            PILImage.fromarray(image).resize((sw, sh), PILImage.Resampling.BILINEAR)
        )
    else:
        small = image
    sH, sW = small.shape[:2]

    # FFT 估计（带范围校验，同原 detect_grid_scale 逻辑）
    est = None
    grid_fft = estimate_grid_fft(rgb_to_gray(image), peak_width=peak_width)
    est = None
    if grid_fft is not None:
        ew, eh = grid_fft
        pxs, pys = W / ew, H / eh
        if (
            min(pxs, pys) >= min_size
            and max(pxs, pys) <= 20.0
            and 1 / 1.5 <= pxs / pys <= 1.5
        ):
            est = (int(round(ew)), int(round(eh)))
    if est is None:
        # detect_grid_scale 失败时返回 (None, None) 元组
        fb = detect_grid_scale(image, peak_width=peak_width, max_ratio=1.5, min_size=min_size)
        if fb is not None and fb[0] is not None:
            est = (int(fb[0]), int(fb[1]))

    def _range(center, lo, hi):
        return sorted(set(range(max(lo, center - search_radius), min(hi, center + search_radius) + 1)))

    lo_w, hi_w = max(2, int(sW / max_size)), min(sW, int(sW / min_size))
    lo_h, hi_h = max(2, int(sH / max_size)), min(sH, int(sH / min_size))
    if est:
        ew, eh = est
        cw = set(_range(ew, lo_w, hi_w))
        ch = set(_range(eh, lo_h, hi_h))
        # 谐波变体（FFT 可能锁定在谐波上）：各给 ±3 小范围
        for k in (2, 3, 1.5, 0.5, 2 / 3, 1 / 3):
            cw.update(range(max(lo_w, int(round(ew * k)) - 3), min(hi_w, int(round(ew * k)) + 3) + 1))
            ch.update(range(max(lo_h, int(round(eh * k)) - 3), min(hi_h, int(round(eh * k)) + 3) + 1))
    else:
        cw, ch = range(lo_w, hi_w + 1), range(lo_h, hi_h + 1)

    # 边界贴合度在原图（或降采样图）上精确计算：网格线位置需与真实格边对齐
    gray_full = rgb_to_gray(image).astype(np.float32)
    gx_full = np.abs(np.diff(gray_full, axis=1))
    gy_full = np.abs(np.diff(gray_full, axis=0))
    mean_all_full = float((gx_full.mean() + gy_full.mean()) / 2)
    if mean_all_full <= 1e-6:
        return None, None  # 无任何边缘（纯色图）-> 非像素风

    def _boundary(gw, gh):
        """网格线落在真实边缘上的贴合度（0~1）；线贴边 → 高。"""
        col_e = [float(gx_full[..., max(0, int(round(i * W / gw)) - 1)].mean()) for i in range(1, gw)]
        row_e = [float(gy_full[max(0, int(round(j * H / gh)) - 1), :].mean()) for j in range(1, gh)]
        if not col_e or not row_e:
            return 0.0
        mean_line = (np.mean(col_e) + np.mean(row_e)) / 2
        return float(mean_line / (mean_all_full + mean_line))

    # 1) 纯度 + 正方形格门槛（排除非像素候选）：
    #    - 格内纯色占比 ≥ 0.40（像素画特征）；
    #    - 格子必须近似正方形（像素画定义）：格纵横比 (W/gw)/(H/gh) ∈ [1/1.6, 1.6]
    #    过细网格的纯度可能更高，因此纯度只作门槛、不作主选择标准
    def _aspect_ok(gw, gh):
        r = (sW / gw) / (sH / gh)
        return 1 / 1.6 <= r <= 1.6

    gated = [
        (gw, gh, _cell_purity(small, gw, gh))
        for gw in cw
        for gh in ch
        if _cell_purity(small, gw, gh) >= 0.40 and _aspect_ok(gw, gh)
    ]
    if not gated:
        return None, None  # 无足够「格内纯色」结构 = 非像素风

    # 2) 选择：边界贴合度最高者（正确网格的线贴在真实格边上，过细/过粗网格更低）；
    #    边界取整到 2 位小数（消除逐线取整噪声），并列时以纯度更高者优先，
    #    再以 FFT 估计接近度打破
    best = min(
        gated,
        key=lambda t: (
            -round(_boundary(t[0], t[1]), 2),
            -t[2],
            (abs(t[0] - est[0]) + abs(t[1] - est[1])) if est else 0,
        ),
    )

    # 3) 纯度峰值锐度：真实像素画在正确网格处纯度显著高于相邻网格（错位即下降）；
    #    平坦/渐变图像在任意网格下纯度都相近（无峰）-> 判为非像素风
    gw, gh = best[0], best[1]
    center_purity = _cell_purity(small, gw, gh)
    neighbor_max = max(
        _cell_purity(small, gw + dx, gh + dy)
        for dx in (-1, 1)
        for dy in (-1, 1)
        if gw + dx >= 2 and gh + dy >= 2
    )
    if center_purity - neighbor_max < 0.05:
        return None, None
    return gw, gh


# ----------------------------
# Grid refinement & sampling
# ----------------------------
def refine_grids(image, grid_x, grid_y, refine_intensity=0.25):
    H, W = image.shape[:2]
    cell_w = W / grid_x
    cell_h = H / grid_y
    gray = rgb_to_gray(image)
    gx, gy = sobel_xy(gray, ksize=3)
    grad_x_sum = np.sum(np.abs(gx), axis=0).reshape(-1)
    grad_y_sum = np.sum(np.abs(gy), axis=1).reshape(-1)

    x_coords = []
    y_coords = []
    x = find_best_grid(W / 2, cell_w, cell_w, grad_x_sum)
    while x < W + cell_w / 2:
        x = find_best_grid(x, cell_w * refine_intensity, cell_w * refine_intensity, grad_x_sum)
        x_coords.append(x)
        x += cell_w
    x = find_best_grid(W / 2, cell_w, cell_w, grad_x_sum) - cell_w
    while x > -cell_w / 2:
        x = find_best_grid(x, cell_w * refine_intensity, cell_w * refine_intensity, grad_x_sum)
        x_coords.append(x)
        x -= cell_w

    y = find_best_grid(H / 2, cell_h, cell_h, grad_y_sum)
    while y < H + cell_h / 2:
        y = find_best_grid(y, cell_h * refine_intensity, cell_h * refine_intensity, grad_y_sum)
        y_coords.append(y)
        y += cell_h
    y = find_best_grid(H / 2, cell_h, cell_h, grad_y_sum) - cell_h
    while y > -cell_h / 2:
        y = find_best_grid(y, cell_h * refine_intensity, cell_h * refine_intensity, grad_y_sum)
        y_coords.append(y)
        y -= cell_h

    return sorted(x_coords), sorted(y_coords)


def sample_center(image, x_coords, y_coords):
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)
    centers_x = ((x[1:] + x[:-1]) * 0.5).astype(np.int32)
    centers_y = ((y[1:] + y[:-1]) * 0.5).astype(np.int32)
    return image[centers_y[:, None], centers_x[None, :]]


def sample_mode_fast(image, x_coords, y_coords, max_colors=64):
    """向量化众数采样：每个网格单元取「最频繁颜色」的真实平均色。

    像素艺术中每个格子应是单一纯色；众数采样比 2-means 更准（不产生混合脏色）。
    全程向量化：PIL MEDIANCUT 量化（C 实现）-> np.add.at + bincount 统计
    每格调色板簇众数 -> 取众数簇像素的平均色。比逐格 Python 循环快数十倍。
    """
    from PIL import Image as PILImage

    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image[..., :3]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    H, W = rgb.shape[:2]

    # 1) 量化到小调色板（MEDIANCUT，无抖动）
    q = PILImage.fromarray(rgb).quantize(
        colors=max_colors,
        method=PILImage.Quantize.MEDIANCUT,
        dither=PILImage.Dither.NONE,
    )
    idx = np.asarray(q)  # (H, W) 调色板索引
    P = max_colors

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    nx, ny = len(x) - 1, len(y) - 1
    if nx < 1 or ny < 1:
        return np.zeros((max(1, ny), max(1, nx), 3), dtype=np.uint8)

    # 2) 每个像素的单元归属（网格线坐标 -> cell id）
    cell_x = np.clip(np.searchsorted(x, np.arange(W), side="right") - 1, 0, nx - 1)
    cell_y = np.clip(np.searchsorted(y, np.arange(H), side="right") - 1, 0, ny - 1)
    cell = (cell_y[:, None] * nx + cell_x[None, :]).ravel()
    n_cells = nx * ny

    # 3) 按 (cell, 调色板簇) 聚合像素数与颜色和（全向量化）
    flat_target = cell.astype(np.int64) * P + idx.ravel().astype(np.int64)
    sums = np.zeros((n_cells * P, 3), dtype=np.int64)
    np.add.at(sums, flat_target, rgb.reshape(-1, 3).astype(np.int64))
    cnt = np.bincount(flat_target, minlength=n_cells * P)

    # 4) 每格众数簇 -> 该簇像素的真实平均色；同时算整体均值
    per_cell_counts = cnt.reshape(n_cells, P)
    total_per_cell = per_cell_counts.sum(axis=1, keepdims=True)
    mode = np.argmax(per_cell_counts, axis=1)
    best = mode.astype(np.int64) + np.arange(n_cells, dtype=np.int64) * P
    mode_means = sums[best].astype(np.float32) / np.maximum(cnt[best], 1)[:, None]

    cell_sums = np.zeros((n_cells, 3), dtype=np.int64)
    np.add.at(cell_sums, cell, rgb.reshape(-1, 3).astype(np.int64))
    cell_cnt = np.bincount(cell, minlength=n_cells)
    overall_means = cell_sums.astype(np.float32) / np.maximum(cell_cnt, 1)[:, None]

    # 众数簇占优（>50%）时用众数均值（消除边界/杂色干扰）；
    # 否则为混合格（如渐变），用整体均值（噪声均匀时均值最优）
    mode_counts = np.take_along_axis(per_cell_counts, mode[:, None], axis=1)[:, 0]
    dominant = (mode_counts / np.maximum(total_per_cell[:, 0], 1)) > 0.5
    means = np.where(dominant[:, None], mode_means, overall_means)
    out = np.clip(means, 0, 255).astype(np.uint8).reshape(ny, nx, 3)
    return out


def sample_mode_exact(image, x_coords, y_coords, dominance=0.35):
    """逐格取「出现最多的精确颜色」；混合格回退整体均值。

    与 sample_mode_fast 的区别：不做 MEDIANCUT 预量化，直接统计原始像素的
    精确颜色众数 —— 消除量化带来的颜色偏移，格色还原更准。
    dominance 为「众数判定阈值」：像素画格内占优纯色通常只有 40%~95%
    （其余为反锯齿/噪声），0.35 足够捕获；真正的渐变混合格（无数占优色）
    才会回退均值，避免脏色。
    全局一次排序分组统计（O(n log n)），网格级速度可接受。
    """
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image[..., :3]
    H, W = rgb.shape[:2]
    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    nx, ny = len(x) - 1, len(y) - 1
    if nx < 1 or ny < 1:
        return np.zeros((max(1, ny), max(1, nx), 3), dtype=np.uint8)
    n_cells = nx * ny

    cell_x = np.clip(np.searchsorted(x, np.arange(W), side="right") - 1, 0, nx - 1)
    cell_y = np.clip(np.searchsorted(y, np.arange(H), side="right") - 1, 0, ny - 1)
    cell = (cell_y[:, None] * nx + cell_x[None, :]).ravel()

    packed = (
        rgb[..., 0].astype(np.int64) << 16
        | rgb[..., 1].astype(np.int64) << 8
        | rgb[..., 2].astype(np.int64)
    )
    flat = packed.ravel()

    # 全局直方图：唯一 (cell, color) 组合及出现次数
    key = cell.astype(np.int64) * (1 << 24) + flat
    uniq, counts = np.unique(key, return_counts=True)
    ucell = uniq >> 24
    upack = uniq & ((1 << 24) - 1)

    # 按 cell 分组，取每格出现次数最多的精确颜色
    order = np.argsort(ucell, kind="stable")
    ucell_s, ucnt_s, upack_s = ucell[order], counts[order], upack[order]
    bounds = np.flatnonzero(np.diff(ucell_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [ucell_s.shape[0]]])

    cell_cnt = np.bincount(cell, minlength=n_cells)
    mode_packed = np.zeros(n_cells, dtype=np.int64)
    dominant = np.zeros(n_cells, dtype=bool)
    for c in range(n_cells):
        s, e = starts[c], ends[c]
        if s >= e:
            continue
        j = s + int(np.argmax(ucnt_s[s:e]))
        mode_packed[c] = upack_s[j]
        dominant[c] = (ucnt_s[j] / max(1, cell_cnt[c])) > dominance

    mode_rgb = np.stack(
        [(mode_packed >> 16) & 255, (mode_packed >> 8) & 255, mode_packed & 255], axis=-1
    ).astype(np.uint8)

    sums = np.zeros((n_cells, 3), dtype=np.int64)
    np.add.at(sums, cell, rgb.reshape(-1, 3).astype(np.int64))
    means = (sums.astype(np.float32) / np.maximum(cell_cnt, 1)[:, None]).astype(np.uint8)

    out = np.where(dominant[:, None], mode_rgb, means)
    return np.clip(out.reshape(ny, nx, 3), 0, 255).astype(np.uint8)


def sample_majority(image, x_coords, y_coords, max_samples=128, iters=6, seed=0):
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]
    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)

    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H)
        y1 = np.clip(y1, 0, H)
        if y1 <= y0:
            y1 = min(y0 + 1, H)
        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W)
            x1 = np.clip(x1, 0, W)
            if x1 <= x0:
                x1 = min(x0 + 1, W)
            cell = img[y0:y1, x0:x1].reshape(-1, C)
            n = cell.shape[0]
            if n == 0:
                out[j, i] = 0
                continue
            if n > max_samples:
                cell = cell[rng.integers(0, n, size=max_samples)]
            c0 = cell[0]
            c1 = cell[np.argmax(((cell - c0) ** 2).sum(1))]
            for _ in range(iters):
                d0 = ((cell - c0) ** 2).sum(1)
                d1 = ((cell - c1) ** 2).sum(1)
                m1 = d1 < d0
                if np.any(~m1):
                    c0 = cell[~m1].mean(0)
                if np.any(m1):
                    c1 = cell[m1].mean(0)
            out[j, i] = c1 if m1.sum() >= (~m1).sum() else c0
    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def sample_median(image, x_coords, y_coords):
    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]
    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)
    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H)
        y1 = np.clip(y1, 0, H)
        if y1 <= y0:
            y1 = min(y0 + 1, H)
        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W)
            x1 = np.clip(x1, 0, W)
            if x1 <= x0:
                x1 = min(x0 + 1, W)
            cell = img[y0:y1, x0:x1].reshape(-1, C)
            if cell.shape[0] == 0:
                out[j, i] = 0
            else:
                out[j, i] = np.median(cell, axis=0)
    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def perfect_pixel_plan(
    image,
    sample_method="center",
    grid_size=None,
    min_size=4.0,
    peak_width=6,
    refine_intensity=0.25,
    fix_square=True,
):
    """检测网格并生成采样计划（供多帧复用同一网格）。

    Returns:
        (refined_w, refined_h, x_coords, y_coords) 或 None（检测失败）。
    """
    H, W = image.shape[:2]
    if grid_size is not None:
        scale_col, scale_row = grid_size
    else:
        # 纯度精修网格大小：FFT 估计 + 候选搜索（修正频率偏移/谐波/非整数格距）
        scale_col, scale_row = detect_grid_size(
            image, peak_width=peak_width, min_size=min_size
        )
        if scale_col is None or scale_row is None:
            logger.debug("Failed to estimate grid size.")
            return None
    size_x = int(round(scale_col))
    size_y = int(round(scale_row))
    x_coords, y_coords = refine_grids(image, size_x, size_y, refine_intensity)
    if len(x_coords) < 2 or len(y_coords) < 2:
        logger.debug("Grid refinement failed (too few grid lines).")
        return None
    refined_w = len(x_coords) - 1
    refined_h = len(y_coords) - 1
    if fix_square and abs(refined_w - refined_h) == 1:
        if refined_w > refined_h:
            if refined_w % 2 == 1:
                x_coords = x_coords[:-1]
            else:
                y_coords = [y_coords[0]] + y_coords
        else:
            if refined_h % 2 == 1:
                y_coords = y_coords[:-1]
            else:
                x_coords = [x_coords[0]] + x_coords
        refined_w = len(x_coords) - 1
        refined_h = len(y_coords) - 1
    logger.debug("Refined grid size: (%d, %d)", refined_w, refined_h)
    return refined_w, refined_h, x_coords, y_coords


def sample_with_plan(image, plan, sample_method="center"):
    """用已有采样计划（x_coords/y_coords）对图像采样。

    majority 使用精确颜色众数采样（不量化，颜色还原最准）；
    center/median 保留原实现。
    """
    _, _, x_coords, y_coords = plan
    if sample_method == "majority":
        return sample_mode_exact(image, x_coords, y_coords)
    if sample_method == "median":
        return sample_median(image, x_coords, y_coords)
    return sample_center(image, x_coords, y_coords)


def get_perfect_pixel(image, sample_method="center", grid_size=None, min_size=4.0, peak_width=6, refine_intensity=0.25, fix_square=True):
    """Perfect Pixel 主入口。

    Args:
        image: RGB ndArray (H * W * 3)
        sample_method: "majority", "center" 或 "median"
        grid_size: (grid_w, grid_h) 手动指定网格，覆盖自动检测
        min_size: 最小可接受的像素单元尺寸
        peak_width: 峰值检测的最小峰宽
        refine_intensity: 网格线精修强度，推荐 [0, 0.5]
        fix_square: 输出接近正方形时强制为正方形

    Returns:
        (refined_w, refined_h, scaled_image)；检测失败时返回 (None, None, image)
    """
    H, W = image.shape[:2]
    plan = perfect_pixel_plan(
        image,
        sample_method=sample_method,
        grid_size=grid_size,
        min_size=min_size,
        peak_width=peak_width,
        refine_intensity=refine_intensity,
        fix_square=fix_square,
    )
    if plan is None:
        return None, None, image
    refined_w, refined_h, _, _ = plan
    scaled_image = sample_with_plan(image, plan, sample_method=sample_method)
    logger.debug("Refined grid size: (%d, %d)", refined_w, refined_h)
    return refined_w, refined_h, scaled_image
