"""Perfect Pixel 算法（内嵌版）测试。

注意：网格检测需要足够分辨率（真实视频帧 256px+ 正常），
测试使用 192x192（16 格 × 12px）贴近真实场景。
"""
from PIL import Image, ImageDraw

from core.processing import pixelizer as px


def make_checker(cells=16, cell_px=12):
    """干净的二色棋盘格：cells×cells 单元格，每格 cell_px 像素。"""
    base = Image.new("RGB", (cells, cells))
    draw = ImageDraw.Draw(base)
    for y in range(cells):
        for x in range(cells):
            draw.rectangle(
                [x, y, x, y], fill=(255, 255, 255) if (x + y) % 2 == 0 else (20, 20, 20)
            )
    return base.resize((cells * cell_px, cells * cell_px), Image.Resampling.NEAREST)


def test_perfect_pixel_detects_grid_and_samples():
    """规整像素网格图：自动检测网格并输出网格对齐的结果。"""
    img = make_checker(cells=16, cell_px=12)  # 192x192，格 12px
    out = px.perfect_pixelize(img)
    assert out is not None
    assert out.mode == "RGBA"
    # 检测到的网格约 16×16（容许 ±3 误差）
    assert 13 <= out.width <= 19 and 13 <= out.height <= 19
    # 输出为纯色格：颜色数很少（黑白 + 少量采样过渡）
    colors = set(out.convert("RGB").getdata())
    assert len(colors) <= 6


def test_perfect_pixel_with_manual_grid():
    img = make_checker(cells=16, cell_px=12)
    out = px.perfect_pixelize(img, grid_size=(16, 16))
    assert out is not None
    assert abs(out.width - 16) <= 1 and abs(out.height - 16) <= 1


def test_perfect_pixel_falls_back_on_solid_image():
    """纯色图不应崩溃（返回 None 或图像均可）。"""
    solid = Image.new("RGB", (192, 192), (120, 120, 120))
    out = px.perfect_pixelize(solid)
    assert out is None or out.size[0] > 0


def test_perfect_pixelize_sequence_empty():
    """空帧序列返回 None（回归测试）。"""
    assert px.perfect_pixelize_sequence([]) is None


def test_sequence_sampling_preserves_cell_colors():
    """逐格采样保留真实格色：棋盘格输出精确还原为源图颜色（无平均混色/颜色损失）。"""
    import numpy as np

    size, cell = 192, 12
    n = size // cell
    frames = []
    for i in range(3):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for cy in range(n):
            for cx in range(n):
                v = 255 if ((cx + cy + i) % 2 == 0) else 20
                arr[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell] = v
        frames.append(Image.fromarray(arr))

    out, grid = px.perfect_pixelize_sequence(frames)
    assert grid == (16, 16)
    for f in out:
        colors = set(f.convert("RGB").getdata())
        assert colors == {(20, 20, 20), (255, 255, 255)}  # 精确两色，无混色
    # 内容相同的帧输出完全一致（帧间一致性 + 循环闭合）
    assert out[0].tobytes() == out[2].tobytes()


def test_perfect_pixel_large_image_downscaled():
    """超大图自动先缩放再处理，不崩溃；检测失败时返回 None（由调用方降级常规缩放）。"""
    img = make_checker(cells=24, cell_px=48)  # 1152x1152 -> 先缩到 768
    out = px.perfect_pixelize(img, max_side=768)
    if out is not None:
        assert out.width >= 12


def test_perfect_pixel_consistent_across_similar_frames():
    """同一内容的不同帧应输出相同尺寸（帧间一致性前提）。"""
    a = make_checker(cells=16, cell_px=12)
    b = make_checker(cells=16, cell_px=12)
    oa, ob = px.perfect_pixelize(a), px.perfect_pixelize(b)
    if oa is not None and ob is not None:
        assert oa.size == ob.size


def test_sample_mode_fast_accuracy():
    """向量化众数采样：带噪声的像素格仍还原真实格色（无混合脏色）。"""
    import numpy as np

    from core.processing import perfect_pixel as pp

    rng = np.random.default_rng(1)
    cells, cell_px = 16, 8
    base = rng.integers(0, 3, size=(cells, cells))
    pal = np.array([[255, 255, 255], [20, 20, 20], [200, 30, 30]], dtype=np.uint8)
    img = pal[base]
    img = np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)
    noise = rng.integers(0, 15, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    x = np.arange(0, 129, cell_px).astype(float)
    y = np.arange(0, 129, cell_px).astype(float)
    out = pp.sample_mode_fast(img, x, y)
    assert out.shape == (cells, cells, 3)
    # 每格颜色接近真实格色（多数格严格一致）
    acc = (np.abs(out.reshape(-1, 3).astype(int) - pal[base.ravel()]).sum(1) < 40).mean()
    assert acc >= 0.95


def test_sample_mode_exact_restores_true_colors():
    """精确颜色众数采样：格内占优纯色时精确还原真实格色（无量化偏移）。

    真实像素画场景：格内多数像素为纯色，仅少量边缘/噪声杂色。
    """
    import numpy as np

    from core.processing import perfect_pixel as pp

    rng = np.random.default_rng(2)
    cells, cell_px = 16, 8
    base = rng.integers(0, 3, size=(cells, cells))
    pal = np.array([[255, 255, 255], [20, 20, 20], [200, 30, 30]], dtype=np.uint8)
    img = pal[base]
    img = np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)
    # 仅 20% 像素加噪声（模拟反锯齿/杂色边缘），80% 为纯格色
    noise_mask = rng.random(img.shape) < 0.2
    noise = rng.integers(0, 15, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise * noise_mask, 0, 255).astype(np.uint8)

    x = np.arange(0, 129, cell_px).astype(float)
    y = np.arange(0, 129, cell_px).astype(float)
    out = pp.sample_mode_exact(img, x, y)
    assert out.shape == (cells, cells, 3)
    # 众数精确颜色应严格等于真实格色
    exact = (out.reshape(-1, 3) == pal[base.ravel()]).all(axis=1)
    assert exact.mean() >= 0.95
