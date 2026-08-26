"""网格大小检测精度回归测试。

覆盖：干净棋盘格（含小格/非整格距）、模糊/反锯齿、非像素（噪声/渐变/平坦块）、
以及"纯度峰值锐度"误判防护（mock 合成帧不应被判为像素风）。
"""
import numpy as np
import pytest
from PIL import Image, ImageFilter

from core.processing import pixelizer as px
from core.processing import perfect_pixel as pp


def _checker(size, cells, cell):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for cy in range(cells):
        for cx in range(cells):
            v = 255 if ((cx + cy) % 2 == 0) else 20
            arr[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell] = v
    return Image.fromarray(arr)


def _detect(img):
    r = px.perfect_pixelize_sequence([img])
    return None if r is None else r[1]


@pytest.mark.parametrize(
    "cells,cell",
    [(16, 12), (16, 13), (24, 8), (32, 6), (10, 19)],
)
def test_clean_checkerboard_grid_exact(cells, cell):
    """干净棋盘格：网格大小精确（此前 24→22、32→24 失准）。"""
    img = _checker(cells * cell, cells, cell)
    assert _detect(img) == (cells, cells)


@pytest.mark.parametrize(
    "cells,cell",
    [(16, 12), (16, 13), (10, 19)],
)
def test_blurred_checkerboard_grid_exact(cells, cell):
    """模糊/反锯齿（模拟 AI 图）：网格大小仍精确（此前 16→14、10→32 失准）。"""
    img = _checker(cells * cell, cells, cell).filter(ImageFilter.GaussianBlur(0.8))
    assert _detect(img) == (cells, cells)


def test_non_integer_cell_spacing_detected():
    """非整数格距（AI 常见）：能检测出真实格数（此前返回 None / 谐波误判）。"""
    for size, cells in [(201, 16), (173, 14)]:
        cell = size / cells
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for cy in range(cells):
            for cx in range(cells):
                x0, y0 = int(cx * cell), int(cy * cell)
                x1, y1 = int((cx + 1) * cell), int((cy + 1) * cell)
                v = 255 if ((cx + cy) % 2 == 0) else 20
                arr[y0:y1, x0:x1] = v
        assert _detect(Image.fromarray(arr)) == (cells, cells)


def test_pure_noise_rejected():
    """纯噪声（非像素）不应被判定为像素网格。"""
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 255, (192, 192, 3), dtype=np.uint8))
    assert _detect(noise) is None


def test_flat_image_rejected():
    """平坦/纯色图：任意网格纯度都高但无峰值 -> 判为非像素。"""
    img = Image.new("RGB", (192, 192), (120, 120, 120))
    assert _detect(img) is None


def test_mock_frame_not_falsely_pixel():
    """mock 合成帧（渐变/平坦背景 + 图形）不应被判为像素风（回归：
    新检测曾因纯度门槛过松将其误判，导致非像素图也导出原生分辨率版）。"""
    import core.api.mock_clients as mc

    seed = mc._hash_seed("walking cycle, side view, 8 frames, natural gait, smooth looping")
    for t in range(4):
        frame = mc._make_frame(256, 256, seed + t, t, 4)
        assert pp.detect_grid_size(np.asarray(frame.convert("RGB"))) == (None, None)


def test_detect_grid_size_returns_none_tuple_consistently():
    """detect_grid_size 失败时返回 (None, None)（元组），调用方需按此解包。"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)  # 纯色
    assert pp.detect_grid_size(img) == (None, None)
