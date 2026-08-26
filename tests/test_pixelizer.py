"""严格像素化算法测试。"""
from PIL import Image

from core.processing import pixelizer as px


def make_image(size=(64, 64), mode="RGB", color=(120, 200, 90)):
    img = Image.new(mode, size, color)
    # 画一个异色方块
    for y in range(10, 20):
        for x in range(10, 20):
            img.putpixel((x, y), (255, 0, 0) if mode == "RGB" else (255, 0, 0, 255))
    return img


def test_resize_nearest_keeps_hard_edges():
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    for y in range(5):
        for x in range(5):
            img.putpixel((x, y), (255, 255, 255))
    out = px.resize_nearest(img, (40, 40))
    assert out.size == (40, 40)
    # 最近邻：像素块保持纯色（无插值灰阶）
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((39, 39)) == (0, 0, 0)
    assert out.getpixel((20, 20)) == (0, 0, 0)


def test_center_crop_to_ratio():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    out = px.center_crop_to_ratio(img, (1, 1))
    assert out.size == (100, 100)  # 高不变，裁宽


def test_pixelize_image_quantizes_colors():
    img = make_image((64, 64))
    out = px.pixelize_image(img, px.PixelizeParams(target_size=(32, 32), max_colors=4))
    assert out.size == (32, 32)
    assert out.mode == "RGBA"
    colors = set(out.convert("RGB").getdata())
    # 量化后颜色数远小于源图
    assert len(colors) <= 8


def test_pixelize_with_fixed_palette():
    img = make_image((64, 64))
    palette = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)]
    params = px.PixelizeParams(target_size=(32, 32), palette=palette)
    out = px.pixelize_image(img, params)
    colors = set(out.convert("RGB").getdata())
    assert colors.issubset(set(palette))


def test_pixelize_frames_share_palette():
    frames = [make_image((64, 64), color=(120 + i, 200, 90)) for i in range(3)]
    out = px.pixelize_frames(frames, px.PixelizeParams(target_size=(32, 32), max_colors=8))
    assert len(out) == 3
    palette_sets = [set(f.convert("RGB").getdata()) for f in out]
    # 各帧使用同一调色板：颜色种类一致
    assert all(len(s) == len(palette_sets[0]) for s in palette_sets)


def test_pixelize_preserves_alpha():
    img = make_image((64, 64), mode="RGBA")
    img.putpixel((0, 0), (10, 10, 10, 0))  # 一个全透明像素
    out = px.pixelize_image(img, px.PixelizeParams(target_size=(32, 32), max_colors=8))
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 0


def test_extract_palette_limits_colors():
    img = make_image((64, 64))
    palette = px.extract_palette(img, 8)
    assert 2 <= len(palette) <= 8
    assert all(len(c) == 3 for c in palette)


def test_extract_dominant_palette_keeps_dominant_colors():
    """频率主导调色板：主色精确保留（不产生 MEDIANCUT 混色）。"""
    # 大量红 + 少量绿 + 更少蓝
    img = Image.new("RGB", (64, 64), (200, 30, 30))
    for i in range(20):
        img.putpixel((i, 0), (20, 200, 30))
    for i in range(3):
        img.putpixel((i, 1), (30, 30, 220))
    palette = px.extract_dominant_palette(img, 2)
    assert (200, 30, 30) in palette
    assert (20, 200, 30) in palette
    assert (30, 30, 220) not in palette  # 频率最低的被排除


def test_no_target_size_keeps_dimensions():
    img = make_image((50, 30))
    out = px.pixelize_image(img, px.PixelizeParams(max_colors=8))
    assert out.size == (50, 30)
