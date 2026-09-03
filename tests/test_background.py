"""背景去除测试。"""
import numpy as np
from PIL import Image

from core.processing import background as bg


def make_image_with_white_bg():
    img = Image.new("RGB", (40, 40), (255, 255, 255))
    for y in range(10, 30):
        for x in range(10, 30):
            img.putpixel((x, y), (200, 50, 50))
    return img


def test_white_background_removed():
    img = make_image_with_white_bg()
    out = bg.remove_white_background(img, tolerance=30)
    assert out.mode == "RGBA"
    arr = np.array(out)
    # 角落（背景）透明
    assert arr[0, 0, 3] == 0
    # 中心（前景）不透明
    assert arr[20, 20, 3] == 255


def test_custom_key_color():
    img = Image.new("RGB", (20, 20), (0, 0, 255))
    out = bg.remove_background(img, key_color=(0, 0, 255), tolerance=10)
    assert np.array(out)[0, 0, 3] == 0


def test_tolerance_gap():
    img = Image.new("RGB", (20, 20), (250, 250, 250))
    # 容差很小 -> 不算背景
    out = bg.remove_background(img, key_color=(255, 255, 255), tolerance=1)
    assert np.array(out)[0, 0, 3] == 255


def test_feather_gives_partial_alpha():
    img = make_image_with_white_bg()
    # 加一个处于过渡带内的中间色像素：到白色的 L1 距离 = 15 ∈ (10, 20]
    # 关闭形态学清理，避免开运算把孤立过渡像素并入背景掩膜
    img.putpixel((5, 35), (252, 252, 246))  # |252-255|*2 + |246-255| = 6 + 9 = 15
    out = bg.remove_background(
        img, key_color=(255, 255, 255), tolerance=10, feather=10, edge_clean=False
    )
    arr = np.array(out)
    alphas = set(arr[..., 3].flatten().tolist())
    # 存在中间 alpha（半透明过渡）
    assert any(0 < a < 255 for a in alphas)
    # 过渡带像素的 alpha 介于 0 与 255 之间（frac = 1 - 5/10 = 0.5 -> ~127）
    # 注意：arr 索引是 [y, x]，putpixel 是 (x=5, y=35)
    assert 0 < arr[35, 5, 3] < 255


def test_input_rgba_passthrough():
    img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
    out = bg.remove_white_background(img)
    assert out.mode == "RGBA"
    assert np.array(out)[0, 0, 3] == 0


def test_whiten_background_light_gray():
    """边缘相连的浅灰背景被强制为纯白，暗色主体不受影响。"""
    img = Image.new("RGB", (64, 64), (230, 230, 230))  # 发灰的"白"
    for y in range(20, 44):
        for x in range(20, 44):
            img.putpixel((x, y), (50, 60, 200))  # 彩色主体
    out = bg.whiten_background(img)
    arr = np.array(out)
    # 角落（背景连通区域）-> 纯白
    assert tuple(arr[2, 2, :3]) == (255, 255, 255)
    assert tuple(arr[0, 60, :3]) == (255, 255, 255)
    # 主体中心保持不变
    assert tuple(arr[32, 32, :3]) == (50, 60, 200)
    # alpha 保留
    assert arr[2, 2, 3] == 255


def test_whiten_background_ignores_dark_edge():
    """深色/彩色边缘不应被误刷白。"""
    img = Image.new("RGB", (32, 32), (30, 40, 50))
    out = bg.whiten_background(img)
    assert tuple(np.array(out)[0, 0, :3]) == (30, 40, 50)


def test_whiten_background_does_not_leak_into_subject():
    """深色内嵌主体（被边框包围、不与边缘连通）不被刷白。"""
    img = Image.new("RGB", (48, 48), (245, 245, 245))
    # 中间画一块深色区域，被深色边框包围（不与边缘连通）
    for y in range(16, 32):
        for x in range(16, 32):
            img.putpixel((x, y), (60, 70, 80))
    for y in range(15, 33):
        img.putpixel((15, y), (20, 20, 20))
        img.putpixel((32, y), (20, 20, 20))
    for x in range(16, 32):
        img.putpixel((x, 15), (20, 20, 20))
        img.putpixel((x, 32), (20, 20, 20))
    out = bg.whiten_background(img)
    arr = np.array(out)
    # 边缘背景已刷白（主体深色 -> 白底）
    assert tuple(arr[2, 2, :3]) == (255, 255, 255)
    # 边框内深色区域保持原样（不连通，不被刷白）
    assert tuple(arr[24, 24, :3]) == (60, 70, 80)
    assert tuple(arr[15, 20, :3]) == (20, 20, 20)


def test_normalize_background_adaptive_white():
    """深色主体 + 浅色背景 -> 背景填充纯白。"""
    img = Image.new("RGB", (64, 64), (230, 230, 230))
    for y in range(20, 44):
        for x in range(20, 44):
            img.putpixel((x, y), (40, 60, 200))
    out, fill, mask = bg.normalize_background(img)
    assert fill == (255, 255, 255)
    arr = np.array(out)
    assert tuple(arr[2, 2, :3]) == (255, 255, 255)
    assert mask is not None and mask[2, 2] and not mask[32, 32]


def test_normalize_background_adaptive_black_for_pale_subject():
    """对象本身是浅色系 -> 背景自动填充纯黑，保证对比度。"""
    img = Image.new("RGB", (64, 64), (230, 230, 230))
    for y in range(20, 44):
        for x in range(20, 44):
            img.putpixel((x, y), (252, 250, 248))  # 极淡色主体
    out, fill, mask = bg.normalize_background(img)
    assert fill == (0, 0, 0)
    arr = np.array(out)
    assert tuple(arr[2, 2, :3]) == (0, 0, 0)
    # 浅色主体保留
    assert tuple(arr[32, 32, :3]) == (252, 250, 248)


def test_normalize_background_solid_image_untouched():
    """整图近一色（无主体）时不处理。"""
    img = Image.new("RGB", (32, 32), (30, 40, 50))
    out, fill, mask = bg.normalize_background(img)
    assert fill is None and mask is None
    assert tuple(np.array(out)[0, 0, :3]) == (30, 40, 50)


def test_apply_background_mask_removes_bg():
    """按掩膜抠图：背景透明、主体不透明。"""
    img = Image.new("RGB", (32, 32), (255, 255, 255))
    mask = np.zeros((32, 32), dtype=bool)
    mask[:16, :] = True  # 上半为背景
    out = bg.apply_background_mask(img, mask, feather=0)
    arr = np.array(out)
    assert arr[4, 4, 3] == 0
    assert arr[24, 16, 3] == 255


def test_remove_background_erode_removes_white_fringe():
    """内缩（erode）消掉对象边缘残留的白边/白晕。"""
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    for y in range(5, 15):
        for x in range(5, 15):
            img.putpixel((x, y), (200, 50, 50))
    # 主体右侧一像素外缘：轻度偏白的"白晕"（L1 距离 60 > 容差 30，原本会残留）
    img.putpixel((15, 10), (255, 225, 225))
    # 无内缩：白晕像素保持不透明（残留白边）
    out0 = bg.remove_background(img, tolerance=30, edge_clean=False)
    assert np.array(out0)[10, 15, 3] == 255
    # 内缩 1px：前景收缩，白晕像素被并入背景（透明）
    out1 = bg.remove_background(img, tolerance=30, edge_clean=False, erode=1)
    assert np.array(out1)[10, 15, 3] == 0
    # 主体核心仍不透明
    assert np.array(out1)[10, 10, 3] == 255


def test_apply_background_mask_erode():
    """掩膜抠图 + 内缩：前景边界向内收缩。"""
    img = Image.new("RGB", (24, 24), (255, 255, 255))
    for y in range(6, 18):
        for x in range(6, 18):
            img.putpixel((x, y), (80, 80, 200))
    mask = np.zeros((24, 24), dtype=bool)
    mask[:, :] = True
    mask[6:18, 6:18] = False  # 前景区域（非背景）
    out = bg.apply_background_mask(img, mask, feather=0, erode=2)
    arr = np.array(out)
    # 前景边缘向内缩 2px：原本前景的最外 2 圈像素变为背景（透明）
    assert arr[7, 10, 3] == 0      # 内缩后边缘处透明
    assert arr[10, 10, 3] == 255   # 中心仍不透明


def test_remove_background_contiguous_protects_interior():
    """contiguous 模式只删与边缘连通的背景，主体内部同色像素保留。

    场景：白底 + 黑色主体，主体内部有一个白色"眼睛"（被黑包围）。
    全局模式会误删内部白色；contiguous 模式保留它。
    """
    img = Image.new("RGB", (12, 12), (255, 255, 255))
    for y in range(3, 9):
        for x in range(2, 10):
            img.putpixel((x, y), (0, 0, 0))
    for y in range(4, 8):
        for x in range(3, 5):
            img.putpixel((x, y), (255, 255, 255))  # 主体内部白色（不是背景）

    out = bg.remove_background(
        img, key_color=(255, 255, 255), tolerance=10, mode="contiguous", edge_clean=False
    )
    assert out.getpixel((0, 0))[3] == 0    # 角落背景透明
    assert out.getpixel((3, 4))[3] == 255  # 内部白色保留

    g = bg.remove_background(
        img, key_color=(255, 255, 255), tolerance=10, mode="global", edge_clean=False
    )
    assert g.getpixel((3, 4))[3] == 0      # 对照组：全局模式误删内部白色


def test_remove_background_hybrid_tolerance():
    """hybrid 模式：连通背景大容差，主体内部同色像素小容差保护。"""
    img = Image.new("RGB", (12, 12), (255, 255, 255))
    for y in range(3, 9):
        for x in range(2, 10):
            img.putpixel((x, y), (0, 0, 0))
    for y in range(4, 8):
        for x in range(3, 5):
            img.putpixel((x, y), (240, 240, 240))  # 与白色 L1 距离 45

    out = bg.remove_background(
        img, key_color=(255, 255, 255), tolerance=60, mode="hybrid", edge_clean=False
    )
    assert out.getpixel((3, 4))[3] == 255  # 非连通容差 30 < 45，保留

    g = bg.remove_background(
        img, key_color=(255, 255, 255), tolerance=60, mode="global", edge_clean=False
    )
    assert g.getpixel((3, 4))[3] == 0      # 对照组：全局容差 60 误删


def test_remove_background_adaptive_region_bonus():
    """adaptive 模式：大面积非连通背景区域获得容差加成，小区域保持小容差。"""
    img = Image.new("RGB", (30, 30), (255, 255, 255))
    for y in range(4, 26):
        for x in range(4, 26):
            img.putpixel((x, y), (0, 0, 0))
    # 浅灰大岛：8x12 = 96 像素（> 阈值 16），与白色 L1 距离 60
    for y in range(6, 18):
        for x in range(6, 14):
            img.putpixel((x, y), (235, 235, 235))
    # 浅灰小岛：2x2 = 4 像素，同样距离 60
    for y in (20, 21):
        for x in (20, 21):
            img.putpixel((x, y), (235, 235, 235))

    out = bg.remove_background(
        img, key_color=(255, 255, 255), tolerance=40, mode="adaptive",
        large_region_threshold=16, large_region_bonus=25,
    )
    assert out.getpixel((8, 10))[3] == 0      # 大区域：40+25=65 ≥ 60 → 移除
    assert out.getpixel((20, 20))[3] == 255   # 小区域：容差 20 < 60 → 保留


def test_remove_background_unknown_mode_raises():
    import pytest

    img = Image.new("RGB", (4, 4), (255, 255, 255))
    with pytest.raises(ValueError):
        bg.remove_background(img, mode="magic")


def test_apply_background_mask_feather_gradient():
    """掩膜抠图 + 羽化：边界带是渐晕过渡（alpha 介于 0-255），羽化数值真正生效。

    回归：旧实现把羽化带固定为 alpha=140 且只有 1px，UI 羽化值不产生差异。
    """
    img = Image.new("RGB", (48, 48), (255, 255, 255))
    for y in range(12, 36):
        for x in range(12, 36):
            img.putpixel((x, y), (80, 80, 200))
    mask = np.zeros((48, 48), dtype=bool)
    mask[:] = True
    mask[12:36, 12:36] = False  # 前景（非背景）

    out = bg.apply_background_mask(img, mask, feather=8)
    arr = np.array(out)
    alphas = set(arr[..., 3].flatten().tolist())
    # 渐晕：存在多种中间 alpha（而非固定单一半透明值）
    mids = [a for a in alphas if 0 < a < 255]
    assert len(mids) >= 3, mids
    # 前景核心不透明、远处背景全透明
    assert arr[24, 24, 3] == 255
    assert arr[2, 2, 3] == 0
    # 距前景边界 4px（掩膜边界 y=12 之外）处 alpha 介于 0-255（过渡带内）
    assert 0 < arr[8, 24, 3] < 255
    # 关闭羽化：边界一刀切（无中间 alpha）
    out0 = bg.apply_background_mask(img, mask, feather=0)
    arr0 = np.array(out0)
    assert set(arr0[..., 3].flatten().tolist()) <= {0, 255}
