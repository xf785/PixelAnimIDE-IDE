"""像素画布数据模型 + 编辑控件测试（模型测试无需 Qt）。"""
from PIL import Image

from core.editing import PixelCanvas


def make_canvas(size=(8, 8)):
    return PixelCanvas(Image.new("RGBA", size, (0, 0, 0, 0)))


def test_pencil_draws_pixel():
    c = make_canvas()
    c.set_pixel(2, 3, (255, 0, 0))
    assert c.get_pixel(2, 3) == (255, 0, 0, 255)
    # 未绘制处保持透明
    assert c.get_pixel(0, 0) == (0, 0, 0, 0)


def test_pencil_out_of_bounds_ignored():
    c = make_canvas()
    c.set_pixel(99, 99, (255, 0, 0))
    c.set_pixel(-1, 0, (255, 0, 0))
    assert c.get_pixel(0, 0) == (0, 0, 0, 0)


def test_eraser_sets_transparent():
    c = make_canvas()
    c.set_pixel(1, 1, (0, 200, 0))
    c.set_pixel(1, 1, (0, 0, 0, 0))
    assert c.get_pixel(1, 1) == (0, 0, 0, 0)


def test_draw_line_connects_points():
    c = make_canvas()
    c.draw_line((0, 0), (4, 0), (255, 255, 255))
    for x in range(5):
        assert c.get_pixel(x, 0) == (255, 255, 255, 255)
    assert c.get_pixel(5, 0) == (0, 0, 0, 0)


def test_flood_fill_replaces_region():
    c = PixelCanvas(Image.new("RGBA", (4, 4), (10, 10, 10, 255)))
    # 在纯色图上填充左上角 -> 全图变色
    c.flood_fill(0, 0, (200, 0, 0, 255))
    for y in range(4):
        for x in range(4):
            assert c.get_pixel(x, y) == (200, 0, 0, 255)


def test_flood_fill_stops_at_boundary():
    img = Image.new("RGBA", (4, 4), (10, 10, 10, 255))
    img.putpixel((2, 2), (255, 255, 255, 255))  # 中央一个“墙”像素
    c = PixelCanvas(img)
    c.flood_fill(0, 0, (0, 0, 200, 255))
    assert c.get_pixel(2, 2) == (255, 255, 255, 255)  # 墙未被填充
    assert c.get_pixel(0, 0) == (0, 0, 200, 255)


def test_undo_redo_cycle():
    c = make_canvas()
    c.set_pixel(1, 1, (255, 0, 0))
    assert c.get_pixel(1, 1) == (255, 0, 0, 255)
    assert c.undo()
    assert c.get_pixel(1, 1) == (0, 0, 0, 0)
    assert c.redo()
    assert c.get_pixel(1, 1) == (255, 0, 0, 255)


def test_undo_empty_returns_false():
    c = make_canvas()
    assert c.undo() is False
    assert c.redo() is False


def test_noop_draw_does_not_pollute_history():
    c = make_canvas()
    c.set_pixel(1, 1, (255, 0, 0))
    c.set_pixel(1, 1, (255, 0, 0))  # 相同颜色 -> 不产生新历史
    assert c.undo()
    assert c.get_pixel(1, 1) == (0, 0, 0, 0)
    assert c.undo() is False  # 只有一次有效操作


def test_replace_image_clears_history():
    c = make_canvas()
    c.set_pixel(1, 1, (255, 0, 0))
    c.replace_image(Image.new("RGBA", (4, 4), (9, 9, 9, 255)))
    assert c.size == (4, 4)
    assert c.undo() is False
    assert c.get_pixel(0, 0) == (9, 9, 9, 255)


def test_color_tuple_normalization():
    c = make_canvas()
    c.set_pixel(0, 0, (1, 2, 3))  # RGB -> RGBA（alpha 255）
    assert c.get_pixel(0, 0) == (1, 2, 3, 255)


# --------------------------------------------------------------------------- #
# 调色板锁定
# --------------------------------------------------------------------------- #
def test_palette_lock_snaps_drawing():
    c = make_canvas()
    c.set_palette([(255, 0, 0, 255), (0, 0, 255, 255)])
    c.set_pixel(1, 1, (240, 10, 10, 255))  # 接近红
    assert c.get_pixel(1, 1) == (255, 0, 0, 255)  # 吸附到最近锁定色


def test_palette_lock_snap_color():
    c = make_canvas()
    c.set_palette([(0, 0, 0, 255), (255, 255, 255, 255)])
    assert c.snap_color((200, 200, 200, 255)) == (255, 255, 255, 255)
    assert c.snap_color((10, 10, 10, 255)) == (0, 0, 0, 255)


def test_palette_clear_unlocks():
    c = make_canvas()
    c.set_palette([(255, 0, 0, 255)])
    assert c.palette is not None
    c.clear_palette()
    assert c.palette is None
    c.set_pixel(1, 1, (12, 34, 56))
    assert c.get_pixel(1, 1) == (12, 34, 56, 255)  # 未锁定 -> 原色


def test_palette_lock_flood_fill_snaps():
    c = PixelCanvas(Image.new("RGBA", (4, 4), (10, 10, 10, 255)))
    c.set_palette([(0, 0, 0, 255), (255, 255, 255, 255)])
    c.flood_fill(0, 0, (230, 230, 230, 255))  # 接近白
    assert c.get_pixel(0, 0) == (255, 255, 255, 255)


def test_palette_lock_eraser_transparent():
    """锁定调色板包含透明时，橡皮仍可擦成透明。"""
    c = make_canvas()
    c.set_palette([(255, 0, 0, 255), (0, 0, 0, 0)])
    c.set_pixel(1, 1, (255, 0, 0, 255))
    c.set_pixel(1, 1, (0, 0, 0, 0))  # 透明
    assert c.get_pixel(1, 1) == (0, 0, 0, 0)


def test_replace_color_global():
    """全局换色：把某颜色的全部色块替换为新色（Krita 式）。"""
    base = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    base.putpixel((0, 0), (0, 255, 0, 255))
    base.putpixel((1, 1), (0, 255, 0, 255))
    c = PixelCanvas(base)
    n = c.replace_color((255, 0, 0), (0, 0, 255))
    assert n == 14  # 16 - 2 个非红像素
    assert c.get_pixel(0, 1) == (0, 0, 255, 255)
    assert c.get_pixel(0, 0) == (0, 255, 0, 255)  # 其它色不受影响
    # 无可替换像素 -> 0 且不产生撤销记录
    assert c.replace_color((9, 9, 9), (1, 1, 1)) == 0
    assert c.undo()  # 只有一次有效替换
    assert c.get_pixel(0, 1) == (255, 0, 0, 255)
    assert c.undo() is False


def test_replace_colors_batch_single_snapshot():
    """批量换色（色族替换用）：一次快照一次提交，可整体撤销。"""
    base = Image.new("RGBA", (3, 3), (200, 30, 30, 255))
    base.putpixel((0, 0), (220, 60, 60, 255))
    base.putpixel((0, 1), (10, 10, 10, 255))
    c = PixelCanvas(base)
    n = c.replace_colors({(200, 30, 30, 255): (0, 0, 255, 255), (220, 60, 60, 255): (20, 30, 255, 255)})
    assert n == 8  # 7 深红 + 1 淡红
    assert c.get_pixel(2, 2) == (0, 0, 255, 255)
    assert c.get_pixel(0, 0) == (20, 30, 255, 255)
    assert c.get_pixel(0, 1) == (10, 10, 10, 255)  # 无关色不动
    # 无变化 -> 0 且不产生撤销记录
    assert c.replace_colors({(1, 1, 1, 1): (2, 2, 2, 2)}) == 0
    assert c.undo()  # 只有一次有效替换
    assert c.get_pixel(2, 2) == (200, 30, 30, 255)
    assert c.undo() is False


def test_paste_image_alpha_composite():
    """粘贴（图层合并）：alpha 合成、越界裁剪、全透明无变化不产生撤销。"""
    c = PixelCanvas(Image.new("RGBA", (8, 8), (255, 255, 255, 255)))
    src = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    src.putpixel((0, 0), (255, 0, 0, 255))   # 不透明红
    src.putpixel((1, 0), (0, 255, 0, 128))   # 半透明绿
    n = c.paste_image(src, 3, 2)
    assert n > 0
    assert c.get_pixel(3, 2) == (255, 0, 0, 255)
    px = c.get_pixel(4, 2)  # 半透明绿叠加白底
    assert px[1] > 200 and px[0] < 200
    assert c.get_pixel(3, 1) == (255, 255, 255, 255)  # 区域外不变
    # 越界裁剪：左上角
    c.paste_image(src, -1, -1)
    assert c.get_pixel(0, 0)[0] == 255  # 红
    # 全透明内容 -> 0 且不产生撤销记录
    assert c.paste_image(Image.new("RGBA", (2, 2), (0, 0, 0, 0)), 5, 5) == 0
    assert c.undo()  # 只有两次有效粘贴


def test_brush_size_square_and_line():
    """方形笔刷：单点与连线都按 size 盖章。"""
    c = PixelCanvas(Image.new("RGBA", (10, 10), (255, 255, 255, 255)))
    c.set_pixel(5, 5, (0, 0, 0), size=3)
    assert c.get_pixel(4, 4) == (0, 0, 0, 255)
    assert c.get_pixel(6, 6) == (0, 0, 0, 255)
    assert c.get_pixel(3, 5) == (255, 255, 255, 255)  # 笔刷外
    c2 = PixelCanvas(Image.new("RGBA", (10, 10), (255, 255, 255, 255)))
    c2.draw_line((2, 2), (8, 2), (255, 0, 0), size=3)
    assert c2.get_pixel(5, 1) == (255, 0, 0, 255)  # 线上方带
    assert c2.get_pixel(5, 3) == (255, 0, 0, 255)  # 线下方带
    assert c2.get_pixel(5, 0) == (255, 255, 255, 255)  # 带外


def test_fill_global_mode():
    """填充方式：全局同色替换（替换画布上所有同色像素）。"""
    c = PixelCanvas(Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    c.set_pixel(0, 0, (0, 255, 0, 255))
    c.replace_color((255, 0, 0), (0, 0, 255))
    assert c.get_pixel(3, 3) == (0, 0, 255, 255)
    assert c.get_pixel(0, 0) == (0, 255, 0, 255)  # 其它色不动


def test_fill_rect():
    """矩形区域填充：替换区域内全部格子、反向坐标、越界裁剪、单次撤销。"""
    c = PixelCanvas(Image.new("RGBA", (8, 8), (255, 255, 255, 255)))
    c.fill_rect(1, 1, 4, 3, (255, 0, 0))
    assert c.get_pixel(4, 3) == (255, 0, 0, 255)
    assert c.get_pixel(0, 0) == (255, 255, 255, 255)
    # 反向坐标（起点/终点任意顺序）
    c.fill_rect(5, 5, 2, 2, (0, 0, 255))
    assert c.get_pixel(2, 2) == (0, 0, 255, 255)
    # 越界自动裁剪
    n = c.fill_rect(-5, -5, 100, 100, (0, 0, 0))
    assert n == 64
    assert c.get_pixel(0, 0) == (0, 0, 0, 255)
    # 无变化 -> 0 且不产生撤销记录
    assert c.fill_rect(0, 0, 0, 0, (0, 0, 0)) == 0
    assert c.undo()  # 只有三次有效填充
    assert c.get_pixel(0, 0) == (255, 255, 255, 255)
    assert c.undo() and c.undo()
    assert c.undo() is False


def test_cluster_color_families_groups_similar():
    """色族聚类：相近红色归一族，与淡红/白/绿分开（贪心距离聚类）。"""
    from ui.widgets.pixel_editor import cluster_color_families

    counts = [
        (100, (255, 0, 0, 255)),      # 红（最高频，作代表）
        (80, (250, 10, 5, 255)),      # 接近红
        (60, (255, 180, 180, 255)),   # 淡红
        (50, (255, 255, 255, 255)),   # 白
        (40, (0, 200, 0, 255)),       # 绿
    ]
    families = cluster_color_families(counts)
    reps = [rep for rep, _total, _m in families]
    assert (255, 0, 0, 255) in reps
    # 红与接近红在同一族（代表为最高频成员 255,0,0）
    fam = next(f for f in families if f[0] == (255, 0, 0, 255))
    assert (250, 10, 5, 255) in fam[2]
    assert (255, 180, 180, 255) not in fam[2]  # 淡红另起一族
    assert len(families) == 4  # 红族 / 淡红 / 白 / 绿


def test_color_family_names():
    """色族中文名：白色族 / 红色族 / 淡红色族 / 深蓝色族 / 灰色族 / 黑色族 / 透明族。"""
    from ui.widgets.pixel_editor import color_family_name

    assert color_family_name((255, 255, 255, 255)) == "白色族"
    assert color_family_name((250, 250, 250, 255)) == "白色族"
    assert color_family_name((255, 0, 0, 255)) == "红色族"
    assert color_family_name((255, 180, 180, 255)) == "淡红色族"
    assert color_family_name((0, 0, 180, 255)) == "蓝色族"
    assert color_family_name((0, 0, 80, 255)) == "深蓝色族"
    assert color_family_name((128, 128, 128, 255)) == "灰色族"
    assert color_family_name((10, 10, 10, 255)) == "黑色族"
    assert color_family_name((0, 0, 0, 0)) == "透明族"


def test_family_replace_mapping_preserves_gradient():
    """色族整体换色：保留族内与代表色的相对差（渐变保留），alpha 保持。"""
    from ui.widgets.pixel_editor import family_replace_mapping

    rep = (255, 0, 0, 255)
    members = [rep, (250, 10, 5, 255), (200, 40, 60, 128)]
    mapping = family_replace_mapping(members, rep, (0, 0, 255, 255))
    assert mapping[rep] == (0, 0, 255, 255)          # 代表色 -> 新基准
    assert mapping[(250, 10, 5, 255)] == (0, 10, 255, 255)  # 保持 +delta
    assert mapping[(200, 40, 60, 128)] == (0, 40, 255, 128)  # alpha 保持原值
    # 越界钳制
    mapping2 = family_replace_mapping(
        [(255, 255, 0, 255), (255, 255, 200, 255)], (255, 255, 0, 255), (0, 0, 255, 255)
    )
    assert mapping2[(255, 255, 0, 255)] == (0, 0, 255, 255)
    assert mapping2[(255, 255, 200, 255)] == (0, 0, 255, 255)  # 255+200 钳制回 255
