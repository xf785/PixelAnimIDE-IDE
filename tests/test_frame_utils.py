"""帧工具测试：PNG/GIF/雪碧图/视频拆帧。"""
from PIL import Image

from core.processing import frame_utils as fu


def make_frames(count=4, size=(16, 16)):
    frames = []
    for i in range(count):
        img = Image.new("RGBA", size, (20 + i * 40, 30, 40, 255))
        frames.append(img)
    return frames


def test_png_roundtrip():
    img = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    data = fu.image_to_bytes(img)
    out = fu.bytes_to_image(data)
    assert out.size == (8, 8)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[:3] == (10, 20, 30)


def test_save_png_sequence(tmp_out):
    frames = make_frames(4)
    paths = fu.save_png_sequence(frames, tmp_out, prefix="f")
    assert len(paths) == 4
    assert all(p.exists() for p in paths)
    assert paths[0].name == "f_0000.png"


def test_frames_to_gif_and_frame_count(tmp_out):
    frames = make_frames(4)
    path = fu.frames_to_gif(frames, tmp_out / "anim.gif", fps=8)
    assert path.exists()
    assert fu.gif_frame_count(path) == 4


def test_frames_to_apng(tmp_out):
    """APNG 导出：帧数正确且保留透明。"""
    frames = make_frames(4)
    frames[0] = Image.new("RGBA", (16, 16), (0, 0, 0, 0))  # 透明首帧
    path = fu.frames_to_apng(frames, tmp_out / "anim.apng", fps=8)
    assert path.exists()
    assert fu.apng_frame_count(path) == 4
    # 首帧保持透明
    with Image.open(path) as img:
        img.seek(0)
        assert img.getpixel((0, 0))[3] == 0


def test_apng_raises_on_empty(tmp_out):
    import pytest

    with pytest.raises(ValueError):
        fu.frames_to_apng([], tmp_out / "e.apng")


def test_gif_with_transparency(tmp_out):
    frames = make_frames(4)
    # 把第一帧弄成全透明，验证透明索引
    frames[0] = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    path = fu.frames_to_gif(frames, tmp_out / "t.gif", fps=8)
    assert path.exists()
    assert fu.gif_frame_count(path) == 4


def test_gif_mixed_transparency(tmp_out):
    """混合透明/不透明帧的 GIF 也保留透明（回归测试）。"""
    frames = make_frames(3)
    frames[0] = Image.new("RGBA", (16, 16), (0, 0, 0, 0))        # 全透明
    frames[1] = Image.new("RGBA", (16, 16), (10, 20, 30, 255))   # 不透明
    frames[2] = Image.new("RGBA", (16, 16), (0, 0, 0, 0))        # 全透明
    path = fu.frames_to_gif(frames, tmp_out / "mixed.gif", fps=8)
    with Image.open(path) as img:
        assert img.info.get("transparency") is not None
        img.seek(0)
        assert img.convert("RGBA").getpixel((0, 0))[3] == 0


def test_crop_sprite_sheet():
    """按 i×j 网格裁切精灵图：尺寸正确、行优先顺序、count 截断。"""
    import pytest

    sheet = Image.new("RGB", (40, 40))
    for r in range(2):
        for c in range(2):
            color = (255, 0, 0) if (r, c) == (0, 0) else ((0, 255, 0) if (r, c) == (0, 1) else ((0, 0, 255) if (r, c) == (1, 0) else (255, 255, 0)))
            for y in range(20):
                for x in range(20):
                    sheet.putpixel((c * 20 + x, r * 20 + y), color)

    frames = fu.crop_sprite_sheet(sheet, 2, 2)
    assert len(frames) == 4
    assert frames[0].size == (20, 20)
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)          # (0,0)
    assert frames[1].getpixel((0, 0)) == (0, 255, 0)          # (0,1)
    assert frames[3].getpixel((0, 0)) == (255, 255, 0)        # (1,1)
    # count 截断
    frames2 = fu.crop_sprite_sheet(sheet, 2, 2, count=3)
    assert len(frames2) == 3
    # 非法参数
    with pytest.raises(ValueError):
        fu.crop_sprite_sheet(sheet, 0, 2)


def test_animation_meta_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        fu.animation_meta([], 8)


def test_sprite_sheet():
    frames = make_frames(4)
    sheet = fu.frames_to_sprite_sheet(frames, columns=2)
    assert sheet.size == (32, 32)


def test_evenly_sample_indices():
    assert fu._evenly_sample_indices(10, 4) == [0, 3, 6, 9]
    assert fu._evenly_sample_indices(4, 10) == list(range(4))
    # 边界：count==1 不应除零（回归测试）
    assert fu._evenly_sample_indices(10, 1) == [0]
    assert fu._evenly_sample_indices(0, 3) == []
    assert fu._evenly_sample_indices(5, 0) == []


def test_sample_frames():
    frames = make_frames(8)
    out = fu.sample_frames(frames, 4)
    assert len(out) == 4
    assert fu.sample_frames(frames, 10) is frames


def test_animation_meta():
    meta = fu.animation_meta(make_frames(4, (16, 16)), fps=8)
    assert meta["frame_count"] == 4
    assert meta["fps"] == 8
    assert meta["width"] == 16
    assert meta["duration_ms_per_frame"] == 125


def make_cyclic_frames(count=12, cycle=4, size=(16, 16)):
    """移动方块动画：位置每 cycle 帧重复一次（可自然循环）。"""
    frames = []
    for i in range(count):
        f = Image.new("RGB", size, (10, 10, 10))
        x = (i * (size[0] - 4) // cycle) % size[0]
        for y in range(6, 10):
            for xx in range(x, min(x + 4, size[0])):
                f.putpixel((xx, y), (255, 255, 255))
        frames.append(f)
    return frames


def test_sample_loop_frames_first_last_consistent():
    """循环抽帧：首尾帧一致，且优先取可自然循环的片段。"""
    frames = make_cyclic_frames(12, cycle=4)
    out = fu.sample_loop_frames(frames, target_count=6, loop=True)
    assert len(out) == 6
    assert out[0].tobytes() == out[-1].tobytes()  # 首尾一致


def test_sample_loop_frames_no_loop():
    """不启用循环闭合时，首尾帧保留原内容。"""
    frames = make_cyclic_frames(12, cycle=4)
    out = fu.sample_loop_frames(frames, target_count=6, loop=False)
    assert len(out) == 6
    # 移动方块：首帧与末帧位置不同
    assert out[0].tobytes() != out[-1].tobytes()


def test_sample_loop_frames_fewer_than_target():
    """帧数不足 target 时保留全部并仍做循环闭合。"""
    frames = make_cyclic_frames(5, cycle=2)
    out = fu.sample_loop_frames(frames, target_count=8, loop=True)
    assert len(out) == 5
    assert out[0].tobytes() == out[-1].tobytes()


def test_sample_frames_preserve_ends():
    """保留首帧与尾帧、中间均匀采样：完整动作都体现。"""
    frames = make_frames(12)  # 12 帧互不相同
    out = fu.sample_frames_preserve_ends(frames, 6)
    assert len(out) == 6
    assert out[0] is frames[0]        # 首帧保留
    assert out[-1] is frames[-1]      # 尾帧保留
    # 中间 4 帧来自中间区域（非首尾）
    assert all(f is not frames[0] and f is not frames[-1] for f in out[1:-1])


def test_sample_frames_preserve_ends_small_counts():
    frames = make_frames(5)
    assert fu.sample_frames_preserve_ends(frames, 1) == [frames[0]]
    assert fu.sample_frames_preserve_ends(frames, 2) == [frames[0], frames[-1]]
    # 不足目标数时全部保留
    assert fu.sample_frames_preserve_ends(frames, 9) == frames
    # 空列表
    assert fu.sample_frames_preserve_ends([], 4) == []


def test_sample_frames_preserve_ends_count_three():
    """count==3 时不再触发除零（回归测试）。"""
    frames = make_frames(5)
    out = fu.sample_frames_preserve_ends(frames, 3)
    assert len(out) == 3
    assert out[0] is frames[0]
    assert out[-1] is frames[-1]


def test_sample_frames_preserve_ends_diverse_picks_distinct():
    """内容感知抽帧：中间有静态重复帧时，跳过重复、选差异最大的姿态。"""
    # 帧 0/11 为端点；帧 1..4 是 4 帧完全相同的静态姿势；帧 5..10 是 6 种不同姿势
    poses = [Image.new("RGB", (16, 16), (c, c, c)) for c in (10, 30, 60, 90, 120, 150)]
    frames = [Image.new("RGB", (16, 16), (200, 0, 0))] + [poses[0]] * 4 + poses[1:] + [
        Image.new("RGB", (16, 16), (0, 200, 0))
    ]

    def uniq(seq):
        return {tuple(f.resize((4, 4)).getdata()) for f in seq}

    even = fu.sample_frames_preserve_ends(frames, 6, diverse=False)
    diverse = fu.sample_frames_preserve_ends(frames, 6, diverse=True)
    assert len(even) == 6 and len(diverse) == 6
    # 均匀采样会重复选中静态姿势；内容感知采样挑出的不同帧更多
    assert len(uniq(diverse)) > len(uniq(even))
    # 首尾帧始终保留
    assert diverse[0] is frames[0] and diverse[-1] is frames[-1]


def test_dedupe_frames():
    """去除完全相同的连续帧（保留每组第一帧）。"""
    a = Image.new("RGB", (8, 8), (255, 0, 0))
    b = Image.new("RGB", (8, 8), (0, 255, 0))
    frames = [a, a, b, b, b, a]
    assert fu.dedupe_frames(frames, threshold=0.01) == [a, b, a]
    assert fu.dedupe_frames([a, a], threshold=0.01) == [a]
    assert fu.dedupe_frames([a], threshold=0.01) == [a]
    # 默认阈值保守：几乎相同但确有差异的帧保留，完全相同的帧去重
    a2 = Image.new("RGB", (8, 8), (0, 255, 0))
    near = Image.new("RGB", (8, 8), (0, 254, 0))  # 绿通道差 1（灰度可分辨）
    assert fu.dedupe_frames([a2, near]) == [a2, near]
    assert fu.dedupe_frames([a2, a2]) == [a2]
    # threshold=0 时只有完全相同的帧被去重
    assert fu.dedupe_frames([a2, near], threshold=0.0) == [a2, near]
    assert fu.dedupe_frames([a2, near], threshold=0.01) == [a2]


def test_crop_sprite_sheet_inset():
    """裁切内缩：去掉格子边缘黑框。"""
    sheet = Image.new("RGB", (40, 40), (255, 0, 0))
    frames = fu.crop_sprite_sheet(sheet, 2, 2, inset=2)
    assert len(frames) == 4
    assert frames[0].size == (16, 16)  # 20 - 2*2
    # inset 过大时被钳制，不产生空帧
    frames2 = fu.crop_sprite_sheet(sheet, 2, 2, inset=100)
    assert frames2[0].size[0] >= 1
    # inset=0 保持原行为
    assert fu.crop_sprite_sheet(sheet, 2, 2, inset=0)[0].size == (20, 20)


def test_compose_sprite_sheet_index():
    """雪碧图合成 + FrameRonin 风格索引 JSON（坐标、尺寸、时间戳）。"""
    frames = make_frames(5)
    sheet, index = fu.compose_sprite_sheet(
        frames, columns=2, spacing=2, timestamps=[0.0, 0.125, 0.25, 0.375, 0.5]
    )
    assert sheet.size == (16 * 2 + 2, 16 * 3 + 2 * 2)  # (34, 52)
    assert index["version"] == "1.0"
    assert index["frame_size"] == {"w": 16, "h": 16}
    assert index["sheet_size"] == {"w": 34, "h": 52}
    assert index["spacing"] == 2
    assert len(index["frames"]) == 5
    f0 = index["frames"][0]
    assert (f0["x"], f0["y"], f0["w"], f0["h"]) == (0, 0, 16, 16)
    assert f0["t"] == 0.0
    assert (index["frames"][2]["x"], index["frames"][2]["y"]) == (0, 18)  # 第二行
    # 单行默认布局
    sheet1, index1 = fu.compose_sprite_sheet(frames)
    assert sheet1.size == (16 * 5, 16)
    assert index1["sheet_size"] == {"w": 80, "h": 16}
    # 自动方形布局（列数 ≈ √N）
    _, indexA = fu.compose_sprite_sheet(frames, auto_square=True)
    assert indexA["sheet_size"]["w"] >= indexA["sheet_size"]["h"]
    # 纵向排列（先填满列）
    _, indexC = fu.compose_sprite_sheet(frames, columns=2, orientation="columns")
    f2 = indexC["frames"][2]
    assert (f2["x"], f2["y"]) == (16, 0)  # 第二列第一行


def test_crop_to_content():
    """内容包围盒裁剪：透明图按 Alpha、不透明图按颜色差异。"""
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    for y in range(5, 10):
        for x in range(3, 8):
            img.putpixel((x, y), (255, 255, 255, 255))
    assert fu.crop_to_content(img).size == (5, 5)
    assert fu.crop_to_content(img, margin=2).size == (9, 9)
    # 不透明图：按与四角颜色的差异
    solid = Image.new("RGB", (10, 10), (255, 255, 255))
    for y in range(2, 8):
        for x in range(4, 6):
            solid.putpixel((x, y), (0, 0, 0))
    assert fu.crop_to_content(solid).size == (2, 6)
    # 整图一色：无内容，原样返回
    flat = Image.new("RGB", (8, 8), (7, 7, 7))
    assert fu.crop_to_content(flat).size == (8, 8)


def test_strip_audio_remux_silent(tmp_out):
    """strip_audio 用 ffmpeg 去除音轨（remux），返回可用视频路径。"""
    import numpy as np

    import imageio.v2 as imageio

    src = tmp_out / "with_audio.mp4"
    writer = imageio.get_writer(str(src), format="ffmpeg", fps=8)
    try:
        for i in range(4):
            writer.append_data(np.full((16, 16, 3), i * 40, dtype=np.uint8))
    finally:
        writer.close()

    dst = fu.strip_audio(src, tmp_out / "silent.mp4")
    assert dst.exists()
    # 静音后的视频仍可拆帧（视频流完好）
    frames = fu.extract_video_frames(dst)
    assert len(frames) >= 1


def test_downscale_bytes():
    """首帧缩放：超大图缩到长边 ≤ max_side；小图原样返回。"""
    big = Image.new("RGB", (2000, 1000), (0, 0, 255))
    big_bytes = fu.image_to_bytes(big)
    small_bytes = fu.downscale_bytes(big_bytes, max_side=512)
    img = fu.bytes_to_image(small_bytes)
    assert max(img.size) <= 512
    assert img.width <= img.height * 2  # 等比

    small = Image.new("RGB", (200, 100), (0, 0, 255))
    small_bytes = fu.image_to_bytes(small)
    assert fu.downscale_bytes(small_bytes, max_side=512) == small_bytes


def test_upscale_to_min_side_bytes_nearest():
    """首帧过小时最近邻放大到最低要求：像素不模糊（硬边方块，无插值中间色）。"""
    tiny = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
    tiny.putpixel((1, 1), (255, 0, 0, 255))  # 单像素红点
    tiny_bytes = fu.image_to_bytes(tiny)
    out = fu.upscale_to_min_side_bytes(tiny_bytes, min_side=16)
    assert len(out) != len(tiny_bytes)  # 确实被放大了
    img = fu.bytes_to_image(out)
    assert max(img.size) >= 16
    assert img.size == (16, 16)
    # NEAREST：红点(1,1) 变成 4x4 硬块（x/y 4..7），没有中间色（无模糊插值）
    block = [img.getpixel((x, y)) for y in range(4, 8) for x in range(4, 8)]
    assert set(block) == {(255, 0, 0, 255)}
    assert img.getpixel((0, 0)) == (255, 255, 255, 255)  # 白色区域保持纯白
    # 已达标原样返回
    ok = Image.new("RGB", (64, 64), (0, 0, 0))
    ok_bytes = fu.image_to_bytes(ok)
    assert fu.upscale_to_min_side_bytes(ok_bytes, min_side=16) == ok_bytes


def test_extract_video_frames_from_mp4(tmp_out):
    """用 imageio-ffmpeg 生成真实 mp4 再拆帧，验证视频路径。"""
    import numpy as np

    import imageio.v2 as imageio

    video_path = tmp_out / "test.mp4"
    writer = imageio.get_writer(str(video_path), format="ffmpeg", fps=8)
    try:
        for i in range(4):
            frame = np.full((24, 32, 3), i * 60, dtype=np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()

    assert video_path.exists() and video_path.stat().st_size > 0
    frames = fu.extract_video_frames(video_path)
    assert len(frames) >= 1
    assert frames[0].mode == "RGBA"


def test_extract_video_frames_sampled(tmp_out):
    import numpy as np

    import imageio.v2 as imageio

    video_path = tmp_out / "test2.mp4"
    writer = imageio.get_writer(str(video_path), format="ffmpeg", fps=8)
    try:
        for i in range(8):
            writer.append_data(np.full((16, 16, 3), 10, dtype=np.uint8))
    finally:
        writer.close()

    frames = fu.extract_video_frames(video_path, max_frames=4)
    assert len(frames) == 4


def _write_mp4(path, frames, fps=8):
    import numpy as np

    import imageio.v2 as imageio

    writer = imageio.get_writer(str(path), format="ffmpeg", fps=fps)
    try:
        for i in range(frames):
            writer.append_data(np.full((16, 16, 3), (i * 16) % 255, dtype=np.uint8))
    finally:
        writer.close()


def test_extract_video_frames_meta_duration(tmp_out):
    """拆帧返回元信息：fps 与时长。"""
    video = tmp_out / "meta.mp4"
    _write_mp4(video, frames=16, fps=8)  # 2s
    frames, meta = fu.extract_video_frames_meta(video)
    assert meta["fps"] == 8
    assert meta["duration"] is not None
    assert abs(meta["duration"] - 2.0) < 0.3
    assert meta["source_frame_count"] == 16
    assert len(frames) == 16


def test_extract_video_frames_max_duration(tmp_out):
    """max_duration 只保留目标时长内的帧。"""
    video = tmp_out / "duration.mp4"
    _write_mp4(video, frames=16, fps=8)  # 2s
    frames, meta = fu.extract_video_frames_meta(video, max_duration=1.0)
    assert 6 <= len(frames) <= 9  # 约 1s * 8fps = 8 帧
    assert meta["duration"] is not None
    assert abs(meta["duration"] - 1.0) < 0.3
