"""--demo 演示入口与 CLI 测试。"""
from core.processing import frame_utils as fu


def test_run_demo_end_to_end(tmp_path, capsys):
    from main import run_demo

    out = tmp_path / "demo"
    result = run_demo(
        description="一只橙色的小猫",
        action="步行",
        output_dir=out,
        frame_count=6,
        fps=8,
        max_colors=8,
        remove_bg=True,
    )
    capsys.readouterr()  # 吞掉进度打印

    assert result.frame_count == 6
    assert result.first_frame.exists()
    assert result.frames_dir.exists()
    assert len(list(result.frames_dir.glob("*.png"))) == 6
    assert result.gif_path.exists()
    assert fu.gif_frame_count(result.gif_path) == 6
    assert result.png_dir and len(list(result.png_dir.glob("*.png"))) == 6
    assert result.project_file.exists()
    # 背景去除后首帧角落透明
    from PIL import Image

    frame = Image.open(sorted(result.frames_dir.glob("*.png"))[0])
    assert frame.mode == "RGBA"


def test_main_cli_demo_flag(tmp_path, capsys):
    from main import main

    out = tmp_path / "cli_demo"
    code = main(["--demo", "--desc", "测试", "--output", str(out), "--frames", "4", "--fps", "8"])
    capsys.readouterr()
    assert code == 0
    assert (out / "export" / "pixel_anim.gif").exists()
