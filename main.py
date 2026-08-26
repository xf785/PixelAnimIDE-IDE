"""PixelAnimIDE 程序入口。

用法：
    python main.py                     # 启动 GUI（缺依赖时自动改用 .venv）
    python main.py --demo              # 无 GUI 演示：模拟 API 跑通 Solo 全流程
    python main.py --demo --desc "..." --output demo_output
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("PixelAnimIDE")


def _ensure_venv() -> None:
    """如果当前解释器缺少 PySide6（如 msys2 的 python 未装依赖），
    自动改用源码树 .venv 里的解释器重新执行。

    若已在使用 .venv 解释器仍缺依赖（虚拟环境损坏），给出明确提示而非死循环。
    """
    try:
        import PySide6  # noqa: F401
        return
    except ImportError:
        pass

    root = Path(__file__).resolve().parent
    venv_python = root / (".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python")
    here = os.path.normcase(os.path.abspath(sys.executable))
    target = os.path.normcase(os.path.abspath(str(venv_python)))
    if venv_python.exists() and here != target:
        print(f"当前 Python 缺少依赖，自动改用虚拟环境: {venv_python}")
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
        return  # pragma: no cover - execv 不返回

    print(
        "错误：缺少 PySide6 等依赖。\n"
        "请使用虚拟环境启动：\n"
        "    .\\.venv\\Scripts\\python.exe main.py\n"
        "或先安装依赖：\n"
        "    python -m pip install -r requirements.txt"
    )
    sys.exit(1)


def build_context():
    """构建全局共享上下文（API 配置管理 + UI 设置）。"""
    from config.api_config import APIConfigManager
    from config.settings import API_CONFIG_FILE, KEYRING_FILE, UI_SETTINGS_FILE
    from ui.app_context import AppContext, UISettings

    api = APIConfigManager(API_CONFIG_FILE)
    return AppContext(api=api, ui_settings=UISettings(UI_SETTINGS_FILE))


def run_gui() -> int:
    """启动桌面 GUI。"""
    from PySide6.QtWidgets import QApplication

    from config.settings import APP_NAME
    from ui.i18n import set_language
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    ctx = build_context()
    set_language(ctx.ui_settings.get("language", "zh"))  # 语言设置（常规 -> 语言）
    window = MainWindow(ctx)
    window.show()
    return app.exec()


def run_demo(
    description: str = "一只拿着剑的橙色小猫，Q 版，侧身站立",
    action: str = "步行",
    output_dir: Path | str = "demo_output",
    frame_count: int = 16,
    fps: int = 8,
    speed: float = 1.0,
    max_colors: int = 16,
    remove_bg: bool = True,
) -> "SoloResult":
    """无 GUI 端到端演示：使用确定性模拟 API 跑通 Solo 全流程。"""
    from core.api.mock_clients import MockImageAPI, MockLLMAPI, MockVideoAPI
    from core.workflow import SoloParams, SoloWorkflow

    params = SoloParams(
        description=description,
        action=action,
        frame_count=frame_count,
        fps=fps,
        speed=speed,
        max_colors=max_colors,
        remove_bg=remove_bg,
        output_dir=Path(output_dir),
    )
    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=MockVideoAPI(),
        params=params,
        progress=_demo_progress,
        log=_demo_log,
    )
    return workflow.run()


def _demo_progress(step: int, total: int, name: str, pct: float, message: str) -> None:
    print(f"  [{step + 1}/{total}] {name} {pct * 100:4.0f}%  {message}")


def _demo_log(level: str, message: str) -> None:
    print(f"    {level.upper():5s} {message}")


def main(argv: Optional[list] = None) -> int:
    _ensure_venv()  # 缺 PySide6 时自动改用 .venv 解释器
    parser = argparse.ArgumentParser(
        prog="PixelAnimIDE",
        description="像素动画 IDE —— 从文本描述到像素动画（Solo 一键 + IDE 分步编辑）",
    )
    parser.add_argument("--demo", action="store_true", help="无 GUI 演示：用模拟 API 跑通全流程")
    parser.add_argument("--desc", default="一只拿着剑的橙色小猫，Q 版，侧身站立", help="演示文本描述")
    parser.add_argument("--action", default="步行", help="动作类型（预设：步行/奔跑/跳跃/攻击等）")
    parser.add_argument("--output", default="demo_output", help="演示输出目录")
    parser.add_argument("--frames", type=int, default=16, help="帧数（默认 16，约 2s）")
    parser.add_argument("--fps", type=int, default=8, help="帧率")
    parser.add_argument("--speed", type=float, default=1.0, help="播放倍速（0.5/1/1.5/2）")
    parser.add_argument("--colors", type=int, default=16, help="最大颜色数")
    parser.add_argument("--no-bg", action="store_true", help="不去除背景")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.demo:
        print(f"=== PixelAnimIDE 演示模式（模拟 API）===")
        print(f"描述: {args.desc}")
        print(f"动作: {args.action} | 帧数: {args.frames} | 帧率: {args.fps} | 倍速: {args.speed} | 颜色: {args.colors}")
        try:
            result = run_demo(
                description=args.desc,
                action=args.action,
                output_dir=args.output,
                frame_count=args.frames,
                fps=args.fps,
                speed=args.speed,
                max_colors=args.colors,
                remove_bg=not args.no_bg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"演示失败: {exc}")
            return 1
        print()
        print("=== 完成 ===")
        print(f"  帧数:   {result.frame_count}（{result.width}x{result.height}）")
        print(f"  首帧:   {result.first_frame}")
        if result.video_path:
            print(f"  视频:   {result.video_path}")
        print(f"  帧目录: {result.frames_dir}")
        if result.gif_path:
            print(f"  GIF:    {result.gif_path}")
        if result.png_dir:
            print(f"  PNG:    {result.png_dir}")
        print(f"  项目:   {result.project_file}")
        return 0

    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
