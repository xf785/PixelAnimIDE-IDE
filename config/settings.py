"""全局配置：应用路径、常量、默认值。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "PixelAnimIDE"
APP_VERSION = "0.1.0"
APP_DISPLAY_NAME = "PixelAnimIDE"

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


def app_root() -> Path:
    """项目源码根目录（含 main.py、core/、ui/ 等）。"""
    return Path(__file__).resolve().parent.parent


def app_data_dir() -> Path:
    """用户数据目录：存放配置、密钥、日志。可通过环境变量覆盖（便于测试）。"""
    override = os.environ.get("PIXELANIMIDE_DATA_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / APP_NAME


DATA_DIR = app_data_dir()
CONFIG_DIR = app_root() / "config"
ASSETS_DIR = app_root() / "assets"
DEFAULT_OUTPUT_DIR = Path.home() / f"{APP_NAME}_Output"

# 运行期数据文件（位于用户数据目录，避免污染源码树）
API_CONFIG_FILE = DATA_DIR / "api_config.json"
KEYRING_FILE = DATA_DIR / ".keyring"
UI_SETTINGS_FILE = DATA_DIR / "ui_settings.json"

# ---------------------------------------------------------------------------
# 业务常量
# ---------------------------------------------------------------------------

# 三种 API 类型（与 core/api/factory.py 对应）
API_KINDS = ("llm", "image", "video")
API_KIND_LABELS = {"llm": "通用文本 API", "image": "图片生成 API", "video": "图转视频 API"}

# 宽高比 -> (w, h)
ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
}

# 常用像素画布尺寸（严格像素化目标）
PIXEL_SIZES = [32, 48, 64, 96, 128, 160, 192, 256]

DEFAULT_FPS = 8
# 默认 1s（8 帧 @ 8fps）；LLM 会按用户描述/动作自动评估时长（如步行→2s、挥砍→1s）
DEFAULT_FRAME_COUNT = 8
DEFAULT_SPEED = 1.0
DEFAULT_MAX_COLORS = 16
DEFAULT_ASPECT = "1:1"

# 导出命名
EXPORT_PREFIX = "pixel_anim"
FRAME_NAME_PATTERN = "frame_{index:04d}.png"
