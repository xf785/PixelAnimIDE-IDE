"""全局配置：应用路径、常量、默认值。"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

APP_NAME = "PixelFoundry"                          # 数据目录 / 输出目录用（无空格）
APP_VERSION = "0.3.0"
APP_DISPLAY_NAME = "PixelFoundry IDE"              # 窗口标题用
APP_FULL_NAME = "PixelFoundry — Pixel Game Asset Foundry"   # 全称（英文）
APP_NAME_ZH = "像素铸造 IDE"                        # 中文名
APP_LEGACY_NAME = "PixelAnimIDE"                   # 旧名（数据目录迁移用）

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


def bundle_root() -> Path:
    """资源根目录：PyInstaller 冻结时为 _MEIPASS（内含 assets/、ui/ 等），否则为源码根。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def app_root() -> Path:
    """项目源码根目录（含 main.py、core/、ui/ 等）。"""
    return bundle_root()


def app_data_dir() -> Path:
    """用户数据目录：存放配置、密钥、日志。可通过环境变量覆盖（便于测试）。"""
    override = os.environ.get("PIXELFOUNDRY_DATA_DIR") or os.environ.get("PIXELANIMIDE_DATA_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    target = base / APP_NAME
    # 旧版本名下的用户数据（配置 / 密钥 / 界面设置）自动迁移，改名不丢配置
    legacy = base / APP_LEGACY_NAME
    if not target.exists() and legacy.exists():
        try:
            legacy.rename(target)
        except OSError:
            import shutil

            shutil.copytree(legacy, target, dirs_exist_ok=True)
        logging.getLogger(APP_NAME).info("用户数据目录已从 %s 迁移到 %s", legacy, target)
    return target


DATA_DIR = app_data_dir()
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
