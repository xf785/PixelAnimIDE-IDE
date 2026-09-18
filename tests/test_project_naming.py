"""项目命名与数据目录迁移的回归测试（改名后防回退）。"""
import importlib
import os
import sys
from pathlib import Path

import pytest


def _reload_settings(tmp_appdata: Path):
    os.environ["APPDATA"] = str(tmp_appdata)
    os.environ.pop("PIXELFOUNDRY_DATA_DIR", None)
    os.environ.pop("PIXELANIMIDE_DATA_DIR", None)
    import config.settings as settings

    return importlib.reload(settings)


def test_app_names_are_pixelfoundry():
    from config.settings import (
        APP_DISPLAY_NAME,
        APP_FULL_NAME,
        APP_NAME,
        APP_NAME_ZH,
        APP_VERSION,
    )

    assert APP_NAME == "PixelFoundry"
    assert APP_DISPLAY_NAME == "PixelFoundry IDE"
    assert APP_FULL_NAME == "PixelFoundry — Pixel Game Asset Foundry"
    assert APP_NAME_ZH == "像素铸造 IDE"
    assert APP_VERSION.count(".") == 2


def test_window_title_uses_display_name(qtbot, tmp_path):
    from config.api_config import APIConfigManager
    from config.settings import APP_DISPLAY_NAME, APP_VERSION
    from core.storage.keyring import Keyring
    from ui.app_context import AppContext, UISettings
    from ui.main_window import MainWindow

    api = APIConfigManager(config_file=tmp_path / "api_config.json", keyring=Keyring(tmp_path / ".keyring"))
    window = MainWindow(AppContext(api=api, ui_settings=UISettings(tmp_path / "ui_settings.json")))
    qtbot.addWidget(window)
    title = window.windowTitle()
    assert APP_DISPLAY_NAME in title and APP_VERSION in title, title


def test_data_dir_migrates_from_legacy_name(tmp_path):
    """改名不应该丢用户配置：旧目录会自动迁移到新名下。"""
    appdata = tmp_path / "AppData"
    legacy = appdata / "PixelAnimIDE"
    legacy.mkdir(parents=True)
    (legacy / "api_config.json").write_text('{"configs": []}', encoding="utf-8")
    (legacy / ".keyring").write_text("secret", encoding="utf-8")

    settings = _reload_settings(appdata)
    try:
        assert settings.DATA_DIR == appdata / "PixelFoundry"
        assert (settings.DATA_DIR / "api_config.json").exists()
        assert (settings.DATA_DIR / ".keyring").read_text(encoding="utf-8") == "secret"
        assert not legacy.exists(), "旧目录应已迁移（重命名）而不是复制残留"
    finally:
        os.environ.pop("APPDATA", None)
        importlib.reload(settings)


def test_data_dir_env_override_supports_both_names(tmp_path):
    new_dir = tmp_path / "new"
    os.environ["PIXELFOUNDRY_DATA_DIR"] = str(new_dir)
    settings = importlib.reload(importlib.import_module("config.settings"))
    assert settings.DATA_DIR == new_dir
    os.environ.pop("PIXELFOUNDRY_DATA_DIR")

    old_dir = tmp_path / "old"
    os.environ["PIXELANIMIDE_DATA_DIR"] = str(old_dir)      # 旧环境变量仍然有效
    settings = importlib.reload(importlib.import_module("config.settings"))
    assert settings.DATA_DIR == old_dir
    os.environ.pop("PIXELANIMIDE_DATA_DIR")
    importlib.reload(importlib.import_module("config.settings"))


def test_existing_data_dir_is_not_touched(tmp_path):
    """新目录已存在时不得动它（避免把用户刚存的配置搬走）。"""
    appdata = tmp_path / "AppData"
    (appdata / "PixelFoundry").mkdir(parents=True)
    (appdata / "PixelFoundry" / "ui_settings.json").write_text("{}", encoding="utf-8")
    legacy = appdata / "PixelAnimIDE"
    legacy.mkdir(parents=True)
    (legacy / "api_config.json").write_text("{}", encoding="utf-8")

    settings = _reload_settings(appdata)
    try:
        assert (settings.DATA_DIR / "ui_settings.json").exists()
        assert legacy.exists(), "新目录已存在时不应迁移/删除旧目录"
    finally:
        os.environ.pop("APPDATA", None)
        importlib.reload(settings)


def test_no_legacy_name_in_sources():
    """源码/文档里不应再出现旧项目名（数据迁移兼容处除外）。"""
    root = Path(__file__).resolve().parent.parent
    # 允许出现旧名的地方：迁移兼容代码，以及本测试自身（需要引用旧名做检查）
    # 允许出现旧名：迁移兼容代码、本测试自身、以及**变更记录文档**（历史沿革需要写明原名）
    allow = {"config/settings.py", "tests/test_project_naming.py", "docs/tilemap_mode_design.md"}
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".toml", ".spec", ".qss", ".bat", ".yml"}:
            continue
        if any(part in {".git", ".venv", "dist", "build", ".tmp", "__pycache__", ".pytest_cache", ".pytest_tmp"} for part in path.parts):
            continue
        if path.relative_to(root).as_posix() in allow:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "PixelAnimIDE" in text or "pixelanimide" in text:
            hits.append(path.relative_to(root).as_posix())
    assert not hits, f"仍残留旧项目名: {hits}"
