"""pytest 根配置：保证 GUI 测试在无显示环境下运行，并提供公共 fixture。"""
import os

# GUI 测试统一使用离屏渲染，避免依赖真实显示器
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

# 保证项目根目录可导入（pyproject 的 pythonpath 也已配置，双保险）
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture()
def tmp_out(tmp_path):
    """独立的临时输出目录。"""
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out
