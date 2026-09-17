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

# 受限环境（Windows 沙箱 / 严格 ACL）下 os.chmod 可能被系统拒绝，而 pytest 创建
# 临时目录（tmp_path fixture）时会调用 chmod，一旦被拒整套测试都会在 setup 阶段报
# PermissionError。这里仅在 chmod 抛权限错误时静默跳过（不影响任何正常权限场景，
# 应用代码本身也不依赖 chmod）。
_orig_chmod = os.chmod


def _safe_chmod(path, mode, *args, **kwargs):
    try:
        return _orig_chmod(path, mode, *args, **kwargs)
    except PermissionError:
        return None
    except OSError:
        return None


if os.name == "nt":
    os.chmod = _safe_chmod

import itertools
import shutil

import pytest

_TMP_COUNTER = itertools.count()


@pytest.fixture()
def tmp_path(request):
    """受限沙箱兼容版临时目录（替代 pytest 内置实现）。

    两个环境坑（Windows 沙箱 / 严格 ACL）：
    1. pytest 内置实现会反复 chmod / 加锁 / 枚举目录，这些操作会被系统拒绝，
       导致所有使用 tmp_path 的测试在 setup 阶段直接 PermissionError；
    2. `tempfile.mkdtemp` 以 0700 权限建目录，会被沙箱判定为「非白名单目录」，
       之后在其内部 mkdir / 写文件同样被拒。
    因此改为：用普通 mkdir（继承工作区权限）建唯一目录，仅做创建与删除，
    语义与内置 fixture 等价。
    """
    base = ROOT / ".tmp" / "pytest_work"
    base.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.node.name)[:24]
    path = base / f"{safe}_{os.getpid()}_{next(_TMP_COUNTER)}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def tmp_out(tmp_path):
    """独立的临时输出目录。"""
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out
