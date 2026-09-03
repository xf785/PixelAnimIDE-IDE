"""后台任务线程：Solo 工作流线程 + 通用函数线程。"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QThread, Signal

from config.api_config import APIConfigManager
from core.api.factory import create_api_client
from core.workflow import (
    SoloParams,
    SoloResult,
    SoloWorkflow,
    SpriteParams,
    SpriteResult,
    SpriteWorkflow,
    WorkflowCancelled,
    WorkflowError,
)
from ui.i18n import tr

SPRITE_API_KINDS = ("llm", "image")


def create_api_clients(api_manager: APIConfigManager, kinds) -> Tuple[Dict, List]:
    """按类型创建 API 客户端（mock 或真实）。

    返回 (clients, opened)：clients 为 kind -> 客户端；opened 为需关闭的列表。
    """
    clients: Dict = {}
    opened: List = []
    for kind in kinds:
        cfg = api_manager.get_default(kind)
        if cfg is None:
            raise WorkflowError(tr("未配置{0} API，请在「设置」中配置或开启模拟 API").format(kind))
        client = create_api_client(kind, cfg)
        opened.append(client)
        clients[kind] = client
    return clients, opened


class FunctionWorker(QThread):
    """在线程中执行任意函数并返回结果（用于连接测试等）。"""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:  # noqa: D102
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class SoloWorker(QThread):
    """Solo 全自动流程后台线程。

    从 APIConfigManager 取每类 API 的默认配置创建客户端（mock 或真实）。
    """

    progress = Signal(int, int, str, float, str)   # step, total, name, pct, message
    log = Signal(str, str)                          # level, message
    prompts_generated = Signal(object)              # dict：图片/动画/负面提示词
    first_frame_ready = Signal(str)                 # 首帧图片路径
    succeeded = Signal(object)                      # SoloResult
    failed = Signal(str)

    def __init__(self, api_manager: APIConfigManager, params: SoloParams, parent=None):
        super().__init__(parent)
        self._api_manager = api_manager
        self._params = params
        self._cancel = threading.Event()
        self._clients = []

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D102
        try:
            clients = {}
            for kind in ("llm", "image", "video"):
                cfg = self._api_manager.get_default(kind)
                if cfg is None:
                    raise WorkflowError(tr("未配置{0} API，请在「设置」中配置或开启模拟 API").format(kind))
                client = create_api_client(kind, cfg)
                self._clients.append(client)
                clients[kind] = client

            workflow = SoloWorkflow(
                llm_api=clients["llm"],
                image_api=clients["image"],
                video_api=clients["video"],
                params=self._params,
                progress=self._on_progress,
                log=self._on_log,
                cancel=self._cancel,
                on_prompts=self._on_prompts,
                on_first_frame=self._on_first_frame,
            )
            result = workflow.run()
            self.succeeded.emit(result)
        except WorkflowCancelled:
            self.failed.emit(tr("任务已取消"))
        except WorkflowError as exc:
            step = tr("（步骤：{0}）").format(exc.step) if exc.step else ""
            self.failed.emit(f"{exc.message}{step}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(tr("未知错误: {0}").format(exc))
        finally:
            for c in self._clients:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass

    def _on_progress(self, step: int, total: int, name: str, pct: float, message: str) -> None:
        self.progress.emit(step, total, name, pct, message)

    def _on_log(self, level: str, message: str) -> None:
        self.log.emit(level, message)

    def _on_prompts(self, prompts: dict) -> None:
        self.prompts_generated.emit(dict(prompts))

    def _on_first_frame(self, path) -> None:
        self.first_frame_ready.emit(str(path))


class IdeStepWorker(QThread):
    """IDE 单个步骤后台线程。

    fn 签名为 fn(log_cb) -> result；log_cb(level, message) 会转发到 log 信号。
    步骤内部抛出的 WorkflowError / 普通异常统一转成 failed 信号。
    """

    log = Signal(str, str)     # level, message
    succeeded = Signal(object)  # 步骤返回值
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # noqa: D102
        try:
            result = self._fn(self._on_log)
            self.succeeded.emit(result)
        except WorkflowError as exc:
            step = tr("（步骤：{0}）").format(exc.step) if exc.step else ""
            self.failed.emit(f"{exc.message}{step}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _on_log(self, level: str, message: str) -> None:
        self.log.emit(level, message)


class SpriteWorker(QThread):
    """精灵图工作流后台线程（文生底图 -> 精灵图 -> 裁切 -> 抠图 -> 导出）。"""

    log = Signal(str, str)
    base_ready = Signal(str)          # 文生对象底图路径
    sheet_ready = Signal(str)         # 精灵图路径
    succeeded = Signal(object)        # SpriteResult
    failed = Signal(str)

    def __init__(self, api_manager: APIConfigManager, params: SpriteParams, parent=None):
        super().__init__(parent)
        self._api_manager = api_manager
        self._params = params
        self._cancel = threading.Event()
        self._clients = []

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D102
        try:
            clients, self._clients = create_api_clients(self._api_manager, SPRITE_API_KINDS)
            workflow = SpriteWorkflow(
                clients["llm"],
                clients["image"],
                log=self._on_log,
                cancel=self._cancel,
                on_base=lambda p: self.base_ready.emit(str(p)),
                on_sheet=lambda p: self.sheet_ready.emit(str(p)),
            )
            result = workflow.run(self._params)
            self.succeeded.emit(result)
        except WorkflowCancelled:
            self.failed.emit(tr("任务已取消"))
        except WorkflowError as exc:
            step = tr("（步骤：{0}）").format(exc.step) if exc.step else ""
            self.failed.emit(f"{exc.message}{step}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(tr("未知错误: {0}").format(exc))
        finally:
            for c in self._clients:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass

    def _on_log(self, level: str, message: str) -> None:
        self.log.emit(level, message)


TILEMAP_API_KINDS = ("image",)


class TilemapWorker(QThread):
    """瓦片地图工作流后台线程（文生瓦片集 → 裁切 → 无缝 → 47/双网格 → 导出）。"""

    log = Signal(str, str)
    sheet_ready = Signal(str)          # 瓦片底图路径
    atlas_ready = Signal(str)          # 瓦片集图路径
    succeeded = Signal(object)         # TilemapResult
    failed = Signal(str)

    def __init__(self, api_manager: APIConfigManager, params, parent=None):
        super().__init__(parent)
        self._api_manager = api_manager
        self._params = params
        self._cancel = threading.Event()
        self._clients: List = []

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D102
        try:
            clients, self._clients = create_api_clients(self._api_manager, TILEMAP_API_KINDS)
            from core.workflow.tilemap_workflow import TilemapWorkflow

            workflow = TilemapWorkflow(
                clients["image"],
                log=self._on_log,
                cancel=self._cancel,
            )
            result = workflow.run(self._params)
            if result.sheet_path:
                self.sheet_ready.emit(str(result.sheet_path))
            if result.atlas_path:
                self.atlas_ready.emit(str(result.atlas_path))
            self.succeeded.emit(result)
        except WorkflowCancelled:
            self.failed.emit(tr("任务已取消"))
        except WorkflowError as exc:
            step = tr("（步骤：{0}）").format(exc.step) if exc.step else ""
            self.failed.emit(f"{exc.message}{step}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(tr("未知错误: {0}").format(exc))
        finally:
            for c in self._clients:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass

    def _on_log(self, level: str, message: str) -> None:
        self.log.emit(level, message)
