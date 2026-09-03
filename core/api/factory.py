"""API 客户端工厂：根据配置创建对应类型的客户端实例。"""
from __future__ import annotations

import logging

from config.settings import API_KINDS

from .base import BaseAPI
from .image_api import ImageAPI
from .llm_api import LLMAPI
from .mock_clients import MockImageAPI, MockLLMAPI, MockVideoAPI
from .video_api import VideoAPI

logger = logging.getLogger("PixelAnimIDE.api.factory")

_REAL = {"llm": LLMAPI, "image": ImageAPI, "video": VideoAPI}
_MOCK = {"llm": MockLLMAPI, "image": MockImageAPI, "video": MockVideoAPI}


def _config_value(config, key: str, default=None):
    """兼容 APIConfig 对象与 dict 的取值（dict 走键、对象走属性）。"""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _use_mock(config) -> bool:
    """是否应使用确定性模拟客户端（params.mock 或 base_url 为 'mock'）。"""
    params = _config_value(config, "params", None) or {}
    base_url = str(_config_value(config, "base_url", "") or "").strip().lower()
    return bool(params.get("mock")) or base_url in ("mock", "mock://")


def create_api_client(kind: str, config) -> BaseAPI:
    """创建 API 客户端。

    config: APIConfig 或 dict。当 config.params.mock 为 True 或
    base_url 为 'mock' 时返回确定性模拟客户端。
    """
    if kind not in API_KINDS:
        raise ValueError(f"未知 API 类型: {kind}（可选: {API_KINDS}）")

    if _use_mock(config):
        logger.info("使用模拟 %s 客户端", kind)
        return _MOCK[kind](config)
    return _REAL[kind](config)


def is_mock_config(config) -> bool:
    """与 create_api_client 用同一套判定逻辑（对象与 dict 配置行为一致）。"""
    return _use_mock(config)
