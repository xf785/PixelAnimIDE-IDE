"""API 客户端工厂：根据配置创建对应类型的客户端实例。"""
from __future__ import annotations

import logging
from typing import Optional

from config.settings import API_KINDS

from .base import BaseAPI
from .image_api import ImageAPI
from .llm_api import LLMAPI
from .mock_clients import MockImageAPI, MockLLMAPI, MockVideoAPI
from .video_api import VideoAPI

logger = logging.getLogger("PixelAnimIDE.api.factory")

_REAL = {"llm": LLMAPI, "image": ImageAPI, "video": VideoAPI}
_MOCK = {"llm": MockLLMAPI, "image": MockImageAPI, "video": MockVideoAPI}


def create_api_client(kind: str, config) -> BaseAPI:
    """创建 API 客户端。

    config: APIConfig 或 dict。当 config.params.mock 为 True 或
    base_url 为 'mock' 时返回确定性模拟客户端。
    """
    if kind not in API_KINDS:
        raise ValueError(f"未知 API 类型: {kind}（可选: {API_KINDS}）")

    params = getattr(config, "params", None) or config.get("params", {}) if isinstance(config, dict) else getattr(config, "params", None)
    base_url = getattr(config, "base_url", None) or (config.get("base_url", "") if isinstance(config, dict) else "")
    use_mock = bool(params and params.get("mock")) or str(base_url).strip().lower() in ("mock", "mock://")

    if use_mock:
        logger.info("使用模拟 %s 客户端", kind)
        return _MOCK[kind](config)
    return _REAL[kind](config)


def is_mock_config(config) -> bool:
    params = getattr(config, "params", None) or {}
    base_url = getattr(config, "base_url", "") or ""
    return bool(params.get("mock")) or str(base_url).strip().lower() in ("mock", "mock://")
