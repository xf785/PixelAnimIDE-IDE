"""core.api —— 第三方 AI API 统一抽象层。

包含：
- base.py         抽象基类 BaseAPI 与统一结果对象 APIResult
- llm_api.py      通用文本 API（OpenAI 兼容 /chat/completions）
- image_api.py    图片生成 API（OpenAI 兼容 /images/generations）
- video_api.py    图转视频 API（通用轮询式任务模型）
- mock_clients.py 确定性模拟客户端（无需密钥即可跑通全流程）
- factory.py      按配置创建客户端实例
"""
from .base import APIResult, BaseAPI
from .factory import create_api_client

__all__ = ["APIResult", "BaseAPI", "create_api_client"]
