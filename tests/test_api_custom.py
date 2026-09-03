"""完全自定义 API 配置测试：LLM / 图片模板请求、额外请求头、自定义响应路径。"""
import io

import httpx
import pytest
from PIL import Image

from config.api_config import FIELD_DEFS
from core.api.image_api import ImageAPI
from core.api.llm_api import LLMAPI
from core.api.video_api import VideoAPI

# 1x1 红色 PNG
_PNG = io.BytesIO()
Image.new("RGB", (1, 1), (255, 0, 0)).save(_PNG, format="PNG")
_PNG_BYTES = _PNG.getvalue()
_PNG_B64 = __import__("base64").b64encode(_PNG_BYTES).decode()


def _capture(callback):
    """构造 MockTransport，返回 (client 参数 dict, 捕获列表)。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode("utf-8", "ignore")
        captured["headers"] = dict(request.headers)
        return callback(request)

    return captured, httpx.MockTransport(handler)


def test_llm_custom_template_text_path_and_headers():
    captured, transport = _capture(
        lambda r: httpx.Response(200, json={"data": {"answer": "你好，世界"}})
    )
    api = LLMAPI(
        {
            "base_url": "https://api.custom.example",
            "api_key": "secret",
            "model": "my-model",
            "params": {
                "custom_request": True,
                "payload_template": '{"model": "$model", "input": "$prompt", "system": "$system", "max_output_tokens": $max_tokens}',
                "text_path": "data.answer",
                "extra_headers": '{"X-Trace-Id": "abc123"}',
            },
        },
        transport=transport,
    )
    result = api.call(prompt="hello", max_tokens=32)
    assert result.ok and result.data == "你好，世界"
    # URL = base + 默认端点
    assert captured["url"] == "https://api.custom.example/chat/completions"
    body = captured["body"]
    assert '"input":"hello"' in body
    assert '"model":"my-model"' in body
    assert '"max_output_tokens":32' in body
    # 额外请求头 + 默认 Bearer
    assert captured["headers"].get("x-trace-id") == "abc123"
    assert captured["headers"].get("authorization") == "Bearer secret"


def test_llm_custom_bad_template_reports_error():
    captured, transport = _capture(lambda r: httpx.Response(200, json={}))
    api = LLMAPI(
        {
            "base_url": "https://x",
            "api_key": "k",
            "model": "m",
            "params": {"custom_request": True, "payload_template": "{not json"},
        },
        transport=transport,
    )
    result = api.call(prompt="p")
    assert not result.ok and "模板不是合法 JSON" in result.message


def test_llm_custom_text_path_list_segments():
    captured, transport = _capture(
        lambda r: httpx.Response(200, json={"output": [{"content": "a"}, "b"]})
    )
    api = LLMAPI(
        {
            "base_url": "https://x",
            "api_key": "k",
            "model": "m",
            "params": {"custom_request": True, "text_path": "output"},
        },
        transport=transport,
    )
    result = api.call(prompt="p")
    assert result.ok and result.data == "a\nb"


def test_image_custom_template_b64_path():
    captured, transport = _capture(
        lambda r: httpx.Response(
            200, json={"output": {"images": [{"b64_json": _PNG_B64}]}}
        )
    )
    api = ImageAPI(
        {
            "base_url": "https://img.custom.example/v1",
            "api_key": "k",
            "model": "m",
            "params": {
                "custom_request": True,
                "payload_template": '{"prompt": "$prompt", "size": "$size", "num_images": $n}',
                "images_path": "output.images",
                "endpoint": "/custom/images",
            },
        },
        transport=transport,
    )
    result = api.call(prompt="cat", size="512x512")
    assert result.ok and len(result.data["images"]) == 1
    assert captured["url"].endswith("/custom/images")
    assert '"prompt":"cat"' in captured["body"]
    assert '"num_images":1' in captured["body"]


def test_image_custom_url_strings_path():
    captured, transport = _capture(
        lambda r: httpx.Response(200, json={"result": ["https://cdn.x/1.png"]})
    )
    api = ImageAPI(
        {
            "base_url": "https://img.x",
            "api_key": "k",
            "model": "m",
            "params": {
                "custom_request": True,
                "payload_template": '{"prompt": "$prompt"}',
                "images_path": "result",
            },
        },
        transport=transport,
    )
    result = api.call(prompt="p")
    assert result.ok and result.data["urls"] == ["https://cdn.x/1.png"]


def test_image_custom_falls_back_default_paths():
    """custom 模板但未配 images_path：回退默认兼容解析（data 数组）。"""
    captured, transport = _capture(lambda r: httpx.Response(200, json={"data": [{"url": "https://x/1.png"}]}))
    api = ImageAPI(
        {
            "base_url": "https://img.x",
            "api_key": "k",
            "model": "m",
            "params": {"custom_request": True, "payload_template": '{"prompt": "$prompt"}'},
        },
        transport=transport,
    )
    result = api.call(prompt="p")
    assert result.ok and result.data["urls"] == ["https://x/1.png"]


def test_video_field_defs_include_custom_provider_and_textarea():
    """视频 provider 含「完全自定义」选项；模板/额外字段用多行 textarea。"""
    kinds = {f["key"] for f in FIELD_DEFS["video"]}
    assert "provider" in kinds and "payload_template" in kinds
    provider_def = next(f for f in FIELD_DEFS["video"] if f["key"] == "provider")
    assert "custom" in [o[0] for o in provider_def["options"]]
    pt = next(f for f in FIELD_DEFS["video"] if f["key"] == "payload_template")
    assert pt["type"] == "textarea"
    assert "extra_headers" in kinds
    # llm/image 新增自定义字段齐全
    for kind in ("llm", "image"):
        keys = {f["key"] for f in FIELD_DEFS[kind]}
        assert {"custom_request", "payload_template", "extra_headers", "request_method"} <= keys
    llm_keys = {f["key"] for f in FIELD_DEFS["llm"]}
    img_keys = {f["key"] for f in FIELD_DEFS["image"]}
    assert "text_path" in llm_keys and "images_path" in img_keys


def test_video_custom_provider_client_accepted():
    """provider=custom 可构造 VideoAPI（提交 URL 可由 submit_url 覆盖）。"""
    api = VideoAPI(
        {
            "base_url": "https://v.x",
            "api_key": "k",
            "model": "m",
            "params": {
                "provider": "custom",
                "submit_url": "{base}/my/submit",
                "poll_url": "{base}/my/poll/{id}",
                "payload_template": '{"model": "$model", "prompt": "$prompt"}',
                "job_id_path": "task_id",
                "status_path": "state",
                "status_success": "done",
                "result_video_url_path": "video.url",
            },
        }
    )
    assert api._submit_url() == "https://v.x/my/submit"
    assert api._poll_url("123") == "https://v.x/my/poll/123"
