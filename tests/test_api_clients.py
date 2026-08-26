"""真实 API 客户端测试（httpx MockTransport 模拟服务端）。"""
import base64
import io

import httpx
import pytest
from PIL import Image

from core.api.base import APIError
from core.api.image_api import ImageAPI
from core.api.llm_api import LLMAPI
from core.api.video_api import VideoAPI


def llm_config(**overrides):
    cfg = {
        "base_url": "http://test.local/v1",
        "api_key": "test-key",
        "model": "gpt-test",
        "params": {"timeout": 30, "max_retries": 0},
    }
    cfg.update(overrides)
    return cfg


def tiny_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
def test_llm_call_payload_and_parse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = request.read().decode()
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello world"}}]})

    transport = httpx.MockTransport(handler)
    api = LLMAPI(llm_config(), transport=transport)
    result = api.call(prompt="你好", system="你是助手", temperature=0.5, max_tokens=100)
    assert result.ok and result.data == "hello world"
    assert captured["url"] == "http://test.local/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    payload = __import__("json").loads(captured["json"])
    assert payload["model"] == "gpt-test"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"] == "你好"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100


def test_llm_parse_alternate_shapes():
    for body in [
        {"choices": [{"text": "alt-text"}]},
        {"output": "output-text"},
        {"response": "response-text"},
    ]:
        api = LLMAPI(llm_config(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
        result = api.call(prompt="p")
        assert result.ok, body


def test_llm_http_error():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    api = LLMAPI(llm_config(), transport=httpx.MockTransport(handler))
    result = api.call(prompt="p")
    assert not result.ok
    assert "401" in result.error


def test_llm_missing_config():
    api = LLMAPI({"base_url": "", "api_key": "", "model": "", "params": {}})
    result = api.call(prompt="p")
    assert not result.ok
    assert "Base URL" in result.error


def test_llm_unparseable_response():
    api = LLMAPI(llm_config(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"foo": 1})))
    result = api.call(prompt="p")
    assert not result.ok


def test_llm_reasoning_content_fallback():
    """DeepSeek 推理模型 content 为空时，回退取 reasoning_content 不报错。"""
    body = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "", "reasoning_content": '{"image_prompt": "a"}'}}
        ]
    }
    api = LLMAPI(llm_config(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    result = api.call(prompt="p")
    assert result.ok
    assert "image_prompt" in result.data


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
def test_image_call_b64():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = request.read().decode()
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    api = ImageAPI(llm_config(base_url="http://img.local/v1"), transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", size="512x512", n=1)
    assert result.ok
    assert len(result.data["images"]) == 1
    img = Image.open(io.BytesIO(result.data["images"][0]))
    assert img.size == (4, 4)
    payload = __import__("json").loads(captured["json"])
    assert payload["size"] == "512x512"
    assert payload["response_format"] == "b64_json"


def test_image_call_url():
    def handler(request):
        return httpx.Response(200, json={"data": [{"url": "http://cdn/x.png"}]})

    api = ImageAPI(llm_config(), transport=httpx.MockTransport(handler))
    result = api.call(prompt="p")
    assert result.ok
    assert result.data["urls"] == ["http://cdn/x.png"]


def test_image_no_images():
    api = ImageAPI(llm_config(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})))
    result = api.call(prompt="p")
    assert not result.ok


def test_image_call_with_reference_image():
    """参考图（图生图）：image 字节以 data URI 写入默认 image 字段。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = __import__("json").loads(request.read().decode())
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    api = ImageAPI(llm_config(base_url="http://img.local/v1"), transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", image=tiny_png_bytes())
    assert result.ok
    assert captured["body"]["image"].startswith("data:image/png;base64,")


def test_image_call_custom_image_field():
    """参考图字段名可配置（image_field）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = __import__("json").loads(request.read().decode())
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    cfg = llm_config(base_url="http://img.local/v1")
    cfg["params"]["image_field"] = "init_image"
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", image=tiny_png_bytes())
    assert result.ok
    assert "init_image" in captured["body"]
    assert "image" not in captured["body"]


def test_image_call_multipart_upload():
    """image_mode=multipart：参考图以 multipart 文件字段上传（gpt.ge 要求）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        captured["url"] = str(request.url)
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    cfg = llm_config(base_url="http://img.local/v1")
    cfg["params"]["image_mode"] = "multipart"
    cfg["params"]["image_field"] = "image"
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", image=tiny_png_bytes())
    assert result.ok
    assert len(result.data["images"]) == 1
    # multipart 请求：boundary 头 + 文件字节原样内嵌
    assert captured["content_type"].startswith("multipart/form-data; boundary=")
    assert tiny_png_bytes() in captured["body"]
    assert b'name="image"' in captured["body"]
    # 文本字段也随 multipart 发送
    assert b'name="prompt"' in captured["body"]
    assert b"a cat" in captured["body"]


def test_image_gptge_auto_multipart():
    """gpt.ge 未显式配置 image_mode 时自动走 multipart（兼容旧保存的配置）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    # 模拟旧版 gpt.ge 预设保存的配置：params 里没有 image_mode
    cfg = llm_config(base_url="https://api.gpt.ge/v1")
    cfg["params"] = {"response_format": "b64_json", "timeout": 30, "max_retries": 0}
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", image=tiny_png_bytes())
    assert result.ok
    assert captured["content_type"].startswith("multipart/form-data; boundary=")


def test_image_gptge_data_uri_explicit_override():
    """gpt.ge 显式配置 image_mode=data_uri 时仍走 JSON data URI。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = __import__("json").loads(request.read().decode())
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    cfg = llm_config(base_url="https://api.gpt.ge/v1")
    cfg["params"]["image_mode"] = "data_uri"
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", image=tiny_png_bytes())
    assert result.ok
    assert captured["content_type"] == "application/json"
    assert captured["body"]["image"].startswith("data:image/png;base64,")


def test_image_size_fallback_retries_larger_size():
    """服务商拒绝小尺寸（unsupported size）时自动换更大档位重试。"""
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.read().decode())
        requested.append(body["size"])
        if body["size"] == "256x256":
            return httpx.Response(
                500,
                json={
                    "error": {
                        "message": "unsupported size: 256x256",
                        "type": "v_api_error",
                        "code": "convert_request_failed",
                    }
                },
            )
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    cfg = llm_config(base_url="http://img.local/v1")
    cfg["params"]["max_retries"] = 0  # 网络层不重试，验证尺寸层回退
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", size="256x256")
    assert result.ok
    assert len(result.data["images"]) == 1
    # 256x256 被拒后依次尝试 512x512（同宽高比放大）
    assert requested[0] == "256x256"
    assert requested[1] == "512x512"


def test_image_size_fallback_multipart_gptge():
    """gpt.ge 完整场景：multipart 上传 + 尺寸被拒 -> 更大尺寸 multipart 重试。"""
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.headers.get("content-type", ""))
        body = request.content
        # 从 multipart 体中解析 size 字段值
        import re

        m = re.search(rb'name="size"\r\n\r\n(\d+x\d+)', body)
        size = m.group(1).decode() if m else None
        if size == "256x256":
            return httpx.Response(
                500,
                json={
                    "error": {
                        "message": "unsupported size: 256x256",
                        "type": "v_api_error",
                        "code": "convert_request_failed",
                    }
                },
            )
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    cfg = llm_config(base_url="https://api.gpt.ge/v1")
    cfg["params"]["max_retries"] = 0
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="a cat", size="256x256", image=tiny_png_bytes())
    assert result.ok
    # 两次都必须是 multipart 上传
    assert all(ct.startswith("multipart/form-data; boundary=") for ct in requested)
    assert len(requested) == 2


def test_image_extract_strips_data_uri_prefix():
    """b64_json 带 data: 前缀时也能正常解码。"""
    from core.api.image_api import ImageAPI

    b64 = "data:image/png;base64," + base64.b64encode(tiny_png_bytes()).decode()
    images, urls = ImageAPI(llm_config())._extract_images({"data": [{"b64_json": b64}]})
    assert len(images) == 1
    img = Image.open(io.BytesIO(images[0]))
    assert img.size == (4, 4)


def test_video_extract_frames_strips_data_uri_prefix():
    """帧序列 b64 带 data: 前缀时也能正常解码。"""
    from core.api.video_api import VideoAPI

    b64 = "data:image/png;base64," + base64.b64encode(tiny_png_bytes()).decode()
    frames = VideoAPI._extract_frames_from({"output": {"frames": [{"b64_json": b64}]}}, "output.frames")
    assert len(frames) == 1


# --------------------------------------------------------------------------- #
# Video（提交 -> 轮询 -> 结果）
# --------------------------------------------------------------------------- #
def video_config(**overrides):
    cfg = llm_config(base_url="http://video.local/v1")
    cfg["params"] = {
        "timeout": 30,
        "max_retries": 0,
        "poll_interval": 0.01,
        "max_polls": 50,
        "status_success": ["succeeded"],
        "status_failure": ["failed"],
    }
    cfg["params"].update(overrides)
    return cfg


def test_video_call_polling_flow():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.method == "POST":
            body = __import__("json").loads(request.read().decode())
            assert body["image"].startswith("data:image/png;base64,")
            assert body["prompt"] == "walk"
            return httpx.Response(200, json={"id": "job-1", "status": "processing"})
        if calls["n"] <= 2:
            return httpx.Response(200, json={"id": "job-1", "status": "processing"})
        return httpx.Response(200, json={"id": "job-1", "status": "succeeded", "output": {"video_url": "http://cdn/out.mp4"}})

    api = VideoAPI(video_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="walk", frames=8, fps=8)
    assert result.ok
    assert result.data["video_url"] == "http://cdn/out.mp4"
    assert calls["n"] >= 3  # 提交 + 至少两次轮询


def test_video_call_failure_status():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job-2", "status": "processing"})
        return httpx.Response(200, json={"id": "job-2", "status": "failed", "error": {"message": "out of quota"}})

    api = VideoAPI(video_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert not result.ok
    assert "out of quota" in result.error


def test_video_call_sync_result():
    def handler(request):
        return httpx.Response(200, json={"id": "job-3", "output": {"video_url": "http://cdn/sync.mp4"}})

    api = VideoAPI(video_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert result.ok
    assert result.data["video_url"] == "http://cdn/sync.mp4"


def test_video_call_frames_directly():
    b64 = base64.b64encode(tiny_png_bytes()).decode()

    def handler(request):
        return httpx.Response(200, json={"id": "job-4", "output": {"frames": [{"b64_json": b64}, {"b64_json": b64}]}})

    api = VideoAPI(video_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert result.ok
    assert len(result.data["frames"]) == 2


def test_video_poll_timeout():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job-5", "status": "processing"})
        return httpx.Response(200, json={"id": "job-5", "status": "processing"})

    cfg = video_config(max_polls=3, poll_interval=0.01)
    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert not result.ok
    assert "超时" in result.error


# --------------------------------------------------------------------------- #
# 端点可配置（解决 404：不同服务商端点路径不同）
# --------------------------------------------------------------------------- #
def test_image_custom_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        b64 = base64.b64encode(tiny_png_bytes()).decode()
        return httpx.Response(200, json={"data": [{"b64_json": b64}]})

    cfg = llm_config(base_url="https://ark.cn-beijing.volces.com/api/v3")
    cfg["params"]["endpoint"] = "/api/v3/images/generations"
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="cat")
    assert result.ok
    assert captured["url"] == "https://ark.cn-beijing.volces.com/api/v3/api/v3/images/generations"


def test_image_full_url_override():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"data": [{"url": "http://cdn/x.png"}]})

    cfg = llm_config(base_url="http://unused")
    cfg["params"]["url"] = "https://gateway.example.com/v1/custom/images"
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    assert api.call(prompt="p").ok
    assert captured["url"] == "https://gateway.example.com/v1/custom/images"


def test_llm_custom_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    cfg = llm_config(base_url="http://x")
    cfg["params"]["endpoint"] = "/v1/chat"
    api = LLMAPI(cfg, transport=httpx.MockTransport(handler))
    assert api.call(prompt="p").ok
    assert captured["url"] == "http://x/v1/chat"


def test_image_response_format_from_params():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"data": [{"url": "http://cdn/x.png"}]})

    cfg = llm_config()
    cfg["params"]["response_format"] = "url"
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    assert api.call(prompt="p").ok
    # httpx json= 序列化为紧凑格式（无空格）
    assert '"response_format":"url"' in captured["body"]


# --------------------------------------------------------------------------- #
# Doubao Seedance（火山方舟）适配
# --------------------------------------------------------------------------- #
def doubao_config(**overrides):
    cfg = llm_config(base_url="https://ark.cn-beijing.volces.com/api/v3")
    cfg["params"] = {
        "timeout": 30,
        "max_retries": 0,
        "provider": "doubao",
        "poll_interval": 0.01,
        "max_polls": 50,
    }
    cfg["params"].update(overrides)
    return cfg


def test_image_404_invalid_url_hint():
    """Base URL 缺少 /v1 时（如 api.gpt.ge），报错应给出排查提示。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": {"message": "Invalid URL (POST /images/generations)"}}
        )

    cfg = llm_config(base_url="https://api.gpt.ge")  # 缺少 /v1
    api = ImageAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="cat")
    assert not result.ok
    assert "https://api.gpt.ge/images/generations" in result.error
    assert "提示" in result.error


def test_doubao_submit_payload_and_endpoints():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["body"] = request.read().decode()
            return httpx.Response(200, json={"id": "cgt-2026-abc", "status": "queued"})
        # 轮询 GET：直接返回成功
        captured["poll_url"] = str(request.url)
        return httpx.Response(
            200,
            json={"id": "cgt-2026-abc", "status": "succeeded", "content": {"video_url": "http://cdn/done.mp4"}},
        )

    api = VideoAPI(doubao_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="a dog running", frames=8, fps=8)
    assert result.ok
    assert result.data["video_url"] == "http://cdn/done.mp4"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    assert captured["poll_url"] == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-2026-abc"
    body = __import__("json").loads(captured["body"])
    assert body["model"] == "gpt-test"
    assert body["content"][0] == {"type": "text", "text": "a dog running"}
    img = body["content"][1]
    assert img["type"] == "image_url"
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    # 8 帧 / 8 fps = 1s -> 换算到 Seedance 支持的 5s
    assert body["duration"] == 5
    # 不发送通用格式的 frame_num/fps/image 平铺字段
    assert "frame_num" not in body and "fps" not in body and "image" not in body


def test_doubao_poll_and_result():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-2026-abc", "status": "queued"})
        assert request.method == "GET"
        assert str(request.url).endswith("/contents/generations/tasks/cgt-2026-abc")
        if calls["n"] <= 2:
            return httpx.Response(200, json={"id": "cgt-2026-abc", "status": "running"})
        return httpx.Response(
            200,
            json={
                "id": "cgt-2026-abc",
                "status": "succeeded",
                "content": {"video_url": "https://tos-cn-beijing.volces.com/video.mp4"},
            },
        )

    api = VideoAPI(doubao_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="run", frames=8, fps=8)
    assert result.ok
    assert result.data["video_url"] == "https://tos-cn-beijing.volces.com/video.mp4"
    assert calls["n"] >= 3


def test_doubao_failure_message():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})
        return httpx.Response(
            200, json={"id": "cgt-1", "status": "failed", "error_message": "balance insufficient"}
        )

    api = VideoAPI(doubao_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert not result.ok
    assert "balance insufficient" in result.error


def test_doubao_explicit_duration_and_extra_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"id": "cgt-2", "status": "succeeded", "content": {"video_url": "http://x/1.mp4"}})

    cfg = doubao_config(extra_payload='{"resolution": "1080p", "watermark": false}')
    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p", duration=10)
    assert result.ok
    body = __import__("json").loads(captured["body"])
    assert body["duration"] == 10
    assert body["resolution"] == "1080p"
    assert body["watermark"] is False


# --------------------------------------------------------------------------- #
# gpt.ge (V-API) 豆包视频适配
# --------------------------------------------------------------------------- #
def gptge_config(**overrides):
    cfg = llm_config(base_url="https://api.gpt.ge")
    cfg["params"] = {
        "timeout": 30,
        "max_retries": 0,
        "provider": "gptge",
        "poll_interval": 0.01,
        "max_polls": 50,
    }
    cfg["params"].update(overrides)
    return cfg


def test_gptge_doubao_video_flow():
    """gpt.ge 豆包视频：提交 /task/volces/seedance，轮询 /task/{id}，结果 content.video_url。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.method == "POST":
            assert str(request.url) == "https://api.gpt.ge/task/volces/seedance"
            body = __import__("json").loads(request.read().decode())
            assert body["content"][1]["type"] == "image_url"
            return httpx.Response(200, json={"id": "cgt-gptge-1"})
        assert request.method == "GET"
        assert str(request.url) == "https://api.gpt.ge/task/cgt-gptge-1"
        return httpx.Response(
            200,
            json={"id": "cgt-gptge-1", "status": "succeeded", "content": {"video_url": "http://cdn/v.mp4"}},
        )

    api = VideoAPI(gptge_config(), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="run", frames=8, fps=8)
    assert result.ok
    assert result.data["video_url"] == "http://cdn/v.mp4"
    assert calls["n"] >= 2


def test_video_last_frame_doubao_appends_image():
    """last_frame=True：doubao/gptge 的 content 数组追加第二张 image_url（首帧即尾帧）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = __import__("json").loads(request.read().decode())
        return httpx.Response(200, json={"id": "j", "status": "succeeded", "content": {"video_url": "http://c/v.mp4"}})

    api = VideoAPI(gptge_config(last_frame=True), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="run", frames=8, fps=8)
    assert result.ok
    content = captured["body"]["content"]
    assert len(content) == 3
    assert content[1]["type"] == "image_url" and content[2]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == content[2]["image_url"]["url"]


def test_video_last_frame_generic_adds_last_image():
    """last_frame=True：通用服务商请求体追加 last_image（尽力而为）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = __import__("json").loads(request.read().decode())
        return httpx.Response(200, json={"id": "j", "output": {"video_url": "http://c/v.mp4"}})

    api = VideoAPI(video_config(last_frame=True), transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="walk", frames=8, fps=8)
    assert result.ok
    assert captured["body"]["last_image"].startswith("data:image/png;base64,")


def test_payload_template_last_image_placeholder():
    """payload_template 支持 $last_image 占位符（与 $image 相同）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = __import__("json").loads(request.read().decode())
        return httpx.Response(200, json={"id": "j", "output": {"video_url": "http://c/v.mp4"}})

    cfg = video_config(payload_template='{"model_name":"$model","image":"$image","last_image":"$last_image"}')
    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p", frames=8, fps=8)
    assert result.ok
    assert captured["body"]["image"].startswith("data:image/png;base64,")
    assert captured["body"]["last_image"] == captured["body"]["image"]


def test_gptge_base_url_with_v1_tolerated():
    """gpt.ge 的 Base URL 误填 /v1 时，任务端点自动回退到根路径。"""
    calls = {"n": 0}
    poll_url = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.method == "POST":
            assert str(request.url) == "https://api.gpt.ge/task/volces/seedance"
            return httpx.Response(200, json={"id": "x"})
        poll_url["url"] = str(request.url)
        return httpx.Response(200, json={"id": "x", "status": "succeeded", "content": {"video_url": "http://c/v.mp4"}})

    cfg = gptge_config(base_url="https://api.gpt.ge/v1")
    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert result.ok
    assert calls["n"] == 2
    assert poll_url["url"] == "https://api.gpt.ge/task/x"


def test_list_models_gptge_base():
    """Base URL 不带 /v1 时，自动请求 /v1/models（gpt.ge 格式）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "doubao-seedance-1-5-pro-251215", "object": "model"},
                    {"id": "gpt-4o", "object": "model"},
                ],
                "object": "list",
            },
        )

    api = VideoAPI(llm_config(base_url="https://api.gpt.ge"), transport=httpx.MockTransport(handler))
    result = api.list_models()
    assert result.ok
    assert captured["url"] == "https://api.gpt.ge/v1/models"
    assert "doubao-seedance-1-5-pro-251215" in result.data


def test_list_models_v1_base():
    """Base URL 已带 /v1 时，直接请求 {base}/models。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    api = LLMAPI(llm_config(), transport=httpx.MockTransport(handler))
    result = api.list_models()
    assert result.ok
    assert result.data == ["m1"]
    assert captured["url"] == "http://test.local/v1/models"


def test_poll_stops_on_succeeded_with_empty_ui_params():
    """设置页表单会把未填的高级字段存成空字符串：轮询必须回退默认值，
    状态为 succeeded 时立即结束，不能继续空转。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job-ok"})
        return httpx.Response(
            200, json={"id": "job-ok", "status": "succeeded", "output": {"video_url": "http://c/v.mp4"}}
        )

    cfg = video_config()
    # 模拟设置页 UI 保存后的字段（空字符串 = 未配置）
    cfg["params"]["status_success"] = ""
    cfg["params"]["status_failure"] = ""
    cfg["params"]["status_path"] = ""
    cfg["params"]["job_id_path"] = ""
    cfg["params"]["result_frames_path"] = ""
    cfg["params"]["result_video_url_path"] = ""

    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert result.ok
    assert result.data["video_url"] == "http://c/v.mp4"
    assert calls["n"] == 2  # 提交 + 1 次轮询即返回，未继续空转


def test_submit_url_base_placeholder():
    """submit_url / poll_url 支持 {base} 占位符。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "job", "status": "succeeded", "output": {"video_url": "http://c/v.mp4"}})

    cfg = video_config()
    cfg["params"]["submit_url"] = "{base}/custom/video"
    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert result.ok
    assert captured["url"] == "http://video.local/v1/custom/video"


# --------------------------------------------------------------------------- #
# 请求体模板 + 可灵式适配
# --------------------------------------------------------------------------- #
def test_payload_template_kling_style():
    """请求体模板：$ 占位符渲染 + 可灵式提交/轮询/结果映射。"""
    captured = {}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.method == "POST":
            captured["url"] = str(request.url)
            captured["body"] = request.read().decode()
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "kling-1"}})
        assert str(request.url) == "https://api.klingai.com/v1/videos/image2video/kling-1"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "task_status": "succeed",
                    "task_result": {"videos": [{"url": "http://cdn/kling.mp4"}]},
                },
            },
        )

    cfg = llm_config(base_url="https://api.klingai.com")
    cfg["params"] = {
        "timeout": 30,
        "max_retries": 0,
        "poll_interval": 0.01,
        "max_polls": 50,
        "payload_template": '{"model_name": "$model", "prompt": "$prompt", "image": "$image", "mode": "std"}',
        "submit_url": "{base}/v1/videos/image2video",
        "poll_url": "{base}/v1/videos/image2video/{id}",
        "job_id_path": "data.task_id",
        "status_path": "data.task_status",
        "status_success": "succeed,success",
        "result_video_url_path": "data.task_result.videos.0.url",
    }
    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="a cat runs", frames=8, fps=8)
    assert result.ok
    assert result.data["video_url"] == "http://cdn/kling.mp4"
    assert captured["url"] == "https://api.klingai.com/v1/videos/image2video"
    body = __import__("json").loads(captured["body"])
    assert body == {
        "model_name": "gpt-test",
        "prompt": "a cat runs",
        "image": "data:image/png;base64," + base64.b64encode(tiny_png_bytes()).decode(),
        "mode": "std",
    }


def test_status_success_as_comma_string():
    """status_success 支持逗号分隔字符串。"""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "j1", "status": "processing"})
        return httpx.Response(200, json={"id": "j1", "status": "done", "output": {"video_url": "http://c/v.mp4"}})

    cfg = video_config(status_success="succeeded,done,finished")
    api = VideoAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(image_bytes=tiny_png_bytes(), prompt="p")
    assert result.ok
    assert result.data["video_url"] == "http://c/v.mp4"


# --------------------------------------------------------------------------- #
# 代理 / SSL 校验 / 网络错误提示
# --------------------------------------------------------------------------- #
def test_proxy_and_verify_ssl_read_from_params():
    """高级参数 proxy / verify_ssl 被 BaseAPI 读取并用于构建客户端。"""
    cfg = llm_config()
    cfg["params"]["proxy"] = "http://127.0.0.1:7890"
    cfg["params"]["verify_ssl"] = False
    api = LLMAPI(cfg)
    assert api._proxy == "http://127.0.0.1:7890"
    assert api._verify_ssl is False
    client = api._http()
    # verify=False 生效：底层传输不使用默认证书验证
    assert client._transport._pool._ssl_context.check_hostname is False
    # 代理挂载到 all:// 前缀（httpx 单代理写法）
    assert any(getattr(m, "pattern", "") == "all://" for m in client._mounts)
    api.close()


def test_verify_ssl_string_false_treated_as_false():
    """手改 JSON 把布尔存成字符串时也能正确识别关闭。"""
    cfg = llm_config()
    cfg["params"]["verify_ssl"] = "false"
    api = LLMAPI(cfg)
    assert api._verify_ssl is False


def test_proxy_ignored_when_custom_transport():
    """测试注入 MockTransport 时，代理配置不影响请求（仍走 mock）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    cfg = llm_config()
    cfg["params"]["proxy"] = "http://127.0.0.1:7890"
    api = LLMAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="hi")
    assert result.ok and result.data == "ok"


def test_friendly_error_ssl_hint():
    """SSL 握手失败的错误信息附带代理/网络排查提示。"""
    api = LLMAPI(llm_config())
    exc = httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
    msg = api._friendly_error(exc)
    assert "SSL" in msg
    assert "代理" in msg


def test_ssl_error_retries_then_friendly_message():
    """SSL 连接错误触发重试，最终提示包含排查建议。"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)")

    cfg = llm_config()
    cfg["params"]["max_retries"] = 1
    api = LLMAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="hi")
    assert not result.ok
    assert "已重试 1 次" in result.error
    assert "SSL" in result.error
    assert "代理" in result.error
    # 排查提示只出现一次（_request 与 call 两处都会经过 _friendly_error）
    assert result.error.count("（提示") == 1


def test_http_status_exhausted_includes_body():
    """5xx 重试耗尽后，最终信息带上状态码与响应体。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "server boom"}})

    cfg = llm_config()
    cfg["params"]["max_retries"] = 1
    api = LLMAPI(cfg, transport=httpx.MockTransport(handler))
    result = api.call(prompt="hi")
    assert not result.ok
    assert "HTTP 500" in result.error
    assert "server boom" in result.error
