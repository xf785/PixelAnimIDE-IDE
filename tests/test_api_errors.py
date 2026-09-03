"""错误提示友好化测试（404 Invalid URL 场景的针对性建议）。"""
from core.api.base import APIError
from core.api.video_api import VideoAPI


def _video_api(base_url: str) -> VideoAPI:
    api = VideoAPI({
        "base_url": base_url,
        "api_key": "k",
        "model": "m",
        "provider": "generic",
    })
    return api


def test_friendly_error_gptge_video_suggests_provider():
    """gpt.ge 视频 404 Invalid URL：提示改用「gpt.ge 豆包视频」适配，而非误导加 /v1。"""
    api = _video_api("https://api.gpt.ge")
    err = APIError(
        "HTTP 404 (POST https://api.gpt.ge/videos/generations): "
        '{"error":{"message":"Invalid URL (POST /videos/generations)"}}'
    )
    hint = api._friendly_error(err)
    assert "gpt.ge" in hint and "服务商适配" in hint
    assert "/task/volces/seedance" in hint
    assert "缺少 /v1" not in hint  # 不再是误导性通用提示


def test_friendly_error_generic_video_keeps_v1_hint():
    """其它 OpenAI 兼容服务（非 gpt.ge）保持「缺少 /v1」提示。"""
    api = _video_api("https://api.other.example")
    err = APIError('HTTP 404 Invalid URL: {"message":"Invalid URL (POST /videos/generations)"}')
    hint = api._friendly_error(err)
    assert "缺少 /v1" in hint
    assert "服务商适配" not in hint


def test_friendly_error_ssl_suggests_proxy():
    api = _video_api("https://api.gpt.ge")
    hint = api._friendly_error(APIError("SSL: UNEXPECTED_EOF_WHILE_READING"))
    assert "代理" in hint
