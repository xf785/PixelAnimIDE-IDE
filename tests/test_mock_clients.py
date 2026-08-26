"""模拟客户端与工厂测试。"""
import io

import pytest
from PIL import Image

from core.api.factory import create_api_client, is_mock_config
from core.api.mock_clients import (
    MockImageAPI,
    MockLLMAPI,
    MockVideoAPI,
    build_mock_prompts,
    parse_size,
)
from config.api_config import APIConfig


def test_mock_llm_prompts():
    api = MockLLMAPI()
    result = api.call(prompt="一只橙色的猫")
    assert result.ok
    prompts = result.data
    assert "image_prompt" in prompts and "animation_prompt" in prompts and "negative_prompt" in prompts
    assert "cat" in prompts["image_prompt"].lower() or "猫" in prompts["image_prompt"] or "orange" in prompts["image_prompt"].lower()


def test_mock_llm_with_action():
    api = MockLLMAPI()
    result = api.call(prompt="角色", action="步行")
    assert "walk" in result.data["animation_prompt"].lower()


def test_mock_image_returns_png():
    api = MockImageAPI()
    result = api.call(prompt="test", size="128x128")
    assert result.ok
    img = Image.open(io.BytesIO(result.data["images"][0]))
    assert img.size == (128, 128)
    assert img.format == "PNG"


def test_mock_video_returns_frames():
    api = MockVideoAPI()
    result = api.call(prompt="walk", frames=6, fps=8)
    assert result.ok
    assert len(result.data["frames"]) == 6
    img = Image.open(io.BytesIO(result.data["frames"][0]))
    assert img.format == "PNG"


def test_mock_deterministic():
    a = MockImageAPI().call(prompt="same", size="64x64").data["images"][0]
    b = MockImageAPI().call(prompt="same", size="64x64").data["images"][0]
    assert a == b


def test_parse_size():
    assert parse_size("512x512") == (512, 512)
    assert parse_size("1024*768") == (1024, 768)
    assert parse_size("800,600") == (800, 600)
    assert parse_size("bad") == (1024, 1024)


def test_factory_returns_mock_for_mock_flag():
    cfg = APIConfig(kind="llm", name="m", base_url="http://x", model="m", params={"mock": True})
    client = create_api_client("llm", cfg)
    assert isinstance(client, MockLLMAPI)
    assert is_mock_config(cfg) is True


def test_factory_returns_mock_for_mock_url():
    cfg = APIConfig(kind="image", name="m", base_url="mock", model="m")
    assert isinstance(create_api_client("image", cfg), MockImageAPI)


def test_factory_returns_real_for_normal_config():
    from core.api.image_api import ImageAPI

    cfg = APIConfig(kind="image", name="m", base_url="http://real/v1", api_key="k", model="m")
    client = create_api_client("image", cfg)
    assert isinstance(client, ImageAPI)


def test_factory_unknown_kind():
    with pytest.raises(ValueError):
        create_api_client("nope", APIConfig(kind="nope", name="x", base_url="http://x", model="m"))


def test_build_mock_prompts_shape():
    prompts = build_mock_prompts("一个骑士", "攻击")
    assert prompts["image_prompt"] and prompts["animation_prompt"] and prompts["negative_prompt"]
    # 附带 LLM 参数建议（frame_count/fps）
    assert isinstance(prompts.get("frame_count"), int) and 4 <= prompts["frame_count"] <= 48
    assert prompts.get("fps") == 8
