"""提示词工具测试。"""
from core.processing import prompt_utils as pu


def test_parse_json_plain():
    text = '{"image_prompt": "a", "animation_prompt": "b", "negative_prompt": "c"}'
    assert pu.parse_json_response(text) == {"image_prompt": "a", "animation_prompt": "b", "negative_prompt": "c"}


def test_parse_json_with_fences():
    text = '```json\n{"image_prompt": "x"}\n```'
    assert pu.parse_json_response(text) == {"image_prompt": "x"}


def test_parse_json_with_noise():
    text = 'Sure! Here you go:\n{"image_prompt": "y", "animation_prompt": "z"}\nHope it helps!'
    data = pu.parse_json_response(text)
    assert data["image_prompt"] == "y"


def test_parse_json_invalid():
    assert pu.parse_json_response("not json at all") == {}
    assert pu.parse_json_response("") == {}


def test_fallback_prompts():
    prompts = pu.build_fallback_prompts("一只龙", "跳跃")
    assert prompts["image_prompt"]
    assert "jump" in prompts["animation_prompt"].lower()
    assert prompts["negative_prompt"]
    # 内置强制项：纯白背景
    assert "white background" in prompts["image_prompt"].lower()
    assert "no gradients" in prompts["image_prompt"].lower()


def test_system_prompt_requires_white_background():
    assert "WHITE background" in pu.SYSTEM_PROMPT


def test_system_prompt_requires_accurate_concise_animation():
    """动画提示词必须忠实于用户动作、禁止虚构、且简洁。"""
    assert "never invent" in pu.SYSTEM_PROMPT
    assert "30 words" in pu.SYSTEM_PROMPT
    assert "STRICT CORRECTION" in pu.STRICT_ANIMATION_CORRECTION
    assert "mounts" in pu.SYSTEM_PROMPT  # 明确禁止虚构坐骑等上下文


def test_fallback_animation_matches_custom_action():
    """自定义动作：动画提示词必须契合用户填写的动作，而不是通用 idle。"""
    p = pu.build_fallback_prompts("角色", "自定义动作XYZ")
    assert "自定义动作XYZ" in p["animation_prompt"]
    assert "idle" not in p["animation_prompt"].lower()
    # 预设动作：使用预设英文
    p2 = pu.build_fallback_prompts("角色", "步行")
    assert "walk" in p2["animation_prompt"].lower()
    # 只有完全没填动作才用 idle
    p3 = pu.build_fallback_prompts("角色", "")
    assert "idle" in p3["animation_prompt"]


def test_is_pixel_prompt():
    assert pu.is_pixel_prompt("a pixel art character")
    assert pu.is_pixel_prompt("像素风格的小猫")
    assert pu.is_pixel_prompt("game sprite, retro")
    assert not pu.is_pixel_prompt("a realistic portrait")
    assert not pu.is_pixel_prompt("")


def test_normalize_prompts_fills_missing():
    data = {"image_prompt": "  hello  "}
    out = pu.normalize_prompts(data, "描述", "步行")
    assert out["image_prompt"] == "hello"
    assert out["animation_prompt"]
    assert out["negative_prompt"]


def test_presets_loaded():
    names = pu.preset_names()
    assert "步行" in names
    assert "攻击" in names
    assert "奔跑" in names


def test_get_preset_and_build():
    assert pu.get_preset("步行")
    assert "walk" in pu.build_animation_prompt("步行").lower()
    # 未知动作给默认值
    assert pu.build_animation_prompt("不存在动作xyz")


def test_build_user_prompt():
    prompt = pu.build_user_prompt("角色A", "攻击")
    assert "角色A" in prompt
    assert "攻击" in prompt


def test_recommended_frames():
    """按动作类型建议帧数：步行 2s、攻击 1.2s、未知动作不干预。"""
    assert pu.recommended_frames("步行", 8) == 16   # 2.0s * 8fps
    assert pu.recommended_frames("奔跑", 8) == 12   # 1.5s * 8fps
    assert pu.recommended_frames("攻击", 8) == 10   # 1.2s * 8fps
    assert pu.recommended_frames("不存在动作xyz", 8) is None
    assert pu.recommended_frames("", 8) is None
    # 帧率变化跟随
    assert pu.recommended_frames("步行", 12) == 24
