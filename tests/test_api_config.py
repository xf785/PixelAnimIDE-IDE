"""API 配置管理测试：CRUD、默认切换、加密落盘、导入导出、服务商预设。"""
import json

import pytest

from config.api_config import APIConfig, APIConfigManager, FIELD_DEFS, PROVIDER_PRESETS
from core.storage.keyring import Keyring


@pytest.fixture()
def manager(tmp_path):
    m = APIConfigManager(
        config_file=tmp_path / "api_config.json",
        keyring=Keyring(tmp_path / ".keyring"),
    )
    return m


def test_add_and_default_auto(manager):
    cfg = APIConfig(kind="llm", name="A", base_url="http://a/v1", api_key="k1", model="m1")
    manager.add(cfg)
    assert manager.get_default("llm").id == cfg.id  # 首套自动默认
    cfg2 = APIConfig(kind="llm", name="B", base_url="http://b/v1", api_key="k2", model="m2")
    manager.add(cfg2)
    assert manager.get_default("llm").id == cfg.id  # 默认不变


def test_set_default(manager):
    a = APIConfig(kind="image", name="A", base_url="http://a", model="m")
    b = APIConfig(kind="image", name="B", base_url="http://b", model="m")
    manager.add(a)
    manager.add(b)
    manager.set_default(b.id)
    assert manager.get_default("image").id == b.id


def test_encrypted_at_rest(manager, tmp_path):
    cfg = APIConfig(kind="llm", name="A", base_url="http://a/v1", api_key="super-secret-key", model="m")
    manager.add(cfg)
    raw = json.loads((tmp_path / "api_config.json").read_text(encoding="utf-8"))
    stored = raw["configs"][0]
    assert stored["api_key_enc"] != "super-secret-key"
    assert "super-secret-key" not in (tmp_path / "api_config.json").read_text(encoding="utf-8")
    # 重新加载后能解密
    manager2 = APIConfigManager(
        config_file=tmp_path / "api_config.json",
        keyring=Keyring(tmp_path / ".keyring"),
    )
    assert manager2.list()[0].api_key == "super-secret-key"


def test_update_and_delete(manager):
    cfg = APIConfig(kind="llm", name="A", base_url="http://a", model="m")
    manager.add(cfg)
    cfg.base_url = "http://new"
    manager.update(cfg)
    assert manager.get(cfg.id).base_url == "http://new"
    assert manager.delete(cfg.id) is True
    assert manager.list("llm") == []
    assert manager.delete(cfg.id) is False


def test_delete_default_promotes_next(manager):
    a = APIConfig(kind="llm", name="A", base_url="http://a", model="m")
    b = APIConfig(kind="llm", name="B", base_url="http://b", model="m")
    manager.add(a)
    manager.add(b)
    manager.delete(a.id)
    assert manager.get_default("llm").id == b.id


def test_export_import_roundtrip(manager, tmp_path):
    manager.add(APIConfig(kind="llm", name="A", base_url="http://a", api_key="k1", model="m"))
    manager.add(APIConfig(kind="image", name="B", base_url="http://b", api_key="k2", model="m"))
    export_path = tmp_path / "export.json"
    manager.export_config(export_path)

    manager2 = APIConfigManager(
        config_file=tmp_path / "api2.json",
        keyring=Keyring(tmp_path / ".keyring2"),
    )
    count = manager2.import_config(export_path)
    assert count == 2
    keys = {c.name: c.api_key for c in manager2.list()}
    assert keys == {"A": "k1", "B": "k2"}


def test_field_defs_complete():
    # 每种 API 都定义 base_url/api_key/model/mock
    for kind, fields in FIELD_DEFS.items():
        keys = [f["key"] for f in fields]
        for required in ("base_url", "api_key", "model", "mock"):
            assert required in keys, f"{kind} 缺少字段 {required}"
    # 视频支持服务商适配、图片/文本支持端点配置
    assert "provider" in [f["key"] for f in FIELD_DEFS["video"]]
    assert "endpoint" in [f["key"] for f in FIELD_DEFS["image"]]
    assert "endpoint" in [f["key"] for f in FIELD_DEFS["llm"]]
    # 三类都支持代理与 SSL 校验开关（网络被拦截时的排查手段）
    for kind, fields in FIELD_DEFS.items():
        keys = [f["key"] for f in fields]
        assert "proxy" in keys and "verify_ssl" in keys, f"{kind} 缺少 proxy/verify_ssl 字段"


def test_invalid_kind_rejected(manager):
    with pytest.raises(ValueError):
        manager.add(APIConfig(kind="unknown", name="x", base_url="http://x", model="m"))


def test_provider_presets_complete():
    """三类 API 都有服务商预设，且字段完整。"""
    for kind in ("llm", "image", "video"):
        presets = PROVIDER_PRESETS[kind]
        assert len(presets) >= 3, f"{kind} 预设太少"
        for preset in presets:
            assert preset["key"] and preset["label"]
            assert "base_url" in preset and "model" in preset
    # 视频预设覆盖三种适配路径
    video_keys = [p["key"] for p in PROVIDER_PRESETS["video"]]
    assert "doubao_ark" in video_keys and "gptge_doubao" in video_keys and "kling" in video_keys
    # 可灵预设包含请求体模板与完整映射
    kling = next(p for p in PROVIDER_PRESETS["video"] if p["key"] == "kling")
    assert "$model" in kling["params"]["payload_template"]
    assert kling["params"]["submit_url"].startswith("{base}/")
    assert kling["params"]["status_success"] == "succeed,success"
    assert kling["params"]["result_video_url_path"].startswith("data.")


def test_preset_params_are_known_fields():
    """预设写入的 params 键都必须在 FIELD_DEFS 中存在对应字段。"""
    for kind, presets in PROVIDER_PRESETS.items():
        field_keys = {f["key"] for f in FIELD_DEFS[kind]}
        for preset in presets:
            for pkey in (preset.get("params") or {}).keys():
                assert pkey in field_keys, f"{kind} 预设 {preset['key']} 的字段 {pkey} 不存在"
            if "endpoint" in preset:
                assert "endpoint" in field_keys
