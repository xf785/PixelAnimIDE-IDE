"""API 配置管理：多套配置的增删改查、默认配置切换、加密落盘、导入导出。

存储格式（JSON，位于用户数据目录）：
{
  "version": 1,
  "configs": [
    {
      "id": "uuid",
      "kind": "llm|image|video",
      "name": "OpenAI",
      "base_url": "...",
      "api_key_enc": "<Fernet 密文>",   # 磁盘上只有密文
      "model": "...",
      "params": {...},
      "is_default": true
    }
  ]
}
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import API_CONFIG_FILE, API_KINDS, KEYRING_FILE
from core.storage.keyring import Keyring

logger = logging.getLogger("PixelAnimIDE.config.api_config")

CONFIG_VERSION = 1

# 各类型 API 的字段定义（供 UI 表单复用）
# type: text/password/int/float/bool/choice；choice 的 options 为 [(value, label)]
# group: "advanced" 表示归入可折叠的「高级选项」分组，缺省为基础字段
FIELD_DEFS: Dict[str, List[dict]] = {
    "llm": [
        {"key": "base_url", "label": "Base URL", "type": "text", "default": "https://api.openai.com/v1", "required": True},
        {"key": "api_key", "label": "API Key", "type": "password", "default": "", "required": True},
        {"key": "model", "label": "模型名称", "type": "text", "default": "gpt-4o-mini", "required": True},
        {"key": "endpoint", "label": "端点路径(可选)", "type": "text", "default": "", "placeholder": "默认 /chat/completions", "group": "advanced"},
        {"key": "temperature", "label": "temperature", "type": "float", "default": 0.7, "min": 0.0, "max": 2.0, "group": "advanced"},
        {"key": "max_tokens", "label": "max_tokens", "type": "int", "default": 2048, "min": 1, "group": "advanced"},
        {"key": "top_p", "label": "top_p", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "group": "advanced"},
        {"key": "timeout", "label": "超时(秒)", "type": "int", "default": 120, "min": 5, "group": "advanced"},
        {"key": "proxy", "label": "代理(可选)", "type": "text", "default": "",
         "placeholder": "http://127.0.0.1:7890（直连被拦截时填写）", "group": "advanced"},
        {"key": "verify_ssl", "label": "校验 SSL 证书", "type": "bool", "default": True, "group": "advanced"},
        {"key": "mock", "label": "使用模拟 API（无需密钥）", "type": "bool", "default": False},
    ],
    "image": [
        {"key": "base_url", "label": "Base URL", "type": "text", "default": "https://api.openai.com/v1", "required": True},
        {"key": "api_key", "label": "API Key", "type": "password", "default": "", "required": True},
        {"key": "model", "label": "模型名称", "type": "text", "default": "gpt-image-1", "required": True},
        {"key": "endpoint", "label": "端点路径(可选)", "type": "text", "default": "", "placeholder": "默认 /images/generations", "group": "advanced"},
        {"key": "response_format", "label": "返回格式", "type": "choice", "default": "b64_json",
         "options": [("b64_json", "base64 (b64_json)"), ("url", "图片 URL")], "group": "advanced"},
        {"key": "image_field", "label": "参考图字段名(图生图)", "type": "text", "default": "image",
         "placeholder": "默认 image；按服务商调整", "group": "advanced"},
        {"key": "image_mode", "label": "参考图上传方式", "type": "choice", "default": "data_uri",
         "options": [("data_uri", "JSON 内嵌 base64 (data URI)"), ("multipart", "multipart 文件上传（gpt.ge 等要求）")],
         "group": "advanced"},
        {"key": "size", "label": "默认尺寸(宽x高)", "type": "text", "default": "1024x1024", "group": "advanced"},
        {"key": "steps", "label": "采样步数", "type": "int", "default": 20, "min": 1, "group": "advanced"},
        {"key": "seed", "label": "种子(-1 随机)", "type": "int", "default": -1, "group": "advanced"},
        {"key": "timeout", "label": "超时(秒)", "type": "int", "default": 180, "min": 5, "group": "advanced"},
        {"key": "proxy", "label": "代理(可选)", "type": "text", "default": "",
         "placeholder": "http://127.0.0.1:7890（直连被拦截时填写）", "group": "advanced"},
        {"key": "verify_ssl", "label": "校验 SSL 证书", "type": "bool", "default": True, "group": "advanced"},
        {"key": "mock", "label": "使用模拟 API（无需密钥）", "type": "bool", "default": False},
    ],
    "video": [
        {"key": "base_url", "label": "Base URL", "type": "text", "default": "https://api.example.com/v1", "required": True},
        {"key": "api_key", "label": "API Key", "type": "password", "default": "", "required": True},
        {"key": "model", "label": "模型名称", "type": "text", "default": "video-model", "required": True},
        {"key": "provider", "label": "服务商适配", "type": "choice", "default": "generic",
         "options": [("generic", "通用（OpenAI 兼容轮询）"), ("doubao", "Doubao Seedance（火山方舟）"), ("gptge", "gpt.ge (V-API) 豆包视频")]},
        {"key": "frames", "label": "默认帧数", "type": "int", "default": 8, "min": 2},
        {"key": "fps", "label": "默认帧率", "type": "int", "default": 8, "min": 1},
        {"key": "last_frame", "label": "首帧同时作为尾帧传入（首尾帧一致）", "type": "bool", "default": False},
        {"key": "prompt_template", "label": "提示词模板", "type": "text", "default": "{prompt}", "group": "advanced"},
        {"key": "submit_url", "label": "提交端点(可选)", "type": "text", "default": "", "group": "advanced"},
        {"key": "poll_url", "label": "轮询端点(可选, 含 {id})", "type": "text", "default": "", "group": "advanced"},
        {"key": "poll_method", "label": "轮询方法", "type": "choice", "default": "GET",
         "options": [("GET", "GET"), ("POST", "POST")], "group": "advanced"},
        {"key": "poll_interval", "label": "轮询间隔(秒)", "type": "int", "default": 5, "min": 1, "group": "advanced"},
        {"key": "max_polls", "label": "最大轮询次数", "type": "int", "default": 120, "min": 1, "group": "advanced"},
        {"key": "job_id_path", "label": "任务ID字段路径", "type": "text", "default": "id", "group": "advanced"},
        {"key": "status_path", "label": "状态字段路径", "type": "text", "default": "status", "group": "advanced"},
        {"key": "status_success", "label": "成功状态(逗号分隔)", "type": "text", "default": "", "group": "advanced"},
        {"key": "status_failure", "label": "失败状态(逗号分隔)", "type": "text", "default": "", "group": "advanced"},
        {"key": "result_video_url_path", "label": "视频URL字段路径", "type": "text", "default": "", "group": "advanced"},
        {"key": "payload_template", "label": "请求体模板(JSON, 可选)", "type": "text", "default": "",
         "placeholder": '如 {"model_name":"$model","image":"$image"}，支持 $model/$prompt/$image/$frames/$fps/$duration',
         "group": "advanced"},
        {"key": "extra_payload", "label": "额外字段(JSON, 可选)", "type": "text", "default": "",
         "placeholder": '如 {"resolution":"1080p","watermark":false}', "group": "advanced"},
        {"key": "timeout", "label": "超时(秒)", "type": "int", "default": 300, "min": 5, "group": "advanced"},
        {"key": "proxy", "label": "代理(可选)", "type": "text", "default": "",
         "placeholder": "http://127.0.0.1:7890（直连被拦截时填写）", "group": "advanced"},
        {"key": "verify_ssl", "label": "校验 SSL 证书", "type": "bool", "default": True, "group": "advanced"},
        {"key": "mock", "label": "使用模拟 API（无需密钥）", "type": "bool", "default": False},
    ],
}

# --------------------------------------------------------------------------- #
# 服务商预设：设置页「服务商预设」下拉一键填充 Base URL / 模型 / 适配参数。
# 每个预设：key/label/base_url/model + 可选 endpoint / params（写入对应字段）。
# --------------------------------------------------------------------------- #
PROVIDER_PRESETS: Dict[str, List[dict]] = {
    "llm": [
        {"key": "openai", "label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
        {"key": "gptge", "label": "gpt.ge (V-API)", "base_url": "https://api.gpt.ge/v1", "model": "gpt-4o-mini"},
        {"key": "deepseek", "label": "DeepSeek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
        {"key": "moonshot", "label": "Moonshot Kimi", "base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
        {"key": "zhipu", "label": "智谱 Zhipu", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
        {"key": "siliconflow", "label": "硅基流动 SiliconFlow", "base_url": "https://api.siliconflow.cn/v1", "model": "deepseek-ai/DeepSeek-V3"},
        {"key": "ark", "label": "火山方舟 Ark", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-1-5-pro-32k-250115"},
        {"key": "dashscope", "label": "通义千问 DashScope", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
        {"key": "hunyuan", "label": "腾讯混元", "base_url": "https://api.hunyuan.cloud.tencent.com/v1", "model": "hunyuan-turbo"},
        {"key": "ollama", "label": "Ollama（本地）", "base_url": "http://localhost:11434/v1", "model": "llama3.1"},
    ],
    "image": [
        {"key": "openai", "label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-image-1", "params": {"response_format": "b64_json"}},
        {"key": "gptge", "label": "gpt.ge (V-API)", "base_url": "https://api.gpt.ge/v1", "model": "gpt-image-1", "params": {"response_format": "b64_json", "image_mode": "multipart"}},
        {"key": "ark", "label": "火山方舟 Seedream", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-seedream-3-0-t2i-250415", "params": {"response_format": "b64_json"}},
        {"key": "zhipu", "label": "智谱 CogView", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "cogview-3-flash", "params": {"response_format": "url"}},
        {"key": "siliconflow", "label": "硅基流动 SiliconFlow", "base_url": "https://api.siliconflow.cn/v1", "model": "black-forest-labs/FLUX.1-schnell", "params": {"response_format": "url"}},
        {"key": "dashscope", "label": "通义万相 DashScope", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "wanx2.1-t2i-turbo", "params": {"response_format": "url"}},
    ],
    "video": [
        {"key": "generic", "label": "通用（OpenAI 兼容轮询）", "base_url": "", "model": "", "params": {"provider": "generic"}},
        {"key": "doubao_ark", "label": "Doubao Seedance（火山方舟）", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-seedance-1-0-pro-250528", "params": {"provider": "doubao", "last_frame": True}},
        {"key": "gptge_doubao", "label": "gpt.ge 豆包视频", "base_url": "https://api.gpt.ge", "model": "doubao-seedance-1-5-pro-251215", "params": {"provider": "gptge", "last_frame": True}},
        {"key": "kling", "label": "快手可灵 Kling", "base_url": "https://api.klingai.com", "model": "kling-v1-6",
         "params": {
             "provider": "generic",
             "payload_template": '{"model_name": "$model", "prompt": "$prompt", "image": "$image", "mode": "std"}',
             "submit_url": "{base}/v1/videos/image2video",
             "poll_url": "{base}/v1/videos/image2video/{id}",
             "job_id_path": "data.task_id",
             "status_path": "data.task_status",
             "status_success": "succeed,success",
             "result_video_url_path": "data.task_result.videos.0.url",
             "poll_interval": 3,
             "max_polls": 180,
         }},
    ],
}


@dataclass
class APIConfig:
    """一套 API 配置（内存中 api_key 为明文，落盘时加密）。"""

    kind: str
    name: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def defaults(cls, kind: str) -> "APIConfig":
        params = {d["key"]: d.get("default") for d in FIELD_DEFS.get(kind, []) if d["key"] not in ("base_url", "api_key", "model")}
        return cls(kind=kind, name=f"{kind} 配置", params=params)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "params": self.params,
            "is_default": self.is_default,
        }


class APIConfigManager:
    """管理三类 API 的配置集合。"""

    def __init__(self, config_file: Path | str = API_CONFIG_FILE, keyring: Optional[Keyring] = None):
        self.config_file = Path(config_file)
        self.keyring = keyring or Keyring(KEYRING_FILE)
        self._configs: List[APIConfig] = []
        self.load()

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        self._configs = []
        if not self.config_file.exists():
            return
        try:
            with open(self.config_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("API 配置读取失败: %s", exc)
            return
        for item in data.get("configs", []):
            try:
                cfg = APIConfig(
                    id=item.get("id") or uuid.uuid4().hex[:12],
                    kind=item["kind"],
                    name=item.get("name", ""),
                    base_url=item.get("base_url", ""),
                    model=item.get("model", ""),
                    params=item.get("params", {}) or {},
                    is_default=bool(item.get("is_default", False)),
                )
                cfg.api_key = self.keyring.decrypt(item.get("api_key_enc", ""))
                if cfg.kind in API_KINDS:
                    self._configs.append(cfg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("跳过损坏的配置项: %s", exc)

    def save(self) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "configs": [
                {**cfg.to_dict(), "api_key_enc": self.keyring.encrypt(cfg.api_key)}
                for cfg in self._configs
            ],
        }
        tmp = self.config_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self.config_file)

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def list(self, kind: Optional[str] = None) -> List[APIConfig]:
        if kind is None:
            return list(self._configs)
        return [c for c in self._configs if c.kind == kind]

    def get(self, cfg_id: str) -> Optional[APIConfig]:
        for c in self._configs:
            if c.id == cfg_id:
                return c
        return None

    def get_default(self, kind: str) -> Optional[APIConfig]:
        for c in self._configs:
            if c.kind == kind and c.is_default:
                return c
        # 没有显式默认时，取该类型第一个
        items = self.list(kind)
        return items[0] if items else None

    # ------------------------------------------------------------------ #
    # 写操作
    # ------------------------------------------------------------------ #
    def add(self, cfg: APIConfig) -> APIConfig:
        if cfg.kind not in API_KINDS:
            raise ValueError(f"未知 API 类型: {cfg.kind}")
        cfg.id = uuid.uuid4().hex[:12]
        if not cfg.is_default:
            cfg.is_default = len(self.list(cfg.kind)) == 0  # 首套自动设为默认
        self._configs.append(cfg)
        self._enforce_single_default(cfg.kind)
        self.save()
        return cfg

    def update(self, cfg: APIConfig) -> None:
        for i, c in enumerate(self._configs):
            if c.id == cfg.id:
                cfg.is_default = c.is_default  # 保留默认标记
                self._configs[i] = cfg
                self._enforce_single_default(cfg.kind)
                self.save()
                return
        raise KeyError(f"配置不存在: {cfg.id}")

    def delete(self, cfg_id: str) -> bool:
        cfg = self.get(cfg_id)
        if cfg is None:
            return False
        self._configs = [c for c in self._configs if c.id != cfg_id]
        # 若删掉的是默认配置，把同类第一个设为默认
        if cfg.is_default:
            first = self.list(cfg.kind)
            if first:
                first[0].is_default = True
        self.save()
        return True

    def set_default(self, cfg_id: str) -> None:
        cfg = self.get(cfg_id)
        if cfg is None:
            raise KeyError(f"配置不存在: {cfg_id}")
        for c in self._configs:
            c.is_default = c.id == cfg_id
        self.save()

    def _enforce_single_default(self, kind: str) -> None:
        defaults = [c for c in self._configs if c.kind == kind and c.is_default]
        if len(defaults) > 1:
            for c in defaults[1:]:
                c.is_default = False

    # ------------------------------------------------------------------ #
    # 测试与导入导出
    # ------------------------------------------------------------------ #
    def test_connection(self, cfg: APIConfig):
        """连通性测试（同步，耗时调用应由 UI 放入后台线程）。"""
        from core.api.factory import create_api_client

        client = create_api_client(cfg.kind, cfg)
        try:
            return client.test_connection()
        finally:
            client.close()

    def export_config(self, path: Path | str) -> Path:
        """导出配置（JSON）。注意：导出文件包含解密后的 API Key，请妥善保管。"""
        path = Path(path)
        payload = {
            "version": CONFIG_VERSION,
            "exported_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "configs": [{**cfg.to_dict(), "api_key": cfg.api_key} for cfg in self._configs],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def import_config(self, path: Path | str, replace: bool = False) -> int:
        """导入配置；replace=True 时先清空现有配置。返回导入条数。"""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("configs", [])
        if not isinstance(items, list):
            raise ValueError("配置文件中没有 configs 列表")
        if replace:
            self._configs = []
        count = 0
        for item in items:
            kind = item.get("kind")
            if kind not in API_KINDS:
                continue
            cfg = APIConfig(
                kind=kind,
                name=item.get("name", ""),
                base_url=item.get("base_url", ""),
                api_key=item.get("api_key", "") or item.get("api_key_enc", ""),
                model=item.get("model", ""),
                params=item.get("params", {}) or {},
                is_default=bool(item.get("is_default", False)),
            )
            self._configs.append(cfg)
            self._enforce_single_default(kind)
            count += 1
        self.save()
        return count
