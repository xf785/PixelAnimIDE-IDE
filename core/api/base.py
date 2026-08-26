"""API 抽象基类与统一结果对象。

设计目标：
- 屏蔽不同服务商差异，返回统一的 APIResult（成功/失败、数据、错误信息）。
- 统一处理超时、重试、日志。
- 所有耗时调用均可被上层（工作流 / UI）异步执行。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("PixelAnimIDE.api")


class APIError(Exception):
    """API 调用失败（网络、超时、HTTP 错误、解析错误等）。"""


class APIUnavailableError(APIError):
    """配置缺失或不完整，无法发起调用。"""


@dataclass
class APIResult:
    """统一 API 调用结果。"""

    ok: bool
    data: Any = None
    error: Optional[str] = None
    raw: Any = None

    @property
    def message(self) -> str:
        if self.ok:
            return "成功"
        return self.error or "调用失败"


def default_retry_policy() -> dict:
    """默认重试策略：最多重试 2 次，退避 0.5s / 1.5s。"""
    return {"max_retries": 2, "backoff": [0.5, 1.5]}


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """兼容 APIConfig 对象与 dict 的取值。"""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class BaseAPI(ABC):
    """API 客户端抽象基类。

    子类需实现 KIND、call() 与 test_connection()。
    config 可以是 APIConfig（config.api_config 中定义）或普通 dict。
    """

    KIND = "base"

    def __init__(self, config: Any, transport: Optional[httpx.BaseTransport] = None):
        self.config = config
        self.base_url = str(_config_get(config, "base_url", "") or "").rstrip("/")
        self.api_key = str(_config_get(config, "api_key", "") or "")
        self.model = str(_config_get(config, "model", "") or "")
        self.params = dict(_config_get(config, "params", None) or {})
        self.timeout = float(self.params.get("timeout", 120))
        self.max_retries = int(self.params.get("max_retries", 2))
        self._proxy = str(self.params.get("proxy", "") or "").strip() or None
        verify = self.params.get("verify_ssl", True)
        self._verify_ssl = verify not in (False, 0, "0", "false", "False", "")
        self._transport = transport
        self._client: Optional[httpx.Client] = None

    # ------------------------------------------------------------------ #
    # HTTP 基础设施
    # ------------------------------------------------------------------ #
    def _resolve_url(self, default_path: str) -> str:
        """解析请求 URL。

        优先使用 params.url（完整 URL 覆盖），其次 base_url + params.endpoint
        （自定义路径，如不同服务商的 /api/v3/images/generations），
        最后 base_url + default_path。
        """
        full = self.params.get("url") or self.params.get("endpoint_url")
        if full:
            return str(full).rstrip("/")
        path = self.params.get("endpoint") or default_path
        return f"{self.base_url}{path}"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _http(self) -> httpx.Client:
        if self._client is None:
            kwargs: dict = {"timeout": self.timeout, "follow_redirects": True}
            if self._transport is not None:
                # 测试注入的 MockTransport 优先
                kwargs["transport"] = self._transport
            else:
                if self._proxy:
                    kwargs["proxy"] = self._proxy
                if not self._verify_ssl:
                    kwargs["verify"] = False
            self._client = httpx.Client(**kwargs)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def __enter__(self) -> "BaseAPI":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 请求与重试
    # ------------------------------------------------------------------ #
    def _request(self, method: str, url: str, multipart: bool = False, **kwargs) -> httpx.Response:
        """带重试的 HTTP 请求；失败抛 APIError。

        multipart=True 时发送 multipart/form-data（用于图片文件上传类服务商），
        此时不设置 Content-Type（由 httpx 自动生成带 boundary 的头）。
        """
        retries = max(0, self.max_retries)
        backoff = [0.5, 1.5, 3.0]
        last_exc: Optional[Exception] = None
        headers = self._headers()
        if multipart:
            headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
        for attempt in range(retries + 1):
            try:
                resp = self._http().request(method, url, headers=headers, **kwargs)
                resp.raise_for_status()
                return resp
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                last_exc = exc
                logger.warning("API 请求网络异常(%s %s): %s", method, url, exc)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body = exc.response.text[:500]
                # 429/5xx 可重试，其余直接抛出（错误信息带完整 URL 便于排查）
                if status in (408, 429) or status >= 500:
                    last_exc = exc
                else:
                    raise APIError(f"HTTP {status} ({method} {url}): {body}") from exc
            if attempt < retries:
                wait = backoff[min(attempt, len(backoff) - 1)]
                logger.info("重试 %s %s（第 %d 次，等待 %.1fs）", method, url, attempt + 1, wait)
                time.sleep(wait)
        raise APIError(f"请求失败（已重试 {retries} 次）: {self._describe_error(last_exc)}") from last_exc

    def _describe_error(self, exc: Exception) -> str:
        """把最终异常转为可读信息（含响应体/排查建议）。"""
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text[:500]
            return f"HTTP {exc.response.status_code}: {body}"
        return self._friendly_error(exc)

    def _post_json(self, url: str, payload: dict) -> dict:
        resp = self._request("POST", url, json=payload)
        return self._parse_json(resp)

    def _post_multipart(self, url: str, data: dict, files: dict) -> dict:
        """multipart/form-data 上传（图片文件等）；data 为文本字段，files 为文件字段。"""
        resp = self._request("POST", url, multipart=True, data=data, files=files)
        return self._parse_json(resp)

    def _get_json(self, url: str, **kwargs) -> dict:
        resp = self._request("GET", url, **kwargs)
        return self._parse_json(resp)

    @staticmethod
    def _parse_json(resp: httpx.Response) -> dict:
        try:
            return resp.json()
        except ValueError as exc:
            raise APIError(f"响应不是合法 JSON: {resp.text[:200]}") from exc

    @staticmethod
    def _dig(obj: Any, path: str) -> Any:
        """按 'a.b.0.c' 形式的路径取值，取不到返回 None。"""
        cur = obj
        for part in path.split("."):
            if cur is None:
                return None
            if part.lstrip("-").isdigit():
                try:
                    cur = cur[int(part)]
                except (IndexError, KeyError, TypeError):
                    return None
            elif isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list):
                # 允许直接按字段名在列表里找
                found = None
                for item in cur:
                    if isinstance(item, dict) and part in item:
                        found = item[part]
                        break
                cur = found
            else:
                return None
        return cur

    # ------------------------------------------------------------------ #
    # 抽象接口
    # ------------------------------------------------------------------ #
    @abstractmethod
    def call(self, **kwargs) -> APIResult:
        """执行一次业务调用，返回统一结果。"""

    @abstractmethod
    def test_connection(self) -> APIResult:
        """连通性测试。"""

    def list_models(self) -> APIResult:
        """查询服务商当前令牌可用的模型列表（OpenAI 兼容 GET /models）。

        自动兼容两种 Base URL 写法：
        - https://api.gpt.ge        -> https://api.gpt.ge/v1/models
        - https://api.openai.com/v1 -> https://api.openai.com/v1/models
        """
        error = self._validate_config()
        if error:
            return APIResult(ok=False, error=error)
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            candidates = [f"{base}/models"]
        else:
            candidates = [f"{base}/v1/models", f"{base}/models"]
        last_err = "无法查询模型列表"
        for url in candidates:
            try:
                data = self._get_json(url)
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                continue
            items = data.get("data") if isinstance(data, dict) else None
            ids = [str(m["id"]) for m in (items or []) if isinstance(m, dict) and m.get("id")]
            if ids:
                return APIResult(ok=True, data=ids, raw=data)
            return APIResult(ok=False, error=f"响应中没有模型列表: {str(data)[:200]}", raw=data)
        return APIResult(ok=False, error=last_err)

    def _friendly_error(self, exc: Exception) -> str:
        """把底层异常转为带排查建议的提示。

        404 + "Invalid URL" 通常是 Base URL 缺少路径前缀（如 /v1），
        例如应填 https://api.gpt.ge/v1 而不是 https://api.gpt.ge。
        SSL 握手失败通常是网络被拦截或需要代理。
        """
        msg = str(exc)
        if "Invalid URL" in msg and "/v1" not in msg:
            msg += "（提示：多为 Base URL 缺少 /v1 等路径前缀所致，请核对服务商要求的完整路径，如 https://api.gpt.ge/v1）"
        if ("SSL" in msg or "TLS" in msg or "EOF" in msg) and "代理" not in msg:
            msg += "（提示：SSL/TLS 握手失败通常是网络被拦截或直连不通。可在 API 配置的高级项「代理」中填写代理地址，如 http://127.0.0.1:7890；或更换网络后重试）"
        return msg

    def _validate_config(self) -> Optional[str]:
        if not self.base_url:
            return "未配置 Base URL"
        if not self.base_url.startswith(("http://", "https://")):
            return f"Base URL 必须以 http:// 或 https:// 开头（当前: {self.base_url}）"
        if not self.api_key:
            return "未配置 API Key"
        return None
