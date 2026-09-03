"""通用文本 API：兼容 OpenAI /chat/completions 的服务商。"""
from __future__ import annotations

import logging
from typing import Optional

from .base import APIResult, BaseAPI

logger = logging.getLogger("PixelAnimIDE.api.llm")


class LLMAPI(BaseAPI):
    KIND = "llm"

    def _chat_url(self) -> str:
        return self._resolve_url("/chat/completions")

    # ------------------------------------------------------------------ #
    def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        action: Optional[str] = None,
    ) -> APIResult:
        """发送聊天补全请求，返回生成文本。

        action 为可选的动作类型提示（已包含在 prompt 文本中），
        供模拟客户端/特定服务商使用；标准 chat/completions 请求忽略它。
        """
        error = self._validate_config()
        if error:
            return APIResult(ok=False, error=error)

        custom = self.custom_enabled()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {"model": self.model, "messages": messages}
        if temperature is not None or "temperature" in self.params:
            payload["temperature"] = float(temperature if temperature is not None else self.params.get("temperature", 0.7))
        if max_tokens is not None or "max_tokens" in self.params:
            payload["max_tokens"] = int(max_tokens if max_tokens is not None else self.params.get("max_tokens", 2048))
        if top_p is not None or "top_p" in self.params:
            payload["top_p"] = float(top_p if top_p is not None else self.params.get("top_p", 1.0))

        url = self._chat_url()
        method = "POST"
        if custom:
            template = str(self.params.get("payload_template") or "")
            if template.strip():
                rendered = self.render_template_payload(
                    template,
                    {
                        "model": self.model,
                        "prompt": prompt,
                        "system": system or "",
                        "max_tokens": str(int(max_tokens if max_tokens is not None else self.params.get("max_tokens", 800))),
                        "temperature": str(float(temperature if temperature is not None else self.params.get("temperature", 0.7))),
                    },
                )
                if rendered is None:
                    return APIResult(ok=False, error="请求体模板不是合法 JSON，请检查「请求体模板(JSON)」")
                payload = rendered
            method = self.custom_method()

        try:
            data = self._post_json(url, payload) if method == "POST" else self._get_json(url)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM 调用失败")
            return APIResult(ok=False, error=self._friendly_error(exc))

        text = self._extract_text_custom(data) if custom else self._extract_text(data)
        if text is None:
            return APIResult(ok=False, error=f"无法从响应中解析文本: {str(data)[:300]}", raw=data)
        return APIResult(ok=True, data=text, raw=data)

    def _extract_text_custom(self, data: dict) -> Optional[str]:
        """完全自定义模式的文本解析：优先按 text_path 取，取不到回退兼容解析。"""
        path = self.params.get("text_path")
        if path:
            val = self._dig(data, path)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list):
                # 数组里拼 content/文本段（部分服务返回分段数组）
                parts = []
                for seg in val:
                    if isinstance(seg, str):
                        parts.append(seg)
                    elif isinstance(seg, dict):
                        c = seg.get("content") or seg.get("text")
                        if isinstance(c, str) and c.strip():
                            parts.append(c)
                if parts:
                    return "\n".join(parts)
            if val is not None:
                s = str(val)
                if s.strip():
                    return s
        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: dict) -> Optional[str]:
        """兼容多种响应结构。

        优先取 content；DeepSeek 等推理模型的 content 可能为空
        （思考内容在 reasoning_content，且 max_tokens 常被推理吃光），
        此时回退取 reasoning_content 兜底（可再配合工作流层提高 max_tokens 重试）。
        """
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content")
                if content and str(content).strip():
                    return content
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                if reasoning and str(reasoning).strip():
                    logger.warning("content 为空，回退使用 reasoning_content（推理模型输出可能被截断）")
                    return reasoning
            if choices[0].get("text"):
                return choices[0]["text"]
        if data.get("output"):
            if isinstance(data["output"], str):
                return data["output"]
            if isinstance(data["output"], list) and data["output"]:
                first = data["output"][0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    return first.get("text") or first.get("content")
        if isinstance(data.get("response"), str):
            return data["response"]
        return None

    # ------------------------------------------------------------------ #
    def test_connection(self) -> APIResult:
        error = self._validate_config()
        if error:
            return APIResult(ok=False, error=error)
        result = self.call(prompt="ping", max_tokens=1, temperature=0.0)
        if result.ok:
            return APIResult(ok=True, data="连接成功", raw=result.raw)
        # 很多服务商对无效请求也会返回 200 + error 字段
        if result.error and "401" in result.error:
            return APIResult(ok=False, error="API Key 无效（HTTP 401）")
        return result
