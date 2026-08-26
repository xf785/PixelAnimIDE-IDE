"""图片生成 API：兼容 OpenAI /images/generations 的服务商（支持 b64_json 或 URL 返回）。"""
from __future__ import annotations

import base64
import logging
from typing import List, Optional

from .base import APIError, APIResult, BaseAPI

logger = logging.getLogger("PixelAnimIDE.api.image")


class ImageAPI(BaseAPI):
    KIND = "image"

    def _generations_url(self) -> str:
        return self._resolve_url("/images/generations")

    # ------------------------------------------------------------------ #
    def _use_multipart_upload(self) -> bool:
        """参考图是否以 multipart 文件上传（而非 JSON data URI）。

        - image_mode=multipart  -> 强制 multipart；
        - image_mode=data_uri   -> 强制 JSON data URI；
        - 未配置时：gpt.ge (V-API) 的 /images/generations 要求 image 字段为
          multipart 文件上传（[]*multipart.FileHeader），因此自动启用 multipart。
        """
        mode = str(self.params.get("image_mode") or "").strip().lower()
        if mode == "multipart":
            return True
        if mode == "data_uri":
            return False
        return "gpt.ge" in (self.base_url or "")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_size_error(exc: Exception) -> bool:
        """错误信息是否表明「请求尺寸不被服务商支持」（如 gpt.ge unsupported size）。"""
        msg = str(exc).lower()
        return (
            "unsupported size" in msg
            or "invalid image size" in msg
            or "invalid size" in msg
            or "size is not supported" in msg
            or "size not supported" in msg
        )

    @staticmethod
    def _fallback_sizes(w: int, h: int, limit: int = 3) -> List[str]:
        """按相同宽高比给出更大的常见档位尺寸（长边 512/768/1024/1536，短边取整到 64 倍数）。

        用于服务商拒绝小尺寸（如 gpt.ge 不支持 256x256）时的自动重试。
        """
        out: List[str] = []
        seen = set()
        for side in (512, 768, 1024, 1536):
            if side <= max(w, h):
                continue
            scale = side / max(w, h)
            nw = max(64, round(w * scale / 64) * 64)
            nh = max(64, round(h * scale / 64) * 64)
            if nw >= nh:
                nw = side
            else:
                nh = side
            nw, nh = min(nw, 2048), min(nh, 2048)
            key = f"{nw}x{nh}"
            if key not in seen:
                seen.add(key)
                out.append(key)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _parse_wh(text: str):
        """'256x256' -> (256, 256)；解析失败返回 (None, None)。"""
        try:
            w, h = str(text).lower().split("x")
            return int(w), int(h)
        except (ValueError, AttributeError):
            return None, None

    # ------------------------------------------------------------------ #
    def call(
        self,
        prompt: str,
        size: Optional[str] = None,
        n: int = 1,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        image: Optional[bytes] = None,
    ) -> APIResult:
        """生成图片，返回 APIResult(data={"images": [bytes], "urls": [str]})。

        image 为参考图字节（图生图/i2i），可选；字段名由 params.image_field
        控制（默认 "image"）。上传方式由 params.image_mode 控制：
        - data_uri（默认）：以 data URI 字符串写入 JSON 请求体；
        - multipart：以 multipart/form-data 文件字段上传（gpt.ge 等要求）。

        服务商拒绝请求尺寸时（错误含 unsupported size 等），自动按同宽高比
        换更大的常见档位重试（最多 3 档），避免 256x256 这类小尺寸被拒。
        """
        error = self._validate_config()
        if error:
            return APIResult(ok=False, error=error)

        size = size or self.params.get("size", "1024x1024")
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "n": int(n),
            "size": size,
            "response_format": self.params.get("response_format", "b64_json"),
        }
        if seed is not None and seed >= 0:
            payload["seed"] = int(seed)
        elif int(self.params.get("seed", -1)) >= 0:
            payload["seed"] = int(self.params["seed"])
        if steps is not None:
            payload["steps"] = int(steps)
        elif self.params.get("steps"):
            payload["steps"] = int(self.params["steps"])
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        field = str(self.params.get("image_field") or "image")
        w, h = self._parse_wh(size)
        sizes = [size] + (self._fallback_sizes(w, h) if w and h else [])
        multipart = bool(image) and self._use_multipart_upload()
        last_exc: Optional[Exception] = None
        try:
            data = None
            for idx, size_str in enumerate(sizes):
                payload["size"] = size_str
                try:
                    if multipart:
                        # gpt.ge 等要求参考图以 multipart 文件字段上传
                        data = self._post_multipart(
                            self._generations_url(),
                            data=payload,
                            files={field: ("image.png", image, "image/png")},
                        )
                    else:
                        if image:
                            payload[field] = self._to_data_uri(image)
                        data = self._post_json(self._generations_url(), payload)
                    break
                except APIError as exc:
                    last_exc = exc
                    if not self._is_size_error(exc) or idx == len(sizes) - 1:
                        raise
                    logger.info("尺寸 %s 不被服务商支持，尝试 %s", size_str, sizes[idx + 1])
        except Exception as exc:  # noqa: BLE001
            logger.exception("生图 API 调用失败")
            return APIResult(ok=False, error=self._friendly_error(exc))

        images, urls = self._extract_images(data)
        if not images and not urls:
            return APIResult(ok=False, error=f"无法从响应中解析图片: {str(data)[:300]}", raw=data)
        return APIResult(ok=True, data={"images": images, "urls": urls}, raw=data)

    @staticmethod
    def _to_data_uri(image: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(image).decode("ascii")

    def _extract_images(self, data: dict):
        """兼容多种响应结构，返回 (images_bytes[], urls[])。"""
        images: List[bytes] = []
        urls: List[str] = []

        items = data.get("data") or []
        if not items and isinstance(data.get("images"), list):
            items = data["images"]
        if not items and isinstance(data.get("output"), list):
            items = data["output"]

        for item in items:
            if not isinstance(item, dict):
                continue
            b64 = item.get("b64_json")
            if b64:
                try:
                    if isinstance(b64, str) and b64.startswith("data:"):
                        b64 = b64.split(",", 1)[1]
                    images.append(base64.b64decode(b64))
                    continue
                except Exception:  # noqa: BLE001
                    pass
            url = item.get("url") or item.get("image_url") or item.get("uri")
            if url:
                urls.append(url)
        return images, urls

    # ------------------------------------------------------------------ #
    def test_connection(self) -> APIResult:
        error = self._validate_config()
        if error:
            return APIResult(ok=False, error=error)
        # 用最小的 n=1 请求探测；生成 1 张图成本可接受，能真实验证密钥与模型权限
        result = self.call(prompt="a single red pixel", n=1)
        if result.ok:
            return APIResult(ok=True, data="连接成功，生图接口可用", raw=result.raw)
        if result.error and "401" in result.error:
            return APIResult(ok=False, error="API Key 无效（HTTP 401）")
        return result
