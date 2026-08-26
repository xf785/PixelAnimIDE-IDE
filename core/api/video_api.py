"""图转视频 API：通用「提交任务 -> 轮询 -> 取结果」模型。

适配主流服务商（如 Luma、Kling、可灵、Runway 等）的常见差异点通过
params 配置项控制：
- submit_url / poll_url   端点覆盖（默认 {base}/videos/generations）
- job_id_path             任务 ID 的 JSON 路径（默认 "id"）
- status_path             状态字段路径（默认 "status"）
- status_success / status_failure  成功/失败状态值列表
- result_video_url_path   结果视频 URL 路径（默认 "output.video_url"）
- result_frames_path      结果帧序列路径（默认 "output.frames"）
- poll_method             轮询方法（默认 GET，个别服务商用 POST）
- max_polls / poll_interval        轮询上限与间隔（秒）
- extra_payload           额外请求字段（JSON 字符串或 dict）

内置服务商适配（params.provider）：
- generic（默认）：OpenAI 兼容轮询式
- doubao / ark：火山方舟 Doubao Seedance（contents/generations/tasks，
  content 数组携带首帧图片，时长由帧数/帧率换算，结果取 content.video_url）
- gptge：gpt.ge (V-API) 网关的豆包视频（提交 {base}/task/volces/seedance，
  轮询 {base}/task/{id}，请求体/结果解析与火山方舟一致）

返回 APIResult(data={"video_url": str|None, "frames": [bytes]|None})。
若服务商直接返回帧序列（base64 图片列表），则 frames 非空。
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import List, Optional

from .base import APIResult, BaseAPI

logger = logging.getLogger("PixelAnimIDE.api.video")

_DOUBAO_PROVIDERS = ("doubao", "ark", "volcengine")
_GPTGE_PROVIDER = "gptge"


class VideoAPI(BaseAPI):
    KIND = "video"

    @property
    def provider(self) -> str:
        return str(self.params.get("provider") or "generic").lower()

    def _is_doubao_style(self) -> bool:
        """请求体/结果解析使用火山方舟 content 数组格式的服务商。"""
        return self.provider in _DOUBAO_PROVIDERS or self.provider == _GPTGE_PROVIDER

    def _last_frame_enabled(self) -> bool:
        """是否把首帧同时作为尾帧传入（首尾帧一致）。"""
        v = self.params.get("last_frame", False)
        return v not in (False, 0, "0", "false", "False", "", None)

    def _task_base(self) -> str:
        """gpt.ge 等网关的任务端点位于根路径（无 /v1），两种填法都容忍。"""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base[:-3]
        return base

    # ------------------------------------------------------------------ #
    def _submit_url(self) -> str:
        override = self.params.get("submit_url")
        if override:
            return str(override).format(base=self.base_url).rstrip("/")
        if self.provider == _GPTGE_PROVIDER:
            return f"{self._task_base()}/task/volces/seedance"
        if self._is_doubao_style():
            return f"{self.base_url}/contents/generations/tasks"
        return f"{self.base_url}/videos/generations"

    def _poll_url(self, job_id: str) -> str:
        template = self.params.get("poll_url")
        if template:
            return str(template).format(base=self.base_url, id=job_id)
        if self.provider == _GPTGE_PROVIDER:
            return f"{self._task_base()}/task/{job_id}"
        if self._is_doubao_style():
            return f"{self.base_url}/contents/generations/tasks/{job_id}"
        return f"{self.base_url}/videos/generations/{job_id}"

    def _result_video_url_path(self) -> str:
        return self.params.get("result_video_url_path") or (
            "content.video_url" if self._is_doubao_style() else "output.video_url"
        )

    # ------------------------------------------------------------------ #
    def _status_set(self, key: str, default: list) -> set:
        """状态值集合，兼容逗号分隔字符串与列表。"""
        value = self.params.get(key)
        if isinstance(value, str):
            parts = {s.strip().lower() for s in value.split(",") if s.strip()}
            # 空字符串（设置页表单未填）视为未配置，回退默认值
            if parts:
                return parts
            return {str(s).lower() for s in default}
        if value:
            return {str(s).lower() for s in value}
        return {str(s).lower() for s in default}

    def _payload_from_template(
        self,
        template: str,
        image_bytes: bytes,
        prompt: str,
        frames: Optional[int],
        fps: Optional[int],
        duration: Optional[float],
    ) -> dict:
        """按 JSON 模板渲染请求体。

        模板为合法 JSON，占位符使用 $ 前缀（避免与 JSON 花括号冲突）：
        $model / $prompt / $image（base64 data URI）/ $frames / $fps / $duration
        """
        frames = int(frames if frames is not None else self.params.get("frames", 8))
        fps = int(fps if fps is not None else self.params.get("fps", 8))
        dur = int(round(float(duration))) if duration else max(5, min(10, round(frames / max(1, fps))))
        text = str(template)
        text = text.replace("$model", self.model)
        text = text.replace("$prompt", prompt)
        text = text.replace("$image", self._to_data_uri(image_bytes))
        text = text.replace("$last_image", self._to_data_uri(image_bytes))
        text = text.replace("$frames", str(frames))
        text = text.replace("$fps", str(fps))
        text = text.replace("$duration", str(dur))
        try:
            return json.loads(text)
        except (ValueError, TypeError) as exc:
            logger.warning("payload_template 渲染后不是合法 JSON（%s）: %s", exc, text[:200])
            return {
                "model": self.model,
                "prompt": prompt,
                "image": self._to_data_uri(image_bytes),
            }

    def _build_payload(
        self,
        image_bytes: bytes,
        prompt: str,
        frames: Optional[int],
        fps: Optional[int],
        duration: Optional[float],
    ) -> dict:
        """按服务商组装请求体（首帧图片始终包含）。"""
        template = self.params.get("payload_template")
        if template:
            return self._payload_from_template(template, image_bytes, prompt, frames, fps, duration)
        if self._is_doubao_style():
            # 火山方舟 Seedance：content 数组携带文本 + 首帧图片（data URI）
            # last_frame 开启时，把同一张图作为尾帧一并传入（首尾帧一致）
            payload: dict = {
                "model": self.model,
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._to_data_uri(image_bytes)}},
                ],
            }
            if self._last_frame_enabled():
                payload["content"].append(
                    {"type": "image_url", "image_url": {"url": self._to_data_uri(image_bytes)}}
                )
            if duration:
                payload["duration"] = int(round(float(duration)))
            elif frames and fps:
                secs = round(int(frames) / max(1, int(fps)))
                payload["duration"] = max(5, min(10, secs))
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "image": self._to_data_uri(image_bytes),
                "frame_num": int(frames if frames is not None else self.params.get("frames", 8)),
                "fps": int(fps if fps is not None else self.params.get("fps", 8)),
            }
            if self._last_frame_enabled():
                # 通用服务商：尽力而为，把首帧同时作为尾帧（字段名可能因服务商而异，
                # 也可用 payload_template + $last_image 精确指定）
                payload["last_image"] = self._to_data_uri(image_bytes)
            template = self.params.get("prompt_template")
            if template and "{prompt}" in template:
                payload["prompt"] = template.format(
                    prompt=prompt,
                    frames=payload["frame_num"],
                    fps=payload["fps"],
                )
            if duration:
                payload["duration"] = float(duration)

        extra = self.params.get("extra_payload")
        if extra:
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except (ValueError, TypeError):
                    logger.warning("extra_payload 不是合法 JSON: %s", extra)
                    extra = {}
            if isinstance(extra, dict):
                payload.update(extra)
        return payload

    # ------------------------------------------------------------------ #
    def call(
        self,
        image_bytes: bytes,
        prompt: str,
        frames: Optional[int] = None,
        fps: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> APIResult:
        """以首帧图片 + 提示词生成动画，返回视频 URL 或帧序列字节。"""
        error = self._validate_config()
        if error:
            return APIResult(ok=False, error=error)

        frames = int(frames if frames is not None else self.params.get("frames", 8))
        fps = int(fps if fps is not None else self.params.get("fps", 8))

        payload = self._build_payload(image_bytes, prompt, frames, fps, duration)

        try:
            data = self._post_json(self._submit_url(), payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("图转视频提交失败")
            return APIResult(ok=False, error=self._friendly_error(exc))

        job_id = self._dig(data, self.params.get("job_id_path") or "id")
        if job_id is None and isinstance(data.get("data"), dict):
            job_id = data["data"].get("id") or data["data"].get("task_id")
        if job_id is None:
            return APIResult(ok=False, error=f"无法从响应中获取任务 ID: {str(data)[:300]}", raw=data)

        # 部分服务商同步返回结果（如直接给 URL），无需轮询
        video_url = self._dig(data, self._result_video_url_path())
        frames_bytes = self._extract_frames_from(data, self.params.get("result_frames_path") or "output.frames")
        if video_url or frames_bytes:
            return APIResult(
                ok=True,
                data={"video_url": video_url, "frames": frames_bytes, "job_id": job_id},
                raw=data,
            )

        poll_result = self._poll(job_id)
        if not poll_result.ok:
            return poll_result
        return APIResult(ok=True, data={**poll_result.data, "job_id": job_id}, raw=poll_result.raw)

    # ------------------------------------------------------------------ #
    def _poll(self, job_id: str) -> APIResult:
        max_polls = int(self.params.get("max_polls", 120))
        interval = float(self.params.get("poll_interval", 5))
        method = str(self.params.get("poll_method", "GET")).upper()
        success = self._status_set("status_success", ["succeeded", "success", "completed", "done", "finished"])
        failure = self._status_set("status_failure", ["failed", "error", "cancelled", "canceled", "expired", "rejected"])
        frames_path = self.params.get("result_frames_path") or "output.frames"

        for attempt in range(max_polls):
            time.sleep(interval)
            try:
                data = self._parse_json(self._request(method, self._poll_url(job_id)))
            except Exception as exc:  # noqa: BLE001
                # 轮询期间的网络抖动不致命，继续尝试
                logger.warning("轮询异常（第 %d 次）: %s", attempt + 1, exc)
                continue

            status = str(self._dig(data, self.params.get("status_path") or "status") or "").lower()
            if status in failure:
                err = (
                    self._dig(data, "error.message")
                    or self._dig(data, "error_message")
                    or self._dig(data, "error")
                    or data
                )
                err = str(err)[:300] if err is not None else ""
                return APIResult(ok=False, error=f"视频任务失败（{status}）: {err}", raw=data)
            if status in success or status in ("", "none"):
                video_url = self._dig(data, self._result_video_url_path())
                frames_bytes = self._extract_frames_from(data, frames_path)
                if video_url or frames_bytes or status in success:
                    return APIResult(
                        ok=True,
                        data={"video_url": video_url, "frames": frames_bytes},
                        raw=data,
                    )
            logger.info("视频任务 %s 轮询中（%d/%d）: %s", job_id, attempt + 1, max_polls, status)

        return APIResult(ok=False, error=f"轮询超时（{max_polls} 次，约 {max_polls * interval:.0f} 秒）")

    @staticmethod
    def _extract_frames_from(data: dict, path: str) -> List[bytes]:
        """从响应中解析帧序列（base64 图片列表 / url 列表）。"""
        frames: List[bytes] = []
        raw = VideoAPI._dig(data, path)
        if not raw:
            return frames
        if isinstance(raw, dict):
            raw = raw.get("frames") or raw.get("images") or raw.get("items")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    b64 = item.get("b64_json") or item.get("base64")
                    if b64:
                        try:
                            if isinstance(b64, str) and b64.startswith("data:"):
                                b64 = b64.split(",", 1)[1]
                            frames.append(base64.b64decode(b64))
                        except Exception:  # noqa: BLE001
                            continue
        return frames

    @staticmethod
    def _to_data_uri(image_bytes: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

    # ------------------------------------------------------------------ #
    def test_connection(self) -> APIResult:
        error = self._validate_config()
        if error:
            return APIResult(ok=False, error=error)
        try:
            # 探测服务是否可达；4xx 也说明服务在、鉴权已生效
            resp = self._request("GET", self.base_url)
            return APIResult(ok=True, data=f"服务可达（HTTP {resp.status_code}）", raw=resp.text[:200])
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "401" in msg or "403" in msg:
                return APIResult(ok=False, error=f"鉴权失败: {msg}")
            return APIResult(ok=False, error=f"连接失败: {msg}")
