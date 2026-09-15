"""LLM provider adapters for direct AlgorithmSpec generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request

from forge.utils.local_env import load_local_env


load_local_env()


@dataclass
class ProviderResponse:
    text: str
    success: bool
    provider: str
    model: str
    latency_seconds: float
    token_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    finish_reason: str | None = None
    model_version: str | None = None


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 60,
        enable_thinking: bool | None = None,
        stream: bool | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key or _api_key_from_env(provider)
        self.base_url = base_url or os.environ.get("LLM_API_BASE_URL") or os.environ.get("LLM_BASE_URL") or _default_base_url(provider)
        self.timeout_seconds = int(timeout_seconds)
        self.is_modelscope = _is_modelscope_base_url(self.base_url)
        configured_thinking = _optional_bool(os.environ.get("LLM_ENABLE_THINKING"))
        self.enable_thinking = enable_thinking if enable_thinking is not None else configured_thinking
        configured_stream = _optional_bool(os.environ.get("LLM_STREAM"))
        selected_stream = stream if stream is not None else configured_stream
        self.stream = self.is_modelscope if selected_stream is None else selected_stream
        if self.enable_thinking is None and self.is_modelscope:
            self.enable_thinking = False
        if self.enable_thinking is None and "glm" in self.model.lower():
            self.enable_thinking = False

    def complete_text(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        task: str | None = None,
    ) -> ProviderResponse:
        started = time.perf_counter()
        if not self.api_key:
            return ProviderResponse(
                text=json.dumps({"error": "missing LLM API key"}),
                success=False,
                provider=self.provider,
                model=self.model,
                latency_seconds=time.perf_counter() - started,
                error="missing LLM API key",
            )
        base_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.stream:
            base_payload["stream"] = True
        if self.enable_thinking is not None and self.is_modelscope:
            base_payload["enable_thinking"] = self.enable_thinking
        elif self.enable_thinking is not None:
            base_payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload_variants = [
            {**base_payload, "response_format": {"type": "json_object"}},
            base_payload,
        ]
        for attempt, request_payload in enumerate(payload_variants):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(request_payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = (
                        _read_streaming_chat_completion(response)
                        if self.stream
                        else json.loads(response.read().decode("utf-8"))
                    )
                choices = payload.get("choices") or []
                choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                message = choice.get("message") or {}
                response_text = str(message.get("content") or "")
                usage = payload.get("usage") or {}
                return ProviderResponse(
                    text=response_text,
                    success=True,
                    provider=self.provider,
                    model=self.model,
                    latency_seconds=time.perf_counter() - started,
                    token_usage={
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                    },
                    finish_reason=normalize_finish_reason(
                        choice.get("finish_reason")
                        or choice.get("finishReason")
                        or payload.get("finish_reason")
                        or payload.get("finishReason")
                        or payload.get("stop_reason")
                    ),
                    model_version=(str(payload.get("model")) if payload.get("model") else None),
                )
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(exc)
                if attempt == 0 and exc.code == 400 and _structured_output_unsupported(detail):
                    continue
                message = f"HTTP {exc.code}: {detail[:1000]}"
                return ProviderResponse(
                    text=json.dumps({"error": message}),
                    success=False,
                    provider=self.provider,
                    model=self.model,
                    latency_seconds=time.perf_counter() - started,
                    error=message,
                )
            except Exception as exc:
                return ProviderResponse(
                    text=json.dumps({"error": str(exc)}),
                    success=False,
                    provider=self.provider,
                    model=self.model,
                    latency_seconds=time.perf_counter() - started,
                    error=str(exc),
                )
        raise RuntimeError("unreachable provider request state")


def normalize_finish_reason(value: Any) -> str | None:
    """Normalize common provider stop-reason spellings for downstream auditing."""

    if value is None:
        return None
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {
        "length",
        "max_tokens",
        "max_token",
        "max_output_tokens",
        "token_limit",
        "model_length",
    }:
        return "length"
    if normalized in {"stop", "end_turn", "eos", "complete", "completed"}:
        return "stop"
    return normalized or None


def build_provider(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: int | None = None,
    enable_thinking: bool | None = None,
    stream: bool | None = None,
) -> OpenAICompatibleProvider:
    provider_name = provider or os.environ.get("LLM_PROVIDER")
    model_name = model or os.environ.get("LLM_MODEL")
    if not provider_name:
        raise ValueError("A production LLM provider must be configured")
    if not model_name:
        raise ValueError("A production LLM model must be configured")
    return OpenAICompatibleProvider(
        provider=provider_name,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=int(timeout_seconds or os.environ.get("LLM_TIMEOUT_SECONDS") or 60),
        enable_thinking=enable_thinking,
        stream=stream,
    )


def _default_base_url(provider: str) -> str:
    if provider == "deepseek":
        return "https://api.deepseek.com"
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_API_BASE_URL") or "https://api.openai.com/v1"


def _api_key_from_env(provider: str) -> str | None:
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")


def _structured_output_unsupported(detail: str) -> bool:
    lowered = detail.lower()
    return "structured output" in lowered or "response_format" in lowered or "spec decoding" in lowered


def _is_modelscope_base_url(base_url: str) -> bool:
    return "api-inference.modelscope.cn" in base_url.casefold()


def _read_streaming_chat_completion(response: Any) -> dict[str, Any]:
    """Aggregate OpenAI-compatible SSE chunks into one completion payload."""

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: Any = None
    model: Any = None
    usage: dict[str, Any] = {}
    event_count = 0

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        chunk = json.loads(data)
        event_count += 1
        if chunk.get("error"):
            raise RuntimeError(str(chunk["error"]))
        if chunk.get("model"):
            model = chunk["model"]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if content:
            content_parts.append(str(content))
        if reasoning:
            reasoning_parts.append(str(reasoning))
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

    if event_count == 0:
        raise ValueError("streaming endpoint returned no SSE completion events")

    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _optional_bool(value: object | None) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid optional boolean: {value!r}")


def redact_secret(text: str, api_key: str | None = None) -> str:
    output = text
    for secret in [api_key, os.environ.get("LLM_API_KEY"), os.environ.get("DEEPSEEK_API_KEY"), os.environ.get("OPENAI_API_KEY")]:
        if secret:
            output = output.replace(secret, "[REDACTED]")
    return output


def provider_config_from_env() -> dict[str, Any]:
    base_url = os.environ.get("LLM_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    configured_stream = _optional_bool(os.environ.get("LLM_STREAM"))
    effective_stream = (
        _is_modelscope_base_url(base_url or "")
        if configured_stream is None
        else configured_stream
    )
    return {
        "provider": os.environ.get("LLM_PROVIDER"),
        "model": os.environ.get("LLM_MODEL"),
        "base_url": base_url,
        "api_key_present": bool(os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        "timeout_seconds": int(os.environ.get("LLM_TIMEOUT_SECONDS") or 60),
        "enable_thinking": _optional_bool(os.environ.get("LLM_ENABLE_THINKING")),
        "stream": effective_stream,
        "config_source": str(Path(".").resolve()),
    }
