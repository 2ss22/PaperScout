"""Thin LLM wrapper.

Both Groq and Ollama expose an OpenAI-compatible API, so a single client and
helper covers both providers. This is intentionally minimal — the interesting
logic lives in the agents, not here.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI

from .config import Settings, VisionSettings, current_configuration
from .models import Usage

_usage: ContextVar[Usage | None] = ContextVar("paperscout_usage", default=None)


@contextmanager
def track_usage(usage: Usage):
    token = _usage.set(usage)
    try:
        yield
    finally:
        _usage.reset(token)


class LLMError(RuntimeError):
    """Raised when the LLM call or response parsing fails."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def get_client() -> OpenAI:
    return _text_client(current_configuration().text)


@lru_cache(maxsize=8)
def _text_client(settings: Settings) -> OpenAI:
    if settings.provider in {"groq", "deepseek"} and not settings.api_key:
        raise LLMError(
            f"尚未设置模型服务 {settings.provider} 的 API 密钥。"
            "请检查 .env 文件。"
        )

    # On corporate networks with SSL inspection, allow skipping verification.
    http_client = httpx.Client(verify=False) if settings.disable_ssl_verify else None

    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        http_client=http_client,
        timeout=90.0,
        max_retries=1,
    )


def get_vision_client() -> OpenAI:
    config = current_configuration()
    return _vision_client(config.vision, config.text.disable_ssl_verify)


@lru_cache(maxsize=8)
def _vision_client(vision_settings: VisionSettings, disable_ssl_verify: bool) -> OpenAI:
    if not vision_settings.enabled:
        raise LLMError('视觉模型未启用，请设置 VISION_MODEL。')
    return OpenAI(base_url=vision_settings.base_url, api_key=vision_settings.api_key or 'ollama',
                  timeout=120.0, max_retries=1,
                  http_client=httpx.Client(verify=not disable_ssl_verify))


def chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.3,
    json_mode: bool = False,
    vision: bool = False,
) -> str:
    """Send a chat-completion request and return the assistant text."""
    config = current_configuration()
    kwargs: dict[str, Any] = {
        "model": config.vision.model if vision else config.text.model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    usage = _usage.get()
    if usage is not None:
        usage.calls += 1
        if vision:
            usage.vision_calls += 1
    try:
        response = (get_vision_client() if vision else get_client()).chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - surface provider errors uniformly
        status = getattr(exc, 'status_code', None)
        raise LLMError(f"模型请求失败：{exc}",
                       retryable=status not in {400, 401, 403, 404, 413, 422}) from exc

    if usage is not None and response.usage is not None:
        usage.calls_with_usage += 1
        usage.prompt_tokens += response.usage.prompt_tokens or 0
        usage.completion_tokens += response.usage.completion_tokens or 0
    if not response.choices:
        raise LLMError("模型返回了空回答。")
    choice = response.choices[0]
    finish_reason = getattr(choice, 'finish_reason', None)
    if finish_reason == 'length':
        raise LLMError('模型输出达到 token 上限而截断（finish_reason=length），未将不完整输出作为有效结果。')
    if finish_reason == 'content_filter' or getattr(choice.message, 'refusal', None):
        raise LLMError('模型服务拒绝生成本次内容，未返回可用结果。')
    content = choice.message.content
    if not content:
        raise LLMError(f"模型返回了空回答（finish_reason={finish_reason or '未提供'}）。")
    return content.strip()


def chat_json(messages: list[dict[str, Any]], *, temperature: float = 0.2, vision: bool = False) -> Any:
    """Chat helper that parses the response as JSON, with a salvage fallback."""
    raw = chat(messages, temperature=temperature, json_mode=True, vision=vision)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models wrap JSON in prose or code fences; salvage the object.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LLMError(f"模型 JSON 无效：第 {exc.lineno} 行第 {exc.colno} 列，{exc.msg}。") from exc
        raise LLMError("模型未返回有效 JSON 对象（可能为空、截断或格式不符）。")
