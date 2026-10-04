"""Per-run configuration: local .env > process environment > Streamlit secrets.

Reading .env never mutates os.environ, so old dotenv values cannot survive edits.
A context snapshot keeps an in-progress research run on one model configuration.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from dotenv import dotenv_values

ENV_FILE = Path(__file__).resolve().parents[1] / '.env'
GROQ_BASE_URL = 'https://api.groq.com/openai/v1'


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    disable_ssl_verify: bool = False


@dataclass(frozen=True)
class VisionSettings:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    enabled: bool
    max_pages: int = 3
    provider: str = ''
    key_source: str = ''


@dataclass(frozen=True)
class Configuration:
    text: Settings
    vision: VisionSettings


_active: ContextVar[Configuration | None] = ContextVar('paperscout_configuration', default=None)


def _values(secrets: Mapping | None = None) -> dict[str, str]:
    values = {k: str(v) for k, v in (secrets or {}).items()}
    values.update(os.environ)
    values.update({k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None})
    return values


def load_settings(values: Mapping[str, str] | None = None) -> Settings:
    env = _values() if values is None else values
    provider = env.get('LLM_PROVIDER', 'groq').lower().strip()
    disable_ssl_verify = env.get('DISABLE_SSL_VERIFY', 'false').lower().strip() in {'1', 'true', 'yes'}
    if provider == 'ollama':
        return Settings('ollama', env.get('OLLAMA_MODEL', 'llama3.1'),
                        env.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1'), 'ollama', disable_ssl_verify)
    if provider == 'deepseek':
        return Settings('deepseek', env.get('DEEPSEEK_MODEL', 'deepseek-flash'),
                        env.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
                        env.get('DEEPSEEK_API_KEY', ''), disable_ssl_verify)
    return Settings('groq', env.get('GROQ_MODEL', 'llama-3.3-70b-versatile'),
                    GROQ_BASE_URL, env.get('GROQ_API_KEY', ''), disable_ssl_verify)


def load_vision_settings(values: Mapping[str, str] | None = None, text: Settings | None = None) -> VisionSettings:
    env = _values() if values is None else values
    text = text or load_settings(env)
    default_model = text.model if text.provider == 'deepseek' and text.model == 'deepseek-flash' else ''
    model = env.get('VISION_MODEL', default_model).strip()
    base_url = env.get('VISION_BASE_URL', '').strip() or text.base_url
    shared_endpoint = base_url.rstrip('/') == text.base_url.rstrip('/')
    explicit_key = env.get('VISION_API_KEY', '').strip()
    api_key = explicit_key or (text.api_key if shared_endpoint else '')
    key_source = 'VISION_API_KEY' if explicit_key else (
        {'groq': 'GROQ_API_KEY', 'deepseek': 'DEEPSEEK_API_KEY', 'ollama': '本地服务'}[text.provider]
        if shared_endpoint else '尚未配置')
    hostname = urlparse(base_url).hostname
    provider = {'api.groq.com': 'groq', 'api.deepseek.com': 'deepseek'}.get(
        hostname, text.provider if shared_endpoint else '自定义服务')
    return VisionSettings(model=model, base_url=base_url, api_key=api_key,
                          enabled=bool(model) and env.get('VISION_ENABLED', 'true').lower().strip() not in {'false', '0', 'no'},
                          max_pages=max(1, min(6, int(env.get('VISION_MAX_PAGES', '3')))),
                          provider=provider, key_source=key_source)


def load_configuration(secrets: Mapping | None = None) -> Configuration:
    values = _values(secrets)
    text = load_settings(values)
    return Configuration(text, load_vision_settings(values, text))


def current_configuration() -> Configuration:
    return _active.get() or load_configuration()


@contextmanager
def configuration_scope(secrets: Mapping | None = None, *, snapshot: Configuration | None = None):
    token = _active.set(snapshot or load_configuration(secrets))
    try:
        yield
    finally:
        _active.reset(token)


class _SettingsView:
    """Keep existing imports live while isolating concurrent Streamlit sessions."""
    def __init__(self, section: str):
        self.section = section

    def __getattr__(self, name):
        return getattr(getattr(current_configuration(), self.section), name)


settings = _SettingsView('text')
vision_settings = _SettingsView('vision')
