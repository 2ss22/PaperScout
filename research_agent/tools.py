"""arXiv access with bounded retries, disk cache and explicit failure states."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import httpx

from .cache_lifecycle import record_cache_file
from .config import settings
from .models import Paper


class ToolError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def base_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def normalize_id(value: str) -> str:
    value = value.strip().removeprefix("http://arxiv.org/abs/").removeprefix("https://arxiv.org/abs/")
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", value):
        raise ToolError("arXiv 返回了无效的论文 ID。")
    return value


class Cache:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.getenv("PAPERSCOUT_CACHE_DIR", ".cache/paperscout"))

    def _path(self, key: str) -> Path:
        return self.root / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def get(self, key: str, ttl: float | None = None):
        path = self._path(key)
        try:
            if path.exists():
                record_cache_file(path)
            if ttl is not None and time.time() - path.stat().st_mtime > ttl:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value) -> None:
        # Concurrent Streamlit sessions must never observe a half-written cache.
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.root, encoding="utf-8", delete=False) as f:
            name = f.name
            json.dump(value, f, ensure_ascii=False)
        path = self._path(key)
        os.replace(name, path)
        record_cache_file(path)

    def binary_path(self, key: str, suffix: str = '.bin') -> Path:
        path = self._path(key).with_suffix(suffix)
        if path.exists():
            record_cache_file(path)
        return path

    def put_bytes(self, key: str, value: bytes, suffix: str = '.bin') -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('wb', dir=self.root, delete=False) as f:
            f.write(value)
            name = f.name
        path = self.binary_path(key, suffix)
        os.replace(name, path)
        record_cache_file(path)
        return path


_rate_lock = threading.Lock()
_last_request = 0.0


class ArxivClient:
    def __init__(self, cache: Cache | None = None, client: httpx.Client | None = None):
        self.cache = cache or Cache()
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(30, connect=10), follow_redirects=True,
            verify=not settings.disable_ssl_verify,
            headers={"User-Agent": "PaperScout/0.1 (academic research prototype)"},
        )
        self._owns_client = client is None
        self.search_diagnostics: list[dict] = []
        self._api_unavailable = False

    def close(self):
        if self._owns_client:
            self.client.close()

    def _request(self, url: str, *, params: dict | None = None, max_bytes: int = 25_000_000) -> bytes:
        global _last_request
        last_error = ""
        status_code = None
        for attempt in range(3):
            with _rate_lock:
                delay = 3.1 - (time.monotonic() - _last_request)
                if delay > 0:
                    time.sleep(delay)
                _last_request = time.monotonic()
            try:
                with self.client.stream("GET", url, params=params) as response:
                    response.raise_for_status()
                    parts, size = [], 0
                    for part in response.iter_bytes():
                        size += len(part)
                        if size > max_bytes:
                            raise ToolError("下载内容超出大小限制。")
                        parts.append(part)
                    return b"".join(parts)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                last_error = f"HTTP {exc.response.status_code}"
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except httpx.RequestError as exc:
                last_error = type(exc).__name__
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise ToolError(f"arXiv 请求失败（{last_error}）；这不代表没有相关论文。", status_code)

    def search(self, query: str, max_results: int = 5) -> list[Paper]:
        from .search import compile_query, plain_query, parse_search_html
        query = query.strip()[:500]
        if not query:
            return []
        max_results = min(max_results, 5)
        self.search_diagnostics = []
        key = f"search-v2:{query}:{max_results}"
        cached = self.cache.get(key, ttl=86400)
        if cached is not None:
            try:
                papers = [Paper.model_validate(p) for p in cached]
                self.search_diagnostics.append({'source': 'cache', 'count': len(papers)})
                return papers
            except (ValueError, TypeError):
                pass
        normalized = compile_query(query)
        if not normalized:
            return []
        try:
            if self._api_unavailable:
                raise ToolError('本次会话的 API 暂不可用，直接使用官方搜索页面。')
            data = self._request('https://export.arxiv.org/api/query', params={
                'search_query': normalized, 'max_results': max_results,
            }, max_bytes=2_000_000)
            papers = parse_feed(data)[:max_results]
            self.search_diagnostics.append({'source': 'arxiv_api', 'query': normalized, 'count': len(papers)})
        except ToolError as exc:
            if exc.status_code in {403, 406, 429}:
                self._api_unavailable = True
            self.search_diagnostics.append({'source': 'arxiv_api', 'error': str(exc)})
            try:
                simplified = plain_query(query)
                data = self._request('https://arxiv.org/search/', params={
                    'query': simplified, 'searchtype': 'all', 'abstracts': 'show',
                    'order': '-announced_date_first', 'size': 50,
                }, max_bytes=3_000_000)
                papers = parse_search_html(data, max_results)
                self.search_diagnostics.append({'source': 'arxiv_html', 'query': simplified, 'count': len(papers)})
            except ToolError as fallback:
                raise ToolError(f'API 与官方搜索页面均失败：{exc}；{fallback}') from fallback
        self.cache.put(key, [p.model_dump() for p in papers])
        return papers

    def pdf(self, paper: Paper) -> bytes:
        normalize_id(paper.id)
        # Resolve HTML-search IDs to an immutable version before caching a PDF.
        if not re.search(r'v\d+$', paper.id):
            self.resolve_version(paper)
        key = f'pdf-v1:{paper.id}'
        path = self.cache.binary_path(key, '.pdf')
        if path.exists() and (re.search(r'v\d+$', paper.id) or time.time() - path.stat().st_mtime < 86400):
            return path.read_bytes()
        data = self._request(paper.pdf_url)
        self.cache.put_bytes(key, data, '.pdf')
        return data

    def resolve_version(self, paper: Paper) -> None:
        from bs4 import BeautifulSoup
        try:
            page = self._request(paper.url, max_bytes=2_000_000).decode()
            versions = re.findall(re.escape(base_id(paper.id)) + r'v(\d+)', page)
            if versions:
                paper.id = base_id(paper.id) + 'v' + str(max(map(int, versions)))
            soup = BeautifulSoup(page, 'html.parser')
            abstract = soup.select_one('blockquote.abstract')
            if abstract:
                descriptor = abstract.select_one('.descriptor')
                if descriptor:
                    descriptor.decompose()
                paper.abstract = abstract.get_text(' ', strip=True)
        except ToolError:
            # A content hash in the visual assets still identifies the exact PDF.
            pass


def parse_feed(data: bytes) -> list[Paper]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(data)
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            raise ToolError("arXiv 返回了非预期格式。")
        result, seen = [], set()
        for entry in root.findall("a:entry", ns):
            def text(tag):
                return " ".join(entry.findtext(f"a:{tag}", default="", namespaces=ns).split())
            if text("id").endswith("/errors"):
                raise ToolError("arXiv 拒绝了查询：" + text("summary")[:200])
            pid = normalize_id(text("id"))
            if base_id(pid) in seen:
                continue
            seen.add(base_id(pid))
            result.append(Paper(id=pid, title=text("title"), abstract=text("summary"),
                                published=text("published"), authors=[
                                    n.text or "" for n in entry.findall("a:author/a:name", ns)]))
        return result
    except ET.ParseError as exc:
        raise ToolError("arXiv 返回了无法解析的检索结果。") from exc
