"""Page-preserving PDF extraction, BM25 retrieval, and literal quote validation."""
from __future__ import annotations

from io import BytesIO
import re
import unicodedata

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from .models import Paper, TextChunk, QuoteSpan
from .tools import ArxivClient, ToolError


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def quote_matches(quote: str, text: str) -> bool:
    quote = normalize_text(quote)
    return len(quote) >= 15 and quote in normalize_text(text)


def quote_spans(chunks: list[TextChunk], width: int = 360) -> list[QuoteSpan]:
    """Tool-owned continuous excerpts; the model selects IDs instead of copying text."""
    spans = []
    for chunk in chunks:
        start = 0
        while start < len(chunk.text):
            end = min(start + width, len(chunk.text))
            if end < len(chunk.text):
                boundary = chunk.text.rfind(' ', start + width // 2, end)
                if boundary > start:
                    end = boundary
            quote = chunk.text[start:end]
            if len(normalize_text(quote)) >= 15:
                spans.append(QuoteSpan(id=f'S{len(spans) + 1:04d}', chunk_id=chunk.id,
                                       start=start, end=end, quote=quote))
            if end == len(chunk.text):
                break
            start = max(start + 1, end - 60)
    return spans


def extract_chunks(data: bytes, paper_id: str) -> list[TextChunk]:
    try:
        reader = PdfReader(BytesIO(data))
        if len(reader.pages) > 80:
            raise ToolError("论文超过首版 80 页阅读限制。")
        chunks = []
        for page_no, page in enumerate(reader.pages, 1):
            text = normalize_text(page.extract_text() or "")
            if len(text) < 50:
                continue
            for start in range(0, len(text), 1500):
                segment = text[start:start + 1800]
                if len(segment) < 50:
                    continue
                chunks.append(TextChunk(id=f"{paper_id}:p{page_no}:c{start // 1500 + 1}",
                                        paper_id=paper_id, page=page_no, text=segment))
        if not chunks:
            raise ToolError("PDF 无可用文本，可能为扫描件；后续尝试视觉 OCR 读取。")
        return chunks
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"PDF 解析失败（{type(exc).__name__}）。") from exc


def read_paper(paper: Paper, client: ArxivClient) -> list[TextChunk]:
    if not re.search(r'v\d+$', paper.id):
        client.resolve_version(paper)
    key = f"chunks-v2:{paper.id}"
    try:
        cached = client.cache.get(key)
        chunks = None
        if cached:
            try:
                chunks = [TextChunk.model_validate(c) for c in cached]
            except (ValueError, TypeError):
                pass
        if not chunks:
            chunks = extract_chunks(client.pdf(paper), paper.id)
            client.cache.put(key, [c.model_dump() for c in chunks])
        paper.read_status = "full_text"
        return chunks
    except (ToolError, OSError) as exc:
        paper.read_status = "abstract_only"
        paper.read_error = str(exc)
        return [TextChunk(id=f"{paper.id}:abstract", paper_id=paper.id, page=None,
                          text=paper.abstract, source="abstract")]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve(chunks: list[TextChunk], query: str, top_k: int = 6) -> list[TextChunk]:
    if not chunks:
        return []
    tokens = [tokenize(c.text) or ["_empty_"] for c in chunks]
    scores = BM25Okapi(tokens).get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [chunks[i] for i in ranked[:top_k]]


def retrieve_context(chunks: list[TextChunk], query: str) -> list[TextChunk]:
    # Reserve space for tables and figures; prose must not crowd them out.
    text = retrieve([c for c in chunks if c.source in {'pdf', 'abstract'}], query, 4)
    visual_chunks = [c for c in chunks if c.source in {'table', 'vision'}]
    figures = retrieve([c for c in visual_chunks if c.source == 'vision' and not c.cells], query, 1)
    tables = retrieve([c for c in visual_chunks if c.cells], query, 1)
    chosen = {c.id for c in figures + tables}
    rest = retrieve([c for c in visual_chunks if c.id not in chosen], query, 4 - len(chosen))
    return text + figures + tables + rest
