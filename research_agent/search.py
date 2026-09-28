"""Query compilation and parsing for official arXiv search fallbacks."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .models import Paper


def search_terms(query: str) -> list[str]:
    # The model proposes concepts, not arbitrary Lucene syntax. Legacy queries
    # still work: remove field/operator syntax and preserve quoted phrases.
    query = re.sub(r'\b(?:all|ti|abs|au|cat):', '', query, flags=re.I)
    parts = re.findall(r'"([^"]+)"|([\w-]+)', query)
    terms = []
    for phrase, word in parts:
        term = (phrase or word).strip()
        if term.upper() in {'AND', 'OR', 'NOT', 'ANDNOT'}:
            continue
        if term and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    return terms[:12]


def compile_query(query: str) -> str:
    # Avoid the invalid/fragile field:(...) form shown in the screenshot.
    terms = search_terms(query)
    if not terms:
        return ''
    if re.search(r'\bOR\b', query):
        # Broad fallback for old boolean plans; relevance is checked by Selector.
        return ' OR '.join(f'all:"{t}"' for t in terms[:6])
    return ' AND '.join(f'all:"{t}"' for t in terms[:8])


def plain_query(query: str) -> str:
    terms = search_terms(query)
    if re.search(r'\bOR\b', query):
        terms = terms[:1]  # avoid AND-ing mutually alternative long phrases
    return ' '.join(terms)[:250]


def parse_search_html(data: bytes, limit: int = 5) -> list[Paper]:
    from .tools import ToolError, normalize_id, base_id
    soup = BeautifulSoup(data, 'html.parser')
    papers, seen = [], set()
    entries = soup.select('li.arxiv-result')
    if not entries:
        text = soup.get_text(' ', strip=True).lower()
        if any(s in text for s in ('sorry, your query', 'no results', 'no matches')):
            return []
        raise ToolError('arXiv 搜索页面未返回可识别的结果，可能被拦截。')
    for entry in entries:
        link = entry.select_one('p.list-title a[href*="/abs/"]')
        title = entry.select_one('p.title')
        if not link or not title:
            continue
        url = urlparse(link.get('href', ''))
        if url.hostname not in {'arxiv.org', 'www.arxiv.org'}:
            continue
        pid = normalize_id(url.path.removeprefix('/abs/'))
        if base_id(pid) in seen:
            continue
        seen.add(base_id(pid))
        abstract = entry.select_one('span.abstract-full') or entry.select_one('p.abstract')
        if abstract:
            for button in abstract.select('a'):
                button.decompose()
        authors = [a.get_text(' ', strip=True) for a in entry.select('p.authors a')]
        papers.append(Paper(id=pid, title=title.get_text(' ', strip=True),
                            abstract=abstract.get_text(' ', strip=True) if abstract else '',
                            authors=authors, search_source='arxiv_html'))
        if len(papers) >= limit:
            break
    if not papers:
        raise ToolError('搜索页面包含结果，但没有可校验的 arXiv 论文链接。')
    return papers
