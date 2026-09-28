"""Retrieve visual pages, preserve table cells, and verify against original pixels."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from . import agents
from .config import vision_settings
from .llm import LLMError
from .models import Claim, Evidence, Paper, TableCell, TextChunk, VisualAsset, VisualReading
from .reader import normalize_text, retrieve
from .tools import ArxivClient, ToolError


def image_bytes(asset: VisualAsset) -> bytes:
    data = Path(asset.image_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != asset.image_sha256:
        raise ToolError('图表原始图片缺失或校验不一致，不能核验视觉证据。')
    return data


def cells_text(cells: list[TableCell]) -> str:
    return '\n'.join(f'row={c.row}, column={c.column}, rowspan={c.row_span}, colspan={c.column_span}, '
                     f'header={c.header}: {c.text}' for c in cells)


def reading_chunks(asset: VisualAsset) -> list[TextChunk]:
    reading = asset.reading
    if reading is None or not reading.readable:
        return []
    result = []
    for i, element in enumerate(reading.elements):
        text = '\n'.join([element.label, '视觉解读（非原文逐字引文）: ' + element.description,
                          '图片可见文字转录: ' + element.visible_text, cells_text(element.cells),
                          '识别限制: ' + (element.limitations or reading.limitations or '未报告')])
        result.append(TextChunk(id=f'{asset.id}:v{i}', paper_id=asset.paper_id, page=asset.page,
                                text=normalize_text(text), source='vision', asset_id=asset.id, cells=element.cells))
    return result


def read_visuals(paper: Paper, client: ArxivClient, query: str, max_pages: int | None = None) -> list[TextChunk]:
    """Returns table/visual chunks; records selected source images on Paper."""
    budget = vision_settings.max_pages if max_pages is None else max(0, min(6, max_pages))
    if budget == 0:
        paper.visual_limitations = ['本次已关闭图表读取。']
        return []
    chunks = []
    paper.assets = []
    paper.visual_limitations = []
    try:
        data = client.pdf(paper)
        pdf_hash = hashlib.sha256(data).hexdigest()
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            if len(doc) > 80:
                raise ToolError('论文超过 80 页图表读取限制。')
            candidates = []
            for i, page in enumerate(doc):
                text = page.get_text()
                area = page.rect.get_area()
                images = page.get_image_info()
                drawings = page.get_drawings()
                meaningful_image = any(pymupdf.Rect(item['bbox']).get_area() > area * 0.02 for item in images)
                meaningful_drawing = any(item['rect'].get_area() > area * 0.01 for item in drawings)
                caption = re.search(r'(?im)^\s*(?:figure|fig\.|table)\s+\d+[\s.:]', text)
                if meaningful_image or meaningful_drawing or caption:
                    candidates.append(TextChunk(id=str(i), paper_id=paper.id, page=i + 1,
                                                text=text or 'scanned page image table figure'))
            paper.visual_pages_available = len(candidates)
            selected = retrieve(candidates, query, budget)
            if len(candidates) > len(selected):
                paper.visual_limitations.append(f'发现 {len(candidates)} 个图表候选页，本次按相关性读取 {len(selected)} 页。')
            for candidate in selected:
                page = doc[int(candidate.id)]
                png = page.get_pixmap(matrix=pymupdf.Matrix(min(2.5, 2200 / max(page.rect.width, page.rect.height)),
                                                           min(2.5, 2200 / max(page.rect.width, page.rect.height))),
                                      alpha=False).tobytes('png')
                digest = hashlib.sha256(png).hexdigest()
                path = client.cache.put_bytes(f'visual-png:{pdf_hash}:{page.number}', png, '.png')
                asset = VisualAsset(id=f'{paper.id}:visual:p{page.number + 1}', paper_id=paper.id, page=page.number + 1,
                                    image_path=str(path.resolve()), image_sha256=digest, pdf_sha256=pdf_hash,
                                    bbox=list(page.rect), caption=normalize_text(candidate.text)[:1500])
                paper.assets.append(asset)
                # Native cells retain row/column positions; image review is still
                # required before these become affirmed quantitative claims.
                try:
                    tables = page.find_tables()
                    for index, table in enumerate(tables.tables[:8]):
                        rows = table.extract()
                        cells = [TableCell(row=r, column=c, text=value, header=r == 0)
                                 for r, row in enumerate(rows[:200]) for c, value in enumerate(row[:100]) if value]
                        if len(cells) < 4:
                            continue
                        chunks.append(TextChunk(id=f'{asset.id}:native-table:{index}', paper_id=paper.id,
                                                page=asset.page, text=normalize_text(cells_text(cells)),
                                                source='table', asset_id=asset.id, cells=cells))
                except Exception as exc:
                    paper.visual_limitations.append(f'第 {asset.page} 页的几何表格解析失败（{type(exc).__name__}），继续视觉读取。')
                if not vision_settings.enabled:
                    asset.error = '未配置视觉模型；已提取截图和表格单元格，尚未理解图片内容。'
                    paper.visual_limitations.append(asset.error)
                    continue
                key = f'visual-reading-v1:{vision_settings.base_url}:{vision_settings.model}:{digest}:{hashlib.sha256(query.encode()).hexdigest()}'
                cached = client.cache.get(key, ttl=86400)
                try:
                    if cached is not None:
                        try:
                            asset.reading = VisualReading.model_validate(cached)
                        except ValueError:
                            asset.reading = None
                    if asset.reading is None:
                        asset.reading = agents.read_visual(png, query, asset.page)
                        client.cache.put(key, asset.reading.model_dump())
                    chunks.extend(reading_chunks(asset))
                    if not asset.reading.readable:
                        asset.error = asset.reading.limitations or '视觉模型未能可靠识别此页。'
                    elif not asset.reading.elements:
                        asset.error = '此候选页没有识别到可用的图像或表格。'
                except (LLMError, OSError, ValueError) as exc:
                    asset.error = str(exc)
                if asset.error:
                    paper.visual_limitations.append(f'第 {asset.page} 页：{asset.error}')
        if paper.read_status == 'abstract_only' and any(c.source == 'vision' for c in chunks):
            paper.read_status = 'visual_only'
    except (ToolError, OSError, RuntimeError, ValueError) as exc:
        paper.visual_limitations.append(f'图表读取未完成：{exc}')
    return chunks


def verify_claim_images(claims: list[Claim], evidence: list[Evidence], papers: list[Paper]) -> None:
    """A text-only critic cannot upgrade a model transcription to original evidence."""
    by_id = {e.id: e for e in evidence}
    assets = {a.id: a for p in papers for a in p.assets}
    pending, required = [], {}
    for claim in claims:
        linked = [by_id[e] for e in claim.evidence_ids if e in by_id and by_id[e].source in {'vision', 'table'}]
        if not linked or claim.status == 'insufficient':
            continue
        if not vision_settings.enabled or any(e.asset_id not in assets for e in linked):
            claim.status, claim.reason = 'insufficient', '缺少原始图表或视觉模型，不能仅根据转录确认结论。'
            continue
        try:
            for e in linked:
                asset = assets[e.asset_id]
                if asset.paper_id != e.paper_id or asset.page != e.page:
                    raise ToolError('图表证据的论文或页码不匹配。')
                required[asset.id] = (asset, image_bytes(asset))
            pending.append(claim)
        except (ToolError, OSError) as exc:
            claim.status, claim.reason = 'insufficient', str(exc)
    if not pending:
        return
    try:
        review = agents.verify_visual([value[1] for value in required.values()], pending,
                                      [e for e in evidence if any(e.id in c.evidence_ids for c in pending)],
                                      list(required))
        rank = {'insufficient': 0, 'partial': 1, 'supported': 2}
        for claim in pending:
            verdicts = [v for v in review.verdicts if v.claim_id == claim.id]
            if len(verdicts) != 1:
                claim.status, claim.reason = 'insufficient', '原始图表核验结果缺失或重复。'
            else:
                verdict = verdicts[0]
                claim.status = min([claim.status, verdict.status], key=rank.get)
                claim.reason += ' 图表复核：' + verdict.reason
    except (LLMError, ValueError) as exc:
        for claim in pending:
            claim.status, claim.reason = 'insufficient', f'图表原图复核未完成：{exc}'
