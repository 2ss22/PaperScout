"""Regression tests for the screenshot failure and pixel-grounded evidence."""
from __future__ import annotations

import hashlib
from io import BytesIO
import zipfile

import httpx
import pymupdf
import pytest

from research_agent import agents, visuals
from research_agent.artifacts import evidence_bundle, table_html
from research_agent.models import (Claim, Evidence, Paper, TableCell, TextChunk, VisualAsset,
                                   VisualElement, VisualReading, VisualReview, Verdict)
from research_agent.search import compile_query, parse_search_html, plain_query
from research_agent.tools import ArxivClient, Cache, ToolError
from research_agent.reader import retrieve_context
from tests.test_pipeline import sample_run

HTML = b'''<html><li class="arxiv-result"><p class="list-title"><a href="https://arxiv.org/abs/2501.00001">arXiv:2501.00001</a></p><p class="title">Visual reasoning</p><p class="authors"><a>A. Author</a></p><span class="abstract-full">The paper studies reasoning. <a>less</a></span></li></html>'''


def test_old_grouped_queries_are_compiled_to_fielded_terms():
    old = 'all:("vision language model" OR "multimodal model") AND abs:("reasoning")'
    compiled = compile_query(old)
    assert ':(' not in compiled
    assert compiled == 'all:"vision language model" OR all:"multimodal model" OR all:"reasoning"'
    assert plain_query(old) == 'vision language model'
    assert compile_query('multimodal reasoning') == 'all:"multimodal" AND all:"reasoning"'


def test_406_recovers_to_official_html_and_remembers_api_failure(tmp_path, monkeypatch):
    monkeypatch.setattr('research_agent.tools.time.sleep', lambda _: None)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(406) if request.url.host == 'export.arxiv.org' else httpx.Response(200, content=HTML)
    client = ArxivClient(Cache(tmp_path), httpx.Client(transport=httpx.MockTransport(handler)))
    first = client.search('multimodal reasoning')
    assert first[0].search_source == 'arxiv_html'
    assert any(d['source'] == 'arxiv_html' for d in client.search_diagnostics)
    client.search('continuous thinking')
    assert len([r for r in calls if r.url.host == 'export.arxiv.org']) == 1
    assert len(calls) == 3


def test_html_empty_results_differ_from_block_pages():
    assert parse_search_html(b'<html>Sorry, your query: test produced no results</html>') == []
    with pytest.raises(ToolError):
        parse_search_html(b'<html>Please verify you are human</html>')
    papers = parse_search_html(HTML)
    assert papers[0].abstract == 'The paper studies reasoning.'
    assert papers[0].authors == ['A. Author']


def fixture_pdf():
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=600)
    page.insert_text((50, 50), 'Table 1. Complex metrics for visual reasoning', fontsize=12)
    for x in (50, 180, 300, 420):
        page.draw_line((x, 100), (x, 220))
    for y in (100, 140, 180, 220):
        page.draw_line((50, y), (420, y))
    for x, y, text in [(60,125,'Model'),(190,125,'Accuracy'),(310,125,'Latency'),
                        (60,165,'Alpha'),(190,165,'90.5'),(310,165,'12'),
                        (60,205,'Beta'),(190,205,'87.0'),(310,205,'8')]:
        page.insert_text((x,y), text, fontsize=11)
    page.draw_line((50,300),(200,250),color=(0,0,1),width=2)
    page.insert_text((50,320),'Figure 1. Increasing performance.',fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def reading():
    return VisualReading(readable=True, elements=[VisualElement(kind='table', label='Table 1',
        description='Alpha 的准确率为 90.5，Beta 为 87.0。', visible_text='Alpha Accuracy 90.5 Beta Accuracy 87.0',
        cells=[TableCell(row=0,column=0,text='Model',row_span=2,header=True),
               TableCell(row=0,column=1,text='Metrics',column_span=2,header=True),
               TableCell(row=1,column=1,text='Accuracy',header=True),
               TableCell(row=1,column=2,text='Latency',header=True),
               TableCell(row=2,column=0,text='Alpha'),TableCell(row=2,column=1,text='90.5'),
               TableCell(row=2,column=2,text='12')])])


def test_visual_reading_indexes_original_pages_cells_and_cache(tmp_path, monkeypatch):
    client = ArxivClient(Cache(tmp_path))
    client.pdf = lambda p: fixture_pdf()
    calls = []
    monkeypatch.setattr(visuals, 'vision_settings', type('Settings', (), dict(enabled=True,max_pages=2,model='test',base_url='test'))())
    def fake(image, query, page):
        calls.append(page)
        assert image.startswith(b'\x89PNG')
        return reading()
    monkeypatch.setattr(agents, 'read_visual', fake)
    p = Paper(id='2501.00001v1',title='Test',abstract='',read_status='full_text')
    try:
        data = fixture_pdf()
        client.pdf = lambda p: data
        chunks = visuals.read_visuals(p, client, 'visual reasoning accuracy')
        assert p.assets and p.assets[0].reading
        assert len(p.assets) == 1
        assert any(c.source == 'table' for c in chunks)
        v = next(c for c in chunks if c.source == 'vision')
        assert v.asset_id == p.assets[0].id and '90.5' in v.text
        assert v.cells[1].column_span == 2
        assert visuals.image_bytes(p.assets[0]).startswith(b'\x89PNG')
        visuals.read_visuals(p, client, 'visual reasoning accuracy')
        assert len(calls) == 1
    finally:
        client.close()


def test_table_html_preserves_spans_and_escapes_content():
    cells = reading().elements[0].cells
    cells[0].text = '<script>unsafe</script>'
    rendered = table_html(cells)
    assert 'rowspan="2"' in rendered and 'colspan="2"' in rendered
    assert '<script>' not in rendered
    with pytest.raises(ValueError):
        table_html([TableCell(row=0,column=0,text='a',column_span=2),TableCell(row=0,column=1,text='overlap')])


def test_retrieval_reserves_space_for_visual_chunks():
    chunks = [TextChunk(id=str(i),paper_id='p',page=1,text='reasoning method model') for i in range(20)]
    chunks.append(TextChunk(id='v',paper_id='p',page=2,text='complex table accuracy',source='vision',asset_id='asset'))
    result = retrieve_context(chunks, 'reasoning method model')
    assert any(c.source == 'vision' for c in result)
    assert len(result) <= 8


def image_run(tmp_path):
    run = sample_run()
    data = b'test-image-bytes'
    path = tmp_path / 'image.png'
    path.write_bytes(data)
    asset = VisualAsset(id='a',paper_id=run.papers[0].id,page=2,image_path=str(path),
                        image_sha256=hashlib.sha256(data).hexdigest(),pdf_sha256='pdfhash',reading=reading())
    run.papers[0].assets = [asset]
    run.evidence[0].source = 'vision'
    run.evidence[0].asset_id = 'a'
    run.claims = [Claim(id='C001',text='准确率为 100%',evidence_ids=['E001'],status='supported')]
    return run


def test_original_image_review_can_downgrade_text_critic(tmp_path, monkeypatch):
    run = image_run(tmp_path)
    monkeypatch.setattr(visuals, 'vision_settings', type('S',(),{'enabled':True})())
    def review(images, claims, evidence, ids):
        assert images == [b'test-image-bytes'] and ids == ['a']
        return VisualReview(verdicts=[Verdict(claim_id='C001',status='insufficient',reason='原图为 90.5 而不是 100')])
    monkeypatch.setattr(agents,'verify_visual',review)
    visuals.verify_claim_images(run.claims,run.evidence,run.papers)
    assert run.claims[0].status == 'insufficient'
    assert '90.5' in run.claims[0].reason


def test_missing_or_tampered_image_cannot_confirm_claim(tmp_path,monkeypatch):
    run = image_run(tmp_path)
    monkeypatch.setattr(visuals, 'vision_settings', type('S',(),{'enabled':True})())
    Path = __import__('pathlib').Path
    Path(run.papers[0].assets[0].image_path).write_bytes(b'tampered')
    visuals.verify_claim_images(run.claims,run.evidence,run.papers)
    assert run.claims[0].status == 'insufficient'


def test_bundle_contains_original_image_and_portable_path(tmp_path):
    import json
    run = image_run(tmp_path)
    with zipfile.ZipFile(BytesIO(evidence_bundle(run))) as archive:
        assert archive.read('images/paper-1-page-2.png') == b'test-image-bytes'
        exported = json.loads(archive.read('run.json'))
        assert exported['papers'][0]['assets'][0]['image_path'] == 'images/paper-1-page-2.png'


def test_multimodal_prompt_contains_pixels(monkeypatch):
    sent = []
    def fake(messages, **kwargs):
        sent.append((messages,kwargs))
        return reading().model_dump()
    monkeypatch.setattr(agents,'chat_json',fake)
    agents.read_visual(b'PNG', 'question', 1)
    messages,kwargs=sent[0]
    assert kwargs['vision'] is True
    assert messages[1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')


def test_extraction_budgets_keep_visual_findings(monkeypatch):
    from research_agent.models import Extraction, ExtractedFinding
    calls = []
    def extract(q,p,chunks):
        calls.append({c.source for c in chunks})
        return Extraction(findings=[ExtractedFinding(kind='method', text='finding',
                           quote='an original source quote', chunk_id=chunks[0].id) for _ in range(8)])
    monkeypatch.setattr(agents,'extract',extract)
    chunks=[TextChunk(id='text',paper_id='p',page=1,text='body'),
            TextChunk(id='image',paper_id='p',page=1,text='transcription',source='vision',asset_id='asset')]
    result=agents.extract_groups('question',Paper(id='p',title='paper',abstract=''),chunks)
    assert calls == [{'pdf'}, {'vision'}]
    assert len(result.findings) == 8
    assert sum(f.chunk_id == 'image' for f in result.findings) == 4


def test_missing_vision_model_never_affirms_transcription(tmp_path,monkeypatch):
    run=image_run(tmp_path)
    monkeypatch.setattr(visuals,'vision_settings',type('S',(),{'enabled':False})())
    visuals.verify_claim_images(run.claims,run.evidence,run.papers)
    assert run.claims[0].status == 'insufficient'


def test_report_labels_visual_transcription_and_page(tmp_path):
    from research_agent.report import render_report
    run=image_run(tmp_path)
    report=render_report(run)
    assert '图表转录（非正文逐字引文' in report
    assert '#page=2' in report
    assert '图表证据索引' in report


def test_scanned_page_can_use_vision_ocr(tmp_path,monkeypatch):
    import io
    from PIL import Image
    data=BytesIO()
    Image.new('RGB',(400,400),'white').save(data,format='PNG')
    doc=pymupdf.open()
    page=doc.new_page(width=400,height=400)
    page.insert_image(page.rect,stream=data.getvalue())
    pdf=doc.tobytes()
    doc.close()
    client=ArxivClient(Cache(tmp_path))
    client.pdf=lambda p: pdf
    monkeypatch.setattr(visuals,'vision_settings',type('S',(),dict(enabled=True,max_pages=1,model='test',base_url='test'))())
    monkeypatch.setattr(agents,'read_visual',lambda *a: reading())
    paper=Paper(id='2501.00001v1',title='scanned',abstract='',read_status='abstract_only')
    try:
        chunks=visuals.read_visuals(paper,client,'accuracy')
        assert any(c.source=='vision' for c in chunks)
        assert paper.read_status=='visual_only'
    finally:
        client.close()
