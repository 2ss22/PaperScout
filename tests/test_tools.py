from io import BytesIO

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from research_agent.models import Paper, TextChunk
from research_agent.reader import extract_chunks, read_paper, retrieve
from research_agent.tools import ArxivClient, Cache, ToolError, parse_feed


FEED = b'''<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2412.06769v2</id><title>Continuous Thought</title><summary>Abstract.</summary><author><name>Author</name></author></entry>
<entry><id>http://arxiv.org/abs/2412.06769v1</id><title>Duplicate</title><summary>Old abstract.</summary></entry>
</feed>'''


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr("research_agent.tools.time.sleep", lambda _: None)


def test_parse_feed_deduplicates_versionless_id():
    papers = parse_feed(FEED)
    assert len(papers) == 1 and papers[0].id == "2412.06769v2"
    assert papers[0].authors == ["Author"]


def test_empty_feed_and_invalid_response_are_distinct():
    assert parse_feed(b'<feed xmlns="http://www.w3.org/2005/Atom"/>') == []
    with pytest.raises(ToolError):
        parse_feed(b'<html>Service unavailable</html>')
    with pytest.raises(ToolError):
        parse_feed(b'broken xml')


def test_arxiv_search_cache_avoids_network(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=FEED)
    client = ArxivClient(Cache(tmp_path), httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(client.search('all:"continuous thought"')) == 1
    assert len(client.search('all:"continuous thought"')) == 1
    assert len(calls) == 1
    assert calls[0].url.params["max_results"] == "5"


def test_retry_limit_and_no_failure_caching(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429)
    client = ArxivClient(Cache(tmp_path), httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ToolError, match="429"):
        client.search("all:test")
    assert len(calls) == 6  # three attempts per official endpoint
    assert not list(tmp_path.glob('*.json'))


def test_network_timeout_is_not_an_empty_result(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)
    client = ArxivClient(Cache(tmp_path), httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ToolError, match="ReadTimeout"):
        client.search("all:test")


def pdf_bytes(text=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    if text:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 750 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extract_real_pdf_keeps_page():
    text = "This test paper evaluates continuous reasoning on synthetic tasks."
    chunks = extract_chunks(pdf_bytes(text), "2401.00001v1")
    assert chunks[0].page == 1
    assert text in chunks[0].text


def test_empty_pdf_falls_back_to_labelled_abstract(tmp_path):
    paper = Paper(id="2401.00001v1", title="Paper", abstract="Known abstract")
    client = ArxivClient(Cache(tmp_path))
    client.pdf = lambda p: pdf_bytes()
    try:
        chunks = read_paper(paper, client)
        assert paper.read_status == "abstract_only"
        assert "OCR" in paper.read_error
        assert chunks[0].page is None and chunks[0].source == "abstract"
    finally:
        client.close()


def test_text_cache_skips_download(tmp_path):
    paper = Paper(id="2401.00001v1", title="Paper", abstract="Abstract")
    calls = []
    client = ArxivClient(Cache(tmp_path))
    def pdf(p):
        calls.append(p.id)
        return pdf_bytes("This test paper evaluates continuous reasoning on synthetic tasks.")
    client.pdf = pdf
    try:
        assert read_paper(paper, client)
        assert read_paper(paper, client)
        assert len(calls) == 1
    finally:
        client.close()


def test_bm25_retrieves_relevant_chunk():
    chunks = [TextChunk(id=str(i), paper_id="p", page=i + 1, text=text) for i, text in enumerate([
        "cats dogs animals wildlife", "latent reasoning compositional benchmark",
        "ocean geography weather patterns", "agriculture crop growth soil",
    ])]
    assert retrieve(chunks, "latent reasoning", top_k=1)[0].id == "1"
