"""Offline regression tests for evidence boundaries and the bounded search loop."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from research_agent import agents, llm, orchestrator
from research_agent.models import (Claim, Draft, DraftClaim, Evidence, Extraction, ExtractedFinding,
                                   Paper, Plan, Review, RunResult, Selection, Selections,
                                   TextChunk, Usage, Verdict)
from research_agent.orchestrator import add_extraction, review_claims, run_research
from research_agent.report import render_report
from research_agent.tools import ToolError

QUOTE = "The model was evaluated on a synthetic reasoning benchmark."


def sample_run():
    paper = Paper(id="2412.06769v1", title="Test paper", abstract=QUOTE, read_status="full_text")
    chunk = TextChunk(id="chunk", paper_id=paper.id, page=2, text=QUOTE)
    evidence = Evidence(id="E001", paper_id=paper.id, chunk_id=chunk.id, page=2, quote=QUOTE, source="pdf")
    run = RunResult(run_id="test", created_at="2026-09-28", question="问题", provider="test", model="test",
                    papers=[paper], chunks=[chunk], evidence=[evidence])
    return run


def test_extraction_discards_fabricated_quotes_and_ids():
    run = sample_run()
    run.evidence = []
    extraction = Extraction(findings=[
        ExtractedFinding(kind="method", text="合理", chunk_id="chunk", quote=QUOTE),
        ExtractedFinding(kind="method", text="编造", chunk_id="chunk", quote="This quote does not exist in this paper."),
        ExtractedFinding(kind="method", text="伪造引用", chunk_id="missing", quote=QUOTE),
    ])
    findings = add_extraction(run, extraction, run.chunks)
    assert len(findings) == len(run.evidence) == 1
    assert len(run.errors) == 2


def test_abstract_cannot_supply_experimental_findings():
    run = sample_run()
    run.evidence = []
    run.chunks[0].source, run.chunks[0].page = "abstract", None
    extraction = Extraction(findings=[ExtractedFinding(kind="dataset", text="不允许", chunk_id="chunk", quote=QUOTE)])
    assert add_extraction(run, extraction, run.chunks) == []


@pytest.mark.parametrize("case", ["missing_id", "wrong_quote", "wrong_page", "wrong_paper", "missing_verdict", "duplicate_verdict", "abstract"])
def test_structural_checks_override_positive_model_verdict(case):
    run = sample_run()
    claim = Claim(id="C001", text="结论", evidence_ids=["E001"], kind="conclusion")
    verdict = Verdict(claim_id="C001", status="supported", reason="模型说支持")
    review = Review(verdicts=[verdict], action="finish", reason="结束")
    if case == "missing_id":
        claim.evidence_ids.append("E999")
    elif case == "wrong_quote":
        run.evidence[0].quote = "This does not occur in the original paper."
    elif case == "wrong_page":
        run.evidence[0].page = 5
    elif case == "wrong_paper":
        run.evidence[0].paper_id = "2412.00000v1"
    elif case == "missing_verdict":
        review.verdicts = []
    elif case == "duplicate_verdict":
        review.verdicts.append(verdict)
    elif case == "abstract":
        run.evidence[0].source = run.chunks[0].source = "abstract"
        run.evidence[0].page = run.chunks[0].page = None
    review_claims([claim], run.evidence, run.chunks, review)
    assert claim.status == "insufficient"


def test_semantic_rejection_even_when_quote_exists():
    run = sample_run()
    claim = Claim(id="C001", text="该模型在所有任务上都最优", evidence_ids=["E001"])
    review = Review(verdicts=[Verdict(claim_id="C001", status="insufficient", reason="原文只说明做了评测，没有比较结果")],
                    action="search", reason="缺少比较", query="all:reasoning")
    review_claims([claim], run.evidence, run.chunks, review)
    assert claim.status == "insufficient"
    assert "没有比较结果" in claim.reason


@pytest.fixture
def pipeline(monkeypatch):
    class Client:
        def __init__(self):
            self.queries = []
        def search(self, query, max_results=5):
            self.queries.append(query)
            # Duplicate first paper between rounds; selection must see only new papers.
            ids = [1, 2, 3] if len(self.queries) <= 2 else [1, 4, 5, 6]
            return [Paper(id=f"2401.{i:05d}v1", title=f"Paper {i}", abstract=QUOTE) for i in ids]
    client = Client()
    monkeypatch.setattr(orchestrator, 'read_visuals', lambda *a: [])
    monkeypatch.setattr(agents, "plan", lambda q: Plan(queries=["all:latent", "all:continuous"]))
    monkeypatch.setattr(agents, "select", lambda q, ps, limit: Selections(papers=[
        Selection(paper_id=p.id, reason="相关") for p in ps[:limit]]))
    def read(p, client):
        p.read_status = "full_text"
        return [TextChunk(id=p.id + ":p1", paper_id=p.id, page=1, text=QUOTE)]
    monkeypatch.setattr(orchestrator, "read_paper", read)
    monkeypatch.setattr(agents, "extract", lambda q, p, cs: Extraction(findings=[
        ExtractedFinding(kind="method", text="在合成基准评测", chunk_id=cs[0].id, quote=QUOTE)]))
    monkeypatch.setattr(agents, "write", lambda q, ps, fs, es: Draft(claims=[
        DraftClaim(text="在合成基准评测", kind="method", evidence_ids=[e.id]) for e in es], gaps=[]))
    def review(q, cs, es, ps, allow_search):
        return Review(verdicts=[Verdict(claim_id=c.id, status="supported" if not allow_search else "partial", reason="范围有限") for c in cs],
                      action="search", query="all:compositional", reason="仍需补充泛化证据")
    monkeypatch.setattr(agents, "critique", review)
    return client


def test_one_supplement_five_papers_and_final_review(pipeline):
    events = list(run_research("测试问题", client=pipeline))
    run = events[-1].data
    assert run.supplemental_rounds == 1
    assert len(run.papers) == 5
    assert len(pipeline.queries) == 3
    assert len(run.drafts) == 2
    assert all(c.status == "supported" for c in run.claims)
    assert "上限" in run.stop_reason
    assert sum(e.stage == "supplement" for e in events) == 1
    assert "#page=1" in run.report
    assert "E001" in run.report
    # Round-trip exported JSON; no computed fields or internal clients leak.
    restored = RunResult.model_validate_json(run.model_dump_json())
    assert restored.report == run.report
    assert restored.trace[-1].stage == "stop"


def test_finish_skips_supplement(pipeline, monkeypatch):
    monkeypatch.setattr(agents, "critique", lambda q, cs, es, ps, allow_search:
                        Review(verdicts=[Verdict(claim_id=c.id, status="supported", reason="支持") for c in cs],
                               action="finish", reason="证据足够"))
    run = list(run_research("问题", client=pipeline))[-1].data
    assert run.supplemental_rounds == 0
    assert len(run.papers) == 3
    assert run.status == "complete"


def test_duplicate_query_does_not_loop(pipeline, monkeypatch):
    monkeypatch.setattr(agents, "critique", lambda *a: Review(verdicts=[], action="search", query="all:latent", reason="再搜"))
    run = list(run_research("问题", client=pipeline))[-1].data
    assert run.supplemental_rounds == 0
    assert "重复" in run.stop_reason
    assert all(c.status == "insufficient" for c in run.claims)


@pytest.mark.parametrize("fails", [False, True])
def test_empty_and_failed_search_are_distinct(pipeline, monkeypatch, fails):
    def search(*a, **kw):
        if fails:
            raise ToolError("检索网络失败")
        return []
    monkeypatch.setattr(pipeline, "search", search)
    run = list(run_research("问题", client=pipeline))[-1].data
    assert not run.papers and not run.claims
    assert bool(run.errors) is fails
    assert run.status == ("failed" if fails else "partial")
    if fails:
        assert run.supplemental_rounds == 0
    assert "不存在" in run.report


def test_model_failure_exports_partial_run(pipeline, monkeypatch):
    def broken(*a):
        raise llm.LLMError("服务暂时不可用")
    monkeypatch.setattr(agents, "critique", broken)
    run = list(run_research("问题", client=pipeline))[-1].data
    assert run.status == "failed"
    assert run.papers
    assert all(c.status == "insufficient" for c in run.claims)
    assert "未完成核验" in run.report


def test_report_withholds_unsupported_claim_and_escapes_model_html():
    run = sample_run()
    run.claims = [Claim(id="C001", text='<script>alert(1)</script> [伪链接](https://evil.test)',
                        evidence_ids=["E001"], status="insufficient", reason="不支持")]
    run.queries = ['<script>alert(2)</script>']
    report = render_report(run)
    assert "<script>" not in report
    assert "https://evil.test" not in report
    assert "待核验说法（不作为结论）" in report
    assert "PDF 第 2 页" in report


def test_structured_output_validation_retries_once(monkeypatch):
    outputs = iter([{"queries": "wrong type"}, {"queries": ["all:reasoning"]}])
    monkeypatch.setattr(agents, "chat_json", lambda *a: next(outputs))
    assert agents.plan("问题").queries == ["all:reasoning"]


def test_usage_is_scoped_and_missing_usage_not_fabricated(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    usage = Usage()
    with llm.track_usage(usage):
        assert llm.chat([]) == "ok"
    llm.chat([])
    assert usage.calls == 1
    assert usage.prompt_tokens == 10
    response.usage = None
    with llm.track_usage(usage):
        llm.chat([])
    assert usage.calls == 2 and usage.calls_with_usage == 1


def test_supplement_failure_preserves_first_review(pipeline, monkeypatch):
    original = pipeline.search
    def search(query, **kwargs):
        if query == 'all:compositional':
            raise ToolError('补搜服务失败')
        return original(query, **kwargs)
    monkeypatch.setattr(pipeline, 'search', search)
    run = list(run_research('问题', client=pipeline))[-1].data
    assert run.supplemental_rounds == 1
    assert len(run.papers) == 3
    assert all(c.status == 'partial' for c in run.claims)
    assert '补搜请求失败' in run.stop_reason
    assert run.status == 'partial'


def test_invalid_selections_cannot_inject_papers(pipeline, monkeypatch):
    monkeypatch.setattr(agents, 'select', lambda *a: Selections(papers=[
        Selection(paper_id='invented-id', reason='伪造论文')]))
    run = list(run_research('问题', client=pipeline))[-1].data
    assert not run.papers
    assert not run.evidence
    assert len(run.candidates) > 0


def test_report_preserves_urls_inside_original_quotes():
    run = sample_run()
    run.evidence[0].quote = 'The code is available at https://example.org/code for reproduction.'
    report = render_report(run)
    assert 'https://example\\.org/code' in report
    assert '（链接略）' not in report
