"""Bounded research state machine. A critic can request exactly one new search."""
from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Iterator
from uuid import uuid4

from . import agents
from .config import settings
from .llm import LLMError, track_usage
from .models import (Claim, Draft, Evidence, Event, Extraction, Paper, Review,
                     RunResult, TextChunk, Trace)
from .reader import normalize_text, quote_matches, read_paper, retrieve_context
from .visuals import read_visuals, verify_claim_images
from .report import render_report
from .tools import ArxivClient, ToolError, base_id


def add_extraction(run: RunResult, extraction: Extraction,
                   chunks: list[TextChunk]) -> list[dict]:
    available = {c.id: c for c in chunks}
    findings = []
    for item in extraction.findings:
        chunk = available.get(item.chunk_id)
        if chunk is None:
            run.errors.append(f'拒绝抽取结果：文本块 {item.chunk_id} 不存在于本次阅读范围；判断：{item.text[:100]}。')
            continue
        if not quote_matches(item.quote, chunk.text):
            run.errors.append(f'拒绝抽取结果：论文 {chunk.paper_id}，PDF 第 {chunk.page} 页，'
                              f'文本块 {chunk.id} 的引文不是连续原文；待核对引文：{item.quote[:200]}。')
            continue
        if chunk.source == "abstract" and item.kind not in {"problem", "method"}:
            run.errors.append(f'拒绝抽取结果：论文 {chunk.paper_id} 的摘要不能支持 {item.kind} 实验细节。')
            continue
        quote = item.quote
        existing = next((e for e in run.evidence if e.chunk_id == chunk.id and
                         normalize_text(e.quote) == normalize_text(quote)), None)
        if existing is None:
            existing = Evidence(id=f"E{len(run.evidence) + 1:03d}", paper_id=chunk.paper_id,
                                chunk_id=chunk.id, page=chunk.page, quote=quote, source=chunk.source,
                                asset_id=chunk.asset_id)
            run.evidence.append(existing)
        findings.append({"kind": item.kind, "text": item.text, "evidence_id": existing.id,
                         "paper_id": chunk.paper_id})
    return findings


def create_claims(draft: Draft) -> list[Claim]:
    return [Claim(id=f"C{i:03d}", text=c.text, kind=c.kind, focus=c.focus, priority=c.priority, scope=c.scope,
                  evidence_ids=list(dict.fromkeys(c.evidence_ids)))
            for i, c in enumerate(draft.claims, 1)]


def review_claims(claims: list[Claim], evidence: list[Evidence], chunks: list[TextChunk],
                  review: Review) -> list[Claim]:
    evidence_by_id = {e.id: e for e in evidence}
    chunk_by_id = {c.id: c for c in chunks}
    verdicts: dict[str, list] = {}
    for verdict in review.verdicts:
        verdicts.setdefault(verdict.claim_id, []).append(verdict)
    for claim in claims:
        claim.evidence_assessments = []
        reason = ""
        if not claim.evidence_ids or any(e not in evidence_by_id for e in claim.evidence_ids):
            reason = "引用缺失或包含不存在的证据 ID。"
        else:
            for eid in claim.evidence_ids:
                e = evidence_by_id[eid]
                c = chunk_by_id.get(e.chunk_id)
                if (c is None or c.paper_id != e.paper_id or c.page != e.page or
                        c.source != e.source or not quote_matches(e.quote, c.text)):
                    reason = "证据与原始文本块、页码或论文不匹配。"
                if c is not None and c.asset_id != e.asset_id:
                    reason = '图表证据的图片关联不一致。'
                if e.source == "abstract" and claim.kind not in {"problem", "method"}:
                    reason = "摘要不能用于支持本报告的实验细节或实验结论。"
        matches = verdicts.get(claim.id, [])
        if reason:
            claim.status, claim.reason = "insufficient", reason
        elif len(matches) != 1:
            claim.status, claim.reason = "insufficient", "审阅结果缺失或重复，未能完成核验。"
        else:
            claim.status, claim.reason = matches[0].status, matches[0].reason
            assessments = matches[0].evidence_assessments
            # Relations may only describe this claim's actual evidence; reject
            # duplicate/conflicting annotations instead of choosing one silently.
            claim.evidence_assessments = [a for a in assessments
                if a.evidence_id in claim.evidence_ids and a.evidence_id in evidence_by_id
                and sum(b.evidence_id == a.evidence_id for b in assessments) == 1]
    return claims


def run_research_legacy(question: str, *, client: ArxivClient | None = None, visual_pages: int | None = None) -> Iterator[Event]:
    started = time.monotonic()
    run = RunResult(run_id=uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
                    question=question.strip(), provider=settings.provider, model=settings.model)
    own_client = client is None
    client = client or ArxivClient()
    findings: list[dict] = []

    def event(stage: str, message: str, **details) -> Event:
        run.trace.append(Trace(stage=stage, message=message,
                               elapsed_seconds=round(time.monotonic() - started, 3), details=details))
        return Event(stage=stage, message=message, data=details or None)

    def call(function, *args, **kwargs):
        # Context only spans a call, never a generator yield / another session.
        with track_usage(run.usage):
            return function(*args, **kwargs)

    try:
        if not run.question:
            raise LLMError("请输入研究问题。")
        yield event("plan", "正在将研究问题转换为英文检索计划…")
        queries = call(agents.plan, run.question).queries[:2]
        yield event("plan", "检索计划已生成。", queries=queries)
        for round_no in range(2):
            seen = {base_id(p.id) for p in run.papers}
            candidates: list[Paper] = []
            search_failures = 0
            for query in queries:
                run.queries.append(query)
                yield event("search", f"正在检索：{query}", round=round_no)
                try:
                    found = client.search(query, max_results=5)
                    diagnostics = getattr(client, 'search_diagnostics', [])
                    if any(d.get('source') == 'arxiv_html' for d in diagnostics):
                        yield event('search', '已自动切换到 arXiv 官方搜索页面，继续筛选论文。', attempts=diagnostics)
                    yield event("search", f"本次查询返回 {len(found)} 篇候选论文。",
                                paper_ids=[p.id for p in found], attempts=diagnostics)
                    for paper in found:
                        if not any(base_id(p.id) == base_id(paper.id) for p in run.candidates):
                            run.candidates.append(paper.model_copy(deep=True))
                        if base_id(paper.id) not in seen:
                            candidates.append(paper)
                            seen.add(base_id(paper.id))
                except (ToolError, OSError) as exc:
                    search_failures += 1
                    run.errors.append(str(exc))
                    yield event("warning", str(exc))
            # Rephrasing a query cannot fix a service-wide transport failure.
            # Do not spend model calls judging a fictitious evidence gap.
            if search_failures == len(queries):
                if not run.papers:
                    run.status = "failed"
                    run.stop_reason = "检索请求全部失败，请检查网络或稍后重试；不能据此判断相关论文不存在。"
                else:
                    run.stop_reason = "补搜请求失败；保留已有证据与审阅结果，剩余缺口仍未知。"
                break
            limit = min(3 if round_no == 0 else 2, 5 - len(run.papers))
            if candidates and limit:
                selection = call(agents.select, run.question, candidates, limit)
                available = {p.id: p for p in candidates}
                selected: set[str] = set()
                for choice in selection.papers:
                    if choice.paper_id not in available or choice.paper_id in selected:
                        continue
                    if len(selected) >= limit:
                        break
                    selected.add(choice.paper_id)
                    paper = available[choice.paper_id]
                    paper.reason = choice.reason
                    run.papers.append(paper)
                    yield event("read", f"正在阅读：{paper.title}", reason=paper.reason)
                    chunks = read_paper(paper, client)
                    yield event('visual', f'正在检索图像与表格：{paper.title}')
                    chunks.extend(call(read_visuals, paper, client, ' '.join(run.queries), visual_pages))
                    run.chunks.extend(chunks)
                    context = retrieve_context(chunks, " ".join(run.queries) +
                                               " method dataset benchmark accuracy evaluation results")
                    yield event('extract', f'正在抽取正文与图表证据：{paper.title}', paper_id=paper.id,
                                chunk_ids=[c.id for c in context])
                    extraction = call(agents.extract_groups, run.question, paper, context)
                    for warning in getattr(extraction, 'warnings', []):
                        message = f'{paper.title}：{warning}'
                        run.errors.append(message)
                        yield event('warning', message, paper_id=paper.id, recoverable=True)
                    findings.extend(add_extraction(run, extraction, context))
                    yield event("read", f"阅读完成：{paper.title}", status=paper.read_status,
                                limitation=paper.read_error, evidence_count=len(run.evidence),
                                visual_pages=len(paper.assets), visual_limitations=paper.visual_limitations)
            yield event("write", "正在基于已匹配的原文证据组织报告…")
            draft = call(agents.write, run.question, run.papers, findings, run.evidence) if run.evidence else Draft(claims=[], gaps=[])
            run.claims = create_claims(draft)
            run.gaps = draft.gaps
            run.drafts.append([c.model_copy(deep=True) for c in run.claims])
            yield event("critique", "正在核验每条结论与原文的支持关系…")
            review = call(agents.critique, run.question, run.claims, run.evidence, run.papers, round_no == 0)
            review_claims(run.claims, run.evidence, run.chunks, review)
            if any(e.source in {'vision', 'table'} for e in run.evidence):
                yield event('critique', '正在对照原始图表复核结论与表格数字…')
                before_images = {c.id: c.status for c in run.claims}
                call(verify_claim_images, run.claims, run.evidence, run.papers)
                if any(c.status != before_images[c.id] for c in run.claims):
                    yield event('critique', '原图复核发现支持范围变化，重新判断是否需要补搜…')
                    # Reconsider routing, without allowing the text reviewer to
                    # upgrade any original-image verdict.
                    review = call(agents.critique, run.question, run.claims, run.evidence, run.papers, round_no == 0)
            counts = {status: sum(c.status == status for c in run.claims)
                      for status in ("supported", "partial", "insufficient")}
            yield event("critique", review.reason, action=review.action, counts=counts)
            if review.action == "finish":
                run.stop_reason = review.reason or "审阅者判断可以结束；未解决问题保留未知。"
                break
            if round_no == 1:
                run.stop_reason = "已达到一次补搜上限；剩余缺口保留未知。"
                break
            query = review.query.strip()[:500]
            if not query or query.casefold() in {q.casefold() for q in run.queries}:
                run.stop_reason = "补搜未提供新的有效查询；停止重复检索并保留未知。"
                break
            run.supplemental_rounds = 1
            queries = [query]
            yield event("supplement", f"启动唯一一次补搜：{review.reason}", query=query)
        if run.status != "failed":
            run.status = "complete" if run.claims and all(c.status == "supported" for c in run.claims) else "partial"
            if run.errors or any(p.read_status != "full_text" for p in run.papers):
                run.status = "partial"
    except (LLMError, ToolError, ValueError, OSError) as exc:
        run.errors.append(str(exc))
        run.status = "failed"
        run.stop_reason = "运行中断；已获得的资料保留，未完成核验的说法不作为结论。"
        for claim in run.claims:
            if claim.status == "unreviewed":
                claim.status, claim.reason = "insufficient", "本次运行未完成核验。"
        yield event("error", str(exc))
    finally:
        if own_client:
            client.close()
    run.elapsed_seconds = round(time.monotonic() - started, 3)
    run.report = render_report(run)
    yield event("stop", run.stop_reason)
    yield Event(stage="done", message="报告与运行记录已生成。", data=run)


def run_research(question: str = '', *, client: ArxivClient | None = None,
                 visual_pages: int | None = None, run_id: str | None = None,
                 resume: bool = False, approve_papers: bool = False,
                 selection: list[str] | None = None, checkpoint_path=None,
                 engine: str = 'langgraph') -> Iterator[Event]:
    if engine == 'legacy':
        if resume:
            raise ValueError('旧版编排不支持检查点恢复。')
        yield from run_research_legacy(question, client=client, visual_pages=visual_pages)
        return
    if engine != 'langgraph':
        raise ValueError('未知编排引擎。')
    from .graph import run_graph
    yield from run_graph(question, client=client, visual_pages=visual_pages, run_id=run_id,
                         resume=resume, approve_papers=approve_papers, selection=selection,
                         checkpoint_path=checkpoint_path)
