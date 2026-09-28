"""Live component smoke test using one explicit paper, not an autonomous-search demo.

Useful when the network allows arXiv pages/PDFs but blocks dynamic API queries.
python -m eval.smoke_reader
"""
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
import time
from uuid import uuid4

from research_agent import agents
from research_agent.config import settings
from research_agent.llm import track_usage
from research_agent.models import Paper, RunResult, Trace
from research_agent.orchestrator import add_extraction, create_claims, review_claims
from research_agent.reader import read_paper, retrieve
from research_agent.report import render_report
from research_agent.tools import ArxivClient


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and attrs.get('name', '').startswith('citation_'):
            self.values.setdefault(attrs['name'], []).append(attrs.get('content', ''))


def main():
    started = time.monotonic()
    client = ArxivClient()
    pid = '2412.06769'
    run = RunResult(run_id=uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
                    question='这篇论文提出了什么连续潜在推理方法？实验数据集、指标及结论分别是什么，哪些结论不能外推？',
                    provider=settings.provider, model=settings.model)
    try:
        print('读取指定论文的官方元数据；这不是自主搜索案例。', flush=True)
        page = client._request(f'https://arxiv.org/abs/{pid}').decode()
        metadata = Metadata()
        metadata.feed(page)
        versions = re.findall(re.escape(pid) + r'v(\d+)', page)
        if not versions:
            raise ValueError('未能确定版本，停止以避免记录不准确的证据来源。')
        versioned_id = pid + 'v' + str(max(map(int, versions)))
        abstract = re.search(r'<blockquote[^>]*class="abstract[^>]*>(.*?)</blockquote>', page, re.S)
        text = unescape(re.sub(r'<[^>]+>', '', abstract.group(1))).strip() if abstract else ''
        paper = Paper(id=versioned_id, title=metadata.values['citation_title'][0],
                      abstract=text, authors=metadata.values.get('citation_author', []),
                      reason='组件冒烟测试：显式指定的公开论文，不是搜索筛选结果。')
        run.papers = [paper]
        print('下载并解析', paper.id, paper.title, flush=True)
        chunks = read_paper(paper, client)
        run.chunks = chunks
        if paper.read_status != 'full_text':
            raise ValueError(paper.read_error)
        context = retrieve(chunks, 'continuous latent reasoning method training curriculum dataset benchmark accuracy results limitations', 6)
        with track_usage(run.usage):
            print('抽取与匹配原文证据…', flush=True)
            extracted = agents.extract(run.question, paper, context)
            findings = add_extraction(run, extracted, context)
            print('写作与语义核验…', flush=True)
            draft = agents.write(run.question, run.papers, findings, run.evidence)
            run.claims = create_claims(draft)
            run.drafts = [[c.model_copy(deep=True) for c in run.claims]]
            run.gaps = draft.gaps
            review = agents.critique(run.question, run.claims, run.evidence, run.papers, False)
            review_claims(run.claims, run.evidence, run.chunks, review)
        run.status = 'complete' if run.claims and all(c.status == 'supported' for c in run.claims) else 'partial'
        run.stop_reason = '指定论文的阅读组件测试完成；未验证自主检索。' + review.reason
    except Exception as exc:
        run.status = 'failed'
        run.errors.append(str(exc))
        run.stop_reason = '指定论文组件测试中断；不作为完整调研成功案例。'
    finally:
        client.close()
    run.elapsed_seconds = round(time.monotonic() - started, 3)
    run.trace.append(Trace(stage='component_smoke', message=run.stop_reason,
                           elapsed_seconds=run.elapsed_seconds, details={'autonomous_search': False}))
    run.report = render_report(run)
    root = Path('eval/runs')
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / ('reader_smoke_' + run.run_id[:8])
    prefix.with_suffix('.json').write_text(run.model_dump_json(indent=2))
    prefix.with_suffix('.md').write_text(run.report)
    print(run.status, len(run.evidence), '条证据', len(run.claims), '条结论', run.errors, flush=True)


if __name__ == '__main__':
    main()
