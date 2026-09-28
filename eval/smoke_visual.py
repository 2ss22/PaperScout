"""Live, labelled component test for figure/table reading and original-image review."""
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4

from research_agent import agents
from research_agent.config import settings
from research_agent.llm import track_usage
from research_agent.models import Paper, RunResult, Trace
from research_agent.orchestrator import add_extraction, create_claims, review_claims
from research_agent.reader import read_paper, retrieve_context
from research_agent.report import render_report
from research_agent.artifacts import evidence_bundle
from research_agent.tools import ArxivClient
from research_agent.visuals import read_visuals, verify_claim_images


def main():
    started = time.monotonic()
    client = ArxivClient()
    run = RunResult(run_id=uuid4().hex,created_at=datetime.now(timezone.utc).isoformat(),
                    question='结合论文的图像和表格，Coconut 与 CoT 在 GSM8k、ProntoQA 和 ProsQA 上的准确率和推理方式有何不同？',
                    provider=settings.provider,model=settings.model)
    p = Paper(id='2412.06769v4',title='Training Large Language Models to Reason in a Continuous Latent Space',abstract='',
              reason='指定论文的图表组件验证，不是自主搜索结果。')
    run.papers = [p]
    try:
        print('读取 PDF 文本与图表…',flush=True)
        chunks = read_paper(p,client)
        with track_usage(run.usage):
            chunks.extend(read_visuals(p,client,'Coconut CoT GSM8k ProntoQA ProsQA accuracy Table Figure',max_pages=2))
            run.chunks = chunks
            print('图表页:',len(p.assets),'视觉成功:',sum(a.reading is not None for a in p.assets),'限制:',p.visual_limitations,flush=True)
            context = retrieve_context(chunks,'Coconut CoT accuracy reasoning Figure Table')
            extracted = agents.extract_groups(run.question,p,context)
            run.errors.extend(extracted.warnings)
            findings = add_extraction(run,extracted,context)
            draft = agents.write(run.question,run.papers,findings,run.evidence)
            run.claims = create_claims(draft)
            run.gaps = draft.gaps
            run.drafts = [[c.model_copy(deep=True) for c in run.claims]]
            print('对照原图核验…',flush=True)
            review = agents.critique(run.question,run.claims,run.evidence,run.papers,False)
            review_claims(run.claims,run.evidence,run.chunks,review)
            verify_claim_images(run.claims,run.evidence,run.papers)
        run.status = 'complete' if run.claims and not run.errors and all(c.status == 'supported' for c in run.claims) else 'partial'
        run.stop_reason = '图表组件验证完成。' + review.reason
    except Exception as exc:
        run.status='failed'
        run.errors.append(str(exc))
        run.stop_reason='图表组件测试中断。'
    finally:
        client.close()
    run.elapsed_seconds=round(time.monotonic()-started,3)
    run.trace=[Trace(stage='visual_smoke',message=run.stop_reason,elapsed_seconds=run.elapsed_seconds,
                     details={'autonomous_search':False})]
    run.report=render_report(run)
    prefix=Path('eval/runs')/f'visual_smoke_{run.run_id[:8]}'
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(run.model_dump_json(indent=2))
    prefix.with_suffix('.md').write_text(run.report)
    prefix.with_suffix('.zip').write_bytes(evidence_bundle(run))
    print(run.status,run.usage.model_dump(),len(run.evidence),'evidence',len(run.claims),'claims',run.errors,flush=True)
    print(prefix,flush=True)


if __name__=='__main__':
    main()
