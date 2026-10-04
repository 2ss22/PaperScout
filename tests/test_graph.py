"""Real SQLite/LangGraph integration with deterministic network/model fixtures."""
import json
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
import pytest

from research_agent import agents, config, graph, llm
from research_agent.checkpoints import list_runs, load_run
from research_agent.models import (Draft, DraftClaim, Extraction, ExtractedFinding, Paper, Plan,
                                   Review, Selection, Selections, TextChunk, Verdict)
from research_agent.orchestrator import run_research

QUOTE = 'The model was evaluated on a synthetic reasoning benchmark.'


def call(name, **args):
    return {'id': uuid4().hex, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}


def assistant(*calls):
    return {'role': 'assistant', 'content': '完成本步。', 'reasoning_content': '保留推理字段',
            **({'tool_calls': list(calls)} if calls else {})}


@pytest.fixture
def flow(monkeypatch, tmp_path):
    # Neither real credentials nor external services are used in offline tests.
    monkeypatch.setattr(config, 'ENV_FILE', tmp_path / 'absent.env')
    monkeypatch.setenv('LLM_PROVIDER', 'deepseek')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-secret-never-persist')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'fixture-model')
    monkeypatch.setenv('PAPERSCOUT_CHECKPOINT_DB', str(tmp_path / 'runs.sqlite'))
    class Client:
        def __init__(self):
            self.queries, self.reads = [], []
        def search(self, query, max_results=5):
            self.queries.append(query)
            ids = [1, 2, 3] if query != 'supplemental reasoning' else [1, 4, 5, 6]
            return [Paper(id=f'2401.{i:05d}v1', title=f'Paper {i}', abstract=QUOTE) for i in ids]
    client = Client()
    monkeypatch.setattr(agents, 'plan', lambda q: Plan(queries=['latent reasoning', 'continuous reasoning']))
    monkeypatch.setattr(agents, 'select', lambda q, ps, limit: Selections(papers=[
        Selection(paper_id=p.id, reason='与问题相关') for p in ps[:limit]]))
    def read(p, c):
        c.reads.append(p.id)
        p.read_status = 'full_text'
        return [TextChunk(id=p.id + ':p1', paper_id=p.id, page=1, text=QUOTE)]
    monkeypatch.setattr(graph, 'read_paper', read)
    def choose(messages, schemas):
        user = json.loads(messages[1]['content'])
        previous = [m for m in messages if m['role'] == 'tool']
        if previous:
            assert all(json.loads(m['content'])['ok'] for m in previous)
            return assistant()
        if 'paper_id' in user:
            return assistant(call('retrieve_passages', paper_id=user['paper_id'], query='reasoning benchmark'))
        return assistant(*(call('search_papers', query=q) for q in user['queries']))
    monkeypatch.setattr(llm, 'chat_tools', choose)
    monkeypatch.setattr(agents, 'extract_groups', lambda q, p, cs: Extraction(findings=[
        ExtractedFinding(kind='method', text='在合成基准评测', chunk_id=cs[0].id, quote=QUOTE)] if cs else []))
    monkeypatch.setattr(agents, 'write', lambda q, ps, fs, es: Draft(claims=[
        DraftClaim(text='在合成基准评测', kind='method', evidence_ids=[e.id]) for e in es], gaps=[]))
    def review(q, cs, es, ps, allow_search):
        return Review(verdicts=[Verdict(claim_id=c.id, status='supported', reason='原文支持') for c in cs],
                      action='finish', reason='证据足够')
    monkeypatch.setattr(agents, 'critique', review)
    monkeypatch.setattr(graph, 'verify_claim_images', lambda *args: None)
    return client


def run(flow, **kwargs):
    return list(run_research('测试问题', client=flow, visual_pages=0, **kwargs))[-1].data


def test_native_graph_completes_and_exports_evidence(flow):
    result = run(flow)
    assert result.engine == 'langgraph'
    assert result.status == 'complete', result.errors
    assert len(result.papers) == len(result.evidence) == 3
    assert len(result.tool_calls) == 5
    assert all(c.status == 'supported' for c in result.claims)
    assert 'E001' in result.report and '#page=1' in result.report
    assert list_runs()[0].run_id == result.run_id
    assert load_run(result.run_id).report == result.report
    assert not result.resumable
    assert not any('offline-secret' in p.read_bytes().decode(errors='ignore')
                   for p in Path(graph.checkpoint_file()).parent.glob('*') if p.is_file())


def test_one_supplement_five_papers(flow, monkeypatch):
    def review(q, cs, es, ps, allow_search):
        return Review(verdicts=[Verdict(claim_id=c.id, status='partial', reason='范围有限') for c in cs],
                      action='search', query='supplemental reasoning', reason='需要补充')
    monkeypatch.setattr(agents, 'critique', review)
    result = run(flow)
    assert result.status == 'partial', result.errors
    assert result.supplemental_rounds == 1
    assert len(result.papers) == 5 and len(set(flow.reads)) == 5
    assert len(flow.queries) == 3
    assert len(result.drafts) == 2
    assert '上限' in result.stop_reason


def test_fresh_graph_resumes_failed_node_without_repeating_read_or_search(flow, monkeypatch):
    original = agents.extract_groups
    attempts = []
    def fail_second(q, p, cs):
        attempts.append(p.id)
        if len(attempts) == 2:
            llm._usage.get().calls += 1
            raise llm.LLMError('模拟服务临时故障')
        return original(q, p, cs)
    monkeypatch.setattr(agents, 'extract_groups', fail_second)
    failed = run(flow)
    assert failed.status == 'failed' and failed.resumable
    assert len(failed.evidence) == 1 and len(flow.reads) == 2
    queries = list(flow.queries)
    # run_research opens a fresh DB connection and compiles a new graph each time.
    resumed = run(flow, run_id=failed.run_id, resume=True)
    assert resumed.status == 'complete', resumed.errors
    assert len(resumed.evidence) == 3 and len(resumed.papers) == 3
    assert len(flow.reads) == 3 and flow.queries == queries
    assert attempts == ['2401.00001v1', '2401.00002v1', '2401.00002v1', '2401.00003v1']
    assert resumed.usage.calls == 1  # Includes the failed model request.
    assert len(resumed.tool_calls) == 5  # No duplicate completed tools.
    again = run(flow, run_id=failed.run_id, resume=True)
    assert again.report == resumed.report and len(flow.reads) == 3


def test_resume_between_two_pending_tools_does_not_reexecute_first(flow, monkeypatch):
    original = flow.search
    def fail(query, **kwargs):
        if query == 'continuous reasoning':
            raise RuntimeError('模拟进程在第二个工具前异常')
        return original(query, **kwargs)
    monkeypatch.setattr(flow, 'search', fail)
    failed = run(flow)
    assert failed.resumable and len(flow.queries) == 1
    monkeypatch.setattr(flow, 'search', original)
    resumed = run(flow, run_id=failed.run_id, resume=True)
    assert resumed.status == 'complete', resumed.errors
    assert flow.queries == ['latent reasoning', 'continuous reasoning']


def test_interrupt_confirmation_validated_before_resuming(flow):
    paused = run(flow, approve_papers=True)
    assert paused.status == 'paused' and paused.resumable
    assert not flow.reads
    assert paused.pending_selection['limit'] == 3
    with pytest.raises(ValueError, match='当前候选'):
        run(flow, run_id=paused.run_id, resume=True, selection=['invented'])
    assert load_run(paused.run_id).status == 'paused'
    resumed = run(flow, run_id=paused.run_id, resume=True, selection=['2401.00002v1'])
    assert resumed.status == 'complete', resumed.errors
    assert flow.reads == ['2401.00002v1'] and len(flow.queries) == 2
    assert resumed.papers[0].id == '2401.00002v1'


def test_same_run_lock_blocks_concurrent_resume(flow):
    paused = run(flow, approve_papers=True)
    path = graph.checkpoint_file().parent / f'{paused.run_id}.lock'
    with FileLock(path):
        with pytest.raises(ValueError, match='另一个会话'):
            run(flow, run_id=paused.run_id, resume=True, selection=[])


def test_resume_rejects_changed_model_but_allows_rotated_key(flow, monkeypatch):
    paused = run(flow, approve_papers=True)
    monkeypatch.setenv('DEEPSEEK_MODEL', 'different-model')
    with pytest.raises(ValueError, match='模型配置'):
        run(flow, run_id=paused.run_id, resume=True, selection=[])
    monkeypatch.setenv('DEEPSEEK_MODEL', 'fixture-model')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'new-secret')
    assert run(flow, run_id=paused.run_id, resume=True, selection=[]).status == 'partial'


def test_bad_tool_and_search_budget_cannot_bypass_executor(flow, monkeypatch):
    def choose(messages, schemas):
        return assistant(call('delete_file', path='/tmp/any'),
                         call('search_papers', query='latent reasoning'),
                         call('search_papers', query='continuous reasoning'))
    monkeypatch.setattr(llm, 'chat_tools', choose)
    # Stop before reader; inspect both rejection messages and actual network calls.
    result = run(flow, approve_papers=True)
    assert result.status == 'paused', result.errors
    assert flow.queries == ['latent reasoning']
    assert [c['ok'] for c in result.tool_calls] == [False, True, False]
    assert '预算' in result.tool_calls[-1]['result']['error']


def test_no_successful_search_is_failure_not_empty_results(flow, monkeypatch):
    from research_agent.tools import ToolError
    def error(*args, **kwargs):
        raise ToolError('检索暂时不可用')
    monkeypatch.setattr(flow, 'search', error)
    result = run(flow)
    assert result.status == 'failed' and not result.resumable
    assert '不能据此判断' in result.stop_reason
    assert not result.claims


def test_separate_python_process_resumes_saved_paper_selection(flow, tmp_path):
    import subprocess
    import sys
    paused = run(flow, approve_papers=True)
    script = '''
import json, sys
from pathlib import Path
from pytest import MonkeyPatch
from tests.test_graph import flow, run
with MonkeyPatch.context() as patch:
    client = flow.__wrapped__(patch, Path(sys.argv[2]))
    result = run(client, run_id=sys.argv[1], resume=True, selection=['2401.00003v1'])
    print(json.dumps({'status': result.status, 'queries': client.queries, 'reads': client.reads,
                      'evidence': len(result.evidence)}))
'''
    output = subprocess.run([sys.executable, '-c', script, paused.run_id, str(tmp_path)],
                            check=True, capture_output=True, text=True, timeout=30)
    data = json.loads(output.stdout)
    assert data == {'status': 'complete', 'queries': [], 'reads': ['2401.00003v1'], 'evidence': 1}


def test_reader_uses_requested_visual_page_and_rejects_out_of_scope(flow, monkeypatch):
    from research_agent.models import VisualAsset
    pages = []
    monkeypatch.setattr(graph, 'list_visual_pages', lambda *a: [{'page': 2, 'excerpt': 'Table 1'}])
    def visual(paper, client, query, max_pages, *, page_numbers):
        pages.extend(page_numbers)
        paper.assets = [VisualAsset(id=paper.id + ':image', paper_id=paper.id, page=2,
                                    image_path='/fixture.png', image_sha256='fixture', pdf_sha256='fixture')]
        return [TextChunk(id=paper.id + ':visual', paper_id=paper.id, page=2, text=QUOTE,
                          source='vision', asset_id=paper.assets[0].id)]
    monkeypatch.setattr(graph, 'read_visuals', visual)
    original = llm.chat_tools
    def choose(messages, schemas):
        user = json.loads(messages[1]['content'])
        if 'paper_id' not in user:
            return original(messages, schemas)
        prior = [m for m in messages if m['role'] == 'tool']
        pid = user['paper_id']
        if not prior:
            return assistant(call('retrieve_passages', paper_id='unselected', query='reasoning'),
                             call('read_visual_page', paper_id=pid, page=3),
                             call('read_visual_page', paper_id=pid, page=2),
                             call('read_visual_page', paper_id=pid, page=2),
                             call('retrieve_passages', paper_id=pid, query='reasoning benchmark'))
        assert [json.loads(m['content'])['ok'] for m in prior] == [False, False, True, True, True]
        return assistant()
    monkeypatch.setattr(llm, 'chat_tools', choose)
    result = list(run_research('测试图表', client=flow, visual_pages=1))[-1].data
    assert result.status == 'complete', result.errors
    assert pages == [2, 2, 2]  # Exactly once per paper; duplicate calls reuse the saved result.
    assert len(result.papers) == 3 and all(len(p.assets) == 1 for p in result.papers)
    assert sum(c['cache_hit'] for c in result.tool_calls) == 3


def test_streamlit_confirmation_and_history_resume(flow, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from research_agent import orchestrator
    original = orchestrator.run_research
    monkeypatch.setattr(orchestrator, 'run_research', lambda **kw: original(client=flow, **kw))
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    app = AppTest.from_file(app_path).run()
    app.checkbox(key='read_visuals').set_value(False)
    app.checkbox(key='approve_papers').set_value(True)
    next(b for b in app.button if b.label.startswith('连续潜在')).click().run()
    assert not app.exception
    paused = app.session_state['paperscout_result']
    assert paused.status == 'paused'
    # A new UI session has no session_state but can load the saved task.
    app = AppTest.from_file(app_path).run()
    app.button(key='load_run').click().run()
    assert app.session_state['paperscout_result'].status == 'paused'
    app.multiselect[0].set_value(['2401.00001v1'])
    app.button(key=f'approve_{paused.run_id}').click().run()
    assert not app.exception
    assert app.session_state['paperscout_result'].status == 'complete'
    assert flow.reads == ['2401.00001v1'] and len(flow.queries) == 2
    assert len(app.get('download_button')) == 3


def test_streamlit_failed_run_resume_button(flow, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from research_agent import orchestrator
    original = agents.write
    def fail(*args):
        raise llm.LLMError('模拟模型故障')
    monkeypatch.setattr(agents, 'write', fail)
    failed = run(flow)
    assert failed.resumable and len(flow.reads) == 3
    monkeypatch.setattr(agents, 'write', original)
    entrypoint = orchestrator.run_research
    monkeypatch.setattr(orchestrator, 'run_research', lambda **kw: entrypoint(client=flow, **kw))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py').run()
    app.button(key='resume_saved').click().run()
    assert not app.exception
    assert app.session_state['paperscout_result'].status == 'complete'
    assert len(flow.reads) == 3 and len(flow.queries) == 2


def test_checkpoint_repairs_history_index_after_abrupt_stop(flow):
    from research_agent.checkpoints import connect, save_run
    paused = run(flow, approve_papers=True)
    stale = paused.model_copy(deep=True)
    stale.status, stale.pending_selection = 'running', {}
    db = connect()
    try:
        save_run(db, stale)
        restored = run(flow, run_id=paused.run_id, resume=True)
        assert restored.status == 'paused' and restored.pending_selection
        completed = run(flow, run_id=paused.run_id, resume=True, selection=['2401.00001v1'])
        save_run(db, stale)  # Simulate dying after final checkpoint but before index update.
        restored = run(flow, run_id=paused.run_id, resume=True)
        assert restored.status == 'complete' and not restored.resumable
        assert restored.report == completed.report
        assert flow.reads == ['2401.00001v1']
    finally:
        db.close()


def test_graph_cannot_publish_fabricated_evidence_id(flow, monkeypatch):
    monkeypatch.setattr(agents, 'write', lambda *a: Draft(claims=[
        DraftClaim(text='缺乏依据的结论', kind='conclusion', evidence_ids=['E999'])], gaps=[]))
    result = run(flow)
    assert result.status == 'partial'
    assert result.claims[0].status == 'insufficient'


def test_image_downgrade_survives_second_text_critic(flow, monkeypatch):
    def downgrade(claims, evidence, papers):
        claims[0].status, claims[0].reason = 'insufficient', '原图不支持此结论'
    monkeypatch.setattr(graph, 'verify_claim_images', downgrade)
    result = run(flow)
    assert result.status == 'partial'
    assert result.claims[0].status == 'insufficient'
    assert result.claims[0].reason == '原图不支持此结论'
