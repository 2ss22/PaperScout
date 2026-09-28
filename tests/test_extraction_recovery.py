"""Bad model output must not discard good evidence or stop unrelated papers."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from research_agent import agents, llm
from research_agent.models import Extraction, ExtractionResult, TextChunk, Usage
from research_agent.orchestrator import run_research, add_extraction
from tests.test_pipeline import QUOTE, sample_run, pipeline


def finding(**changes):
    return dict(kind='method', text='在合成基准评测', chunk_id='chunk', quote=QUOTE, **changes)


def test_retry_receives_field_error_and_can_correct_it(monkeypatch):
    bad = finding()
    bad['kind'] = 'datasets'
    messages = []
    outputs = iter([{'findings': [bad]}, {'findings': [finding()]}])
    def chat(payload):
        messages.append(deepcopy(payload))
        return next(outputs)
    monkeypatch.setattr(agents, 'chat_json', chat)
    result = agents.ask(Extraction, 'test', {})
    assert len(result.findings) == 1
    assert 'findings.0.kind' in messages[1][-1]['content']
    assert 'literal_error' in messages[1][-1]['content']


def test_retry_exhausted_preserves_valid_items_but_not_invalid_quotes(monkeypatch):
    bad = finding()
    bad['quote'] = 'short'
    invented = finding()
    invented['quote'] = 'This sentence does not exist in the original source.'
    output = {'findings': [finding(), bad, invented]}
    monkeypatch.setattr(agents, 'chat_json', lambda *a: output)
    result = agents.ask(Extraction, 'test', {})
    assert len(result.findings) == 2
    assert result.warnings and 'string_too_short' in result.warnings[0]
    run = sample_run()
    run.evidence = []
    accepted = add_extraction(run, result, run.chunks)
    assert len(accepted) == 1
    assert run.evidence[0].quote == QUOTE


def test_overlong_finding_list_is_bounded_after_retry(monkeypatch):
    monkeypatch.setattr(agents, 'chat_json', lambda *a: {'findings': [finding()] * 9})
    result = agents.ask(Extraction, 'test', {})
    assert len(result.findings) == 8 and result.warnings


def test_all_invalid_reports_specific_error(monkeypatch):
    monkeypatch.setattr(agents, 'chat_json', lambda *a: {'findings': [{'kind': 'method'}]})
    with pytest.raises(llm.LLMError, match=r'findings.0.text'):
        agents.ask(Extraction, 'test', {})


@pytest.mark.parametrize('failed_source', ['pdf', 'vision'])
def test_modalities_fail_independently(monkeypatch, failed_source):
    run = sample_run()
    chunks = run.chunks + [TextChunk(id='visual', paper_id=run.papers[0].id, page=2,
                                   text=QUOTE, source='vision', asset_id='image')]
    calls = []
    def extract(q, p, group):
        calls.append(group[0].source)
        if group[0].source == failed_source:
            raise llm.LLMError('模拟模型持续返回损坏的 JSON')
        record = finding()
        record['chunk_id'] = group[0].id
        return Extraction(findings=[record])
    monkeypatch.setattr(agents, 'extract', extract)
    result = agents.extract_groups('问题', run.papers[0], chunks)
    assert calls == ['pdf', 'vision']
    assert len(result.findings) == 1
    assert len(result.warnings) == 1


def test_one_paper_failure_does_not_abort_other_papers_or_review(pipeline, monkeypatch):
    original = agents.extract
    def extract(q, p, cs):
        if p.title == 'Paper 2':
            raise llm.LLMError('findings.0.quote: string_too_short')
        return original(q, p, cs)
    monkeypatch.setattr(agents, 'extract', extract)
    events = list(run_research('问题', client=pipeline))
    run = events[-1].data
    assert run.status == 'partial'
    assert len(run.papers) == 5
    assert len(run.evidence) == 4
    assert run.claims and all(c.status == 'supported' for c in run.claims)
    assert any('Paper 2' in error and 'string_too_short' in error for error in run.errors)
    assert any(e.stage == 'warning' and e.data['recoverable'] for e in events)
    assert '运行中断' not in run.stop_reason


def test_all_extractions_fail_bounded_stop_without_claims(pipeline, monkeypatch):
    def extract(*args):
        raise llm.LLMError('模拟 JSON 错误')
    monkeypatch.setattr(agents, 'extract', extract)
    run = list(run_research('问题', client=pipeline))[-1].data
    assert run.status == 'partial'
    assert not run.claims and not run.evidence
    assert len(run.errors) == 5
    assert run.supplemental_rounds == 1
    assert '上限' in run.stop_reason


def test_truncation_is_reported_and_usage_preserved(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',
                               message=SimpleNamespace(content='{"findings": []}'))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    monkeypatch.setattr(llm, 'get_client', lambda: fake)
    usage = Usage()
    with llm.track_usage(usage), pytest.raises(llm.LLMError, match='finish_reason=length'):
        llm.chat([])
    assert usage.calls == usage.calls_with_usage == 1
    assert usage.completion_tokens == 20


def test_model_not_found_does_not_retry_or_skip_through_every_paper(monkeypatch):
    calls = []
    def chat(*args):
        calls.append(1)
        raise llm.LLMError('HTTP 404 model_not_found', retryable=False)
    monkeypatch.setattr(agents, 'chat_json', chat)
    run = sample_run()
    with pytest.raises(llm.LLMError, match='不进行格式重试') as exc:
        agents.extract_groups('问题', run.papers[0], run.chunks)
    assert len(calls) == 1
    assert not exc.value.retryable


def test_provider_configuration_error_is_classified(monkeypatch):
    class ProviderError(Exception):
        status_code = 404
    def create(**kwargs):
        raise ProviderError('model_not_found')
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, 'get_client', lambda: fake)
    with pytest.raises(llm.LLMError) as exc:
        llm.chat([])
    assert not exc.value.retryable
