"""Program-owned quotation spans preserve source text without fuzzy matching."""
import json

from research_agent import agents
from research_agent.models import AnchoredExtraction, Extraction, ExtractedFinding, TextChunk
from research_agent.orchestrator import add_extraction, review_claims
from research_agent.reader import quote_spans, quote_matches
from tests.test_pipeline import sample_run


def test_spans_preserve_exact_characters_and_contiguous_offsets():
    run = sample_run()
    text = ('“Accuracy” is 62.89% — not 82.89%; 16× fewer tokens.\n'
            'Visual reasoning improves on this benchmark.\u00a0') * 16
    run.chunks[0].text = text
    spans = quote_spans(run.chunks)
    assert len(spans) > 1
    assert len({s.id for s in spans}) == len(spans)
    covered = set()
    for span in spans:
        assert span.quote == text[span.start:span.end]
        assert quote_matches(span.quote, text)
        assert len(span.quote) <= 360
        covered.update(range(span.start, span.end))
    assert covered == set(range(len(text)))


def test_model_selects_span_and_cannot_rewrite_numbers_or_punctuation(monkeypatch):
    run = sample_run()
    run.chunks[0].text = 'The model achieves 62.89% accuracy — with 16× fewer tokens.\nLimited to this benchmark.'
    def chat(messages):
        data = json.loads(messages[1]['content'])
        span = data['spans'][0]
        assert '62.89%' in span['quote']
        assert 'span_id' in messages[0]['content']
        return {'findings': [{'kind': 'metric', 'text': '该模型准确率为 62.89%', 'span_id': span['id']}]}
    monkeypatch.setattr(agents, 'chat_json', chat)
    result = agents.extract('问题', run.papers[0], run.chunks)
    assert result.findings[0].quote == run.chunks[0].text
    run.evidence = []
    assert len(add_extraction(run, result, run.chunks)) == 1
    assert run.evidence[0].quote == run.chunks[0].text
    assert not run.errors


def test_invalid_span_id_is_rejected_but_valid_sibling_survives(monkeypatch):
    run = sample_run()
    monkeypatch.setattr(agents, 'chat_json', lambda *a: {'findings': [
        {'kind':'method','text':'有效','span_id':'S0001'},
        {'kind':'method','text':'未知来源','span_id':'S9999'}]})
    result = agents.extract('问题',run.papers[0],run.chunks)
    assert len(result.findings) == 1
    assert len(result.warnings) == 1 and 'S9999' in result.warnings[0]
    assert run.papers[0].id in result.warnings[0]


def test_malformed_anchored_sibling_still_salvages_valid_record(monkeypatch):
    run = sample_run()
    monkeypatch.setattr(agents, 'chat_json', lambda *a: {'findings': [
        {'kind':'method','text':'有效','span_id':'S0001'},
        {'kind':'wrong','text':'无效','span_id':'S0002'}]})
    result = agents.extract('问题',run.papers[0],run.chunks)
    assert len(result.findings) == 1 and result.warnings


def test_same_text_other_chunk_does_not_retarget_wrong_reference():
    run=sample_run()
    run.evidence=[]
    original=run.chunks[0].text
    run.chunks.append(TextChunk(id='other',paper_id='different-paper',page=9,text='A completely different measurement is reported here.'))
    extraction=Extraction(findings=[ExtractedFinding(kind='method',text='错误引用',chunk_id='other',quote=original)])
    assert not add_extraction(run,extraction,run.chunks)
    assert 'different-paper' in run.errors[0] and '第 9 页' in run.errors[0] and 'other' in run.errors[0]
    assert '连续原文' in run.errors[0]


def test_quote_guard_still_rejects_rewritten_numbers_and_stitched_sentences():
    original='The model achieved 62.89% accuracy. It used 16 tokens. Further work is needed.'
    assert not quote_matches('The model achieved 82.89% accuracy.',original)
    assert not quote_matches('The model achieved 62.89% accuracy. Further work is needed.',original)


def test_real_quote_does_not_make_overstated_claim_supported(monkeypatch):
    from research_agent.models import Claim, Review, Verdict
    run=sample_run()
    claim=Claim(id='C1',text='所有真实世界任务都最优',evidence_ids=['E001'])
    review=Review(verdicts=[Verdict(claim_id='C1',status='insufficient',reason='片段只提及合成基准，不能支持所有任务')],
                  action='finish',reason='保留未知')
    review_claims([claim],run.evidence,run.chunks,review)
    assert claim.status=='insufficient'
