"""Answer grounding, bidirectional citations, portable exports and UI controls."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

import pytest
from streamlit.testing.v1 import AppTest

from research_agent import agents
from research_agent.artifacts import evidence_bundle
from research_agent.models import (Claim, Draft, DraftClaim, EvidenceAssessment, Gap, Review, RunResult, Verdict)
from research_agent.orchestrator import create_claims, review_claims
from research_agent.report import (answer_claims, evidence_connections, evidence_markdown, render_report)
from tests.test_pipeline import sample_run
from tests.test_multimodal import image_run, fixture_pdf


def example():
    run = sample_run()
    run.claims = [Claim(id='C001', text='该方法已在合成推理基准上评测。', kind='conclusion',
        evidence_ids=['E001'], status='supported', reason='原文明确描述了评测范围。',
        priority='primary', focus='在哪些任务上有证据？', scope='仅合成推理基准。',
        evidence_assessments=[EvidenceAssessment(evidence_id='E001', role='support', explanation='说明了评测对象。')])]
    run.evidence.append(run.evidence[0].model_copy(update={'id': 'E002'}))
    return run


def test_core_answer_precedes_sources_and_keeps_partial_qualifier():
    run = example()
    run.claims[0].status = 'partial'
    run.claims[0].reason = '只能支持合成任务范围，不能外推。'
    report = render_report(run)
    core = report.split('## 关键论文对比')[0]
    assert 'C001 · 部分支持' in core and '不能外推' in core
    assert '仅合成推理基准' in core and '[E001](#evidence-e001)' in core
    assert report.index('## 核心回答') < report.index('## 检索范围与论文')
    assert 'E002' not in core


@pytest.mark.parametrize('bad', ['insufficient', 'unreviewed', 'missing_id', 'quote_mismatch'])
def test_unsupported_or_broken_saved_claim_never_enters_core(bad):
    run = example()
    if bad in {'insufficient', 'unreviewed'}:
        run.claims[0].status = bad
    elif bad == 'missing_id':
        run.claims[0].evidence_ids = ['E999']
    else:
        run.evidence[0].quote = 'An invented quote about a different finding.'
    report = render_report(run)
    core = report.split('## 关键论文对比')[0]
    assert '没有足够的已核验证据' in core
    assert run.claims[0].text not in core
    assert '待核验说法（不作为结论）' in report
    if bad == 'quote_mismatch':
        assert '结构未通过检查' in evidence_connections(run, 'E001')[0]['reason']


def test_review_preserves_scope_and_only_matching_unique_relations():
    run = example()
    run.claims = create_claims(Draft(claims=[DraftClaim(text='判断', kind='method', evidence_ids=['E001', 'E002'],
        focus='如何评测？', priority='primary', scope='合成任务')], gaps=[]))
    relation = EvidenceAssessment(evidence_id='E002', role='background', explanation='背景')
    review = Review(action='finish', reason='完成', verdicts=[Verdict(claim_id='C001', status='partial', reason='范围受限',
        evidence_assessments=[EvidenceAssessment(evidence_id='E001', role='limitation', explanation='仅合成任务'),
                              relation, relation, relation.model_copy(update={'evidence_id': 'E999'})])])
    review_claims(run.claims, run.evidence, run.chunks, review)
    assert run.claims[0].scope == '合成任务' and run.claims[0].priority == 'primary'
    assert len(run.claims[0].evidence_assessments) == 1
    conn = evidence_connections(run, 'E001')[0]
    assert conn['role'] == '限制' and conn['status'] == 'partial'
    assert conn['explanation'] == '仅合成任务'


def test_visual_downgrade_is_visible_even_when_text_assessment_supported():
    run = example()
    run.claims[0].status, run.claims[0].reason = 'insufficient', '原图不支持数字归属。'
    card = evidence_markdown(run, run.evidence[0])
    assert '证据不足' in card and '原图不支持数字归属' in card
    assert '文本审阅关系（支持）' in card
    assert '未用于肯定回答' in card


def test_full_export_resolves_bidirectional_links_and_retains_unused_sources():
    run = example()
    run.gaps = [Gap(question='迁移到真实任务是否仍成立？', experiment='加入真实任务与相同预算基线。',
                    evidence_ids=['E002', 'E999'], criterion='在相同条件下改善则支持假设，否则不支持。')]
    report = render_report(run)
    assert '](#claim-c001)' in report
    assert '未用于肯定回答的资料' in report and '尚未被任何结论引用' in report
    assert 'E999' not in report
    targets = re.findall(r'\]\(#([^)]+)\)', report)
    anchors = re.findall(r'<a id="([^"]+)"></a>', report)
    assert set(targets) <= set(anchors)
    assert len(anchors) == len(set(anchors))
    assert 'G1' in evidence_markdown(run, run.evidence[1])
    run.report = report
    with zipfile.ZipFile(BytesIO(evidence_bundle(run))) as z:
        assert z.read('report.md').decode() == report
        restored = RunResult.model_validate_json(z.read('run.json'))
    assert restored.claims[0].evidence_assessments[0].role == 'support'
    assert len(restored.evidence) == 2


def test_external_metadata_cannot_insert_markup_or_fake_source_links():
    run = example()
    bad = '<script>alert(1)</script> [伪来源](https://evil.test)'
    run.claims[0].scope = bad
    run.claims[0].evidence_assessments[0].explanation = bad
    report = render_report(run)
    assert '<script>' not in report and 'https://evil.test' not in report
    assert '#evidence-e001' in report


def test_old_records_load_and_render_without_added_metadata():
    raw = example().model_dump()
    for c in raw['claims']:
        for key in ('focus', 'priority', 'scope', 'evidence_assessments'):
            c.pop(key)
    run = RunResult.model_validate(raw)
    assert run.claims[0].priority == 'detail'
    assert '## 核心回答' in render_report(run)
    assert evidence_connections(run, 'E001')[0]['role'] == '未单独标注'


def test_new_writer_and_critic_require_grounded_conditions(monkeypatch):
    calls = []
    def ask(schema, instruction, payload):
        calls.append((schema, instruction, payload))
        return Draft(claims=[], gaps=[]) if schema is Draft else Review(verdicts=[], action='finish', reason='结束')
    monkeypatch.setattr(agents, 'ask', ask)
    run = example()
    agents.write(run.question, run.papers, [], run.evidence)
    agents.critique(run.question, run.claims, run.evidence, run.papers, False)
    assert 'priority=primary' in calls[0][1] and 'criterion' in calls[0][1]
    assert 'text 与 scope' in calls[1][1]
    assert calls[1][2]['claims'][0]['scope'] == run.claims[0].scope


def test_ui_shows_collapsed_sources_and_filters_without_model_calls(monkeypatch):
    monkeypatch.setattr(agents, 'write', lambda *a: pytest.fail('Changing layout must not call a model'))
    run = example()
    run.report = '旧版报告内容'
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py')
    app.session_state['paperscout_result'] = run
    app.run()
    assert not app.exception
    assert any('## 核心回答' in m.value for m in app.markdown)
    assert not any(m.value == '旧版报告内容' for m in app.markdown)
    assert any('E001 · 展开原文' in e.label and not e.proto.expanded for e in app.expander)
    assert any('E002 · 未用于肯定回答' in e.label for e in app.expander)
    app.radio(key='evidence_scope').set_value('未用于肯定回答').run()
    assert not app.exception
    assert any('E002 · 关联判断：尚未被结论引用' in e.label for e in app.expander)
    assert len(app.get('download_button')) == 3


def test_visual_index_thumbnail_association_and_integrity_stay_in_bundle(tmp_path):
    import pymupdf
    run = image_run(tmp_path)
    with pymupdf.open(stream=fixture_pdf(), filetype='pdf') as doc:
        png = doc[0].get_pixmap().tobytes('png')
    asset = run.papers[0].assets[0]
    Path(asset.image_path).write_bytes(png)
    asset.image_sha256 = hashlib.sha256(png).hexdigest()
    run.chunks[0].source, run.chunks[0].asset_id = 'vision', asset.id
    run.claims[0].reason = '用于测试关联；不代表真人核验。'
    run.report = render_report(run)
    assert '回答引用' in run.report and '关联判断' in run.report
    assert asset.image_sha256 not in run.report
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py')
    app.session_state['paperscout_result'] = run
    app.run()
    assert not app.exception
    app.checkbox(key='used_visuals').set_value(True).run()
    assert not app.exception
    assert any('第 2 页 · 回答引用' in e.label for e in app.expander)
    with zipfile.ZipFile(BytesIO(evidence_bundle(run))) as z:
        manifest = json.loads(z.read('images/manifest.json'))
        assert manifest[0]['sha256'] == asset.image_sha256
        assert z.read(manifest[0]['file']) == png


def test_explicit_primary_selection_does_not_pull_details_into_core():
    run = example()
    run.claims.append(run.claims[0].model_copy(update={'id': 'C002', 'priority': 'detail', 'text': '另一个详细发现'}))
    report = render_report(run)
    core = report.split('## 分问题分析')[0]
    assert '另一个详细发现' not in core
    assert '另一个详细发现' in report
    assert run.claims[0].reason not in core  # Full reasoning remains in the evidence card.
    assert run.claims[0].reason in report
