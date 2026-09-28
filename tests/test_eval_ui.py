import csv
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from eval.evaluator import FIELDS, prepare, score
from research_agent.models import Claim, Event
from research_agent.report import render_report
from tests.test_pipeline import sample_run


def test_annotation_export_starts_unlabelled_and_preserves_quotes(tmp_path):
    run = sample_run()
    run.claims = [Claim(id="C001", text="在合成任务上评测", evidence_ids=["E001"], status="partial")]
    run.drafts = [[run.claims[0].model_copy(update={"status": "unreviewed"})]]
    path = tmp_path / 'run.json'
    path.write_text(run.model_dump_json())
    output = tmp_path / 'annotations.csv'
    assert prepare([str(path)], output) == 1
    with output.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['human_support'] == ''
    assert '#page=2' in rows[0]['evidence']
    metrics = score(output)
    assert metrics['pending'] == 1
    assert metrics['without_verification']['grounded_rate'] is None


def test_scoring_tracks_coverage_and_does_not_reward_empty_output(tmp_path):
    output = tmp_path / 'labels.csv'
    # Synthetic unit-test labels, not project benchmark measurements.
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (human, machine) in enumerate([('supported', 'supported'), ('insufficient', 'insufficient'), ('supported', 'partial')]):
            writer.writerow(dict(run_id='test', claim_id=str(i), machine_status=machine,
                                 human_support=human, human_relevant='yes', reviewer='synthetic-test'))
    metrics = score(output)
    assert metrics['without_verification']['useful_answers'] == 2
    assert metrics['verified_assertions']['useful_answers'] == 1
    assert metrics['without_verification']['unsupported_assertions'] == 1
    assert metrics['verified_assertions']['unsupported_assertions'] == 0
    assert metrics['qualified_or_withheld'] == 2


def test_ui_result_survives_rerun_and_exposes_downloads():
    run = sample_run()
    run.status = 'partial'
    run.stop_reason = '测试结束'
    run.report = render_report(run)
    calls = []
    def fake(question, **kwargs):
        calls.append(question)
        yield Event(stage='done', message='完成', data=run)
    with patch('research_agent.orchestrator.run_research', fake):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py').run()
        assert not app.exception
        assert app.title[0].value == '🔎 PaperScout · 论文调研助手'
        app.button[0].click().run()
        assert not app.exception
        assert len(app.get('download_button')) == 3
        assert len(app.tabs) == 4
        app.run()
        assert not app.exception
        assert len(calls) == 1
        assert app.session_state['paperscout_result'].run_id == 'test'
        assert len(app.get('download_button')) == 3
