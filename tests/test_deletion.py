"""Deletion isolation, shared caches, active leases, legacy caches and UI state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

from filelock import FileLock
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
import pytest
from streamlit.testing.v1 import AppTest

from research_agent import graph
from research_agent.cache_lifecycle import cache_session, metadata_dir, read_references, track_cache
from research_agent.checkpoints import connect, load_run, save_run
from research_agent.deletion import delete_run, preview_deletion
from research_agent.graph import ResearchState
from research_agent.models import VisualAsset
from research_agent.orchestrator import run_research
from research_agent.tools import Cache
from scripts import research
from tests.test_graph import flow
from tests.test_pipeline import sample_run
from tests.test_multimodal import reading


def cached_run(cache, number=1):
    run = sample_run()
    run.run_id, run.cache_root = uuid4().hex, str(cache.root.resolve())
    run.question, run.queries = f'调研 {number}', [f'query {number}']
    paper = run.papers[0]
    paper.id = f'2401.{number:05d}v1'
    run.chunks[0].paper_id = run.evidence[0].paper_id = paper.id
    pdf = ('PDF fixture ' + paper.id).encode()
    png = ('PNG fixture ' + paper.id).encode()
    pdf_hash = hashlib.sha256(pdf).hexdigest()
    pdf_path = cache.put_bytes(f'pdf-v1:{paper.id}', pdf, '.pdf')
    chunks_key = f'chunks-v2:{paper.id}'
    cache.put(chunks_key, [c.model_dump() for c in run.chunks])
    image = cache.put_bytes(f'visual-png:{pdf_hash}:1', png, '.png')
    visual = reading()
    visual.elements[0].description += paper.id  # Different reading per paper.
    paper.assets = [VisualAsset(id=paper.id + ':visual', paper_id=paper.id, page=2, image_path=str(image),
                               image_sha256=hashlib.sha256(png).hexdigest(), pdf_sha256=pdf_hash, reading=visual)]
    visual_key = 'visual-reading-v1:fixture:' + paper.id
    cache.put(visual_key, visual.model_dump())
    search_key = f'search-v2:{run.queries[0]}:5'
    cache.put(search_key, [paper.model_dump(exclude={'assets'})])
    return run, {pdf_path, image, cache._path(chunks_key), cache._path(visual_key), cache._path(search_key)}


def saved(run, path=None, *, checkpoints=True):
    db = connect(path)
    try:
        save_run(db, run)
        if checkpoints:
            builder = StateGraph(ResearchState)
            builder.add_node('save', lambda state: state)
            builder.add_edge(START, 'save'); builder.add_edge('save', END)
            builder.compile(checkpointer=SqliteSaver(db)).invoke({'run': run.model_dump()},
                {'configurable': {'thread_id': run.run_id}})
    finally:
        db.close()


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / 'cache')


def test_delete_legacy_record_removes_pdf_text_visual_search_and_graph_state(cache):
    run, files = cached_run(cache)
    saved(run)
    cache.put('unrelated', {'leave': 'alone'})
    unrelated = cache._path('unrelated')
    plan = preview_deletion(run.run_id)
    assert set(plan.files) == files and plan.bytes == sum(p.stat().st_size for p in files)
    assert all(p.exists() for p in files)  # Preview never deletes.
    removed = delete_run(run.run_id)
    assert len(removed.files) == 5 and not any(p.exists() for p in files)
    assert unrelated.exists() and load_run(run.run_id) is None
    db = connect()
    try:
        for table, column in [('paperscout_runs','id'), ('checkpoints','thread_id'), ('writes','thread_id')]:
            assert db.execute(f'SELECT COUNT(*) FROM {table} WHERE {column}=?', (run.run_id,)).fetchone()[0] == 0
    finally:
        db.close()
    with pytest.raises(ValueError, match='不存在'):
        delete_run(run.run_id)


def test_shared_papers_preserved_and_exclusive_files_deleted(cache):
    common, shared = cached_run(cache, 1)
    unique, exclusive = cached_run(cache, 2)
    target = common.model_copy(deep=True)
    target.run_id = uuid4().hex
    target.papers.extend(unique.papers); target.chunks.extend(unique.chunks)
    target.queries.extend(unique.queries)
    saved(common); saved(target)
    plan = delete_run(target.run_id)
    assert set(plan.files) == exclusive and plan.shared_files == len(shared)
    assert all(p.exists() for p in shared) and not any(p.exists() for p in exclusive)
    assert load_run(common.run_id) is not None
    # The last owner can reclaim files after the other record is gone.
    assert set(delete_run(common.run_id).files) == shared


def test_paused_candidate_protects_every_version_of_shared_paper(cache):
    target, files = cached_run(cache)
    other = sample_run()
    other.run_id, other.cache_root = uuid4().hex, str(cache.root.resolve())
    other.papers, other.chunks, other.evidence = [], [], []
    other.candidates = [target.papers[0].model_copy(update={'id': '2401.00001v2', 'assets': []})]
    saved(target); saved(other)
    plan = delete_run(target.run_id)
    assert all(p.exists() for p in files) and plan.shared_files == 5


def test_record_only_deletion_keeps_all_cached_files(cache):
    run, files = cached_run(cache); saved(run)
    result = delete_run(run.run_id, clean_cache=False)
    assert result.files == [] and all(p.exists() for p in files)
    assert load_run(run.run_id) is None


def test_active_cache_session_blocks_deletion(cache):
    run, files = cached_run(cache); saved(run)
    with cache_session(cache.root, uuid4().hex):
        with pytest.raises(ValueError, match='使用缓存'):
            delete_run(run.run_id)
    assert load_run(run.run_id) is not None and all(p.exists() for p in files)


def test_active_run_lock_blocks_deletion_even_before_leases(cache, tmp_path):
    run, files = cached_run(cache); saved(run)
    with FileLock(tmp_path / f'{run.run_id}.lock'):
        with pytest.raises(ValueError, match='正在执行'):
            delete_run(run.run_id)
    assert all(p.exists() for p in files)


def test_new_cache_reference_manifest_captures_writes_reads_and_binary_hits(cache, tmp_path):
    run, _ = cached_run(cache); saved(run)
    with track_cache(cache.root, run.run_id, tmp_path / 'checkpoints.sqlite'):
        assert cache.get(f'chunks-v2:{run.papers[0].id}')
        cache.binary_path(f'pdf-v1:{run.papers[0].id}', '.pdf')
        cache.put('reading-with-unknown-key', {'new': 'result'})
        cache.put_bytes('another-image', b'image', '.png')
    manifest = metadata_dir(cache.root) / f'{run.run_id}.json'
    names = read_references(manifest)
    assert len(names) == 4
    arbitrary = cache._path('reading-with-unknown-key')
    result = delete_run(run.run_id)
    assert arbitrary in result.files and not arbitrary.exists() and not manifest.exists()


def test_recheck_after_preview_preserves_new_owner(cache):
    target, files = cached_run(cache); saved(target)
    assert len(preview_deletion(target.run_id).files) == 5
    other = target.model_copy(deep=True); other.run_id = uuid4().hex; saved(other)
    assert delete_run(target.run_id).files == []
    assert all(p.exists() for p in files)


def test_failed_unlink_retains_record_for_retry(cache, monkeypatch):
    run, files = cached_run(cache); saved(run)
    original = Path.unlink
    fail_at = sorted(files)[1]
    def blocked(path, *args, **kwargs):
        if path == fail_at:
            raise PermissionError('fixture permission failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', blocked)
    with pytest.raises(ValueError, match='记录保留'):
        delete_run(run.run_id)
    assert load_run(run.run_id) is not None
    assert not sorted(files)[0].exists() and fail_at.exists()
    monkeypatch.setattr(Path, 'unlink', original)
    assert len(delete_run(run.run_id).files) == 4


def test_database_failure_rolls_back_history_and_checkpoints_for_retry(cache):
    run, files = cached_run(cache); saved(run)
    db = connect()
    try:
        db.execute("CREATE TRIGGER refuse_delete BEFORE DELETE ON paperscout_runs "
                   "BEGIN SELECT RAISE(ABORT, 'fixture database failure'); END")
        db.commit()
    finally:
        db.close()
    with pytest.raises(ValueError, match='数据库删除失败'):
        delete_run(run.run_id)
    assert load_run(run.run_id) is not None and not any(p.exists() for p in files)
    db = connect()
    try:
        assert db.execute('SELECT COUNT(*) FROM checkpoints WHERE thread_id=?', (run.run_id,)).fetchone()[0] > 0
        db.execute('DROP TRIGGER refuse_delete'); db.commit()
    finally:
        db.close()
    assert delete_run(run.run_id).files == [] and load_run(run.run_id) is None


def test_cache_cleanup_never_follows_exported_paths_or_symlinks(cache, tmp_path):
    run, files = cached_run(cache)
    foreign = tmp_path / ('a' * 64 + '.png'); foreign.write_bytes(b'outside-cache')
    run.papers[0].assets[0].image_path = str(foreign)
    link = cache.binary_path('link-to-foreign', '.png'); link.symlink_to(foreign)
    with track_cache(cache.root, run.run_id, 'fixture.sqlite'):
        cache.binary_path('link-to-foreign', '.png')
    saved(run)
    delete_run(run.run_id)
    assert foreign.read_bytes() == b'outside-cache' and not link.exists()


def test_invalid_manifest_path_never_deletes_foreign_file(cache, tmp_path):
    run, files = cached_run(cache); saved(run)
    foreign = tmp_path / 'foreign.txt'; foreign.write_text('keep')
    (metadata_dir(cache.root) / f'{run.run_id}.json').write_text(json.dumps({'files':['../foreign.txt']}))
    with pytest.raises(ValueError, match='无效文件名'):
        delete_run(run.run_id)
    assert foreign.read_text() == 'keep' and all(p.exists() for p in files)
    assert load_run(run.run_id) is not None


def test_deleted_run_cannot_resume_or_reappear(flow, cache):
    flow.cache = cache
    result = list(run_research('问题', client=flow, visual_pages=0))[-1].data
    delete_run(result.run_id)
    with pytest.raises(ValueError, match='找不到'):
        list(run_research(run_id=result.run_id, resume=True, client=flow))
    assert load_run(result.run_id) is None


def test_graph_tracks_actual_cache_use_per_node(flow, cache, monkeypatch):
    flow.cache = cache
    original = graph.read_paper
    def read(paper, client):
        chunks = original(paper, client)
        client.cache.put(f'chunks-v2:{paper.id}', [c.model_dump() for c in chunks])
        client.cache.put_bytes(f'pdf-v1:{paper.id}', b'fixture-pdf', '.pdf')
        return chunks
    monkeypatch.setattr(graph, 'read_paper', read)
    result = list(run_research('问题', client=flow, visual_pages=0))[-1].data
    assert result.cache_root == str(cache.root.resolve())
    names = read_references(metadata_dir(cache.root) / f'{result.run_id}.json')
    assert len(names) == 6
    assert len(delete_run(result.run_id).files) == 6


def test_node_failure_keeps_cache_references_before_checkpoint_is_saved(flow, cache, monkeypatch):
    flow.cache = cache
    def read(paper, client):
        client.cache.put('half-read-paper', {'temporary': 'reading'})
        raise RuntimeError('fixture reader interrupted')
    monkeypatch.setattr(graph, 'read_paper', read)
    result = list(run_research('问题', client=flow, visual_pages=0))[-1].data
    file = cache._path('half-read-paper')
    assert file.exists() and file.name in read_references(metadata_dir(cache.root) / f'{result.run_id}.json')
    assert file in delete_run(result.run_id).files and not file.exists()


def test_latest_checkpoint_protects_shared_assets_even_when_history_index_is_stale(cache):
    target, files = cached_run(cache); saved(target)
    other = target.model_copy(deep=True); other.run_id = uuid4().hex; saved(other)
    stale = other.model_copy(deep=True)
    stale.papers, stale.candidates, stale.chunks, stale.queries = [], [], [], []
    db = connect()
    try:
        save_run(db, stale)
    finally:
        db.close()
    assert delete_run(target.run_id).files == [] and all(p.exists() for p in files)


def test_reference_from_other_database_protects_shared_cache(cache):
    target, _ = cached_run(cache); saved(target)
    with track_cache(cache.root, target.run_id, 'first.sqlite'):
        cache.put('unclassified-cache', {'keep': True})
    with track_cache(cache.root, uuid4().hex, 'another.sqlite'):
        cache.get('unclassified-cache')
    shared = cache._path('unclassified-cache')
    result = delete_run(target.run_id)
    assert shared.exists() and shared not in result.files and result.shared_files == 1


def test_unreadable_other_manifest_preserves_potentially_shared_cache(cache):
    target, files = cached_run(cache); saved(target)
    (metadata_dir(cache.root) / f'{uuid4().hex}.json').write_text('{broken')
    result = delete_run(target.run_id)
    assert result.files == [] and result.warnings and all(p.exists() for p in files)


@pytest.mark.parametrize('keep_cache', [False, True])
def test_cli_previews_until_explicit_confirmation(cache, monkeypatch, capsys, keep_cache):
    run, files = cached_run(cache); saved(run)
    arguments = ['research', 'delete', run.run_id] + (['--keep-cache'] if keep_cache else [])
    monkeypatch.setattr(sys, 'argv', arguments)
    research.main()
    assert '预览' in capsys.readouterr().out
    assert load_run(run.run_id) is not None and all(p.exists() for p in files)
    monkeypatch.setattr(sys, 'argv', arguments + ['--yes'])
    research.main()
    assert '已删除' in capsys.readouterr().out and load_run(run.run_id) is None
    assert all(p.exists() == keep_cache for p in files)


def test_ui_requires_confirmation_deletes_files_and_clears_current_record(cache, monkeypatch):
    run, files = cached_run(cache); saved(run)
    # The preview itself does not need decoding thumbnails in this UI test.
    run.papers[0].assets = []
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py')
    app.session_state['paperscout_result'] = run
    app.run()
    assert not app.exception and app.button(key='delete_saved').disabled
    app.checkbox(key=f'confirm_delete_{run.run_id}').set_value(True).run()
    assert not app.button(key='delete_saved').disabled
    app.button(key='delete_saved').click().run()
    assert not app.exception
    assert load_run(run.run_id) is None and not any(p.exists() for p in files)
    assert 'paperscout_result' not in app.session_state
    assert any('已删除' in s.value for s in app.success)


def test_ui_deletion_switches_to_remaining_record_and_preserves_its_display(cache):
    target, files = cached_run(cache); saved(target)
    other, shared = cached_run(cache, 2); saved(other)
    other.papers[0].assets = []
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py')
    app.session_state['paperscout_result'] = other
    app.run()
    app.selectbox(key='saved_run').set_value(target.run_id).run()
    app.checkbox(key=f'confirm_delete_{target.run_id}').set_value(True).run()
    app.button(key='delete_saved').click().run()
    assert not app.exception and app.selectbox(key='saved_run').value == other.run_id
    assert app.session_state['paperscout_result'].run_id == other.run_id
    assert load_run(other.run_id) is not None and all(p.exists() for p in shared)
    assert load_run(target.run_id) is None and not any(p.exists() for p in files)
