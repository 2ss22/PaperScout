"""Delete one saved research run and its exclusive local cache files."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sqlite3

from filelock import FileLock, Timeout
from langgraph.checkpoint.sqlite import SqliteSaver

from .cache_lifecycle import CACHE_FILENAME, cleanup_lock, ensure_idle, metadata_dir, read_references
from .checkpoints import checkpoint_file, connect, validate_run_id
from .models import RunResult
from .tools import Cache, base_id


@dataclass
class DeletionPreview:
    run_id: str
    question: str
    cache_root: Path
    files: list[Path] = field(default_factory=list)
    shared_files: int = 0
    bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def counts(self):
        return {suffix: sum(p.suffix == suffix for p in self.files) for suffix in ('.pdf', '.png', '.json', '.bin')}


def _records(db):
    records = [RunResult.model_validate_json(row[0]) for row in db.execute('SELECT result FROM paperscout_runs')]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # A process can stop between its graph checkpoint and history index update.
    # Include both copies so unfinished reads still protect their cache files.
    if 'checkpoints' in tables:
        saver = SqliteSaver(db)
        for thread_id, in db.execute('SELECT DISTINCT thread_id FROM checkpoints').fetchall():
            saved = saver.get_tuple({'configurable': {'thread_id': thread_id}})
            raw = saved.checkpoint.get('channel_values', {}).get('run') if saved else None
            if raw:
                records.append(RunResult.model_validate(raw))
    return records


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _asset_names(asset, root):
    # Recover relocated historical screenshots from their program-owned key,
    # rather than treating an arbitrary exported image path as deletion authority.
    names = {Cache(root).binary_path(f'visual-png:{asset.pdf_sha256}:{asset.page - 1}', '.png').name}
    path = Path(asset.image_path)
    if path.parent.resolve() == root and CACHE_FILENAME.fullmatch(path.name):
        names.add(path.name)
    return names


def _collect(records, root):
    paper_ids, candidate_ids, queries, names, readings = set(), set(), set(), set(), set()
    cache = Cache(root)
    for run in records:
        # Same paper in another record/version is treated conservatively as shared.
        paper_ids.update(base_id(p.id) for p in run.papers)
        candidate_ids.update(base_id(p.id) for p in run.candidates + run.papers)
        queries.update(run.queries)
        for paper in run.papers:
            names.add(cache.binary_path(f'pdf-v1:{paper.id}', '.pdf').name)
            names.add(cache._path(f'chunks-v2:{paper.id}').name)
            for asset in paper.assets:
                names.update(_asset_names(asset, root))
                if asset.reading:
                    readings.add(_fingerprint(asset.reading.model_dump()))
        for query in run.queries:
            for limit in range(1, 6):
                names.add(cache._path(f'search-v2:{query.strip()[:500]}:{limit}').name)
    return paper_ids, candidate_ids, queries, names, readings


def _plan(db, run_id, root, clean_cache):
    records = _records(db)
    selected = [r for r in records if r.run_id == run_id]
    if not selected:
        raise ValueError('此调研记录不存在或已经删除。')
    remaining = [r for r in records if r.run_id != run_id]
    plan = DeletionPreview(run_id, selected[0].question, root)
    if not clean_cache:
        return plan
    target_papers, target_candidates, _, target_names, target_readings = _collect(selected, root)
    other_papers, other_candidates, _, other_names, other_readings = _collect(remaining, root)
    target_manifest = metadata_dir(root) / f'{run_id}.json'
    if target_manifest.exists():
        target_names.update(read_references(target_manifest))
    for path in metadata_dir(root).glob('*.json'):
        if path == target_manifest:
            continue
        try:
            other_names.update(read_references(path))
        except (OSError, ValueError):
            # Unknown references cannot safely be treated as unshared.
            other_names.update(p.name for p in root.iterdir() if CACHE_FILENAME.fullmatch(p.name))
            plan.warnings.append('部分缓存关联记录无法读取，本次保留可能共享的文件。')

    # Protect every version of a shared paper before identifying old visual JSON.
    for run in selected:
        for paper in run.papers:
            if base_id(paper.id) in other_papers | other_candidates:
                other_names.update([Cache(root).binary_path(f'pdf-v1:{paper.id}', '.pdf').name,
                                    Cache(root)._path(f'chunks-v2:{paper.id}').name])
                for asset in paper.assets:
                    other_names.update(_asset_names(asset, root))
                other_readings.update(_fingerprint(a.reading.model_dump()) for a in paper.assets if a.reading)

    # Reconstruct older runs, whose hash filenames predate reference manifests.
    for path in root.glob('*.json'):
        if path.is_symlink() or not CACHE_FILENAME.fullmatch(path.name):
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(value, list):
            chunk_ids = {base_id(v['paper_id']) for v in value if isinstance(v, dict) and isinstance(v.get('paper_id'), str)}
            candidates = {base_id(v['id']) for v in value if isinstance(v, dict) and
                          isinstance(v.get('id'), str) and 'abstract' in v}
            if chunk_ids.intersection(target_papers) or candidates.intersection(target_candidates):
                target_names.add(path.name)
            if chunk_ids.intersection(other_papers | other_candidates) or candidates.intersection(other_candidates):
                other_names.add(path.name)
        elif isinstance(value, dict) and 'readable' in value and 'elements' in value:
            digest = _fingerprint(value)
            if digest in target_readings:
                target_names.add(path.name)
            if digest in other_readings:
                other_names.add(path.name)
    existing = {name for name in target_names if CACHE_FILENAME.fullmatch(name)
                and ((root / name).is_file() or (root / name).is_symlink())}
    plan.shared_files = len(existing.intersection(other_names))
    plan.files = [root / name for name in sorted(existing - other_names)]
    plan.bytes = sum(p.lstat().st_size for p in plan.files)
    return plan


@contextmanager
def _locked_run(run_id, path, cache_root):
    run_id = validate_run_id(run_id)
    target = checkpoint_file(path)
    if not target.exists():
        raise ValueError('没有已保存的调研记录。')
    # Read only the index to locate a run's registered root. Explicit override
    # supports older records or a moved cache directory.
    db = connect(target)
    try:
        row = db.execute('SELECT result FROM paperscout_runs WHERE id=?', (run_id,)).fetchone()
        saved_root = RunResult.model_validate_json(row[0]).cache_root if row else ''
    finally:
        db.close()
    root = Path(cache_root or saved_root or Cache().root).resolve()
    try:
        with FileLock(str(target.parent / f'{run_id}.lock'), timeout=0), cleanup_lock(root):
            ensure_idle(root)
            db = connect(target)
            try:
                # Also catch older running processes that predate cache leases.
                ids = [row[0] for row in db.execute('SELECT id FROM paperscout_runs')]
                for other_id in ids:
                    if other_id == run_id:
                        continue
                    try:
                        with FileLock(str(target.parent / f'{validate_run_id(other_id)}.lock'), timeout=0):
                            pass
                    except Timeout as exc:
                        raise ValueError('有调研正在执行，请等待调研结束后再删除。') from exc
                yield db, root
            finally:
                db.close()
    except Timeout as exc:
        raise ValueError('此调研正在执行或已有删除操作，请稍后重试。') from exc


def preview_deletion(run_id, path=None, *, cache_root=None, clean_cache=True):
    try:
        with _locked_run(run_id, path, cache_root) as (db, root):
            return _plan(db, run_id, root, clean_cache)
    except sqlite3.Error as exc:
        raise ValueError('无法读取调研数据库，请检查文件权限或占用情况后重试。') from exc


def delete_run(run_id, path=None, *, cache_root=None, clean_cache=True):
    """Recompute sharing under the lock; a preview is never authority to unlink."""
    try:
        return _delete_run(run_id, path, cache_root=cache_root, clean_cache=clean_cache)
    except sqlite3.Error as exc:
        raise ValueError('调研数据库删除失败；已清理的缓存不会恢复，调研记录保留以便重试。') from exc


def _delete_run(run_id, path, *, cache_root, clean_cache):
    with _locked_run(run_id, path, cache_root) as (db, root):
        plan = _plan(db, run_id, root, clean_cache)
        for file in plan.files:
            try:
                file.unlink(missing_ok=True)
            except OSError as exc:
                raise ValueError(f'缓存文件 {file.name} 清理失败；已清理的文件不会恢复，调研记录保留以便重试。') from exc
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        with db:
            for table in ('writes', 'checkpoints'):
                if table in tables:
                    db.execute(f'DELETE FROM {table} WHERE thread_id=?', (run_id,))
            db.execute('DELETE FROM paperscout_runs WHERE id=?', (run_id,))
        manifest = metadata_dir(root) / f'{run_id}.json'
        try:
            manifest.unlink(missing_ok=True)
        except OSError:
            plan.warnings.append('调研已删除，但缓存关联清单未能清理；可能会影响后续缓存空间回收。')
        return plan
