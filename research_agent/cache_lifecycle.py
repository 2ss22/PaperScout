"""Per-run cache references and leases coordinated with explicit cleanup."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import tempfile

from filelock import FileLock, Timeout

CACHE_FILENAME = re.compile(r'[a-f0-9]{64}\.(?:json|pdf|png|bin)')
_active = ContextVar('paperscout_cache_references', default=None)


def metadata_dir(root):
    root = Path(root).resolve()
    directory = root / '.runs'
    if directory.is_symlink():
        raise ValueError('缓存关联目录不能是符号链接。')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cleanup_lock(root):
    directory = metadata_dir(root)
    return FileLock(str(directory / 'cleanup.lock'), timeout=0)


def read_references(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not isinstance(value.get('files'), list):
        raise ValueError('缓存关联记录无效。')
    if any(not isinstance(name, str) or not CACHE_FILENAME.fullmatch(name) for name in value['files']):
        raise ValueError('缓存关联记录包含无效文件名。')
    return set(value['files'])


@contextmanager
def cache_session(root, run_id):
    """Register a lease before reading any files; cleanup blocks new leases."""
    if root is None:
        yield
        return
    lease = FileLock(str(metadata_dir(root) / f'{run_id}.lock'), timeout=0)
    try:
        with cleanup_lock(root):
            lease.acquire()
    except Timeout as exc:
        raise ValueError('正在清理缓存或当前调研正在执行，请稍后重试。') from exc
    try:
        yield
    finally:
        lease.release()


def ensure_idle(root):
    for path in metadata_dir(root).glob('*.lock'):
        if path.name == 'cleanup.lock':
            continue
        lease = FileLock(str(path), timeout=0)
        try:
            lease.acquire()
        except Timeout as exc:
            raise ValueError('有调研正在使用缓存，请等待调研结束后再删除。') from exc
        else:
            lease.release()


@contextmanager
def track_cache(root, run_id, checkpoint_path):
    if root is None:
        yield
        return
    root = Path(root).resolve()
    path = metadata_dir(root) / f'{run_id}.json'
    files = read_references(path) if path.exists() else set()
    token = _active.set({'root': root, 'path': path, 'files': files, 'run_id': run_id,
                         'checkpoint_path': str(checkpoint_path)})
    try:
        yield
    finally:
        _active.reset(token)


def record_cache_file(path):
    tracking = _active.get()
    if tracking is None:
        return
    path = Path(path)
    if path.parent.resolve() != tracking['root'] or not CACHE_FILENAME.fullmatch(path.name):
        return
    if path.name in tracking['files']:
        return
    tracking['files'].add(path.name)
    value = {'run_id': tracking['run_id'], 'checkpoint_path': tracking['checkpoint_path'],
             'files': sorted(tracking['files'])}
    with tempfile.NamedTemporaryFile('w', dir=tracking['path'].parent, encoding='utf-8', delete=False) as f:
        json.dump(value, f, ensure_ascii=False)
        temporary = f.name
    os.replace(temporary, tracking['path'])
