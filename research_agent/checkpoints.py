"""Local SQLite checkpoints and a compact index of resumable research jobs."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone

from .models import RunResult


def checkpoint_file(path=None) -> Path:
    return Path(path or os.getenv('PAPERSCOUT_CHECKPOINT_DB',
                str(Path(os.getenv('PAPERSCOUT_CACHE_DIR', '.cache/paperscout')) / 'checkpoints.sqlite'))).resolve()


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise ValueError('运行 ID 必须是 32 位十六进制标识。')
    return run_id


def connect(path=None):
    target = checkpoint_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target, timeout=30, check_same_thread=False)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS paperscout_runs '
               '(id TEXT PRIMARY KEY, updated TEXT NOT NULL, result TEXT NOT NULL)')
    db.commit()
    return db


def save_run(db, run: RunResult):
    db.execute('INSERT INTO paperscout_runs VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE '
               'SET updated=excluded.updated, result=excluded.result',
               (run.run_id, datetime.now(timezone.utc).isoformat(), run.model_dump_json()))
    db.commit()


def load_run(run_id: str, path=None) -> RunResult | None:
    validate_run_id(run_id)
    if not checkpoint_file(path).exists():
        return None
    db = connect(path)
    try:
        row = db.execute('SELECT result FROM paperscout_runs WHERE id=?', (run_id,)).fetchone()
        return RunResult.model_validate_json(row[0]) if row else None
    finally:
        db.close()


def list_runs(path=None, limit=15) -> list[RunResult]:
    if not checkpoint_file(path).exists():
        return []
    db = connect(path)
    try:
        rows = db.execute('SELECT result FROM paperscout_runs ORDER BY updated DESC LIMIT ?', (min(limit, 50),)).fetchall()
        return [RunResult.model_validate_json(row[0]) for row in rows]
    finally:
        db.close()
