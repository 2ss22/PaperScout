"""Keep offline tests from touching the user's persisted research sessions."""
import pytest


@pytest.fixture(autouse=True)
def isolated_checkpoints(monkeypatch, tmp_path):
    monkeypatch.setenv('PAPERSCOUT_CHECKPOINT_DB', str(tmp_path / 'checkpoints.sqlite'))
