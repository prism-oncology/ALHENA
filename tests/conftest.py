"""Settings every test runs under."""

from __future__ import annotations

import pytest

from alhena._progress import PROGRESS_ENV_VAR


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep progress bars off, and point the download cache at an empty folder."""
    monkeypatch.setenv(PROGRESS_ENV_VAR, "1")
    monkeypatch.setenv("ALHENA_CACHE", str(tmp_path_factory.mktemp("alhena_cache")))
    monkeypatch.delenv("ALHENA_BACKBONE", raising=False)
