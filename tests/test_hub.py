"""Tests for locating, fetching and checking the released artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import errors
from safetensors.torch import save_file

import alhena
from alhena import BackboneNotFoundError, BundleError
from alhena._constants import (
    ADAPTERS_FILE,
    BUNDLE_CONFIG_FILE,
    HEADS_FILE,
    HUB_REVISION,
    REFERENCES_REPO,
    REGISTRY_FILE,
    WEIGHTS_REPO,
)
from alhena.hub import (
    BUNDLE_FILES,
    SHARD_DIRECTORY,
    bundle_fingerprint,
    cache_root,
    ensure_bundle,
    ensure_reference,
    ensure_registry,
    file_digest,
    find_backbone,
)

HEADS = {
    "img_tok_proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    "st_node_proj.weight": torch.eye(3),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FakeHub:
    """Serves repository files from local folders, standing in for `hf_hub_download`."""

    def __init__(self, repositories: dict[str, Path]) -> None:
        self.repositories = repositories
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(
        self, repo_id: str, filename: str, cache_dir: str, revision: str | None = None
    ) -> str:
        self.calls.append((repo_id, filename, revision))
        folder = "models--" + repo_id.replace("/", "--")
        target = Path(cache_dir) / folder / "snapshots" / str(revision) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((self.repositories[repo_id] / filename).read_bytes())
        return str(target)


def serve(monkeypatch: pytest.MonkeyPatch, repositories: dict[str, Path]) -> FakeHub:
    hub = FakeHub(repositories)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub)
    return hub


def fail_with(monkeypatch: pytest.MonkeyPatch, kind: type[Exception], message: str) -> None:
    """Make every download raise a huggingface-hub error, built without an HTTP response."""
    error = kind.__new__(kind)
    Exception.__init__(error, message)

    def hf_hub_download(**kwargs: object) -> str:
        raise error

    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)


def write_bundle(directory: Path, teacher: dict[str, str] | None = None) -> Path:
    """Write the bundle files and a configuration declaring their digests."""
    directory.mkdir(parents=True)
    (directory / ADAPTERS_FILE).write_bytes(b"adapter weights")
    save_file(HEADS, str(directory / HEADS_FILE))
    config: dict[str, object] = {
        "sha256": {name: sha256((directory / name).read_bytes()) for name in BUNDLE_FILES}
    }
    if teacher is not None:
        config["teacher"] = teacher
    (directory / BUNDLE_CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
    return directory


def write_references(directory: Path, shards: dict[str, bytes]) -> Path:
    """Write a registry declaring the digest of each shard, and the shards."""
    entries: dict[str, dict[str, str]] = {}
    for name, payload in shards.items():
        filename = f"{SHARD_DIRECTORY}/{name}.safetensors"
        (directory / filename).parent.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(payload)
        entries[name] = {"file": filename, "sha256": sha256(payload)}
    registry = json.dumps({"shards": entries})
    (directory / REGISTRY_FILE).write_text(registry, encoding="utf-8")
    return directory


def write_weights(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "pytorch_model.bin"
    path.write_bytes(b"backbone weights")
    return path


# --- Cache ----------------------------------------------------------------------------------


def test_cache_root_reads_alhena_cache_unless_given_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("ALHENA_CACHE", str(tmp_path / "declared"))
    assert cache_root() == tmp_path / "declared"
    assert (tmp_path / "declared").is_dir()
    assert cache_root(tmp_path / "given") == tmp_path / "given"


def test_cache_root_defaults_to_a_user_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name in ("ALHENA_CACHE", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    root = cache_root()
    assert root.is_dir()
    assert "alhena" in root.relative_to(tmp_path).parts


def test_file_digest_matches_hashlib(tmp_path: Path):
    payload = bytes(range(256)) * 50
    path = tmp_path / "shard.safetensors"
    path.write_bytes(payload)
    assert file_digest(path) == sha256(payload)
    assert file_digest(path, chunk_size=1000) == sha256(payload)


# --- Downloads ------------------------------------------------------------------------------


def test_ensure_bundle_fetches_the_pinned_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    remote = write_bundle(tmp_path / "remote")
    hub = serve(monkeypatch, {WEIGHTS_REPO: remote})
    directory = ensure_bundle(cache_dir=tmp_path / "cache")
    assert directory.is_relative_to(tmp_path / "cache")
    for name in (BUNDLE_CONFIG_FILE, *BUNDLE_FILES):
        assert (directory / name).read_bytes() == (remote / name).read_bytes()
    assert {revision for _, _, revision in hub.calls} == {HUB_REVISION}


def test_ensure_bundle_refuses_a_file_with_another_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    remote = write_bundle(tmp_path / "remote")
    (remote / ADAPTERS_FILE).write_bytes(b"other weights")
    serve(monkeypatch, {WEIGHTS_REPO: remote})
    with pytest.raises(BundleError, match="SHA-256"):
        ensure_bundle(cache_dir=tmp_path / "cache")


def test_ensure_bundle_refuses_a_path_to_no_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    hub = serve(monkeypatch, {})
    with pytest.raises(BundleError, match="No bundle directory"):
        ensure_bundle(tmp_path / "models" / "bundle")
    assert hub.calls == []


def test_ensure_bundle_reports_a_gated_repository(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fail_with(monkeypatch, errors.GatedRepoError, "403 Forbidden")
    with pytest.raises(BundleError, match="hf auth login"):
        ensure_bundle(cache_dir=tmp_path / "cache")


def test_ensure_registry_fetches_the_pinned_or_named_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    remote = write_references(tmp_path / "remote", {"breast": b"breast cells"})
    hub = serve(monkeypatch, {REFERENCES_REPO: remote})
    path = ensure_registry(cache_dir=tmp_path / "cache")
    assert path.read_bytes() == (remote / REGISTRY_FILE).read_bytes()
    ensure_registry(cache_dir=tmp_path / "cache", revision="abc1234")
    assert hub.calls == [
        (REFERENCES_REPO, REGISTRY_FILE, HUB_REVISION),
        (REFERENCES_REPO, REGISTRY_FILE, "abc1234"),
    ]


def test_ensure_reference_fetches_the_shard_of_a_tissue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    shards = {"breast": b"breast cells", "intestine": b"intestine cells"}
    hub = serve(monkeypatch, {REFERENCES_REPO: write_references(tmp_path / "remote", shards)})
    path = ensure_reference("Colon", cache_dir=tmp_path / "cache")
    assert path.read_bytes() == b"intestine cells"
    assert hub.calls == [
        (REFERENCES_REPO, REGISTRY_FILE, HUB_REVISION),
        (REFERENCES_REPO, f"{SHARD_DIRECTORY}/intestine.safetensors", HUB_REVISION),
    ]


def test_ensure_reference_refuses_a_shard_with_another_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    remote = write_references(tmp_path / "remote", {"breast": b"breast cells"})
    (remote / SHARD_DIRECTORY / "breast.safetensors").write_bytes(b"other cells")
    serve(monkeypatch, {REFERENCES_REPO: remote})
    with pytest.raises(alhena.ReferenceError, match="SHA-256"):
        ensure_reference("Breast", cache_dir=tmp_path / "cache")


def test_ensure_reference_reports_a_transport_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fail_with(monkeypatch, errors.HfHubHTTPError, "503 Service Unavailable")
    with pytest.raises(alhena.ReferenceError, match="503 Service Unavailable"):
        ensure_reference("Breast", cache_dir=tmp_path / "cache")


# --- Backbone -------------------------------------------------------------------------------


def test_find_backbone_reads_alhena_backbone_unless_given_a_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    declared = write_weights(tmp_path / "declared")
    explicit = write_weights(tmp_path / "explicit")
    monkeypatch.setenv("ALHENA_BACKBONE", str(declared.parent))
    assert find_backbone() == declared
    assert find_backbone(explicit) == explicit


def test_find_backbone_reports_weights_found_nowhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # huggingface-hub reads HF_HUB_CACHE once, at import, so the value it read is replaced.
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(BackboneNotFoundError, match="ALHENA_BACKBONE"):
        find_backbone()


# --- Identity and names ---------------------------------------------------------------------


def test_bundle_fingerprint_is_computed_from_the_weights(tmp_path: Path):
    teacher = {"scconcept_model": "model-a", "scconcept_version": "v9"}
    directory = write_bundle(tmp_path / "bundle", teacher=teacher)
    adapters = sha256((directory / ADAPTERS_FILE).read_bytes())
    heads = [
        sha256(HEADS[name].numpy().tobytes())
        for name in ("img_tok_proj.weight", "st_node_proj.weight")
    ]
    fields = ["alhena-cell-space-v1", adapters, *heads, "model-a", "v9"]
    cell_space = sha256("|".join(fields).encode("utf-8"))
    assert bundle_fingerprint(directory) == (adapters[:16], cell_space[:16])
