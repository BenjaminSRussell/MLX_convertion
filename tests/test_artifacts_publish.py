"""Artifact layout + metadata.json (#11) and HF publish dry-run (#12)."""
import json

import pytest

from scripts import publish_hf, upload
from utils.artifacts import artifact_dir, build_metadata, read_metadata, verify_files, write_metadata


@pytest.fixture()
def art(tmp_path):
    d = artifact_dir("typeform/distilbert-base-uncased-mnli", 8, root=tmp_path / "artifacts")
    d.mkdir(parents=True)
    (d / "weights.npz").write_bytes(b"\x00" * 128)
    (d / "config.json").write_text(json.dumps({"hidden": 768}))
    meta = build_metadata("typeform/distilbert-base-uncased-mnli", 8, d, strategy="encoder",
                          metrics={"min_cosine": 0.9999}, git_sha="abc1234", run_id=7)
    write_metadata(d, meta)
    return d


def test_layout_and_metadata(art):
    assert art.parts[-2:] == ("typeform__distilbert-base-uncased-mnli", "8bit")
    meta = read_metadata(art)
    assert meta["strategy"] == "encoder" and meta["run_id"] == 7
    assert {f["path"] for f in meta["files"]} == {"weights.npz", "config.json"}
    assert verify_files(art) == []
    (art / "weights.npz").write_bytes(b"tampered")
    assert any("mismatch" in p for p in verify_files(art))


def test_publish_dry_run_prints_plan_and_card(art, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert publish_hf.main([str(art), "--namespace", "someone"]) == 0
    out = capsys.readouterr().out
    assert "someone/distilbert-base-uncased-mnli-mlx-8bit" in out
    assert "weights.npz" in out and "min_cosine" in out
    assert "{{" not in out
    assert not (art / "README.md").exists()  # dry-run writes nothing


def test_publish_execute_requires_token(art, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert publish_hf.main([str(art), "--execute"]) == 3


def test_upload_dry_run_from_artifact_dir(art, capsys):
    assert upload.main(["--artifact-dir", str(art), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "weights.npz" in out
