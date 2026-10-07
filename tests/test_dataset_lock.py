"""Dataset pinning / checksum verification (#9)."""
from pathlib import Path

import pytest
import yaml

from utils.dataset_lock import (DatasetChecksumError, fingerprint_rows, fingerprint_split,
                                verify, verify_fixture)

ROOT = Path(__file__).resolve().parents[1]


def test_committed_fixtures_match_lock():
    cfg = yaml.safe_load((ROOT / "config" / "dataset_fixtures.yaml").read_text())
    for name, fx in cfg["fixtures"].items():
        verify_fixture(ROOT / fx["path"], fx["sha256"])


def test_wrong_sha_fails():
    cfg = yaml.safe_load((ROOT / "config" / "dataset_fixtures.yaml").read_text())
    fx = cfg["fixtures"]["mnli_sample"]
    with pytest.raises(DatasetChecksumError):
        verify_fixture(ROOT / fx["path"], "0" * 64)


def test_fingerprint_is_order_and_content_sensitive():
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert fingerprint_rows(rows) == fingerprint_rows([{"b": "x", "a": 1}, {"b": "y", "a": 2}])
    assert fingerprint_rows(rows) != fingerprint_rows(list(reversed(rows)))
    assert fingerprint_rows(rows) != fingerprint_rows([{"a": 1, "b": "x"}, {"a": 2, "b": "z"}])


def test_unpinned_passes_and_split_fingerprint_limits_rows():
    verify("abc", None, "unpinned")
    rows = [{"i": i} for i in range(10)]
    assert fingerprint_split(rows, n=3) == fingerprint_rows(rows[:3])
