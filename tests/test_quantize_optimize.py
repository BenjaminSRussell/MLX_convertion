"""quantize.py / optimize.py on a fixture encoder (#5)."""
import json

import numpy as np
import pytest

from scripts import optimize, quantize


@pytest.fixture()
def encoder_npz(tmp_path):
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((512, 128)).astype(np.float32)
    weights = {
        "embeddings.word_embeddings.weight": emb,
        "lm_head.weight": emb.copy(),  # tied
        "encoder.layer.0.attention.query.weight": rng.standard_normal((128, 128)).astype(np.float32),
        "encoder.layer.0.attention.query.bias": rng.standard_normal(128).astype(np.float32),
        "encoder.layer.0.output.LayerNorm.weight": np.ones(128, dtype=np.float32),
        "optimizer.state.0.exp_avg": np.zeros((4, 4), dtype=np.float32),
    }
    path = tmp_path / "encoder.npz"
    np.savez(path, **weights)
    return path, weights


@pytest.mark.parametrize("bits,min_cos", [(8, 0.9999), (4, 0.99)])
def test_quantize_roundtrip_accuracy(bits, min_cos):
    w = np.random.default_rng(1).standard_normal((64, 256)).astype(np.float32)
    restored = quantize.dequantize_array(quantize.quantize_array(w, bits=bits, group_size=64), bits=bits)
    assert restored.shape == w.shape
    cos = float(np.dot(w.ravel(), restored.ravel()) / (np.linalg.norm(w) * np.linalg.norm(restored)))
    assert cos >= min_cos


def test_quantize_npz_smaller_and_gate(encoder_npz, tmp_path):
    src, _ = encoder_npz
    dst = tmp_path / "q8.npz"
    meta = quantize.quantize_npz(src, dst, bits=8)
    assert meta["output_bytes"] < meta["input_bytes"]
    assert meta["min_cosine"] > 0.999
    assert "encoder.layer.0.output.LayerNorm.weight" in meta["kept"]
    assert quantize.main([str(src), str(tmp_path / "q4.npz"), "--bits", "4", "--min-cosine", "0.98"]) == 0
    assert quantize.main([str(src), str(tmp_path / "q4b.npz"), "--bits", "4", "--min-cosine", "0.99999"]) == 1


def test_quantize_model_passthrough_for_non_mlx():
    sentinel = object()
    assert quantize.quantize_model(sentinel, {"bits": 8}) is sentinel
    with pytest.raises(ValueError):
        quantize.quantize_model(sentinel, {"bits": 3})


def test_optimize_drops_ties_and_restores(encoder_npz, tmp_path):
    src, weights = encoder_npz
    dst = tmp_path / "opt.npz"
    report = optimize.optimize_npz(src, dst)
    assert report["dropped"] == ["optimizer.state.0.exp_avg"]
    assert report["tied"] == {"lm_head.weight": "embeddings.word_embeddings.weight"}
    assert report["output_bytes"] < report["input_bytes"]
    with np.load(dst) as data:
        restored = optimize.restore_ties({k: data[k] for k in data.files})
    np.testing.assert_array_equal(restored["lm_head.weight"], weights["lm_head.weight"])
    assert "optimizer.state.0.exp_avg" not in restored


def test_optimize_cli_fp16(encoder_npz, tmp_path, capsys):
    src, _ = encoder_npz
    assert optimize.main([str(src), str(tmp_path / "o16.npz"), "--fp16", "--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["fp16"] is True
    with np.load(tmp_path / "o16.npz") as data:
        assert data["encoder.layer.0.attention.query.weight"].dtype == np.float16
