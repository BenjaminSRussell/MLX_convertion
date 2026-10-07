"""Run registry (#6), gate report (#8), stage resume (#13)."""
import pytest

from scripts import report_gates
from utils.run_registry import RunRegistry
from verification.quality_gate import QualityGateEnforcer, QualityGateResult


@pytest.fixture()
def reg(tmp_path):
    return RunRegistry(tmp_path / "runs.db")


def test_successful_run_records_metrics(reg):
    rid = reg.start_run("bert-nli", strategy="encoder", bits=8)
    reg.add_artifact(rid, "weights", "artifacts/bert-nli/8/weights.npz")
    reg.finish_run(rid, "passed", {"latency_ms": 12.5, "cosine": 0.999})
    run = reg.get_run(rid)
    assert run["status"] == "passed"
    assert run["metrics"]["cosine"] == pytest.approx(0.999)
    assert run["artifacts"][0]["kind"] == "weights"
    assert len(reg.list_runs()) == 1


def test_failed_gate_still_recorded(reg):
    result = QualityGateEnforcer().enforce_all_gates(model_name="demo")
    rid = QualityGateEnforcer().record_to_registry(result, registry=reg, strategy="encoder", bits=8)
    run = reg.get_run(rid)
    assert run["status"] == "failed"
    assert run["metrics"]["gate_layer"] == 0.0


def test_report_exit_code(reg, tmp_path, capsys):
    ok = QualityGateResult(model_name="good", passed=True,
                           gates_passed={g: True for g in ("layer", "task", "parity", "performance")})
    reg.record_gate_result(ok)
    assert report_gates.main(["--db", str(reg.db_path)]) == 0
    assert "good" in capsys.readouterr().out
    bad = QualityGateResult(model_name="bad", passed=False, gates_passed={"parity": False})
    reg.record_gate_result(bad)
    assert report_gates.main(["--db", str(reg.db_path)]) == 1
    assert report_gates.main(["--db", str(reg.db_path), "--model", "good"]) == 0


def test_resume_skips_done_and_retries_failed(reg):
    rid = reg.start_run("m")
    calls = []

    def convert():
        calls.append("convert")
        return "out/convert"

    attempts = {"n": 0}

    def quantize():
        attempts["n"] += 1
        calls.append("quantize")
        if attempts["n"] == 1:
            raise RuntimeError("boom")
        return "out/q"

    with pytest.raises(RuntimeError):
        reg.run_stages(rid, [("convert", convert), ("quantize", quantize)])
    assert reg.stage_status(rid, "convert") == "done"
    assert reg.stage_status(rid, "quantize") == "failed"

    outcome = reg.run_stages(rid, [("convert", convert), ("quantize", quantize)])
    assert outcome == {"convert": "skipped", "quantize": "done"}
    assert calls == ["convert", "quantize", "quantize"]
    assert reg.get_run(rid)["stages"]["quantize"]["attempts"] == 2
