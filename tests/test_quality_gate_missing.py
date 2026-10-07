"""Required gates without data must fail (no vacuous pass)."""
from verification.quality_gate import QualityGateEnforcer


def test_missing_inputs_fail_required_gates():
    enforcer = QualityGateEnforcer()
    result = enforcer.enforce_all_gates(model_name="demo-model")
    assert result.passed is False
    assert result.summary["gates_passed"] == 0
    assert result.summary["gates_failed"] == 4
    for gate in ("layer", "task", "parity", "performance"):
        assert result.gates_passed.get(gate) is False
        assert result.summary["gate_status_detail"][gate] == "missing"


def test_unknown_task_type_fails():
    enforcer = QualityGateEnforcer()
    result = enforcer.enforce_all_gates(
        model_name="demo-model",
        task_data={"task_type": "not-a-real-task"},
        required_gates=["task"],
    )
    assert result.passed is False
    assert result.gates_passed.get("task") is False
