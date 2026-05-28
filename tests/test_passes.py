"""Tests for neuromorf.passes.validate_structure."""

import pytest
import numpy as np

from neuromorf.ir import NeuromorphIR, Neuron, Synapse
from neuromorf.passes.validate_structure import (
    validate_structure,
    StructureValidationError,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _neuron(nid: str) -> Neuron:
    """Minimal LIF neuron with a single-element threshold."""
    return Neuron(
        id=nid,
        type="LIF",
        params={"tau": np.array([20.0]), "v_threshold": np.array([1.0])},
    )


def _synapse(src: str, dst: str) -> Synapse:
    return Synapse(src_id=src, dst_id=dst, weight=np.array([1.0]))


def _make_ir(*synapses: Synapse, extra_neurons: dict | None = None) -> NeuromorphIR:
    """Build a NeuromorphIR with base neurons n0/n1/n2 plus optional extras."""
    neurons = {
        "n0": _neuron("n0"),
        "n1": _neuron("n1"),
        "n2": _neuron("n2"),
    }
    if extra_neurons:
        neurons.update(extra_neurons)
    return NeuromorphIR(
        target_hardware="cpu",
        neurons=neurons,
        synapses=list(synapses),
        input_neuron_ids=["n0"],
        output_neuron_ids=["n2"],
    )


# ---------------------------------------------------------------------------
# Test 1-3: Valid graph passes cleanly
# ---------------------------------------------------------------------------

class TestValidGraph:
    def test_valid_graph_no_error(self):
        ir = _make_ir(_synapse("n0", "n1"), _synapse("n1", "n2"))
        validate_structure(ir)  # must not raise

    def test_valid_graph_returns_ir(self):
        ir = _make_ir(_synapse("n0", "n1"), _synapse("n1", "n2"))
        result = validate_structure(ir)
        assert result is ir  # same object, mutated in-place

    def test_valid_graph_appends_log_entry(self):
        ir = _make_ir(_synapse("n0", "n1"), _synapse("n1", "n2"))
        validate_structure(ir)
        assert len(ir.transformation_log) == 1
        entry = ir.transformation_log[0]
        assert entry["pass"] == "ValidateStructure"
        assert entry["status"] == "passed"
        assert entry["affected_ids"] == []
        assert entry["params"] == {"allow_cycles": False}

    def test_log_entry_reflects_allow_cycles_param(self):
        ir = _make_ir(_synapse("n0", "n1"))
        validate_structure(ir, allow_cycles=True)
        entry = ir.transformation_log[-1]
        assert entry["params"]["allow_cycles"] is True

    def test_empty_graph_passes(self):
        ir = _make_ir()  # no synapses at all
        validate_structure(ir)
        assert ir.transformation_log[-1]["status"] == "passed"


# ---------------------------------------------------------------------------
# Test 4-6: Missing synapse endpoints
# ---------------------------------------------------------------------------

class TestMissingEndpoints:
    def test_missing_src_id_raises(self):
        ir = _make_ir(_synapse("ghost", "n1"))
        with pytest.raises(StructureValidationError):
            validate_structure(ir)

    def test_missing_dst_id_raises(self):
        ir = _make_ir(_synapse("n0", "ghost"))
        with pytest.raises(StructureValidationError):
            validate_structure(ir)

    def test_error_message_contains_missing_id(self):
        ir = _make_ir(_synapse("n0", "ghost_dst"))
        with pytest.raises(StructureValidationError, match="ghost_dst"):
            validate_structure(ir)

    def test_error_message_contains_src_and_dst(self):
        ir = _make_ir(_synapse("n0", "ghost_dst"))
        with pytest.raises(StructureValidationError, match="n0"):
            validate_structure(ir)

    def test_error_message_format(self):
        ir = _make_ir(_synapse("n0", "missing"))
        with pytest.raises(StructureValidationError, match="ValidateStructure"):
            validate_structure(ir)

    def test_all_missing_endpoints_reported_in_one_error(self):
        """Two synapses with different missing ids -> both ids in one exception."""
        ir = _make_ir(
            _synapse("ghost_a", "n1"),
            _synapse("n0", "ghost_b"),
        )
        with pytest.raises(StructureValidationError) as exc_info:
            validate_structure(ir)
        msg = str(exc_info.value)
        assert "ghost_a" in msg
        assert "ghost_b" in msg


# ---------------------------------------------------------------------------
# Test 7: Duplicate synapses
# ---------------------------------------------------------------------------

class TestDuplicateSynapse:
    def test_duplicate_raises(self):
        ir = _make_ir(_synapse("n0", "n1"), _synapse("n0", "n1"))
        with pytest.raises(StructureValidationError):
            validate_structure(ir)

    def test_error_message_contains_both_ids(self):
        ir = _make_ir(_synapse("n0", "n1"), _synapse("n0", "n1"))
        with pytest.raises(StructureValidationError, match="n0") as exc_info:
            validate_structure(ir)
        assert "n1" in str(exc_info.value)

    def test_non_duplicate_different_direction_ok(self):
        """n0→n1 and n1→n0 are two different synapses, not duplicates."""
        ir = _make_ir(_synapse("n0", "n1"), _synapse("n1", "n0"))
        validate_structure(ir, allow_cycles=True)  # cycle present, but not duplicate

    def test_duplicate_error_message_format(self):
        ir = _make_ir(_synapse("n0", "n1"), _synapse("n0", "n1"))
        with pytest.raises(StructureValidationError, match="Duplicate synapse"):
            validate_structure(ir)

    def test_all_duplicate_pairs_reported_in_one_error(self):
        """Two distinct duplicate pairs -> both pairs in one exception."""
        ir = _make_ir(
            _synapse("n0", "n1"), _synapse("n0", "n1"),   # pair A duplicated
            _synapse("n1", "n2"), _synapse("n1", "n2"),   # pair B duplicated
        )
        with pytest.raises(StructureValidationError) as exc_info:
            validate_structure(ir)
        msg = str(exc_info.value)
        # Both duplicate pairs must be described
        assert msg.count("appears more than once") == 2


# ---------------------------------------------------------------------------
# Test 8-9: Cycle detection
# ---------------------------------------------------------------------------

class TestCycles:
    def _cyclic_ir(self) -> NeuromorphIR:
        """A→B→C→A triangle."""
        return _make_ir(
            _synapse("n0", "n1"),
            _synapse("n1", "n2"),
            _synapse("n2", "n0"),
        )

    def test_cycle_raises_by_default(self):
        with pytest.raises(StructureValidationError):
            validate_structure(self._cyclic_ir())

    def test_cycle_raises_when_allow_cycles_false(self):
        with pytest.raises(StructureValidationError):
            validate_structure(self._cyclic_ir(), allow_cycles=False)

    def test_cycle_error_message_format(self):
        with pytest.raises(StructureValidationError, match="Cycle detected"):
            validate_structure(self._cyclic_ir())

    def test_cycle_does_not_raise_when_allowed(self):
        ir = self._cyclic_ir()
        validate_structure(ir, allow_cycles=True)  # must not raise

    def test_cycle_logs_warning_when_allowed(self):
        ir = self._cyclic_ir()
        validate_structure(ir, allow_cycles=True)
        warning = next(
            e for e in ir.transformation_log if e.get("status") == "warning"
        )
        assert warning["pass"] == "ValidateStructure"
        assert "Cycle" in warning["message"]
        assert "cycle_ids" in warning

    def test_cycle_warning_log_contains_node_ids(self):
        ir = self._cyclic_ir()
        validate_structure(ir, allow_cycles=True)
        warning = next(e for e in ir.transformation_log if e.get("status") == "warning")
        # At least one cycle node must appear in cycle_ids
        assert len(warning["cycle_ids"]) > 0
        for nid in warning["cycle_ids"]:
            assert nid in ir.neurons

    def test_passed_entry_still_added_when_cycle_allowed(self):
        ir = self._cyclic_ir()
        validate_structure(ir, allow_cycles=True)
        passed = next(
            e for e in ir.transformation_log if e.get("status") == "passed"
        )
        assert passed["params"]["allow_cycles"] is True

    def test_self_loop_detected(self):
        """Single neuron with a self-synapse is a cycle."""
        ir = _make_ir(_synapse("n0", "n0"))
        with pytest.raises(StructureValidationError, match="Cycle"):
            validate_structure(ir)
