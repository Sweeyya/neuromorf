"""Tests for neuromorf.frontend.nir_parser."""

import pytest
import numpy as np
import nir

from neuromorf.frontend.nir_parser import parse, UnsupportedNIRNodeError
from neuromorf.ir import NeuromorphIR, Neuron, Synapse


# ---------------------------------------------------------------------------
# Helpers — reusable NIR graph factories
# ---------------------------------------------------------------------------

def _lif_node(tau=20.0, r=1.0, v_leak=0.0, v_threshold=1.0):
    return nir.LIF(
        tau=np.array([tau]),
        r=np.array([r]),
        v_leak=np.array([v_leak]),
        v_threshold=np.array([v_threshold]),
    )


def _if_node(r=1.0, v_threshold=1.0):
    return nir.IF(
        r=np.array([r]),
        v_threshold=np.array([v_threshold]),
    )


def _minimal_linear_lif_graph():
    """Input → Linear(1×4) → LIF(1 neuron) → Output"""
    return nir.NIRGraph(
        nodes={
            "input":  nir.Input(input_type={"input": np.array([4])}),
            "linear": nir.Linear(weight=np.ones((1, 4))),
            "lif":    _lif_node(),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("input", "linear"), ("linear", "lif"), ("lif", "output")],
    )


# ---------------------------------------------------------------------------
# Test 1: parse minimal graph (input, linear, lif, output)
# ---------------------------------------------------------------------------

class TestMinimalGraph:
    def test_returns_neuromorf_ir(self):
        ir = parse(_minimal_linear_lif_graph())
        assert isinstance(ir, NeuromorphIR)

    def test_default_target_is_cpu(self):
        ir = parse(_minimal_linear_lif_graph())
        assert ir.target_hardware == "cpu"

    def test_loihi2_target_accepted(self):
        ir = parse(_minimal_linear_lif_graph(), target="loihi2")
        assert ir.target_hardware == "loihi2"


# ---------------------------------------------------------------------------
# Test 2: input_neuron_ids populated correctly
# ---------------------------------------------------------------------------

class TestInputNeuronIds:
    def test_single_input_id(self):
        ir = parse(_minimal_linear_lif_graph())
        assert ir.input_neuron_ids == ["input"]

    def test_multiple_input_ids(self):
        graph = nir.NIRGraph(
            nodes={
                "in_a":  nir.Input(input_type={"input": np.array([4])}),
                "in_b":  nir.Input(input_type={"input": np.array([4])}),
                "lin_a": nir.Linear(weight=np.ones((1, 4))),
                "lin_b": nir.Linear(weight=np.ones((1, 4))),
                "lif_a": _lif_node(),
                "lif_b": _lif_node(),
                "out_a": nir.Output(output_type={"output": np.array([1])}),
                "out_b": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[
                ("in_a", "lin_a"), ("lin_a", "lif_a"), ("lif_a", "out_a"),
                ("in_b", "lin_b"), ("lin_b", "lif_b"), ("lif_b", "out_b"),
            ],
        )
        ir = parse(graph)
        assert set(ir.input_neuron_ids) == {"in_a", "in_b"}


# ---------------------------------------------------------------------------
# Test 3: output_neuron_ids populated correctly
# ---------------------------------------------------------------------------

class TestOutputNeuronIds:
    def test_single_output_id(self):
        ir = parse(_minimal_linear_lif_graph())
        assert ir.output_neuron_ids == ["output"]

    def test_multiple_output_ids(self):
        graph = nir.NIRGraph(
            nodes={
                "in_a":  nir.Input(input_type={"input": np.array([4])}),
                "in_b":  nir.Input(input_type={"input": np.array([4])}),
                "lin_a": nir.Linear(weight=np.ones((1, 4))),
                "lin_b": nir.Linear(weight=np.ones((1, 4))),
                "lif_a": _lif_node(),
                "lif_b": _lif_node(),
                "out_a": nir.Output(output_type={"output": np.array([1])}),
                "out_b": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[
                ("in_a", "lin_a"), ("lin_a", "lif_a"), ("lif_a", "out_a"),
                ("in_b", "lin_b"), ("lin_b", "lif_b"), ("lif_b", "out_b"),
            ],
        )
        ir = parse(graph)
        assert set(ir.output_neuron_ids) == {"out_a", "out_b"}


# ---------------------------------------------------------------------------
# Test 4: Input/Output nodes NOT in neurons dict
# ---------------------------------------------------------------------------

class TestInputOutputNotInNeurons:
    def test_input_not_in_neurons(self):
        ir = parse(_minimal_linear_lif_graph())
        assert "input" not in ir.neurons

    def test_output_not_in_neurons(self):
        ir = parse(_minimal_linear_lif_graph())
        assert "output" not in ir.neurons

    def test_only_lif_in_neurons(self):
        ir = parse(_minimal_linear_lif_graph())
        assert set(ir.neurons.keys()) == {"lif"}


# ---------------------------------------------------------------------------
# Test 5: Linear node NOT in neurons dict
# ---------------------------------------------------------------------------

class TestLinearNotInNeurons:
    def test_linear_not_in_neurons(self):
        ir = parse(_minimal_linear_lif_graph())
        assert "linear" not in ir.neurons

    def test_affine_not_in_neurons(self):
        W = np.arange(4, dtype=float).reshape(1, 4)
        graph = nir.NIRGraph(
            nodes={
                "input":  nir.Input(input_type={"input": np.array([4])}),
                "affine": nir.Affine(weight=W, bias=np.zeros(1)),
                "lif":    _lif_node(),
                "output": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "affine"), ("affine", "lif"), ("lif", "output")],
        )
        ir = parse(graph)
        assert "affine" not in ir.neurons


# ---------------------------------------------------------------------------
# Test 6: Synapse has correct weight from Linear/Affine node
# ---------------------------------------------------------------------------

class TestSynapseWeight:
    def test_linear_weight_folded_into_synapse(self):
        W = np.ones((1, 4))
        ir = parse(_minimal_linear_lif_graph())
        folded = next(s for s in ir.synapses if s.src_id == "input")
        assert folded.dst_id == "lif"
        np.testing.assert_array_equal(folded.weight, W)

    def test_affine_weight_folded_into_synapse(self):
        W = np.arange(4, dtype=float).reshape(1, 4)
        graph = nir.NIRGraph(
            nodes={
                "input":  nir.Input(input_type={"input": np.array([4])}),
                "affine": nir.Affine(weight=W, bias=np.zeros(1)),
                "lif":    _lif_node(),
                "output": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "affine"), ("affine", "lif"), ("lif", "output")],
        )
        ir = parse(graph)
        folded = next(s for s in ir.synapses if s.src_id == "input")
        assert folded.dst_id == "lif"
        np.testing.assert_array_equal(folded.weight, W)

    def test_direct_connection_weight_shape(self):
        """LIF → Output direct edge gets np.ones shaped to the connection."""
        ir = parse(_minimal_linear_lif_graph())
        direct = next(s for s in ir.synapses if s.src_id == "lif")
        assert direct.dst_id == "output"
        # LIF has 1 neuron, Output has 1 unit → weight shape (1, 1)
        assert direct.weight.shape == (1, 1)
        np.testing.assert_array_equal(direct.weight, np.ones((1, 1)))

    def test_synapse_src_and_dst_are_real_ids(self):
        """After weight folding, no synapse endpoint should be a weight node."""
        ir = parse(_minimal_linear_lif_graph())
        for s in ir.synapses:
            assert s.src_id != "linear", "Linear node leaked into synapse src_id"
            assert s.dst_id != "linear", "Linear node leaked into synapse dst_id"


# ---------------------------------------------------------------------------
# Test 7: unknown node type raises UnsupportedNIRNodeError
# ---------------------------------------------------------------------------

class TestUnknownNodeRaises:
    def test_delay_node_raises(self):
        graph = nir.NIRGraph(
            nodes={
                "input":  nir.Input(input_type={"input": np.array([1])}),
                "delay":  nir.Delay(delay=np.array([1.0])),
                "output": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "delay"), ("delay", "output")],
        )
        with pytest.raises(UnsupportedNIRNodeError, match="NIR Parser"):
            parse(graph)

    def test_error_message_contains_node_id(self):
        graph = nir.NIRGraph(
            nodes={
                "input":    nir.Input(input_type={"input": np.array([1])}),
                "bad_node": nir.Delay(delay=np.array([1.0])),
                "output":   nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "bad_node"), ("bad_node", "output")],
        )
        with pytest.raises(UnsupportedNIRNodeError, match="bad_node"):
            parse(graph)

    def test_error_message_contains_node_type(self):
        graph = nir.NIRGraph(
            nodes={
                "input":    nir.Input(input_type={"input": np.array([1])}),
                "bad_node": nir.Delay(delay=np.array([1.0])),
                "output":   nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "bad_node"), ("bad_node", "output")],
        )
        with pytest.raises(UnsupportedNIRNodeError, match="Delay"):
            parse(graph)


# ---------------------------------------------------------------------------
# LIF / IF param extraction (regression coverage)
# ---------------------------------------------------------------------------

class TestLIFParams:
    def test_lif_param_keys(self):
        ir = parse(_minimal_linear_lif_graph())
        params = ir.neurons["lif"].params
        for key in ("tau", "r", "v_leak", "v_threshold", "v_reset"):
            assert key in params, f"Missing LIF param: {key}"

    def test_lif_param_values_are_ndarrays(self):
        ir = parse(_minimal_linear_lif_graph())
        for key, val in ir.neurons["lif"].params.items():
            assert isinstance(val, np.ndarray), f"Param {key!r} is not ndarray"

    def test_lif_tau_value(self):
        ir = parse(_minimal_linear_lif_graph())
        np.testing.assert_array_equal(ir.neurons["lif"].params["tau"], np.array([20.0]))


class TestIFParams:
    def _if_graph(self):
        return nir.NIRGraph(
            nodes={
                "input":  nir.Input(input_type={"input": np.array([1])}),
                "if_n":   _if_node(r=2.0, v_threshold=0.5),
                "output": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "if_n"), ("if_n", "output")],
        )

    def test_if_param_keys(self):
        ir = parse(self._if_graph())
        params = ir.neurons["if_n"].params
        assert "r" in params
        assert "v_threshold" in params

    def test_if_r_value(self):
        ir = parse(self._if_graph())
        np.testing.assert_array_equal(ir.neurons["if_n"].params["r"], np.array([2.0]))

    def test_if_v_threshold_value(self):
        ir = parse(self._if_graph())
        np.testing.assert_array_equal(
            ir.neurons["if_n"].params["v_threshold"], np.array([0.5])
        )
