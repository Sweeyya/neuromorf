"""Tests for neuromorf.backends.cpu_backend.CPUBackend."""

import pytest
import numpy as np

from neuromorf.ir import NeuromorphIR, Neuron, Synapse
from neuromorf.backends.cpu_backend import CPUBackend


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _if_ir() -> NeuromorphIR:
    """Single IF neuron that serves as both input and output.

    R=1, v_threshold=1.0, v_reset=0.0.
    One unit: input I >= 1.0 fires it in a single timestep.
    """
    neuron = Neuron("n0", "IF", {
        "r":           np.array([1.0]),
        "v_threshold": np.array([1.0]),
        "v_reset":     np.array([0.0]),
    })
    return NeuromorphIR(
        target_hardware="cpu",
        neurons={"n0": neuron},
        synapses=[],
        input_neuron_ids=["n0"],
        output_neuron_ids=["n0"],
    )


def _lif_ir() -> NeuromorphIR:
    """Single LIF neuron that serves as both input and output.

    tau=20, R=1, v_threshold=1.0, v_leak=0.0, v_reset=0.0.
    One unit: input I >= 1.0 fires it in a single timestep (since v_leak=0,
    dv = R*I = I, same as IF for the first step from v=0).
    """
    neuron = Neuron("n0", "LIF", {
        "tau":         np.array([20.0]),
        "r":           np.array([1.0]),
        "v_threshold": np.array([1.0]),
        "v_leak":      np.array([0.0]),
        "v_reset":     np.array([0.0]),
    })
    return NeuromorphIR(
        target_hardware="cpu",
        neurons={"n0": neuron},
        synapses=[],
        input_neuron_ids=["n0"],
        output_neuron_ids=["n0"],
    )


def _two_neuron_ir() -> NeuromorphIR:
    """n0 (IF, input) -> synapse(weight=[[2.0]]) -> n1 (IF, output).

    n0 fires when it receives I >= 1.0; its spike (1.0) then drives n1
    with effective current 2.0, which fires n1 (threshold=1.0) one timestep
    later.
    """
    n0 = Neuron("n0", "IF", {
        "r": np.array([1.0]), "v_threshold": np.array([1.0]),
        "v_reset": np.array([0.0]),
    })
    n1 = Neuron("n1", "IF", {
        "r": np.array([1.0]), "v_threshold": np.array([1.0]),
        "v_reset": np.array([0.0]),
    })
    syn = Synapse(src_id="n0", dst_id="n1", weight=np.array([[2.0]]))
    return NeuromorphIR(
        target_hardware="cpu",
        neurons={"n0": n0, "n1": n1},
        synapses=[syn],
        input_neuron_ids=["n0"],
        output_neuron_ids=["n1"],
    )


# ---------------------------------------------------------------------------
# Test 1: initialize_state structure
# ---------------------------------------------------------------------------

class TestInitializeState:
    def test_has_entry_per_neuron(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        assert "n0" in state

    def test_v_key_present(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        assert "v" in state["n0"]

    def test_v_shape_matches_neuron_size(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        assert state["n0"]["v"].shape == (1,)

    def test_v_initialized_to_zeros(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        np.testing.assert_array_equal(state["n0"]["v"], np.zeros(1))

    def test_no_i_key_for_if_neuron(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        assert "i" not in state["n0"]

    def test_lif_neuron_has_no_i_key(self):
        backend = CPUBackend(_lif_ir())
        state = backend.initialize_state()
        assert "i" not in state["n0"]


# ---------------------------------------------------------------------------
# Test 2: IF neuron fires when input exceeds threshold
# ---------------------------------------------------------------------------

class TestIFFires:
    def test_fires_at_threshold(self):
        ir = _if_ir()
        backend = CPUBackend(ir)
        state = backend.initialize_state()
        input_data = np.array([[1.1]])   # (1 timestep, 1 input neuron)
        output, _ = backend.run(input_data, state, num_timesteps=1)
        assert output[0, 0] == 1.0

    def test_output_is_one_not_more(self):
        ir = _if_ir()
        backend = CPUBackend(ir)
        state = backend.initialize_state()
        output, _ = backend.run(np.array([[2.0]]), state, num_timesteps=1)
        assert output[0, 0] == 1.0   # fires exactly once, not twice


# ---------------------------------------------------------------------------
# Test 3: LIF neuron fires when input exceeds threshold
# ---------------------------------------------------------------------------

class TestLIFFires:
    def test_lif_fires_above_threshold(self):
        ir = _lif_ir()
        backend = CPUBackend(ir)
        state = backend.initialize_state()
        # With v_leak=0, tau=20, R=1: dv = (0-0)/20 + 1*1.1 = 1.1 -> fires
        output, _ = backend.run(np.array([[1.1]]), state, num_timesteps=1)
        assert output[0, 0] == 1.0

    def test_lif_v_resets_after_firing(self):
        ir = _lif_ir()
        backend = CPUBackend(ir)
        state = backend.initialize_state()
        _, new_state = backend.run(np.array([[1.1]]), state, num_timesteps=1)
        np.testing.assert_array_equal(new_state["n0"]["v"], np.array([0.0]))


# ---------------------------------------------------------------------------
# Test 4: neuron does NOT fire when input is below threshold
# ---------------------------------------------------------------------------

class TestNoFire:
    def test_if_no_fire_below_threshold(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        output, _ = backend.run(np.array([[0.5]]), state, num_timesteps=1)
        assert output[0, 0] == 0.0

    def test_output_all_zeros(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        output, _ = backend.run(np.zeros((3, 1)), state, num_timesteps=3)
        np.testing.assert_array_equal(output, np.zeros((3, 1)))

    def test_sub_threshold_v_accumulates(self):
        """v should increase but not reach threshold."""
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        _, new_state = backend.run(np.array([[0.5]]), state, num_timesteps=1)
        # v = 0 + 1.0 * 0.5 = 0.5 (below threshold=1.0)
        np.testing.assert_array_almost_equal(new_state["n0"]["v"], np.array([0.5]))


# ---------------------------------------------------------------------------
# Test 5: state persists correctly across two run() calls
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_v_accumulates_across_calls(self):
        """Two successive calls with I=0.6 each -> fires on second call."""
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()

        # Run 1: I=0.6 -> v=0.6, no fire
        out1, state = backend.run(np.array([[0.6]]), state, num_timesteps=1)
        assert out1[0, 0] == 0.0

        # Run 2: same I=0.6 -> v=0.6+0.6=1.2 >= 1.0 -> fires
        out2, state = backend.run(np.array([[0.6]]), state, num_timesteps=1)
        assert out2[0, 0] == 1.0

    def test_original_state_not_mutated(self):
        """run() must not modify the state dict passed in."""
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        v_before = state["n0"]["v"].copy()
        backend.run(np.array([[0.9]]), state, num_timesteps=1)
        np.testing.assert_array_equal(state["n0"]["v"], v_before)


# ---------------------------------------------------------------------------
# Test 6: output shape is (num_timesteps, num_output_neurons)
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_shape_single_neuron_multi_timestep(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        output, _ = backend.run(np.zeros((5, 1)), state, num_timesteps=5)
        assert output.shape == (5, 1)

    def test_shape_two_neuron_graph(self):
        ir = _two_neuron_ir()   # output_neuron_ids has one entry (n1)
        backend = CPUBackend(ir)
        state = backend.initialize_state()
        output, _ = backend.run(np.zeros((4, 1)), state, num_timesteps=4)
        assert output.shape == (4, 1)

    def test_dtype_is_float64(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()
        output, _ = backend.run(np.zeros((2, 1)), state, num_timesteps=2)
        assert output.dtype == np.float64


# ---------------------------------------------------------------------------
# Test 7: reset works by calling initialize_state() again
# ---------------------------------------------------------------------------

class TestReset:
    def test_reinitialize_clears_v(self):
        backend = CPUBackend(_if_ir())
        state = backend.initialize_state()

        # Run once to accumulate some v
        _, state = backend.run(np.array([[0.7]]), state, num_timesteps=1)
        assert state["n0"]["v"][0] > 0.0

        # Reinitialize
        state = backend.initialize_state()
        np.testing.assert_array_equal(state["n0"]["v"], np.zeros(1))

    def test_reinitialized_run_matches_fresh_run(self):
        """After reset, same input -> same output as a fresh run."""
        backend = CPUBackend(_if_ir())
        input_data = np.array([[1.1]])

        state1 = backend.initialize_state()
        out1, _ = backend.run(input_data, state1, num_timesteps=1)

        # Accumulate dirty state, then reinitialize
        dirty_state = backend.initialize_state()
        _, dirty_state = backend.run(np.array([[0.5]]), dirty_state, 1)
        fresh_state = backend.initialize_state()
        out2, _ = backend.run(input_data, fresh_state, num_timesteps=1)

        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Test 8: two runs with same input and state give identical output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_identical_outputs_same_input(self):
        backend = CPUBackend(_if_ir())
        input_data = np.array([[1.1], [0.3], [0.8], [1.2]])
        state = backend.initialize_state()

        out1, _ = backend.run(input_data, state, num_timesteps=4)
        out2, _ = backend.run(input_data, state, num_timesteps=4)

        np.testing.assert_array_equal(out1, out2)

    def test_identical_new_states(self):
        backend = CPUBackend(_if_ir())
        input_data = np.array([[0.6]])
        state = backend.initialize_state()

        _, s1 = backend.run(input_data, state, num_timesteps=1)
        _, s2 = backend.run(input_data, state, num_timesteps=1)

        np.testing.assert_array_equal(s1["n0"]["v"], s2["n0"]["v"])


# ---------------------------------------------------------------------------
# Bonus: two-neuron synapse propagation (verifies two-buffer design)
# ---------------------------------------------------------------------------

class TestSynapsePropagation:
    def test_spike_propagates_to_downstream_neuron(self):
        """n0 fires at t=0 (I=1.1 > 1.0); n1 receives weight*spike=2.0 at t=1."""
        ir = _two_neuron_ir()
        backend = CPUBackend(ir)
        state = backend.initialize_state()
        # t=0: n0 gets I=1.1 -> fires; n1 gets no synaptic input yet (two-buffer)
        # t=1: n0 gets I=0   -> no fire; n1 gets 2.0 from n0's t=0 spike -> fires
        input_data = np.array([[1.1], [0.0]])
        output, _ = backend.run(input_data, state, num_timesteps=2)
        assert output[0, 0] == 0.0   # n1 not yet fired at t=0
        assert output[1, 0] == 1.0   # n1 fires at t=1
