"""Tests for neuromorf.ir — Neuron, Synapse, and NeuromorphIR."""

import pytest
import numpy as np

from neuromorf.ir import (
    Neuron,
    Synapse,
    NeuromorphIR,
    VersionMismatchError,
    VALID_NEURON_TYPES,
)


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------

class TestNeuron:
    def test_basic_construction(self):
        n = Neuron(id="n0", type="LIF", params={"tau": np.array([20.0])})
        assert n.id == "n0"
        assert n.type == "LIF"
        np.testing.assert_array_equal(n.params["tau"], np.array([20.0]))

    def test_defaults(self):
        n = Neuron(id="n1", type="IF")
        assert n.params == {}
        assert n.initial_membrane_potential is None
        assert n.metadata == {}

    def test_all_valid_types_accepted(self):
        for ntype in VALID_NEURON_TYPES:
            n = Neuron(id="x", type=ntype)
            assert n.type == ntype

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown neuron type"):
            Neuron(id="bad", type="SpikyBoi")

    def test_repr_shows_id_type_param_keys(self):
        n = Neuron(
            id="n2",
            type="CubaLIF",
            params={"tau": np.array([10.0]), "v_threshold": np.array([0.5])},
        )
        r = repr(n)
        assert "n2" in r
        assert "CubaLIF" in r
        # param keys appear, not values
        assert "tau" in r
        assert "v_threshold" in r

    def test_initial_membrane_potential(self):
        v0 = np.zeros(8)
        n = Neuron(id="n3", type="LIF", initial_membrane_potential=v0)
        np.testing.assert_array_equal(n.initial_membrane_potential, v0)

    def test_metadata_stored(self):
        n = Neuron(id="n4", type="I", metadata={"layer": "input"})
        assert n.metadata["layer"] == "input"


# ---------------------------------------------------------------------------
# Synapse
# ---------------------------------------------------------------------------

class TestSynapse:
    def test_basic_construction(self):
        w = np.array([[0.5, 0.1], [0.2, 0.9]])
        s = Synapse(src_id="n0", dst_id="n1", weight=w)
        assert s.src_id == "n0"
        assert s.dst_id == "n1"
        np.testing.assert_array_equal(s.weight, w)
        assert s.delay == 1

    def test_custom_delay(self):
        s = Synapse(src_id="a", dst_id="b", weight=np.array([1.0]), delay=3)
        assert s.delay == 3

    def test_delay_zero_raises(self):
        with pytest.raises(ValueError, match="delay must be >= 1"):
            Synapse(src_id="a", dst_id="b", weight=np.array([1.0]), delay=0)

    def test_delay_negative_raises(self):
        with pytest.raises(ValueError, match="delay must be >= 1"):
            Synapse(src_id="a", dst_id="b", weight=np.array([1.0]), delay=-2)

    def test_repr_contains_fields(self):
        w = np.array([0.7])
        s = Synapse(src_id="pre", dst_id="post", weight=w, delay=2)
        r = repr(s)
        assert "pre" in r
        assert "post" in r
        assert "0.7" in r
        assert "2" in r


# ---------------------------------------------------------------------------
# NeuromorphIR
# ---------------------------------------------------------------------------

class TestNeuromorphIR:
    def _make_graph(self, **kwargs):
        defaults = dict(target_hardware="cpu")
        defaults.update(kwargs)
        return NeuromorphIR(**defaults)

    def test_basic_construction(self):
        g = self._make_graph()
        assert g.version == "1.0"
        assert g.target_hardware == "cpu"
        assert g.neurons == {}
        assert g.synapses == []
        assert g.input_neuron_ids == []
        assert g.output_neuron_ids == []
        assert g.transformation_log == []
        assert g.metadata == {}

    def test_loihi2_target_accepted(self):
        g = self._make_graph(target_hardware="loihi2")
        assert g.target_hardware == "loihi2"

    def test_invalid_hardware_raises(self):
        with pytest.raises(ValueError, match="Unknown target hardware"):
            self._make_graph(target_hardware="tpu")

    def test_version_mismatch_raises(self):
        with pytest.raises(VersionMismatchError, match="Unsupported IR version"):
            self._make_graph(version="2.0")

    def test_neurons_dict(self):
        n = Neuron(id="n0", type="LIF")
        g = self._make_graph(neurons={"n0": n})
        assert g.neurons["n0"] is n

    def test_synapses_list(self):
        s = Synapse(src_id="n0", dst_id="n1", weight=np.array([1.0]))
        g = self._make_graph(synapses=[s])
        assert g.synapses[0] is s

    def test_io_neuron_ids(self):
        g = self._make_graph(
            input_neuron_ids=["n0", "n1"],
            output_neuron_ids=["n5"],
        )
        assert g.input_neuron_ids == ["n0", "n1"]
        assert g.output_neuron_ids == ["n5"]

    def test_simulation_config(self):
        cfg = {"dt": 0.001, "steps": 100}
        g = self._make_graph(simulation_config=cfg)
        assert g.simulation_config["dt"] == 0.001

    def test_transformation_log(self):
        g = self._make_graph(transformation_log=["quantize_weights"])
        assert "quantize_weights" in g.transformation_log

    def test_repr(self):
        g = self._make_graph(
            neurons={"n0": Neuron(id="n0", type="IF")},
            synapses=[Synapse(src_id="n0", dst_id="n1", weight=np.array([1.0]))],
        )
        r = repr(g)
        assert "1.0" in r
        assert "cpu" in r
        assert "neurons=1" in r
        assert "synapses=1" in r

    def test_full_roundtrip(self):
        """Build a small two-neuron graph end-to-end."""
        n_in = Neuron(
            id="input",
            type="I",
            params={"bias": np.array([0.1])},
            metadata={"role": "input"},
        )
        n_out = Neuron(
            id="output",
            type="LIF",
            params={"tau": np.array([20.0]), "v_threshold": np.array([1.0])},
            initial_membrane_potential=np.zeros(1),
        )
        syn = Synapse(src_id="input", dst_id="output", weight=np.array([[0.8]]))

        g = NeuromorphIR(
            target_hardware="loihi2",
            neurons={"input": n_in, "output": n_out},
            synapses=[syn],
            input_neuron_ids=["input"],
            output_neuron_ids=["output"],
            simulation_config={"dt": 0.001, "steps": 200},
        )

        assert len(g.neurons) == 2
        assert len(g.synapses) == 1
        assert g.neurons["input"].type == "I"
        assert g.neurons["output"].params["tau"][0] == 20.0
        np.testing.assert_array_equal(g.synapses[0].weight, np.array([[0.8]]))
