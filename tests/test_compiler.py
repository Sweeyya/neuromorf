"""Tests for neuromorf.compiler (Compiler class and compile() function)."""

import pytest
import numpy as np
import nir

from neuromorf.compiler import Compiler, compile
from neuromorf.backends.cpu_backend import CPUBackend
from neuromorf.backends.lava_backend import LavaBackend


# ---------------------------------------------------------------------------
# Shared NIR graph fixture
# ---------------------------------------------------------------------------

def _simple_nir_graph():
    """Input -> Linear(1x1) -> LIF -> Output - minimal valid NIR graph."""
    return nir.NIRGraph(
        nodes={
            "input":  nir.Input(input_type={"input": np.array([1])}),
            "linear": nir.Linear(weight=np.ones((1, 1))),
            "lif":    nir.LIF(
                          tau=np.array([20.0]),
                          r=np.array([1.0]),
                          v_leak=np.array([0.0]),
                          v_threshold=np.array([1.0]),
                      ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("input", "linear"), ("linear", "lif"), ("lif", "output")],
    )


# ---------------------------------------------------------------------------
# TestCompileFunction - module-level compile() convenience wrapper
# ---------------------------------------------------------------------------

class TestCompileFunction:
    def test_cpu_returns_cpu_backend(self):
        backend = compile(_simple_nir_graph(), target="cpu")
        assert isinstance(backend, CPUBackend)

    def test_loihi2_returns_lava_backend(self):
        backend = compile(_simple_nir_graph(), target="loihi2")
        assert isinstance(backend, LavaBackend)

    def test_default_target_is_cpu(self):
        backend = compile(_simple_nir_graph())
        assert isinstance(backend, CPUBackend)

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target hardware"):
            compile(_simple_nir_graph(), target="fpga")


# ---------------------------------------------------------------------------
# TestCompilerClass - Compiler class interface
# ---------------------------------------------------------------------------

class TestCompilerClass:
    def test_cpu_target_returns_cpu_backend(self):
        compiler = Compiler(target_hardware="cpu")
        backend = compiler.compile(_simple_nir_graph())
        assert isinstance(backend, CPUBackend)

    def test_loihi2_target_returns_lava_backend(self):
        compiler = Compiler(target_hardware="loihi2")
        backend = compiler.compile(_simple_nir_graph())
        assert isinstance(backend, LavaBackend)

    def test_unknown_target_raises_clear_error(self):
        compiler = Compiler(target_hardware="fpga")
        with pytest.raises(ValueError, match="Unknown target hardware"):
            compiler.compile(_simple_nir_graph())

    def test_error_message_includes_supported_targets(self):
        compiler = Compiler(target_hardware="fpga")
        with pytest.raises(ValueError) as exc_info:
            compiler.compile(_simple_nir_graph())
        msg = str(exc_info.value)
        assert "cpu" in msg
        assert "loihi2" in msg


# ---------------------------------------------------------------------------
# TestFullPipeline - end-to-end pipeline behaviour
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_full_pipeline_no_errors(self):
        backend = compile(_simple_nir_graph())
        assert backend is not None

    def test_pipeline_applies_all_passes(self):
        """Transformation log must contain entries for all three passes."""
        compiler = Compiler(target_hardware="cpu")
        backend = compiler.compile(_simple_nir_graph())
        pass_names = [entry["pass"] for entry in backend.ir.transformation_log]
        assert "ValidateStructure" in pass_names
        assert "ValidateNeuronTypes" in pass_names
        assert "QuantizeWeights" in pass_names

    def test_weights_quantized_to_int8(self):
        """All synapse weights must be dtype int8 after compilation."""
        backend = compile(_simple_nir_graph())
        for syn in backend.ir.synapses:
            assert syn.weight.dtype == np.int8


# ---------------------------------------------------------------------------
# TestSaveIntermediates - debug snapshot files
# ---------------------------------------------------------------------------

class TestSaveIntermediates:
    def test_creates_debug_files(self, tmp_path):
        compiler = Compiler(
            target_hardware="cpu",
            save_intermediates=True,
            debug_output_dir=str(tmp_path),
        )
        compiler.compile(_simple_nir_graph())
        for fname in [
            "00_parsed.txt",
            "01_validated_structure.txt",
            "02_validated_types.txt",
            "03_quantized.txt",
        ]:
            assert (tmp_path / fname).exists(), f"Missing debug file: {fname}"

    def test_debug_files_are_nonempty(self, tmp_path):
        compiler = Compiler(
            target_hardware="cpu",
            save_intermediates=True,
            debug_output_dir=str(tmp_path),
        )
        compiler.compile(_simple_nir_graph())
        for fname in [
            "00_parsed.txt",
            "01_validated_structure.txt",
            "02_validated_types.txt",
            "03_quantized.txt",
        ]:
            content = (tmp_path / fname).read_text().strip()
            assert content != "", f"Debug file is empty: {fname}"

    def test_no_debug_files_when_flag_false(self, tmp_path):
        compiler = Compiler(
            target_hardware="cpu",
            save_intermediates=False,
            debug_output_dir=str(tmp_path),
        )
        compiler.compile(_simple_nir_graph())
        assert list(tmp_path.iterdir()) == [], "Expected no files in debug dir"


# ---------------------------------------------------------------------------
# TestCompilationReport - stdout report printed on success
# ---------------------------------------------------------------------------

class TestCompilationReport:
    def test_report_is_printed(self, capsys):
        compile(_simple_nir_graph())
        out = capsys.readouterr().out
        assert "neuromorf compilation report" in out

    def test_report_contains_target(self, capsys):
        compile(_simple_nir_graph(), target="cpu")
        out = capsys.readouterr().out
        assert "target: cpu" in out

    def test_report_contains_neuron_count(self, capsys):
        compile(_simple_nir_graph())
        out = capsys.readouterr().out
        assert "neurons:" in out

    def test_report_contains_status_ready(self, capsys):
        compile(_simple_nir_graph())
        out = capsys.readouterr().out
        assert "status: ready" in out

    def test_report_contains_pass_names(self, capsys):
        compile(_simple_nir_graph())
        out = capsys.readouterr().out
        assert "ValidateStructure" in out
        assert "ValidateNeuronTypes" in out
        assert "QuantizeWeights" in out
