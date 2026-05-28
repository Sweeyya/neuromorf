"""Main compiler orchestrator for neuromorf.

Exposes two public interfaces:

- :class:`Compiler` - configurable compilation with optional debug snapshots
  and a printed compilation report.
- :func:`compile` - one-call convenience wrapper around ``Compiler.compile()``.

Pipeline (in order)
-------------------
1. ``parse()``                - NIR graph -> NeuromorphIR
2. ``validate_structure()``   - check graph integrity (endpoints, duplicates, cycles)
3. ``validate_neuron_types()``- check hardware compatibility
4. ``quantize_weights()``     - float32 -> int8 (clips overflow with warning)
5. backend selection          - returns CPUBackend or LavaBackend

Note: ``compile`` shadows the Python builtin of the same name. This is
intentional - the module is never used in a context that requires the builtin.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from neuromorf.frontend.nir_parser import parse as _parse
from neuromorf.passes.validate_structure import validate_structure
from neuromorf.passes.validate_neuron_types import validate_neuron_types
from neuromorf.passes.quantize_weights import quantize_weights
from neuromorf.backends.cpu_backend import CPUBackend
from neuromorf.backends.lava_backend import LavaBackend
from neuromorf.ir import NeuromorphIR


_VALID_TARGETS = {"cpu", "loihi2"}


# ---------------------------------------------------------------------------
# Compiler class
# ---------------------------------------------------------------------------

class Compiler:
    """Orchestrates the full neuromorf compilation pipeline.

    Parameters
    ----------
    target_hardware:
        Compilation target - ``"cpu"`` or ``"loihi2"``.
    save_intermediates:
        If ``True``, write the IR to ``debug_output_dir`` after each pass.
        Files are named ``00_parsed.txt`` through ``03_quantized.txt``.
    debug_output_dir:
        Directory for intermediate IR snapshots. Created automatically when
        ``save_intermediates=True``. Defaults to ``"./debug/"``.

    Examples
    --------
    >>> compiler = Compiler(target_hardware="cpu")
    >>> backend = compiler.compile(nir_graph)

    >>> compiler = Compiler("loihi2", save_intermediates=True, debug_output_dir="/tmp/dbg")
    >>> backend = compiler.compile(nir_graph)
    """

    def __init__(
        self,
        target_hardware: str,
        save_intermediates: bool = False,
        debug_output_dir: str = "./debug/",
    ) -> None:
        self._target = target_hardware
        self._save = save_intermediates
        self._debug_dir = Path(debug_output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, nir_graph) -> CPUBackend | LavaBackend:
        """Run the full compilation pipeline on *nir_graph*.

        Parameters
        ----------
        nir_graph:
            A :class:`nir.NIRGraph` to compile.

        Returns
        -------
        CPUBackend | LavaBackend
            A ready-to-use backend wrapping the compiled IR.

        Raises
        ------
        ValueError
            If ``target_hardware`` is not ``"cpu"`` or ``"loihi2"``.
        UnsupportedNIRNodeError
            If the graph contains unsupported NIR node types.
        StructureValidationError
            If the graph has missing endpoints, duplicate synapses, or cycles.
        NeuronTypeValidationError
            If any neuron type is not supported on the target hardware.
        """
        # Validate target BEFORE parse so the error is clear and no partial
        # work is done (NeuromorphIR also validates, but its message is less
        # actionable in this context).
        if self._target not in _VALID_TARGETS:
            raise ValueError(
                f"ERROR: Compiler - Unknown target hardware\n"
                f"Problem: '{self._target}' is not a supported compilation target\n"
                f"Supported: {', '.join(sorted(_VALID_TARGETS))}\n"
                f"Fix: Use target='cpu' or target='loihi2'"
            )

        # Step 1: parse NIR -> NeuromorphIR
        ir = _parse(nir_graph, target=self._target)
        self._save_intermediate(ir, "00_parsed.txt")

        # Step 2: structural validation
        ir = validate_structure(ir)
        self._save_intermediate(ir, "01_validated_structure.txt")

        # Step 3: neuron-type validation
        ir = validate_neuron_types(ir)
        self._save_intermediate(ir, "02_validated_types.txt")

        # Step 4: weight quantization
        ir = quantize_weights(ir)
        self._save_intermediate(ir, "03_quantized.txt")

        # Print compilation report
        self._print_report(ir)

        # Step 5: select and return backend
        if self._target == "cpu":
            return CPUBackend(ir)
        return LavaBackend(ir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_intermediate(self, ir: NeuromorphIR, filename: str) -> None:
        """Write an IR snapshot to *debug_output_dir/filename* if enabled."""
        if not self._save:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        # Use ir.to_text() if the IR ever gains that method; fall back to str().
        text = ir.to_text() if hasattr(ir, "to_text") else str(ir)
        (self._debug_dir / filename).write_text(text, encoding="utf-8")

    def _print_report(self, ir: NeuromorphIR) -> None:
        """Print a human-readable compilation summary to stdout."""
        # Neuron type breakdown, e.g. "3 (IF: 1, LIF: 2)"
        type_counts = Counter(n.type for n in ir.neurons.values())
        breakdown = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))
        neuron_summary = (
            f"{len(ir.neurons)} ({breakdown})" if breakdown else str(len(ir.neurons))
        )

        # Clipping warnings are stored in the QuantizeWeights log entry.
        warnings: list[str] = []
        for entry in ir.transformation_log:
            if entry.get("pass") == "QuantizeWeights":
                warnings = entry.get("warnings", [])
                break
        warnings_str = "\n  ".join(warnings) if warnings else "none"

        print(
            "neuromorf compilation report\n"
            f"target: {ir.target_hardware}\n"
            f"neurons: {neuron_summary}\n"
            f"synapses: {len(ir.synapses)}\n"
            "passes: ValidateStructure, ValidateNeuronTypes, QuantizeWeights\n"
            f"warnings: {warnings_str}\n"
            "status: ready"
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def compile(nir_graph, target: str = "cpu") -> CPUBackend | LavaBackend:
    """Compile a NIR graph to a backend in one call.

    A convenience wrapper around :class:`Compiler`. Equivalent to::

        Compiler(target_hardware=target).compile(nir_graph)

    Parameters
    ----------
    nir_graph:
        A :class:`nir.NIRGraph` to compile.
    target:
        Compilation target - ``"cpu"`` (default) or ``"loihi2"``.

    Returns
    -------
    CPUBackend | LavaBackend
        A ready-to-use backend wrapping the compiled IR.

    Examples
    --------
    >>> backend = compile(nir_graph, target="cpu")
    >>> spikes, state = backend.run(input_data, backend.initialize_state(), num_timesteps=10)

    >>> backend = compile(nir_graph, target="loihi2")
    >>> backend.generate("model_loihi2.py")
    """
    return Compiler(target_hardware=target).compile(nir_graph)
