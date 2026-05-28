"""Lava backend: Python code generation for Intel Loihi 2.

This module provides :class:`LavaBackend`, which generates a valid Python
source file that uses the Lava SDK to run a compiled :class:`NeuromorphIR`
graph on Intel Loihi 2.

**This file never imports lava-nc.** Only the *generated* file imports lava.
This means LavaBackend works on any machine, even without lava-nc installed.

Usage::

    backend = LavaBackend(ir)
    code = backend.generate("model_loihi2.py")

The generated file follows the patterns documented in
``.claude/skills/lava_backend.md`` exactly.

Supported neuron types (v1.0)
------------------------------
- IF  - mapped to Lava LIF with du=1 (no leak, full integration)
- LIF - mapped to Lava LIF with du=0, dv derived from tau

Not yet implemented
--------------------
- CubaLIF, CubaLI, LI, I
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np

from neuromorf.ir import NeuromorphIR, Neuron


# ---------------------------------------------------------------------------
# Name-mangling helpers
# ---------------------------------------------------------------------------

def _safe_id(nid: str) -> str:
    """Sanitize a neuron id into a valid Python identifier fragment."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", nid)


def _proc_name(nid: str) -> str:
    """Variable name for the Lava Process representing neuron *nid*."""
    return f"proc_{_safe_id(nid)}"


def _weight_name(src_id: str, dst_id: str) -> str:
    """Variable name for the weight array of a synapse."""
    return f"w_{_safe_id(src_id)}_to_{_safe_id(dst_id)}"


def _dense_name(src_id: str, dst_id: str) -> str:
    """Variable name for the Dense process of a synapse."""
    return f"dense_{_safe_id(src_id)}_to_{_safe_id(dst_id)}"


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _neuron_size(neuron: Neuron) -> int:
    """Return the number of units represented by *neuron*."""
    if neuron.type in ("LIF", "CubaLIF", "CubaLI", "LI"):
        return int(neuron.params["tau"].size)
    if neuron.type == "IF":
        return int(neuron.params["r"].size)
    return 1  # fallback


def _array_repr(arr: np.ndarray) -> str:
    """Render a numpy array as a Python list literal for generated code."""
    return f"np.array({arr.tolist()})"


# ---------------------------------------------------------------------------
# LavaBackend
# ---------------------------------------------------------------------------

class LavaBackend:
    """Code-generation backend that produces a Lava SDK Python script.

    The generated script can be run directly on a machine with lava-nc
    installed (or in simulation mode via ``Loihi2SimCfg``).

    Parameters
    ----------
    ir:
        A compiled IR graph (all passes applied, target_hardware="loihi2").
        ``ir.simulation_config.get("num_steps", 100)`` sets the run length.

    Examples
    --------
    >>> backend = LavaBackend(ir)
    >>> code = backend.generate("model_loihi2.py")
    """

    def __init__(self, ir: NeuromorphIR) -> None:
        self.ir = ir
        self._num_steps: int = int(ir.simulation_config.get("num_steps", 100))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, output_path: str) -> str:
        """Generate a Lava SDK Python script and write it to *output_path*.

        Parameters
        ----------
        output_path:
            File path where the generated ``.py`` script will be written.

        Returns
        -------
        str
            The generated Python source code (same content as the written file).

        Raises
        ------
        NotImplementedError
            If any neuron in ``ir.neurons`` has a type not supported by this
            backend (currently only IF and LIF are supported).
        """
        # ------------------------------------------------------------------
        # 1. Validate all neuron types BEFORE building any output, so we
        #    never write a partially-generated file on error.
        # ------------------------------------------------------------------
        for nid, neuron in self.ir.neurons.items():
            if neuron.type not in ("IF", "LIF"):
                raise NotImplementedError(
                    f"LavaBackend: neuron type {neuron.type!r} is not supported "
                    f"in v1.0 (neuron id: {nid!r}).\n"
                    f"Supported types: IF, LIF.\n"
                    f"Fix: Use a supported neuron type or extend LavaBackend."
                )

        # ------------------------------------------------------------------
        # 2. Build code line by line
        # ------------------------------------------------------------------
        lines: List[str] = []

        # Header
        lines += [
            "# Auto-generated by neuromorf LavaBackend. Do not edit manually.",
            "",
        ]

        # Imports (section 2)
        lines += [
            "import numpy as np",
            "from lava.proc.lif.process import LIF",
            "from lava.proc.dense.process import Dense",
            "from lava.magma.core.run_conditions import RunSteps",
            "from lava.magma.core.run_configs import Loihi2SimCfg",
            "",
        ]

        # Weight arrays (section 3)
        lines.append("# Weight arrays")
        for syn in self.ir.synapses:
            wname = _weight_name(syn.src_id, syn.dst_id)
            lines.append(f"{wname} = {_array_repr(syn.weight)}")
        lines.append("")

        # Neuron processes (section 4)
        lines.append("# Neuron processes")
        for nid, neuron in self.ir.neurons.items():
            n = _neuron_size(neuron)
            pname = _proc_name(nid)
            vth = int(neuron.params["v_threshold"][0])

            if neuron.type == "IF":
                # du=1: full current decay each step = integrate-and-fire (no leak)
                lines.append(
                    f"{pname} = LIF(shape=({n},), du=1, dv=0, vth={vth}, bias_mant=0)"
                )
            elif neuron.type == "LIF":
                # dv derived from tau (approximation for fixed-point Loihi 2 decay)
                dv = int(neuron.params["tau"][0])
                lines.append(
                    f"{pname} = LIF(shape=({n},), du=0, dv={dv}, vth={vth}, bias_mant=0)"
                )
        lines.append("")

        # Dense connections (section 5)
        lines.append("# Dense (synaptic) connections")
        for syn in self.ir.synapses:
            wname = _weight_name(syn.src_id, syn.dst_id)
            dname = _dense_name(syn.src_id, syn.dst_id)
            lines.append(f"{dname} = Dense(weights={wname})")
        lines.append("")

        # Wire processes (section 6)
        lines.append("# Wire processes")
        for syn in self.ir.synapses:
            src_proc = _proc_name(syn.src_id)
            dst_proc = _proc_name(syn.dst_id)
            dname = _dense_name(syn.src_id, syn.dst_id)
            lines.append(
                f"{src_proc}.out_ports.s_out.connect({dname}.in_ports.s_in)"
            )
            lines.append(
                f"{dname}.out_ports.a_out.connect({dst_proc}.in_ports.a_in)"
            )
        lines.append("")

        # Run call (section 7) — Lava propagates run() to all connected processes
        lines.append("# Run")
        lines.append(f"num_steps = {self._num_steps}")
        if self.ir.neurons:
            first_proc = _proc_name(next(iter(self.ir.neurons)))
            lines.append(
                f"{first_proc}.run("
                f"condition=RunSteps(num_steps=num_steps), "
                f"run_cfg=Loihi2SimCfg())"
            )
        lines.append("")

        code = "\n".join(lines)

        # ------------------------------------------------------------------
        # 3. Write to disk
        # ------------------------------------------------------------------
        Path(output_path).write_text(code, encoding="utf-8")

        return code
