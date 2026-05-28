"""CPU backend: NumPy-based SNN simulation.

This module provides :class:`CPUBackend`, the reference runtime for the
neuromorf compiler pipeline. It simulates IF and LIF neuron dynamics using
pure NumPy, one timestep at a time.

Design principles followed
--------------------------
- **Explicit state** (CLAUDE.md #3): state dicts are passed in/out every call;
  the backend never stores or resets state internally.
- **Deterministic** (CLAUDE.md #5): the input state is deep-copied at the start
  of every ``run()`` call, so re-using the same (input_data, state) pair always
  produces bit-identical output.
- **Fail loudly** (CLAUDE.md #1): unsupported neuron types or input encodings
  raise clear ``NotImplementedError`` messages rather than silently degrading.

Supported neuron types (v1.0)
------------------------------
- ``IF``  - Integrate-and-Fire (Euler, dt=1)
- ``LIF`` - Leaky Integrate-and-Fire (Euler, dt=1)

Not yet implemented (v1.1+)
----------------------------
- ``CubaLIF``, ``CubaLI``, ``LI``, ``I``
- Input encodings: ``"poisson"``, ``"latency"``
- Synaptic delay > 1 (stored in Synapse but not simulated)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neuromorf.ir import NeuromorphIR, Neuron, Synapse


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _neuron_size(neuron: Neuron) -> int:
    """Return the number of units represented by *neuron*.

    Derived from the primary parameter that determines the neuron's shape:
    - LIF / CubaLIF / CubaLI / LI → ``params["tau"].size``
    - IF → ``params["r"].size``
    - I  → ``params["r"].size`` (or 1 if r is absent)
    """
    if neuron.type in ("LIF", "CubaLIF", "CubaLI", "LI"):
        return int(neuron.params["tau"].size)
    if neuron.type == "IF":
        return int(neuron.params["r"].size)
    if neuron.type == "I":
        return int(neuron.params.get("r", np.array([1.0])).size)
    return 1  # fallback - should not be reached for validated graphs


# ---------------------------------------------------------------------------
# CPUBackend
# ---------------------------------------------------------------------------

class CPUBackend:
    """NumPy-based simulation backend for a compiled :class:`NeuromorphIR` graph.

    Parameters
    ----------
    ir:
        A compiled IR graph (all passes applied).  The backend reads
        ``ir.simulation_config.get("input_encoding", "direct")`` to select
        how external input is encoded.

    Examples
    --------
    >>> backend = CPUBackend(ir)
    >>> state = backend.initialize_state()
    >>> spikes, state = backend.run(input_data, state, num_timesteps=10)
    """

    def __init__(self, ir: NeuromorphIR) -> None:
        self.ir = ir
        self._input_encoding: str = ir.simulation_config.get("input_encoding", "direct")

        # Pre-build an incoming-synapse index for fast per-neuron lookup.
        # _incoming[nid] = list of Synapse objects whose dst_id == nid
        self._incoming: Dict[str, List[Synapse]] = {nid: [] for nid in ir.neurons}
        for syn in ir.synapses:
            if syn.dst_id in self._incoming:
                self._incoming[syn.dst_id].append(syn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_state(self) -> dict:
        """Create a zeroed state dict for every neuron in the graph.

        Returns
        -------
        dict
            Mapping ``neuron_id → {"v": np.zeros(n)}`` for all neuron types,
            plus ``"i": np.zeros(n)`` for ``CubaLIF`` and ``CubaLI`` neurons.
        """
        state: dict = {}
        for nid, neuron in self.ir.neurons.items():
            n = _neuron_size(neuron)
            entry: dict = {"v": np.zeros(n)}
            if neuron.type in ("CubaLIF", "CubaLI"):
                entry["i"] = np.zeros(n)
            state[nid] = entry
        return state

    def run(
        self,
        input_data: np.ndarray,
        state: dict,
        num_timesteps: int,
    ) -> tuple[np.ndarray, dict]:
        """Simulate *num_timesteps* of the network.

        Parameters
        ----------
        input_data:
            Shape ``(num_timesteps, num_input_neurons)``.  Each value is the
            direct current injected into the corresponding input neuron at
            that timestep (``input_encoding="direct"`` only).
        state:
            Neuron state dict produced by :meth:`initialize_state` or
            returned by a previous :meth:`run` call.  **Never mutated** -
            a deep copy is made internally before simulation begins.
        num_timesteps:
            Number of simulation steps to run.

        Returns
        -------
        output_spikes : np.ndarray
            Shape ``(num_timesteps, num_output_neurons)``.  Each entry is the
            total spike count across all units of the corresponding output
            neuron at that timestep.
        new_state : dict
            Updated neuron state after ``num_timesteps`` steps.

        Raises
        ------
        NotImplementedError
            If ``input_encoding`` is not ``"direct"``.
            If any neuron type other than IF or LIF is encountered.
        """
        # ------------------------------------------------------------------
        # 1. Validate input encoding
        # ------------------------------------------------------------------
        if self._input_encoding in ("poisson", "latency"):
            raise NotImplementedError(
                f"input_encoding={self._input_encoding!r} is not yet supported; "
                f"coming in v1.1. Supported now: 'direct'"
            )
        if self._input_encoding != "direct":
            raise NotImplementedError(
                f"Unknown input_encoding={self._input_encoding!r}. "
                f"Supported: 'direct'"
            )

        # ------------------------------------------------------------------
        # 2. Deep-copy state - caller's dict is never mutated
        # ------------------------------------------------------------------
        state = {nid: {k: v.copy() for k, v in s.items()} for nid, s in state.items()}

        # ------------------------------------------------------------------
        # 3. Allocate output buffer
        # ------------------------------------------------------------------
        n_out = len(self.ir.output_neuron_ids)
        output_spikes = np.zeros((num_timesteps, n_out), dtype=np.float64)

        # Two-buffer approach:
        #   current_spikes - spikes that fired in the *previous* timestep and
        #                    are now delivered (default delay = 1 timestep).
        #   next_spikes    - spikes fired during *this* timestep; will become
        #                    current_spikes for the next iteration.
        #
        # NOTE: Synaptic delay > 1 (stored in Synapse.delay) is not simulated
        # in v1.0. All synapses are treated as delay=1. (v1.1 will add a
        # per-synapse ring-buffer to the state dict.)
        current_spikes: Dict[str, np.ndarray] = {
            nid: np.zeros(_neuron_size(self.ir.neurons[nid]))
            for nid in self.ir.neurons
        }

        # ------------------------------------------------------------------
        # 4. Timestep loop
        # ------------------------------------------------------------------
        for t in range(num_timesteps):
            next_spikes: Dict[str, np.ndarray] = {
                nid: np.zeros(_neuron_size(self.ir.neurons[nid]))
                for nid in self.ir.neurons
            }

            for nid, neuron in self.ir.neurons.items():
                n = _neuron_size(neuron)
                I = np.zeros(n)

                # ---- Synaptic input from previous-timestep spikes ----------
                # weight shape: (n_dst, n_src); spikes shape: (n_src,)
                # → I shape: (n_dst,) via matrix-vector product
                for syn in self._incoming[nid]:
                    if syn.src_id in current_spikes:
                        I += syn.weight @ current_spikes[syn.src_id]

                # ---- Direct current injection for input neurons -------------
                # input_data[t, idx] is a scalar broadcast to all n units
                if nid in self.ir.input_neuron_ids:
                    idx = self.ir.input_neuron_ids.index(nid)
                    I += float(input_data[t, idx])

                # ---- Neuron dynamics (Euler integration, dt=1) -------------
                v = state[nid]["v"]

                if neuron.type == "IF":
                    R = neuron.params.get("r", np.ones(n))
                    v = v + R * I

                elif neuron.type == "LIF":
                    tau    = neuron.params["tau"]
                    R      = neuron.params.get("r", np.ones(n))
                    v_leak = neuron.params.get("v_leak", np.zeros(n))
                    v = v + (v_leak - v) / tau + R * I

                else:
                    raise NotImplementedError(
                        f"CPUBackend: neuron type {neuron.type!r} is not "
                        f"implemented in v1.0. Supported types: IF, LIF."
                    )

                # ---- Threshold check + reset --------------------------------
                v_thr   = neuron.params["v_threshold"]
                v_reset = neuron.params.get("v_reset", np.zeros_like(v_thr))
                fired   = v >= v_thr

                next_spikes[nid]   = fired.astype(np.float64)
                state[nid]["v"]    = np.where(fired, v_reset, v)

            # ---- Collect output spikes -------------------------------------
            for oi, out_id in enumerate(self.ir.output_neuron_ids):
                if out_id in next_spikes:
                    output_spikes[t, oi] = next_spikes[out_id].sum()

            current_spikes = next_spikes

        return output_spikes, state
