"""Compiler pass: QuantizeWeights.

Pass #3 in the compiler pipeline. Converts every Synapse.weight from
floating-point to int8 so downstream backends can use fixed-point arithmetic
on target hardware.

When any weight value falls outside the int8 range [-128, 127] it is clipped
to the boundary and a warning is recorded in the transformation log.  No
exception is raised for clipping — the pass never fails on out-of-range
values, it just reports them loudly (CLAUDE.md principle #1: "Fail loudly").

Neuron params (tau, v_threshold, r, etc.) are never touched by this pass, per
CLAUDE.md principle #4 ("Respect the model").
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from neuromorf.ir import NeuromorphIR, Synapse


INT8_MIN: int = -128
INT8_MAX: int = 127


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quantize_synapse(syn: Synapse) -> Optional[str]:
    """Quantize *syn.weight* to int8 in-place.

    Returns a warning string if any element was clipped, or ``None`` if all
    values were already in range.
    """
    w = syn.weight
    clipped_mask = (w < INT8_MIN) | (w > INT8_MAX)

    if clipped_mask.any():
        pct = 100.0 * clipped_mask.sum() / w.size
        max_mag = float(np.abs(w).max())
        warning = (
            f"Synapse {syn.src_id}->{syn.dst_id}: "
            f"{pct:.2f}% of weights clipped "
            f"(max original magnitude: {max_mag:.4f})"
        )
        syn.weight = np.clip(w, INT8_MIN, INT8_MAX).astype(np.int8)
        return warning

    syn.weight = w.astype(np.int8)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_weights(ir: NeuromorphIR) -> NeuromorphIR:
    """Convert all synapse weights from float to int8.

    Each weight matrix is clipped to [-128, 127] and cast to ``np.int8``.
    Out-of-range clips are recorded as warning strings in the log entry rather
    than raising exceptions.

    Neuron params are never modified.

    Parameters
    ----------
    ir:
        The IR graph to quantize.  Modified in-place (synapse weights and
        transformation_log).

    Returns
    -------
    NeuromorphIR
        The same *ir* object, with a log entry appended.
    """
    affected_ids: List[str] = []
    warnings: List[str] = []

    for syn in ir.synapses:
        affected_ids.append(f"{syn.src_id}->{syn.dst_id}")
        warning = _quantize_synapse(syn)
        if warning:
            warnings.append(warning)

    ir.transformation_log.append({
        "pass": "QuantizeWeights",
        "status": "passed",
        "affected_ids": affected_ids,
        "params": {"dtype": "int8", "range": [INT8_MIN, INT8_MAX]},
        "warnings": warnings,
    })
    return ir
