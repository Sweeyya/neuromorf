"""Compiler pass: ValidateNeuronTypes.

Pass #2 in the compiler pipeline. Checks that every neuron's type is supported
on the compilation target declared in ir.target_hardware.

Per CLAUDE.md design principle #2 ("No silent conversions"), unsupported types
are never auto-converted - the pass raises loudly and tells the user exactly
what to fix.  All unsupported neurons are collected before raising so the user
sees every problem in a single error message.
"""

from __future__ import annotations

from typing import List

from neuromorf.ir import NeuromorphIR


class NeuronTypeValidationError(Exception):
    """Raised when a neuron type is not supported on the target hardware."""


# Hardware support table - the single source of truth for which neuron types
# are available on each compilation target.  Both targets currently expose the
# full set; future hardware restrictions are a one-line change here.
HARDWARE_SUPPORT: dict[str, frozenset[str]] = {
    "cpu":    frozenset({"IF", "LIF", "CubaLIF", "CubaLI", "LI", "I"}),
    "loihi2": frozenset({"IF", "LIF", "CubaLIF", "CubaLI", "LI", "I"}),
}


# ---------------------------------------------------------------------------
# Formatting helper (self-contained; each pass owns its own copy)
# ---------------------------------------------------------------------------

def _fmt_items(items: List[str]) -> str:
    """Format error items: plain for one, numbered blocks for many."""
    if len(items) == 1:
        return items[0]
    return "\n\n".join(f"[{i + 1}] {item}" for i, item in enumerate(items))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_neuron_types(ir: NeuromorphIR) -> NeuromorphIR:
    """Verify that every neuron type is supported on ``ir.target_hardware``.

    Collects *all* unsupported neurons before raising, so the error message
    lists every problem at once rather than forcing repeated re-runs.

    Parameters
    ----------
    ir:
        The IR graph to validate.  Modified in-place (transformation_log only).

    Returns
    -------
    NeuromorphIR
        The same *ir* object, with a log entry appended.

    Raises
    ------
    NeuronTypeValidationError
        If ``ir.target_hardware`` is not in the support table, or if any
        neuron type is not supported on that hardware.
    """
    # Defence-in-depth: NeuromorphIR.__post_init__ already rejects unknown
    # targets, but the pass must not silently assume the table stays in sync.
    if ir.target_hardware not in HARDWARE_SUPPORT:
        raise NeuronTypeValidationError(
            f"ERROR: ValidateNeuronTypes unknown target hardware\n"
            f"Problem: '{ir.target_hardware}' is not in the hardware support table\n"
            f"Supported hardware: {sorted(HARDWARE_SUPPORT.keys())}\n"
            f"Fix: Use a supported target hardware"
        )

    supported = HARDWARE_SUPPORT[ir.target_hardware]
    errors: List[str] = []

    for neuron_id, neuron in ir.neurons.items():
        if neuron.type not in supported:
            errors.append(
                f"Neuron: {neuron_id}\n"
                f"Type: {neuron.type}\n"
                f"Problem: Target hardware '{ir.target_hardware}' "
                f"does not support {neuron.type}\n"
                f"Supported on {ir.target_hardware}: "
                f"{', '.join(sorted(supported))}\n"
                f"Fix: Use a supported neuron type or target a different hardware"
            )

    if errors:
        n = len(errors)
        header = (
            f"ERROR: ValidateNeuronTypes unsupported neuron type "
            f"({n} error{'s' if n > 1 else ''})"
        )
        raise NeuronTypeValidationError(header + "\n\n" + _fmt_items(errors))

    ir.transformation_log.append({
        "pass": "ValidateNeuronTypes",
        "status": "passed",
        "affected_ids": [],
        "params": {},
    })
    return ir
