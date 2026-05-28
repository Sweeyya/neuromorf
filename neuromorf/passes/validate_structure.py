"""Compiler pass: ValidateStructure.

Pass #1 in the compiler pipeline. Checks three graph integrity constraints:

1. Every synapse's src_id and dst_id exist in ir.neurons.
2. No two synapses share the same (src_id, dst_id) pair.
3. The directed graph formed by synapses is acyclic (unless allow_cycles=True).

Checks 1 and 2 collect *all* violations before raising, so the user sees every
problem in one error message rather than fixing them one at a time.

On success the pass appends a record to ir.transformation_log and returns the
same ir object unchanged.  All errors are raised as :class:`StructureValidationError`
with a structured message that tells the user exactly what is wrong and how to
fix it.
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx

from neuromorf.ir import NeuromorphIR, Synapse


class StructureValidationError(Exception):
    """Raised when ValidateStructure finds a graph integrity violation."""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_items(items: List[str]) -> str:
    """Format a list of error items, numbered if more than one."""
    if len(items) == 1:
        return items[0]
    return "\n\n".join(f"[{i + 1}] {item}" for i, item in enumerate(items))


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------

def _check_endpoints(ir: NeuromorphIR) -> None:
    """Collect all synapses that reference a missing neuron id; raise once.

    Valid endpoints are all ids in ``ir.neurons`` plus any ids in
    ``ir.input_neuron_ids`` and ``ir.output_neuron_ids``.  The latter
    represent NIR Input/Output boundary nodes that are not real neurons
    but are legitimate synapse endpoints produced by the NIR parser.
    """
    valid_ids = (
        set(ir.neurons.keys())
        | set(ir.input_neuron_ids)
        | set(ir.output_neuron_ids)
    )
    errors: List[str] = []
    for syn in ir.synapses:
        for missing_id in (syn.src_id, syn.dst_id):
            if missing_id not in valid_ids:
                errors.append(
                    f"Neuron:  {missing_id}\n"
                    f"Problem: Synapse {syn.src_id} -> {syn.dst_id} references "
                    f"'{missing_id}' which is not in ir.neurons\n"
                    f"Fix:     Add a Neuron with id='{missing_id}' to ir.neurons, "
                    f"or remove the synapse"
                )
    if errors:
        n = len(errors)
        header = (
            f"ERROR: ValidateStructure Missing neuron id "
            f"({n} error{'s' if n > 1 else ''})"
        )
        raise StructureValidationError(header + "\n\n" + _fmt_items(errors))


def _check_duplicates(ir: NeuromorphIR) -> None:
    """Collect all duplicate (src_id, dst_id) pairs; raise once."""
    seen: set[tuple[str, str]] = set()
    reported: set[tuple[str, str]] = set()
    errors: List[str] = []

    for syn in ir.synapses:
        pair = (syn.src_id, syn.dst_id)
        if pair in seen and pair not in reported:
            reported.add(pair)
            errors.append(
                f"Neuron:  {syn.src_id}\n"
                f"Problem: Synapse {syn.src_id} -> {syn.dst_id} appears more than once\n"
                f"Fix:     Remove the duplicate Synapse from ir.synapses"
            )
        seen.add(pair)

    if errors:
        n = len(errors)
        header = (
            f"ERROR: ValidateStructure Duplicate synapse "
            f"({n} error{'s' if n > 1 else ''})"
        )
        raise StructureValidationError(header + "\n\n" + _fmt_items(errors))


def _check_cycles(ir: NeuromorphIR, allow_cycles: bool) -> None:
    """Detect cycles; raise or log a warning depending on allow_cycles."""
    G = nx.DiGraph()
    G.add_nodes_from(ir.neurons.keys())
    G.add_edges_from((s.src_id, s.dst_id) for s in ir.synapses)

    try:
        cycle_edges = nx.find_cycle(G)
    except nx.NetworkXNoCycle:
        return  # clean graph - nothing to do

    # Build an ordered list of unique node ids around the cycle
    cycle_ids = list(dict.fromkeys(e[0] for e in cycle_edges))
    cycle_str = " -> ".join(cycle_ids) + f" -> {cycle_ids[0]}"

    if allow_cycles:
        ir.transformation_log.append({
            "pass": "ValidateStructure",
            "status": "warning",
            "message": f"Cycle detected: {cycle_str}",
            "cycle_ids": cycle_ids,
        })
    else:
        raise StructureValidationError(
            f"ERROR: ValidateStructure Cycle detected\n"
            f"Neuron:  {cycle_ids[0]}\n"
            f"Problem: Cycle: {cycle_str}\n"
            f"Fix:     Remove the cycle, or pass allow_cycles=True "
            f"if this is intentional (e.g. RNN)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_structure(
    ir: NeuromorphIR,
    allow_cycles: bool = False,
) -> NeuromorphIR:
    """Validate the structural integrity of a :class:`NeuromorphIR` graph.

    Checks (in order):

    1. All synapse endpoints exist in ``ir.neurons``.
    2. No duplicate ``(src_id, dst_id)`` synapse pairs.
    3. The synapse graph is acyclic (unless *allow_cycles* is ``True``).

    Checks 1 and 2 accumulate *all* violations before raising, so the error
    message lists every problem at once.

    Parameters
    ----------
    ir:
        The IR graph to validate.  Modified in-place (transformation_log only).
    allow_cycles:
        If ``False`` (default), a detected cycle raises
        :class:`StructureValidationError`.  If ``True``, the cycle is logged as
        a warning in ``ir.transformation_log`` and execution continues (use
        this for intentionally recurrent graphs such as RNNs).

    Returns
    -------
    NeuromorphIR
        The same *ir* object, with a log entry appended.

    Raises
    ------
    StructureValidationError
        Listing every endpoint or duplicate violation found, or the first
        cycle detected.
    """
    _check_endpoints(ir)
    _check_duplicates(ir)
    _check_cycles(ir, allow_cycles)

    ir.transformation_log.append({
        "pass": "ValidateStructure",
        "status": "passed",
        "affected_ids": [],
        "params": {"allow_cycles": allow_cycles},
    })
    return ir
