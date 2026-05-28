"""NIR → NeuromorphIR parser.

Converts a :class:`nir.NIRGraph` (the standard interchange format produced by
snnTorch / the NIR ecosystem) into our internal :class:`NeuromorphIR`
representation.

Public API
----------
    from neuromorf.frontend.nir_parser import parse

    ir = parse(nir_graph, target="loihi2")

Supported NIR node types
------------------------
- ``nir.Input``   - marks graph input boundary; not added to ``neurons``
- ``nir.Output``  - marks graph output boundary; not added to ``neurons``
- ``nir.LIF``     - Leaky Integrate-and-Fire neuron
- ``nir.IF``      - Integrate-and-Fire neuron
- ``nir.Affine``  - weight matrix + bias; folded into Synapse weight
- ``nir.Linear``  - weight matrix; folded into Synapse weight

Weight-node folding
-------------------
NIR edges carry no weight information - weights live on ``Affine`` / ``Linear``
nodes.  The parser "folds" each weight node transparently:

    A  →  [Affine/Linear W]  →  B
    becomes  Synapse(src_id=A, dst_id=B, weight=W.weight)

Direct real-to-real edges (where neither endpoint is a weight node) receive a
placeholder weight ``np.ones((n_dst, n_src))`` shaped to match the connection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import nir
import numpy as np

from neuromorf.ir import NeuromorphIR, Neuron, Synapse


class UnsupportedNIRNodeError(Exception):
    """Raised when an NIR node type has no mapping to NeuromorphIR.

    The message follows the project-wide structured error format so that
    toolchain consumers can parse it consistently.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unsupported_error(node_id: str, node: nir.NIRNode) -> UnsupportedNIRNodeError:
    node_type = type(node).__name__
    return UnsupportedNIRNodeError(
        f"ERROR: NIR Parser - Unsupported node type\n"
        f"Node:      {node_id}\n"
        f"Type:      {node_type}\n"
        f"Problem:   No mapping from this NIR node type to NeuromorphIR\n"
        f"Supported: Input, Output, LIF, IF, Affine, Linear\n"
        f"Fix:       Use a NIR graph that only contains supported node types, "
        f"or add a converter for {node_type} to nir_parser.py"
    )


def _node_size(node: nir.NIRNode) -> int:
    """Return the number of units represented by an NIR node.

    Used to shape the placeholder weight matrix for direct real→real edges.
    """
    if isinstance(node, nir.Input):
        return int(np.prod(node.input_type["input"]))
    if isinstance(node, nir.Output):
        return int(np.prod(node.output_type["output"]))
    if isinstance(node, nir.LIF):
        return int(np.prod(node.tau.shape))
    if isinstance(node, nir.IF):
        return int(np.prod(node.r.shape))
    return 1  # fallback - should not be reached for valid, supported nodes


def _parse_lif(node_id: str, node: nir.LIF) -> Neuron:
    params: dict = {
        "tau": node.tau,
        "r": node.r,
        "v_threshold": node.v_threshold,
        "v_leak": node.v_leak,
    }
    if node.v_reset is not None:
        params["v_reset"] = node.v_reset
    return Neuron(id=node_id, type="LIF", params=params)


def _parse_if(node_id: str, node: nir.IF) -> Neuron:
    params: dict = {
        "r": node.r,
        "v_threshold": node.v_threshold,
    }
    if node.v_reset is not None:
        params["v_reset"] = node.v_reset
    return Neuron(id=node_id, type="IF", params=params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(
    nir_graph: nir.NIRGraph,
    target: str = "cpu",
) -> NeuromorphIR:
    """Parse a :class:`nir.NIRGraph` into a :class:`NeuromorphIR`.

    Parameters
    ----------
    nir_graph:
        The NIR graph to parse.
    target:
        Compilation target - ``"cpu"`` or ``"loihi2"``.

    Returns
    -------
    NeuromorphIR
        The parsed internal representation, ready for compiler passes.

    Raises
    ------
    UnsupportedNIRNodeError
        If the graph contains an NIR node type that has no mapping.
    """
    neurons: Dict[str, Neuron] = {}
    input_neuron_ids: List[str] = []
    output_neuron_ids: List[str] = []
    # weight_nodes: node_id → weight np.ndarray (for Affine/Linear)
    weight_nodes: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Pass 1: walk nodes
    # ------------------------------------------------------------------
    for node_id, node in nir_graph.nodes.items():
        if isinstance(node, nir.Input):
            input_neuron_ids.append(node_id)

        elif isinstance(node, nir.Output):
            output_neuron_ids.append(node_id)

        elif isinstance(node, nir.LIF):
            neurons[node_id] = _parse_lif(node_id, node)

        elif isinstance(node, nir.IF):
            neurons[node_id] = _parse_if(node_id, node)

        elif isinstance(node, nir.Affine):
            weight_nodes[node_id] = node.weight

        elif isinstance(node, nir.Linear):
            weight_nodes[node_id] = node.weight

        else:
            raise _unsupported_error(node_id, node)

    # ------------------------------------------------------------------
    # Pass 2: walk edges → build synapses, folding weight nodes
    # ------------------------------------------------------------------
    edges: List[Tuple[str, str]] = nir_graph.edges
    synapses: List[Synapse] = []

    for src_id, dst_id in edges:
        # Case A: src is a weight node - already handled by the leg that
        # created the synapse when processing the *incoming* edge to this
        # weight node.  Skip to avoid double-counting.
        if src_id in weight_nodes:
            continue

        # Case B: dst is a weight node - fold it transparently.
        # For every edge leaving dst_id, create one Synapse from src_id to
        # that successor, carrying the weight matrix from the weight node.
        if dst_id in weight_nodes:
            for s2, d2 in edges:
                if s2 == dst_id:
                    synapses.append(
                        Synapse(
                            src_id=src_id,
                            dst_id=d2,
                            weight=weight_nodes[dst_id],
                        )
                    )

        # Case C: direct real→real edge (Input→LIF, LIF→LIF, LIF→Output…)
        # Use a unit weight matrix shaped to match the connection dimensions.
        else:
            n_src = _node_size(nir_graph.nodes[src_id])
            n_dst = _node_size(nir_graph.nodes[dst_id])
            synapses.append(
                Synapse(
                    src_id=src_id,
                    dst_id=dst_id,
                    weight=np.ones((n_dst, n_src)),
                )
            )

    return NeuromorphIR(
        target_hardware=target,
        neurons=neurons,
        synapses=synapses,
        input_neuron_ids=input_neuron_ids,
        output_neuron_ids=output_neuron_ids,
        simulation_config={},
    )


# Backward-compatible alias - prefer `parse` in new code.
parse_nir = parse
