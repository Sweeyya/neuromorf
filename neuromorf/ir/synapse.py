"""Synapse (directed weighted edge) in the NeuromorphIR graph."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Synapse:
    """A directed connection between two neurons.

    Attributes:
        src_id: Identifier of the pre-synaptic neuron.
        dst_id: Identifier of the post-synaptic neuron.
        weight: Connection weight matrix as a NumPy array.
        delay: Synaptic delay in timesteps (must be >= 1).
    """

    src_id: str
    dst_id: str
    weight: np.ndarray
    delay: int = 1

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError(f"delay must be >= 1, got {self.delay}")

    def __repr__(self) -> str:
        return (
            f"Synapse(src_id={self.src_id!r}, dst_id={self.dst_id!r}, "
            f"weight={self.weight}, delay={self.delay})"
        )
