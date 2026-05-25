"""Neuron node in the NeuromorphIR graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


VALID_NEURON_TYPES = {"IF", "LIF", "CubaLIF", "CubaLI", "LI", "I"}


@dataclass(eq=False)
class Neuron:
    """A single neuron node.

    Attributes:
        id: Unique string identifier within the graph.
        type: Neuron model type — one of IF, LIF, CubaLIF, CubaLI, LI, I.
        params: Model parameters keyed by name; values are NumPy arrays
            (e.g. {"tau": np.array([20.0]), "v_threshold": np.array([1.0])}).
        initial_membrane_potential: Optional initial membrane voltage array.
        metadata: Free-form key/value store for toolchain annotations.
    """

    id: str
    type: str
    params: dict = field(default_factory=dict)
    initial_membrane_potential: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.type not in VALID_NEURON_TYPES:
            raise ValueError(
                f"Unknown neuron type {self.type!r}. "
                f"Valid types: {sorted(VALID_NEURON_TYPES)}"
            )

    def __repr__(self) -> str:
        param_keys = list(self.params.keys())
        return f"Neuron(id={self.id!r}, type={self.type!r}, params={param_keys})"
