"""Top-level NeuromorphIR graph and related errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .neuron import Neuron
from .synapse import Synapse


SUPPORTED_VERSION = "1.0"
VALID_HARDWARE_TARGETS = {"loihi2", "cpu"}


class VersionMismatchError(Exception):
    """Raised when a NeuromorphIR graph carries an unsupported version string."""


@dataclass
class NeuromorphIR:
    """The complete intermediate representation of a neuromorphic model.

    Attributes:
        target_hardware: Compilation target - "loihi2" or "cpu".
        neurons: Neuron registry mapping id → Neuron.
        synapses: Ordered list of synaptic connections.
        input_neuron_ids: IDs of neurons that receive external input.
        output_neuron_ids: IDs of neurons whose spikes are read out.
        simulation_config: Backend-specific simulation parameters.
        version: IR schema version; only "1.0" is currently supported.
        transformation_log: Ordered record of passes applied to this graph.
        metadata: Free-form key/value store for toolchain annotations.
    """

    target_hardware: str
    neurons: Dict[str, Neuron] = field(default_factory=dict)
    synapses: List[Synapse] = field(default_factory=list)
    input_neuron_ids: List[str] = field(default_factory=list)
    output_neuron_ids: List[str] = field(default_factory=list)
    simulation_config: dict = field(default_factory=dict)
    version: str = SUPPORTED_VERSION
    transformation_log: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != SUPPORTED_VERSION:
            raise VersionMismatchError(
                f"Unsupported IR version {self.version!r}. "
                f"Expected {SUPPORTED_VERSION!r}."
            )
        if self.target_hardware not in VALID_HARDWARE_TARGETS:
            raise ValueError(
                f"Unknown target hardware {self.target_hardware!r}. "
                f"Valid targets: {sorted(VALID_HARDWARE_TARGETS)}"
            )

    def __repr__(self) -> str:
        return (
            f"NeuromorphIR(version={self.version!r}, "
            f"target_hardware={self.target_hardware!r}, "
            f"neurons={len(self.neurons)}, "
            f"synapses={len(self.synapses)})"
        )
