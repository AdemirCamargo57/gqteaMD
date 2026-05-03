"""Shared force-provider protocols and result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from gqteaMD.core.state import State, System


@dataclass(frozen=True)
class ForceResult:
    """Potential energy, Cartesian forces, and provider-specific metadata."""

    energy: float
    forces: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


class ForceProvider(Protocol):
    """Interface implemented by every force backend."""

    def compute(self, system: System, state: State) -> ForceResult:
        """Compute potential energy and forces for the current state."""
