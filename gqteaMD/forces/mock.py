"""Simple analytic force providers used by tests and examples."""

from __future__ import annotations

import numpy as np

from gqteaMD.core.state import State, System
from gqteaMD.forces.base import ForceResult


class HarmonicForceProvider:
    """Simple independent harmonic wells, useful for tests and dry runs."""

    def __init__(self, k_ev_per_angstrom2: float = 1.0, center: np.ndarray | None = None):
        """Store the harmonic force constant and optional equilibrium center."""
        self.k = float(k_ev_per_angstrom2)
        self.center = center

    def compute(self, system: System, state: State) -> ForceResult:
        """Compute harmonic restoring forces for all atoms."""
        center = np.zeros_like(state.positions) if self.center is None else np.asarray(self.center, dtype=float)
        displacement = system.cell.minimum_image_displacement(state.positions - center)
        forces = -self.k * displacement
        energy = float(0.5 * self.k * np.sum(displacement**2))
        return ForceResult(energy=energy, forces=forces, metadata={"provider": "harmonic"})
