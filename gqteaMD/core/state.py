"""System and state snapshots used by the MD engine."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from gqteaMD.core.cell import Cell
from gqteaMD.core.units import BOLTZMANN_EV_PER_K


@dataclass(frozen=True)
class System:
    """Immutable atom labels, masses, and cell shared by an MD run."""

    symbols: list[str]
    masses: np.ndarray
    cell: Cell

    def __post_init__(self) -> None:
        """Normalize and validate the mass array."""
        masses = np.asarray(self.masses, dtype=float)
        if masses.shape != (len(self.symbols),):
            raise ValueError("Mass array shape must be (natoms,)")
        if np.any(masses <= 0):
            raise ValueError("All atomic masses must be positive")
        object.__setattr__(self, "masses", masses)

    @property
    def natoms(self) -> int:
        """Return the number of atoms in the system."""
        return len(self.symbols)


@dataclass(frozen=True)
class State:
    """Mutable-in-time MD quantities stored as an immutable snapshot."""

    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    energy: float | None = None
    total_energy: float | None = None
    step: int = 0
    time_fs: float = 0.0
    image_flags: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalize array fields and create missing image flags."""
        positions = np.asarray(self.positions, dtype=float)
        velocities = np.asarray(self.velocities, dtype=float)
        forces = np.asarray(self.forces, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Positions must have shape (natoms, 3)")
        if velocities.shape != positions.shape:
            raise ValueError("Velocities must have the same shape as positions")
        if forces.shape != positions.shape:
            raise ValueError("Forces must have the same shape as positions")
        image_flags = self.image_flags
        if image_flags is None:
            image_flags = np.zeros_like(positions, dtype=int)
        else:
            image_flags = np.asarray(image_flags, dtype=int)
            if image_flags.shape != positions.shape:
                raise ValueError("Image flags must have the same shape as positions")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "image_flags", image_flags)

    def with_updates(self, **changes: object) -> "State":
        """Return a copy of this state with selected fields replaced."""
        return replace(self, **changes)

    def unwrapped_positions(self, cell: Cell) -> np.ndarray:
        """Reconstruct continuous coordinates from wrapped positions and images."""
        return self.positions + self.image_flags * cell.lengths

    def kinetic_energy(self, masses: np.ndarray) -> float:
        """Compute kinetic energy in electronvolts from velocities and masses."""
        return float(0.5 * np.sum(masses[:, None] * self.velocities**2) / 0.009648533212331)

    def temperature(self, masses: np.ndarray) -> float:
        """Estimate instantaneous temperature from the kinetic energy."""
        dof = 3 * len(masses)
        if dof == 0:
            return 0.0
        return 2.0 * self.kinetic_energy(masses) / (dof * BOLTZMANN_EV_PER_K)
