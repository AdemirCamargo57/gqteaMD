"""Classical harmonic-bond and Lennard-Jones force provider."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt

import numpy as np

from gqteaMD.core.state import State, System
from gqteaMD.forces.base import ForceResult


@dataclass(frozen=True)
class HarmonicBond:
    """Parameters for one harmonic bond interaction."""

    atom_i: int
    atom_j: int
    k_ev_per_angstrom2: float
    r0_angstrom: float


@dataclass(frozen=True)
class LennardJonesType:
    """Lennard-Jones epsilon and sigma for one atom type."""

    epsilon_ev: float
    sigma_angstrom: float


class ClassicalForceProvider:
    """Harmonic bonds plus Lennard-Jones nonbonded interactions."""

    def __init__(
        self,
        atom_types: list[str],
        bonds: list[HarmonicBond] | None = None,
        lennard_jones: dict[str, LennardJonesType] | None = None,
        cutoff_angstrom: float | None = None,
        exclude_bonded: bool = True,
    ):
        """Store force-field parameters and precompute bonded exclusions."""
        self.atom_types = list(atom_types)
        self.bonds = list(bonds or [])
        self.lennard_jones = dict(lennard_jones or {})
        self.cutoff_angstrom = None if cutoff_angstrom is None else float(cutoff_angstrom)
        self.exclude_bonded = bool(exclude_bonded)
        self._bonded_pairs = {tuple(sorted((bond.atom_i, bond.atom_j))) for bond in self.bonds}

        if self.cutoff_angstrom is not None and self.cutoff_angstrom <= 0:
            raise ValueError("Classical cutoff_angstrom must be positive")

    def compute(self, system: System, state: State) -> ForceResult:
        """Compute classical bonded and nonbonded forces for a state."""
        if len(self.atom_types) != system.natoms:
            raise ValueError("Classical atom_types length must match the number of atoms")

        forces = np.zeros_like(state.positions)
        bond_energy = self._compute_bonds(system, state, forces)
        lj_energy = self._compute_lennard_jones(system, state, forces)
        energy = bond_energy + lj_energy
        return ForceResult(
            energy=energy,
            forces=forces,
            metadata={
                "provider": "classical",
                "bond_energy": bond_energy,
                "lennard_jones_energy": lj_energy,
            },
        )

    def _compute_bonds(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate harmonic bond energy and forces."""
        energy = 0.0
        for bond in self.bonds:
            _validate_atom_index(bond.atom_i, system.natoms)
            _validate_atom_index(bond.atom_j, system.natoms)
            if bond.atom_i == bond.atom_j:
                raise ValueError("Classical bond cannot connect an atom to itself")
            if bond.k_ev_per_angstrom2 < 0:
                raise ValueError("Classical bond force constants must be non-negative")
            if bond.r0_angstrom <= 0:
                raise ValueError("Classical bond equilibrium distances must be positive")

            displacement = system.cell.minimum_image_displacement(
                state.positions[bond.atom_i] - state.positions[bond.atom_j]
            )
            distance = float(np.linalg.norm(displacement))
            if distance == 0.0:
                raise ValueError("Classical bond has zero atom-atom distance")

            extension = distance - bond.r0_angstrom
            energy += 0.5 * bond.k_ev_per_angstrom2 * extension**2
            force = -bond.k_ev_per_angstrom2 * extension * displacement / distance
            forces[bond.atom_i] += force
            forces[bond.atom_j] -= force

        return float(energy)

    def _compute_lennard_jones(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate Lennard-Jones nonbonded energy and forces."""
        energy = 0.0
        for atom_i, atom_j in combinations(range(system.natoms), 2):
            pair = (atom_i, atom_j)
            if self.exclude_bonded and pair in self._bonded_pairs:
                continue

            type_i = self.atom_types[atom_i]
            type_j = self.atom_types[atom_j]
            if type_i not in self.lennard_jones or type_j not in self.lennard_jones:
                continue

            displacement = system.cell.minimum_image_displacement(
                state.positions[atom_i] - state.positions[atom_j]
            )
            distance2 = float(np.dot(displacement, displacement))
            if distance2 == 0.0:
                raise ValueError("Classical Lennard-Jones pair has zero atom-atom distance")
            if self.cutoff_angstrom is not None and distance2 > self.cutoff_angstrom**2:
                continue

            params_i = self.lennard_jones[type_i]
            params_j = self.lennard_jones[type_j]
            if params_i.epsilon_ev < 0 or params_j.epsilon_ev < 0:
                raise ValueError("Classical Lennard-Jones epsilon values must be non-negative")
            if params_i.sigma_angstrom <= 0 or params_j.sigma_angstrom <= 0:
                raise ValueError("Classical Lennard-Jones sigma values must be positive")
            epsilon = sqrt(params_i.epsilon_ev * params_j.epsilon_ev)
            sigma = 0.5 * (params_i.sigma_angstrom + params_j.sigma_angstrom)

            inv_r2 = 1.0 / distance2
            sigma2_over_r2 = sigma * sigma * inv_r2
            sigma6_over_r6 = sigma2_over_r2**3
            sigma12_over_r12 = sigma6_over_r6**2
            energy += 4.0 * epsilon * (sigma12_over_r12 - sigma6_over_r6)
            force = 24.0 * epsilon * (2.0 * sigma12_over_r12 - sigma6_over_r6) * inv_r2 * displacement
            forces[atom_i] += force
            forces[atom_j] -= force

        return float(energy)


def _validate_atom_index(index: int, natoms: int) -> None:
    """Raise a clear error when a force-field atom index is invalid."""
    if index < 0 or index >= natoms:
        raise ValueError(f"Classical atom index out of range: {index}")
