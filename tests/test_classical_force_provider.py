"""Tests for the classical bonded and Lennard-Jones force provider."""

import numpy as np
import pytest

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.forces.classical import ClassicalForceProvider, HarmonicBond, LennardJonesType


def _state(positions):
    """Create a minimal state for classical force tests."""
    positions = np.array(positions, dtype=float)
    return State(
        positions=positions,
        velocities=np.zeros_like(positions),
        forces=np.zeros_like(positions),
    )


def test_harmonic_bond_energy_and_forces():
    """A stretched harmonic bond should produce restoring forces."""
    system = System(["A", "A"], np.array([1.0, 1.0]), Cell(10.0, 10.0, 10.0))
    provider = ClassicalForceProvider(
        atom_types=["A", "A"],
        bonds=[HarmonicBond(atom_i=0, atom_j=1, k_ev_per_angstrom2=2.0, r0_angstrom=1.0)],
    )

    result = provider.compute(system, _state([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))

    assert result.energy == pytest.approx(0.25)
    assert result.metadata["bond_energy"] == pytest.approx(0.25)
    assert result.forces[0] == pytest.approx([1.0, 0.0, 0.0])
    assert result.forces[1] == pytest.approx([-1.0, 0.0, 0.0])


def test_lennard_jones_energy_and_forces():
    """A Lennard-Jones pair should produce finite energy and opposite forces."""
    system = System(["A", "A"], np.array([1.0, 1.0]), Cell(10.0, 10.0, 10.0))
    provider = ClassicalForceProvider(
        atom_types=["A", "A"],
        lennard_jones={"A": LennardJonesType(epsilon_ev=0.5, sigma_angstrom=1.0)},
    )

    result = provider.compute(system, _state([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    assert result.energy == pytest.approx(-0.03076171875)
    assert result.metadata["lennard_jones_energy"] == pytest.approx(-0.03076171875)
    assert result.forces[0] == pytest.approx([0.0908203125, 0.0, 0.0])
    assert result.forces[1] == pytest.approx([-0.0908203125, 0.0, 0.0])


def test_lennard_jones_uses_minimum_image():
    """Nonbonded distances should use periodic minimum-image convention."""
    system = System(["A", "A"], np.array([1.0, 1.0]), Cell(10.0, 10.0, 10.0))
    provider = ClassicalForceProvider(
        atom_types=["A", "A"],
        lennard_jones={"A": LennardJonesType(epsilon_ev=0.5, sigma_angstrom=1.0)},
    )

    wrapped = provider.compute(system, _state([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
    near = provider.compute(system, _state([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

    assert wrapped.energy == pytest.approx(near.energy)
    assert wrapped.forces == pytest.approx(near.forces)


def test_bonded_pairs_are_excluded_from_lennard_jones_by_default():
    """Bonded atom pairs should not also receive Lennard-Jones interactions."""
    system = System(["A", "A"], np.array([1.0, 1.0]), Cell(10.0, 10.0, 10.0))
    provider = ClassicalForceProvider(
        atom_types=["A", "A"],
        bonds=[HarmonicBond(atom_i=0, atom_j=1, k_ev_per_angstrom2=2.0, r0_angstrom=1.0)],
        lennard_jones={"A": LennardJonesType(epsilon_ev=0.5, sigma_angstrom=1.0)},
    )

    result = provider.compute(system, _state([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))

    assert result.metadata["lennard_jones_energy"] == pytest.approx(0.0)
