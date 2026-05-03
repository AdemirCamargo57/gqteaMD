"""Tests for UFF topology detection and force evaluation."""

import numpy as np
import pytest

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.forces.uff import (
    COULOMB_EV_ANGSTROM_PER_E2,
    UFFForceProvider,
    _bond_rest_length,
    _params,
    assign_uff_atom_types,
    build_uff_topology,
    detect_bonds,
    generate_angles,
    generate_inversions,
    generate_torsions,
    validate_uff_topology,
)
from gqteaMD.forces.uff_parameters import supported_uff_atom_types, validate_uff_parameter_table


def _state(positions):
    """Create a minimal state for UFF force tests."""
    positions = np.array(positions, dtype=float)
    return State(positions=positions, velocities=np.zeros_like(positions), forces=np.zeros_like(positions))


def _finite_difference_force(system, provider, positions, atom, axis, step=1.0e-6):
    """Return a finite-difference force component from provider energies."""
    plus = np.array(positions, dtype=float, copy=True)
    minus = np.array(positions, dtype=float, copy=True)
    plus[atom, axis] += step
    minus[atom, axis] -= step
    velocities = np.zeros_like(plus)
    plus_state = State(positions=plus, velocities=velocities, forces=np.zeros_like(plus))
    minus_state = State(positions=minus, velocities=velocities, forces=np.zeros_like(minus))
    derivative = (provider.compute(system, plus_state).energy - provider.compute(system, minus_state).energy) / (2.0 * step)
    return -derivative


def test_uff_detects_water_bonds_and_angle():
    """Water should produce two bonds, one angle, and expected UFF types."""
    system = System(["O", "H", "H"], np.array([15.999, 1.008, 1.008]), Cell(20.0, 20.0, 20.0))
    state = _state(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.92662721, 0.0],
        ]
    )

    bonds = detect_bonds(system, state)
    topology = build_uff_topology(system, state)

    assert bonds == [(0, 1), (0, 2)]
    assert topology.atom_types == ["O_3", "H_", "H_"]
    assert topology.angles == [(1, 0, 2)]


def test_uff_detects_methane_topology_and_atom_types():
    """Methane should produce tetrahedral bonding and carbon sp3 typing."""
    system = System(["C", "H", "H", "H", "H"], np.array([12.011, 1.008, 1.008, 1.008, 1.008]), Cell(20, 20, 20))
    state = _state(
        [
            [0.0, 0.0, 0.0],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [-0.629118, 0.629118, -0.629118],
            [0.629118, -0.629118, -0.629118],
        ]
    )

    topology = build_uff_topology(system, state)

    assert len(topology.bonds) == 4
    assert len(topology.angles) == 6
    assert len(topology.torsions) == 0
    assert len(topology.inversions) == 0
    assert topology.atom_types == ["C_3", "H_", "H_", "H_", "H_"]


def test_uff_assigns_common_element_types():
    """Supported elements should map to the intended simple UFF types."""
    symbols = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    bonds = [(0, 1), (1, 2), (2, 3), (5, 6)]

    assert assign_uff_atom_types(symbols, bonds) == [
        "H_",
        "C_3",
        "N_2",
        "O_2",
        "F_",
        "P_3+3",
        "S_3+2",
        "Cl",
        "Br",
        "I_",
    ]


def test_generate_angles_from_bonds():
    """Angle generation should enumerate neighbor pairs around each center."""
    assert generate_angles(4, [(0, 1), (1, 2), (1, 3)]) == [(0, 1, 2), (0, 1, 3), (2, 1, 3)]


def test_generate_torsions_from_bonds():
    """Torsion generation should enumerate unique four-atom paths."""
    assert generate_torsions(4, [(0, 1), (1, 2), (2, 3)]) == [(0, 1, 2, 3)]


def test_generate_inversions_from_three_coordinate_centers():
    """Inversion generation should mark three-coordinate central atoms."""
    assert generate_inversions(4, [(0, 1), (0, 2), (0, 3)]) == [(0, 1, 2, 3)]


def test_uff_topology_carries_bonded_extension_fields():
    """Topology should include bond orders, exclusions, and future interaction lists."""
    system = System(["C", "C", "H", "H", "H", "H"], np.array([12.011, 12.011, 1.008, 1.008, 1.008, 1.008]), Cell(20, 20, 20))
    state = _state(
        [
            [0.0, 0.0, 0.0],
            [1.34, 0.0, 0.0],
            [-0.6, 0.9, 0.0],
            [-0.6, -0.9, 0.0],
            [1.94, 0.9, 0.0],
            [1.94, -0.9, 0.0],
        ]
    )

    topology = build_uff_topology(system, state, charges=[-0.1, -0.1, 0.05, 0.05, 0.05, 0.05])

    assert topology.atom_types[:2] == ["C_2", "C_2"]
    assert topology.bond_orders[(0, 1)] == 2.0
    assert topology.charges == [-0.1, -0.1, 0.05, 0.05, 0.05, 0.05]
    assert topology.torsions
    assert (0, 1) in topology.excluded_pairs
    assert topology.one_four_pairs


def test_uff_explicit_bonds_override_distance_detection():
    """Explicit UFF bonds should replace radius-based bond detection."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(20, 20, 20))
    state = _state([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    topology = build_uff_topology(system, state, bonds=[(0, 1)], bond_orders={(0, 1): 1.0})

    assert topology.bonds == [(0, 1)]
    assert topology.bond_orders == {(0, 1): 1.0}


def test_uff_explicit_empty_angles_disable_generated_angles():
    """Explicit empty angle lists should be honored instead of regenerated."""
    system = System(["O", "H", "H"], np.array([15.999, 1.008, 1.008]), Cell(20, 20, 20))
    state = _state([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.92662721, 0.0]])

    topology = build_uff_topology(system, state, angles=[])

    assert topology.bonds == [(0, 1), (0, 2)]
    assert topology.angles == []


def test_uff_rejects_explicit_angle_not_supported_by_bonds():
    """User-supplied bonded terms should be consistent with the bond graph."""
    system = System(["O", "H", "H"], np.array([15.999, 1.008, 1.008]), Cell(20, 20, 20))
    state = _state([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.92662721, 0.0]])

    with pytest.raises(ValueError, match="angle atoms"):
        build_uff_topology(system, state, bonds=[(0, 1)], angles=[(1, 0, 2)])


def test_bond_order_correction_shortens_double_bonds():
    """UFF bond-order correction should shorten inferred double bonds."""
    single = _bond_rest_length(_params("C_2"), _params("C_2"), 1.0)
    double = _bond_rest_length(_params("C_2"), _params("C_2"), 2.0)

    assert double < single


def test_uff_angle_forces_match_finite_difference():
    """Analytic angle forces should match a finite-difference energy derivative."""
    system = System(["O", "H", "H"], np.array([15.999, 1.008, 1.008]), Cell(20.0, 20.0, 20.0))
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.02, 0.0, 0.0],
            [-0.35, 0.90, 0.0],
        ],
        dtype=float,
    )
    state = _state(positions)
    provider = UFFForceProvider(cutoff_angstrom=10.0)
    result = provider.compute(system, state)
    angle = provider.topology.angles[0]
    step = 1.0e-6
    plus = np.array(positions, copy=True)
    minus = np.array(positions, copy=True)
    plus[1, 1] += step
    minus[1, 1] -= step
    derivative = (provider._angle_energy(system, plus, angle) - provider._angle_energy(system, minus, angle)) / (2.0 * step)

    assert result.forces[1, 1] == pytest.approx(-derivative, rel=1.0e-5, abs=1.0e-5)


def test_uff_bond_forces_match_finite_difference():
    """Bond forces should be the negative gradient of the bond energy."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(20, 20, 20))
    positions = np.array([[0.0, 0.0, 0.0], [0.90, 0.0, 0.0]], dtype=float)
    state = _state(positions)
    provider = UFFForceProvider(
        bonds=[(0, 1)],
        bond_orders={(0, 1): 1.0},
        angles=[],
        torsions=[],
        inversions=[],
        use_neighbor_list=False,
    )

    result = provider.compute(system, state)
    finite_difference_force = _finite_difference_force(system, provider, positions, atom=1, axis=0)

    assert result.forces[1, 0] == pytest.approx(finite_difference_force, rel=1.0e-5, abs=1.0e-5)


def test_uff_lennard_jones_forces_match_finite_difference():
    """Lennard-Jones forces should be consistent with the LJ energy."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(20, 20, 20))
    positions = np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]], dtype=float)
    state = _state(positions)
    provider = UFFForceProvider(nonbonded_exclusions="none", use_neighbor_list=False)

    result = provider.compute(system, state)
    finite_difference_force = _finite_difference_force(system, provider, positions, atom=1, axis=0)

    assert result.forces[1, 0] == pytest.approx(finite_difference_force, rel=1.0e-5, abs=1.0e-5)


def test_uff_coulomb_forces_match_finite_difference():
    """Fixed-charge Coulomb forces should be consistent with the Coulomb energy."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(20, 20, 20))
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float)
    state = _state(positions)
    provider = UFFForceProvider(
        charges=[1.0, -1.0],
        nonbonded_exclusions="none",
        use_neighbor_list=False,
    )

    result = provider.compute(system, state)
    finite_difference_force = _finite_difference_force(system, provider, positions, atom=1, axis=0)

    assert result.forces[1, 0] == pytest.approx(finite_difference_force, rel=1.0e-5, abs=1.0e-5)


def test_validate_uff_topology_rejects_bad_charges():
    """Charges should be per-atom when present."""
    system = System(["O", "H", "H"], np.array([15.999, 1.008, 1.008]), Cell(20.0, 20.0, 20.0))
    state = _state(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.92662721, 0.0],
        ]
    )
    topology = build_uff_topology(system, state)
    bad_topology = type(topology)(
        atom_types=topology.atom_types,
        bonds=topology.bonds,
        angles=topology.angles,
        bond_orders=topology.bond_orders,
        charges=[0.0],
    )

    with pytest.raises(ValueError, match="charges length"):
        validate_uff_topology(system.natoms, bad_topology)


def test_uff_parameter_table_validates():
    """Built-in UFF parameter values should pass structural validation."""
    validate_uff_parameter_table()
    assert "C_3" in supported_uff_atom_types()


def test_uff_provider_returns_finite_energy_and_forces():
    """UFF force evaluation should return finite energy and force arrays."""
    system = System(["O", "H", "H"], np.array([15.999, 1.008, 1.008]), Cell(20.0, 20.0, 20.0))
    state = _state(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.92662721, 0.0],
        ]
    )
    provider = UFFForceProvider(cutoff_angstrom=10.0)

    result = provider.compute(system, state)

    assert np.isfinite(result.energy)
    assert result.forces.shape == (3, 3)
    assert np.all(np.isfinite(result.forces))
    assert result.metadata["provider"] == "uff"
    assert result.metadata["bond_count"] == 2
    assert result.metadata["angle_count"] == 1
    assert result.metadata["torsion_energy"] == 0.0
    assert result.metadata["inversion_energy"] == 0.0
    assert result.metadata["electrostatic_energy"] == 0.0


def test_uff_provider_includes_finite_torsion_and_inversion_terms():
    """A twisted ethene-like molecule should activate torsion and inversion terms."""
    system = System(["C", "C", "H", "H", "H", "H"], np.array([12.011, 12.011, 1.008, 1.008, 1.008, 1.008]), Cell(20, 20, 20))
    state = _state(
        [
            [0.0, 0.0, 0.0],
            [1.34, 0.0, 0.0],
            [-0.6, 0.9, 0.0],
            [-0.6, -0.9, 0.0],
            [1.94, 0.6, 0.6],
            [1.94, -0.6, -0.6],
        ]
    )
    provider = UFFForceProvider(cutoff_angstrom=10.0)

    result = provider.compute(system, state)

    assert result.metadata["torsion_count"] > 0
    assert result.metadata["inversion_count"] > 0
    assert result.metadata["torsion_energy"] >= 0.0
    assert result.metadata["inversion_energy"] >= 0.0
    assert np.all(np.isfinite(result.forces))


def test_uff_fixed_charge_electrostatics_for_nonbonded_pair():
    """Explicit charges should add Coulomb energy and forces for nonexcluded pairs."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(20, 20, 20))
    state = _state([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    provider = UFFForceProvider(charges=[1.0, -1.0])

    result = provider.compute(system, state)

    assert result.metadata["electrostatics"] is True
    assert result.metadata["electrostatic_energy"] == pytest.approx(-COULOMB_EV_ANGSTROM_PER_E2 / 5.0)
    assert result.forces[0, 0] > 0.0
    assert result.forces[1, 0] < 0.0


def test_uff_default_electrostatics_excludes_bonded_pairs():
    """Default UFF nonbonded exclusions should skip 1-2 Coulomb interactions."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(20, 20, 20))
    state = _state([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    provider = UFFForceProvider(charges=[1.0, -1.0])

    result = provider.compute(system, state)

    assert result.metadata["bond_count"] == 1
    assert result.metadata["electrostatic_energy"] == 0.0


def test_uff_one_four_electrostatic_scaling():
    """1-4 Coulomb interactions should use the configured scale factor."""
    system = System(["C", "C", "C", "C"], np.array([12.011, 12.011, 12.011, 12.011]), Cell(20, 20, 20))
    state = _state([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    provider = UFFForceProvider(
        charges=[1.0, 0.0, 0.0, 1.0],
        electrostatic_14_scale=0.5,
        bond_detection_scale=1.05,
    )

    result = provider.compute(system, state)

    assert (0, 3) in provider.topology.one_four_pairs
    assert result.metadata["electrostatic_energy"] == pytest.approx(0.5 * COULOMB_EV_ANGSTROM_PER_E2 / 4.5)


def test_uff_shifted_lj_cutoff_changes_energy_not_force_shape():
    """Shifted LJ cutoff should keep finite forces and alter only the reported pair energy."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(20, 20, 20))
    state = _state([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    plain = UFFForceProvider(cutoff_angstrom=10.0, lj_cutoff_mode="plain").compute(system, state)
    shifted = UFFForceProvider(cutoff_angstrom=10.0, lj_cutoff_mode="shift").compute(system, state)

    assert shifted.metadata["lennard_jones_energy"] != pytest.approx(plain.metadata["lennard_jones_energy"])
    assert shifted.forces.shape == plain.forces.shape


def test_uff_neighbor_list_matches_all_pairs_nonbonded():
    """Neighbor-list nonbonded evaluation should match all-pairs evaluation."""
    system = System(["H", "H", "H", "H"], np.array([1.008, 1.008, 1.008, 1.008]), Cell(30, 30, 30))
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    state = _state(positions)
    kwargs = {
        "cutoff_angstrom": 8.0,
        "charges": [0.5, -0.5, 0.25, -0.25],
        "nonbonded_exclusions": "none",
        "neighbor_skin_angstrom": 1.5,
    }

    listed = UFFForceProvider(use_neighbor_list=True, **kwargs).compute(system, state)
    all_pairs = UFFForceProvider(use_neighbor_list=False, **kwargs).compute(system, state)

    assert listed.energy == pytest.approx(all_pairs.energy)
    assert listed.forces == pytest.approx(all_pairs.forces)
    assert listed.metadata["neighbor_pair_count"] < system.natoms * (system.natoms - 1) // 2
    assert listed.metadata["neighbor_list_rebuilds"] == 1


def test_uff_neighbor_list_rebuilds_after_large_displacement():
    """Neighbor lists should rebuild when atoms move more than half the skin."""
    system = System(["H", "H"], np.array([1.008, 1.008]), Cell(30, 30, 30))
    provider = UFFForceProvider(cutoff_angstrom=8.0, neighbor_skin_angstrom=1.0, nonbonded_exclusions="none")
    first_state = _state([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    second_state = _state([[0.0, 0.0, 0.0], [4.7, 0.0, 0.0]])

    provider.compute(system, first_state)
    result = provider.compute(system, second_state)

    assert result.metadata["neighbor_list_rebuilds"] == 2


def test_uff_rejects_unsupported_elements():
    """Unsupported elements should fail during UFF atom typing."""
    with pytest.raises(ValueError, match="atom typing is not implemented"):
        assign_uff_atom_types(["He"], [])
