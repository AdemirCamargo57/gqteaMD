"""Minimal Universal Force Field topology and force implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import acos, atan2, cos, log, radians, sin, sqrt

import numpy as np

from gqteaMD.core.state import State, System
from gqteaMD.forces.base import ForceResult
from gqteaMD.forces.uff_parameters import (
    COVALENT_RADII,
    KCAL_PER_MOL_TO_EV,
    UFFAtomParameters,
    get_uff_parameters,
)


FINITE_DIFFERENCE_ANGSTROM = 1.0e-6
UFF_BOND_ORDER_SCALE = 0.1332
COULOMB_EV_ANGSTROM_PER_E2 = 14.3996454784255
TORSION_FINITE_DIFFERENCE_ANGSTROM = 1.0e-6
INVERSION_FINITE_DIFFERENCE_ANGSTROM = 1.0e-6
MINIMUM_SIN_ANGLE = 1.0e-8
NONBONDED_EXCLUSION_POLICIES = {"none", "exclude_12", "exclude_12_13"}
LJ_CUTOFF_MODES = {"plain", "shift"}


@dataclass(frozen=True)
class UFFTopology:
    """Detected and derived UFF topology for one system."""

    atom_types: list[str]
    bonds: list[tuple[int, int]]
    angles: list[tuple[int, int, int]]
    bond_orders: dict[tuple[int, int], float] = field(default_factory=dict)
    torsions: list[tuple[int, int, int, int]] = field(default_factory=list)
    inversions: list[tuple[int, int, int, int]] = field(default_factory=list)
    charges: list[float] | None = None
    bonded_pairs: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    angle_pairs: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    excluded_pairs: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    one_four_pairs: frozenset[tuple[int, int]] = field(default_factory=frozenset)


@dataclass
class _NeighborList:
    """Cached nonbonded candidate pairs for one UFF force provider."""

    pairs: list[tuple[int, int]]
    reference_positions: np.ndarray
    cutoff_with_skin: float


class UFFForceProvider:
    """First-milestone UFF provider: bond stretch, angle bend, and Lennard-Jones."""

    def __init__(
        self,
        bond_detection_scale: float = 1.2,
        cutoff_angstrom: float | None = None,
        atom_types: list[str] | None = None,
        bonds: list[tuple[int, int]] | None = None,
        bond_orders: dict[tuple[int, int], float] | None = None,
        angles: list[tuple[int, int, int]] | None = None,
        torsions: list[tuple[int, int, int, int]] | None = None,
        inversions: list[tuple[int, int, int, int]] | None = None,
        charges: list[float] | None = None,
        electrostatics: bool | None = None,
        nonbonded_exclusions: str = "exclude_12_13",
        lj_14_scale: float = 1.0,
        electrostatic_14_scale: float = 1.0,
        lj_cutoff_mode: str = "plain",
        use_neighbor_list: bool = True,
        neighbor_skin_angstrom: float = 2.0,
    ):
        """Configure topology detection and optional nonbonded cutoff."""
        self.bond_detection_scale = float(bond_detection_scale)
        self.cutoff_angstrom = None if cutoff_angstrom is None else float(cutoff_angstrom)
        self.atom_types_override = None if atom_types is None else list(atom_types)
        self.bonds_override = None if bonds is None else [tuple(bond) for bond in bonds]
        self.bond_orders_override = None if bond_orders is None else dict(bond_orders)
        self.angles_override = None if angles is None else [tuple(angle) for angle in angles]
        self.torsions_override = None if torsions is None else [tuple(torsion) for torsion in torsions]
        self.inversions_override = None if inversions is None else [tuple(inversion) for inversion in inversions]
        self.charges_override = None if charges is None else [float(charge) for charge in charges]
        self.electrostatics = self.charges_override is not None if electrostatics is None else bool(electrostatics)
        self.nonbonded_exclusions = nonbonded_exclusions
        self.lj_14_scale = float(lj_14_scale)
        self.electrostatic_14_scale = float(electrostatic_14_scale)
        self.lj_cutoff_mode = lj_cutoff_mode
        self.use_neighbor_list = bool(use_neighbor_list)
        self.neighbor_skin_angstrom = float(neighbor_skin_angstrom)
        self._topology: UFFTopology | None = None
        self._neighbor_list: _NeighborList | None = None
        self._neighbor_list_rebuilds = 0

        if self.bond_detection_scale <= 0:
            raise ValueError("UFF bond_detection_scale must be positive")
        if self.cutoff_angstrom is not None and self.cutoff_angstrom <= 0:
            raise ValueError("UFF cutoff_angstrom must be positive")
        if self.nonbonded_exclusions not in NONBONDED_EXCLUSION_POLICIES:
            raise ValueError(f"UFF nonbonded_exclusions must be one of {sorted(NONBONDED_EXCLUSION_POLICIES)}")
        if self.lj_14_scale < 0.0:
            raise ValueError("UFF lj_14_scale must be non-negative")
        if self.electrostatic_14_scale < 0.0:
            raise ValueError("UFF electrostatic_14_scale must be non-negative")
        if self.lj_cutoff_mode not in LJ_CUTOFF_MODES:
            raise ValueError(f"UFF lj_cutoff_mode must be one of {sorted(LJ_CUTOFF_MODES)}")
        if self.neighbor_skin_angstrom < 0.0:
            raise ValueError("UFF neighbor_skin_angstrom must be non-negative")

    @property
    def topology(self) -> UFFTopology | None:
        """Return the cached topology after the first force evaluation."""
        return self._topology

    def compute(self, system: System, state: State) -> ForceResult:
        """Compute UFF bond, angle, and Lennard-Jones forces."""
        if self._topology is None:
            self._topology = build_uff_topology(
                system,
                state,
                bond_detection_scale=self.bond_detection_scale,
                atom_types=self.atom_types_override,
                bonds=self.bonds_override,
                bond_orders=self.bond_orders_override,
                angles=self.angles_override,
                torsions=self.torsions_override,
                inversions=self.inversions_override,
                charges=self.charges_override,
            )
        if len(self._topology.atom_types) != system.natoms:
            raise ValueError("UFF topology atom count does not match the system")

        forces = np.zeros_like(state.positions)
        bond_energy = self._compute_bonds(system, state, forces)
        angle_energy = self._compute_angles(system, state, forces)
        torsion_energy = self._compute_torsions(system, state, forces)
        inversion_energy = self._compute_inversions(system, state, forces)
        lj_energy = self._compute_lennard_jones(system, state, forces)
        electrostatic_energy = self._compute_electrostatics(system, state, forces)
        energy = bond_energy + angle_energy + torsion_energy + inversion_energy + lj_energy + electrostatic_energy
        return ForceResult(
            energy=energy,
            forces=forces,
            metadata={
                "provider": "uff",
                "atom_types": self._topology.atom_types,
                "bond_count": len(self._topology.bonds),
                "angle_count": len(self._topology.angles),
                "torsion_count": len(self._topology.torsions),
                "inversion_count": len(self._topology.inversions),
                "bond_energy": bond_energy,
                "angle_energy": angle_energy,
                "torsion_energy": torsion_energy,
                "inversion_energy": inversion_energy,
                "lennard_jones_energy": lj_energy,
                "electrostatic_energy": electrostatic_energy,
                "electrostatics": self.electrostatics,
                "nonbonded_exclusions": self.nonbonded_exclusions,
                "lj_14_scale": self.lj_14_scale,
                "electrostatic_14_scale": self.electrostatic_14_scale,
                "lj_cutoff_mode": self.lj_cutoff_mode,
                "use_neighbor_list": self.use_neighbor_list,
                "neighbor_skin_angstrom": self.neighbor_skin_angstrom,
                "neighbor_pair_count": self._neighbor_pair_count(system),
                "neighbor_list_rebuilds": self._neighbor_list_rebuilds,
            },
        )

    def _compute_bonds(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate UFF bond-stretch energy and forces."""
        assert self._topology is not None
        energy = 0.0
        for atom_i, atom_j in self._topology.bonds:
            params_i = _params(self._topology.atom_types[atom_i])
            params_j = _params(self._topology.atom_types[atom_j])
            bond_order = self._topology.bond_orders[_normalize_pair(atom_i, atom_j)]
            r0 = _bond_rest_length(params_i, params_j, bond_order)
            k = _bond_force_constant(params_i, params_j, r0)

            displacement = system.cell.minimum_image_displacement(state.positions[atom_i] - state.positions[atom_j])
            distance = float(np.linalg.norm(displacement))
            if distance == 0.0:
                raise ValueError("UFF bond has zero atom-atom distance")

            extension = distance - r0
            energy += 0.5 * k * extension * extension
            force = -k * extension * displacement / distance
            forces[atom_i] += force
            forces[atom_j] -= force

        return float(energy)

    def _compute_angles(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate UFF angle-bend energy and analytic forces."""
        assert self._topology is not None
        energy = 0.0
        for angle in self._topology.angles:
            energy += self._add_angle_energy_and_forces(system, state.positions, forces, angle)
        return float(energy)

    def _add_angle_energy_and_forces(
        self,
        system: System,
        positions: np.ndarray,
        forces: np.ndarray,
        angle: tuple[int, int, int],
    ) -> float:
        """Accumulate the harmonic angle energy and analytic forces for one angle."""
        assert self._topology is not None
        atom_i, atom_j, atom_k = angle
        params_i = _params(self._topology.atom_types[atom_i])
        params_j = _params(self._topology.atom_types[atom_j])
        params_k = _params(self._topology.atom_types[atom_k])
        vector_ji = system.cell.minimum_image_displacement(positions[atom_i] - positions[atom_j])
        vector_jk = system.cell.minimum_image_displacement(positions[atom_k] - positions[atom_j])
        norm_ji = float(np.linalg.norm(vector_ji))
        norm_jk = float(np.linalg.norm(vector_jk))
        if norm_ji == 0.0 or norm_jk == 0.0:
            raise ValueError("UFF angle has zero atom-atom distance")
        cosine = float(np.dot(vector_ji, vector_jk) / (norm_ji * norm_jk))
        cosine = max(-1.0, min(1.0, cosine))
        theta = acos(cosine)
        sine = max(sqrt(max(1.0 - cosine * cosine, 0.0)), MINIMUM_SIN_ANGLE)
        theta0 = radians(params_j.theta0_degrees)
        bond_order_ij = self._topology.bond_orders[_normalize_pair(atom_i, atom_j)]
        bond_order_jk = self._topology.bond_orders[_normalize_pair(atom_j, atom_k)]
        k = _angle_force_constant(params_i, params_j, params_k, bond_order_ij, bond_order_jk)
        delta = theta - theta0
        energy = 0.5 * k * delta * delta
        prefactor = k * delta / sine
        force_i = prefactor * (vector_jk / (norm_ji * norm_jk) - cosine * vector_ji / (norm_ji * norm_ji))
        force_k = prefactor * (vector_ji / (norm_ji * norm_jk) - cosine * vector_jk / (norm_jk * norm_jk))
        force_j = -force_i - force_k
        forces[atom_i] += force_i
        forces[atom_j] += force_j
        forces[atom_k] += force_k
        return float(energy)

    def _angle_energy(
        self,
        system: System,
        positions: np.ndarray,
        angle: tuple[int, int, int],
    ) -> float:
        """Return the harmonic angle-bend energy for one angle."""
        assert self._topology is not None
        atom_i, atom_j, atom_k = angle
        params_i = _params(self._topology.atom_types[atom_i])
        params_j = _params(self._topology.atom_types[atom_j])
        params_k = _params(self._topology.atom_types[atom_k])
        theta = _angle_rad(system, positions, atom_i, atom_j, atom_k)
        theta0 = radians(params_j.theta0_degrees)
        bond_order_ij = self._topology.bond_orders[_normalize_pair(atom_i, atom_j)]
        bond_order_jk = self._topology.bond_orders[_normalize_pair(atom_j, atom_k)]
        k = _angle_force_constant(params_i, params_j, params_k, bond_order_ij, bond_order_jk)
        delta = theta - theta0
        return float(0.5 * k * delta * delta)

    def _compute_torsions(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate UFF torsion energy and finite-difference forces."""
        assert self._topology is not None
        energy = 0.0
        for torsion in self._topology.torsions:
            barrier, periodicity, phase = self._torsion_parameters(torsion)
            if barrier <= 0.0:
                continue
            energy += self._torsion_energy(system, state.positions, torsion)
            _add_finite_difference_forces(
                forces,
                state.positions,
                torsion,
                lambda positions, torsion=torsion: self._torsion_energy(system, positions, torsion),
                step=TORSION_FINITE_DIFFERENCE_ANGSTROM,
            )
        return float(energy)

    def _torsion_energy(
        self,
        system: System,
        positions: np.ndarray,
        torsion: tuple[int, int, int, int],
    ) -> float:
        """Return UFF cosine torsion energy for one torsion."""
        barrier, periodicity, phase = self._torsion_parameters(torsion)
        if barrier <= 0.0:
            return 0.0
        phi = _dihedral_rad(system, positions, torsion)
        return float(0.5 * barrier * (1.0 - cos(periodicity * phase) * cos(periodicity * phi)))

    def _torsion_parameters(self, torsion: tuple[int, int, int, int]) -> tuple[float, int, float]:
        """Return barrier, periodicity, and phase for a generated torsion."""
        assert self._topology is not None
        atom_i, atom_j, atom_k, atom_l = torsion
        params_j = _params(self._topology.atom_types[atom_j])
        params_k = _params(self._topology.atom_types[atom_k])
        type_j = self._topology.atom_types[atom_j]
        type_k = self._topology.atom_types[atom_k]
        type_i = self._topology.atom_types[atom_i]
        type_l = self._topology.atom_types[atom_l]
        bond_order = self._topology.bond_orders[_normalize_pair(atom_j, atom_k)]
        hybrid_j = _hybridization(type_j)
        hybrid_k = _hybridization(type_k)

        if hybrid_j == "sp3" and hybrid_k == "sp3":
            barrier = sqrt(params_j.vsp3_kcal_per_mol * params_k.vsp3_kcal_per_mol) * KCAL_PER_MOL_TO_EV
            return barrier, 3, radians(60.0)
        if hybrid_j == "sp2" and hybrid_k == "sp2":
            barrier = _sp2_torsion_barrier(params_j, params_k, bond_order) * KCAL_PER_MOL_TO_EV
            return barrier, 2, radians(180.0)
        if {hybrid_j, hybrid_k} == {"sp2", "sp3"} and (type_i.endswith("_2") or type_l.endswith("_2")):
            return 2.0 * KCAL_PER_MOL_TO_EV, 3, radians(180.0)
        return 0.0, 1, 0.0

    def _compute_inversions(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate out-of-plane inversion energy and finite-difference forces."""
        assert self._topology is not None
        energy = 0.0
        for inversion in self._topology.inversions:
            barrier = self._inversion_barrier(inversion)
            if barrier <= 0.0:
                continue
            energy += self._inversion_energy(system, state.positions, inversion)
            _add_finite_difference_forces(
                forces,
                state.positions,
                inversion,
                lambda positions, inversion=inversion: self._inversion_energy(system, positions, inversion),
                step=INVERSION_FINITE_DIFFERENCE_ANGSTROM,
            )
        return float(energy)

    def _inversion_energy(
        self,
        system: System,
        positions: np.ndarray,
        inversion: tuple[int, int, int, int],
    ) -> float:
        """Return harmonic out-of-plane inversion energy for one center."""
        barrier = self._inversion_barrier(inversion)
        if barrier <= 0.0:
            return 0.0
        gamma = _out_of_plane_angle_rad(system, positions, inversion)
        return float(0.5 * barrier * gamma * gamma)

    def _inversion_barrier(self, inversion: tuple[int, int, int, int]) -> float:
        """Return a conservative UFF-like inversion barrier for a center atom."""
        assert self._topology is not None
        center, atom_i, atom_j, atom_k = inversion
        center_type = self._topology.atom_types[center]
        neighbor_types = [self._topology.atom_types[atom] for atom in (atom_i, atom_j, atom_k)]
        if center_type.startswith("C_2") or center_type.startswith("C_R"):
            value = 50.0 if any(atom_type.startswith("O_2") for atom_type in neighbor_types) else 6.0
        elif center_type.startswith("N_"):
            value = 6.0
        elif center_type.startswith("P_"):
            value = 6.0
        else:
            value = 0.0
        return value * KCAL_PER_MOL_TO_EV

    def _compute_lennard_jones(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate UFF Lennard-Jones energy and forces."""
        assert self._topology is not None
        energy = 0.0
        for atom_i, atom_j in self._nonbonded_pairs(system, state):
            scale = self._nonbonded_scale(atom_i, atom_j, self.lj_14_scale)
            if scale == 0.0:
                continue

            displacement = system.cell.minimum_image_displacement(state.positions[atom_i] - state.positions[atom_j])
            distance2 = float(np.dot(displacement, displacement))
            if distance2 == 0.0:
                raise ValueError("UFF Lennard-Jones pair has zero atom-atom distance")
            if self.cutoff_angstrom is not None and distance2 > self.cutoff_angstrom**2:
                continue

            distance = sqrt(distance2)
            params_i = _params(self._topology.atom_types[atom_i])
            params_j = _params(self._topology.atom_types[atom_j])
            x_ij = sqrt(params_i.x1 * params_j.x1)
            epsilon = sqrt(params_i.d1_kcal_per_mol * params_j.d1_kcal_per_mol) * KCAL_PER_MOL_TO_EV
            ratio = x_ij / distance
            ratio6 = ratio**6
            ratio12 = ratio6 * ratio6
            pair_energy = epsilon * (ratio12 - 2.0 * ratio6)
            if self.cutoff_angstrom is not None and self.lj_cutoff_mode == "shift":
                cutoff_ratio = x_ij / self.cutoff_angstrom
                cutoff_ratio6 = cutoff_ratio**6
                pair_energy -= epsilon * (cutoff_ratio6 * cutoff_ratio6 - 2.0 * cutoff_ratio6)
            energy += scale * pair_energy
            force = scale * 12.0 * epsilon * (x_ij**12 / distance**14 - x_ij**6 / distance**8) * displacement
            forces[atom_i] += force
            forces[atom_j] -= force

        return float(energy)

    def _compute_electrostatics(self, system: System, state: State, forces: np.ndarray) -> float:
        """Accumulate Coulomb energy and forces for explicit fixed charges."""
        assert self._topology is not None
        if not self.electrostatics:
            return 0.0
        if self._topology.charges is None:
            raise ValueError("UFF electrostatics requires per-atom charges")

        energy = 0.0
        for atom_i, atom_j in self._nonbonded_pairs(system, state):
            scale = self._nonbonded_scale(atom_i, atom_j, self.electrostatic_14_scale)
            if scale == 0.0:
                continue
            displacement = system.cell.minimum_image_displacement(state.positions[atom_i] - state.positions[atom_j])
            distance2 = float(np.dot(displacement, displacement))
            if distance2 == 0.0:
                raise ValueError("UFF Coulomb pair has zero atom-atom distance")
            if self.cutoff_angstrom is not None and distance2 > self.cutoff_angstrom**2:
                continue
            distance = sqrt(distance2)
            charge_product = self._topology.charges[atom_i] * self._topology.charges[atom_j]
            prefactor = scale * COULOMB_EV_ANGSTROM_PER_E2 * charge_product
            energy += prefactor / distance
            force = prefactor * displacement / (distance2 * distance)
            forces[atom_i] += force
            forces[atom_j] -= force
        return float(energy)

    def _nonbonded_scale(self, atom_i: int, atom_j: int, one_four_scale: float) -> float:
        """Return the nonbonded scale for a pair under the configured policy."""
        assert self._topology is not None
        pair = _normalize_pair(atom_i, atom_j)
        if self.nonbonded_exclusions in {"exclude_12", "exclude_12_13"} and pair in self._topology.bonded_pairs:
            return 0.0
        if self.nonbonded_exclusions == "exclude_12_13" and pair in self._topology.angle_pairs:
            return 0.0
        if pair in self._topology.one_four_pairs:
            return one_four_scale
        return 1.0

    def _nonbonded_pairs(self, system: System, state: State):
        """Return nonbonded candidate pairs, using a cached neighbor list when possible."""
        if not self.use_neighbor_list or self.cutoff_angstrom is None:
            return combinations(range(system.natoms), 2)
        self._ensure_neighbor_list(system, state)
        assert self._neighbor_list is not None
        return iter(self._neighbor_list.pairs)

    def _ensure_neighbor_list(self, system: System, state: State) -> None:
        """Build or refresh the Verlet neighbor list if atoms moved too far."""
        cutoff_with_skin = self.cutoff_angstrom + self.neighbor_skin_angstrom
        if self._neighbor_list is not None and self._neighbor_list.cutoff_with_skin == cutoff_with_skin:
            displacements = system.cell.minimum_image_displacement(state.positions - self._neighbor_list.reference_positions)
            max_displacement = float(np.max(np.linalg.norm(displacements, axis=1))) if system.natoms else 0.0
            if max_displacement <= 0.5 * self.neighbor_skin_angstrom:
                return

        cutoff2 = cutoff_with_skin * cutoff_with_skin
        pairs: list[tuple[int, int]] = []
        for atom_i, atom_j in combinations(range(system.natoms), 2):
            displacement = system.cell.minimum_image_displacement(state.positions[atom_i] - state.positions[atom_j])
            if float(np.dot(displacement, displacement)) <= cutoff2:
                pairs.append((atom_i, atom_j))
        self._neighbor_list = _NeighborList(
            pairs=pairs,
            reference_positions=np.array(state.positions, dtype=float, copy=True),
            cutoff_with_skin=cutoff_with_skin,
        )
        self._neighbor_list_rebuilds += 1

    def _neighbor_pair_count(self, system: System) -> int:
        """Return the current number of candidate nonbonded pairs."""
        if self.use_neighbor_list and self.cutoff_angstrom is not None and self._neighbor_list is not None:
            return len(self._neighbor_list.pairs)
        return system.natoms * (system.natoms - 1) // 2


def build_uff_topology(
    system: System,
    state: State,
    bond_detection_scale: float = 1.2,
    atom_types: list[str] | None = None,
    bonds: list[tuple[int, int]] | None = None,
    bond_orders: dict[tuple[int, int], float] | None = None,
    angles: list[tuple[int, int, int]] | None = None,
    torsions: list[tuple[int, int, int, int]] | None = None,
    inversions: list[tuple[int, int, int, int]] | None = None,
    charges: list[float] | None = None,
) -> UFFTopology:
    """Build atom types, bonds, and angles for the UFF provider."""
    normalized_bonds = (
        [_normalize_pair(atom_i, atom_j) for atom_i, atom_j in detect_bonds(system, state, bond_detection_scale)]
        if bonds is None
        else [_normalize_pair(int(atom_i), int(atom_j)) for atom_i, atom_j in bonds]
    )
    assigned_types = list(atom_types) if atom_types is not None else assign_uff_atom_types(system.symbols, normalized_bonds)
    resolved_angles = (
        generate_angles(system.natoms, normalized_bonds)
        if angles is None
        else [(int(atom_i), int(atom_j), int(atom_k)) for atom_i, atom_j, atom_k in angles]
    )
    resolved_torsions = (
        generate_torsions(system.natoms, normalized_bonds)
        if torsions is None
        else [(int(atom_i), int(atom_j), int(atom_k), int(atom_l)) for atom_i, atom_j, atom_k, atom_l in torsions]
    )
    resolved_inversions = (
        generate_inversions(system.natoms, normalized_bonds)
        if inversions is None
        else [(int(center), int(atom_i), int(atom_j), int(atom_k)) for center, atom_i, atom_j, atom_k in inversions]
    )
    normalized_bond_orders = _normalized_bond_orders(
        system,
        state,
        normalized_bonds,
        assigned_types,
        bond_orders,
    )
    bonded_pairs = frozenset(_normalize_pair(atom_i, atom_j) for atom_i, atom_j in normalized_bonds)
    angle_pairs = frozenset(_normalize_pair(atom_i, atom_k) for atom_i, _atom_j, atom_k in resolved_angles)
    excluded_pairs = frozenset(bonded_pairs | angle_pairs)
    one_four_pairs = frozenset(_normalize_pair(atom_i, atom_l) for atom_i, _atom_j, _atom_k, atom_l in resolved_torsions)
    topology = UFFTopology(
        atom_types=assigned_types,
        bonds=normalized_bonds,
        angles=resolved_angles,
        bond_orders=normalized_bond_orders,
        torsions=resolved_torsions,
        inversions=resolved_inversions,
        charges=None if charges is None else list(charges),
        bonded_pairs=bonded_pairs,
        angle_pairs=angle_pairs,
        excluded_pairs=excluded_pairs,
        one_four_pairs=one_four_pairs,
    )
    validate_uff_topology(system.natoms, topology)
    return topology


def detect_bonds(system: System, state: State, bond_detection_scale: float = 1.2) -> list[tuple[int, int]]:
    """Detect covalent bonds from element radii and current distances."""
    bonds: list[tuple[int, int]] = []
    for atom_i, atom_j in combinations(range(system.natoms), 2):
        symbol_i = system.symbols[atom_i]
        symbol_j = system.symbols[atom_j]
        if symbol_i not in COVALENT_RADII or symbol_j not in COVALENT_RADII:
            raise ValueError(f"UFF has no bond-detection radius for {symbol_i}-{symbol_j}")

        displacement = system.cell.minimum_image_displacement(state.positions[atom_i] - state.positions[atom_j])
        distance = float(np.linalg.norm(displacement))
        threshold = bond_detection_scale * (COVALENT_RADII[symbol_i] + COVALENT_RADII[symbol_j])
        if 0.1 < distance <= threshold:
            bonds.append((atom_i, atom_j))
    return bonds


def assign_uff_atom_types(symbols: list[str], bonds: list[tuple[int, int]]) -> list[str]:
    """Assign UFF atom types from elements and local bond graph features."""
    neighbors = _neighbors_by_atom(len(symbols), bonds)
    degrees = [len(atom_neighbors) for atom_neighbors in neighbors]

    atom_types: list[str] = []
    for atom_index, (symbol, degree) in enumerate(zip(symbols, degrees, strict=True)):
        if symbol == "H":
            atom_types.append("H_")
        elif symbol == "C":
            atom_types.append(_assign_carbon_type(atom_index, symbols, neighbors))
        elif symbol == "N":
            atom_types.append("N_2" if degree <= 2 else "N_3")
        elif symbol == "O":
            atom_types.append("O_2" if degree <= 1 else "O_3")
        elif symbol == "F":
            atom_types.append("F_")
        elif symbol == "P":
            atom_types.append("P_3+3")
        elif symbol == "S":
            atom_types.append("S_3+2")
        elif symbol == "Cl":
            atom_types.append("Cl")
        elif symbol == "Br":
            atom_types.append("Br")
        elif symbol == "I":
            atom_types.append("I_")
        else:
            raise ValueError(f"UFF atom typing is not implemented for element {symbol!r}")
    return atom_types


def generate_angles(natoms: int, bonds: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Generate all unique bonded angles from the bond graph."""
    neighbors = _neighbors_by_atom(natoms, bonds)

    angles: list[tuple[int, int, int]] = []
    for center, center_neighbors in enumerate(neighbors):
        for atom_i, atom_k in combinations(sorted(center_neighbors), 2):
            angles.append((atom_i, center, atom_k))
    return angles


def generate_torsions(natoms: int, bonds: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Generate unique torsions from the bond graph."""
    neighbors = _neighbors_by_atom(natoms, bonds)
    torsions: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for atom_j, atom_k in sorted(_normalize_pair(atom_i, atom_j) for atom_i, atom_j in bonds):
        for atom_i in sorted(neighbors[atom_j] - {atom_k}):
            for atom_l in sorted(neighbors[atom_k] - {atom_j}):
                if len({atom_i, atom_j, atom_k, atom_l}) != 4:
                    continue
                torsion = (atom_i, atom_j, atom_k, atom_l)
                reverse = (atom_l, atom_k, atom_j, atom_i)
                canonical = min(torsion, reverse)
                if canonical in seen:
                    continue
                seen.add(canonical)
                torsions.append(torsion)
    return torsions


def generate_inversions(natoms: int, bonds: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Generate inversion centers as center-neighbor-neighbor-neighbor tuples."""
    neighbors = _neighbors_by_atom(natoms, bonds)
    inversions: list[tuple[int, int, int, int]] = []
    for center, center_neighbors in enumerate(neighbors):
        if len(center_neighbors) != 3:
            continue
        atom_i, atom_j, atom_k = sorted(center_neighbors)
        inversions.append((center, atom_i, atom_j, atom_k))
    return inversions


def validate_uff_topology(natoms: int, topology: UFFTopology) -> None:
    """Validate topology indices, parameters, and derived interaction lists."""
    if len(topology.atom_types) != natoms:
        raise ValueError("UFF atom_types length must match the number of atoms")
    for atom_type in topology.atom_types:
        _params(atom_type)
    if topology.charges is not None and len(topology.charges) != natoms:
        raise ValueError("UFF charges length must match the number of atoms")

    bonds = [_normalize_pair(atom_i, atom_j) for atom_i, atom_j in topology.bonds]
    if len(set(bonds)) != len(bonds):
        raise ValueError("UFF topology contains duplicate bonds")
    bonded_pairs = set(bonds)
    for atom_i, atom_j in bonds:
        _validate_atom_index(atom_i, natoms, "bond")
        _validate_atom_index(atom_j, natoms, "bond")
        if atom_i == atom_j:
            raise ValueError("UFF bond cannot connect an atom to itself")

    expected_bond_orders = set(bonds)
    if set(topology.bond_orders) != expected_bond_orders:
        raise ValueError("UFF bond_orders must contain exactly one value for each bond")
    for pair, order in topology.bond_orders.items():
        if order <= 0:
            raise ValueError(f"UFF bond order for pair {pair} must be positive")

    for atom_i, atom_j, atom_k in topology.angles:
        for atom in (atom_i, atom_j, atom_k):
            _validate_atom_index(atom, natoms, "angle")
        if len({atom_i, atom_j, atom_k}) != 3:
            raise ValueError("UFF angle atoms must be distinct")
        if _normalize_pair(atom_i, atom_j) not in bonded_pairs or _normalize_pair(atom_j, atom_k) not in bonded_pairs:
            raise ValueError("UFF angle atoms must follow two bonded pairs")

    for atom_i, atom_j, atom_k, atom_l in topology.torsions:
        for atom in (atom_i, atom_j, atom_k, atom_l):
            _validate_atom_index(atom, natoms, "torsion")
        if len({atom_i, atom_j, atom_k, atom_l}) != 4:
            raise ValueError("UFF torsion atoms must be distinct")
        if (
            _normalize_pair(atom_i, atom_j) not in bonded_pairs
            or _normalize_pair(atom_j, atom_k) not in bonded_pairs
            or _normalize_pair(atom_k, atom_l) not in bonded_pairs
        ):
            raise ValueError("UFF torsion atoms must follow three bonded pairs")

    for center, atom_i, atom_j, atom_k in topology.inversions:
        for atom in (center, atom_i, atom_j, atom_k):
            _validate_atom_index(atom, natoms, "inversion")
        if len({center, atom_i, atom_j, atom_k}) != 4:
            raise ValueError("UFF inversion atoms must be distinct")
        if any(_normalize_pair(center, atom) not in bonded_pairs for atom in (atom_i, atom_j, atom_k)):
            raise ValueError("UFF inversion neighbors must be bonded to the center atom")


def _neighbors_by_atom(natoms: int, bonds: list[tuple[int, int]]) -> list[set[int]]:
    """Return bonded-neighbor sets for each atom."""
    neighbors = [set() for _ in range(natoms)]
    for atom_i, atom_j in bonds:
        _validate_atom_index(atom_i, natoms, "bond")
        _validate_atom_index(atom_j, natoms, "bond")
        if atom_i == atom_j:
            raise ValueError("UFF bond cannot connect an atom to itself")
        neighbors[atom_i].add(atom_j)
        neighbors[atom_j].add(atom_i)
    return neighbors


def _assign_carbon_type(atom_index: int, symbols: list[str], neighbors: list[set[int]]) -> str:
    """Assign the best currently supported carbon type from graph context."""
    degree = len(neighbors[atom_index])
    if degree == 3:
        return "C_2"
    if degree == 2 and all(symbols[neighbor] in {"C", "N", "O", "S"} for neighbor in neighbors[atom_index]):
        return "C_2"
    return "C_3"


def _normalized_bond_orders(
    system: System,
    state: State,
    bonds: list[tuple[int, int]],
    atom_types: list[str],
    bond_orders: dict[tuple[int, int], float] | None,
) -> dict[tuple[int, int], float]:
    """Return normalized bond-order values, inferring simple orders when absent."""
    if bond_orders is None:
        return _infer_bond_orders(system, state, bonds, atom_types)
    normalized = {_normalize_pair(atom_i, atom_j): float(order) for (atom_i, atom_j), order in bond_orders.items()}
    missing = set(bonds) - set(normalized)
    extra = set(normalized) - set(bonds)
    if missing or extra:
        raise ValueError("UFF bond_orders must contain exactly one value for each bond")
    return normalized


def _infer_bond_orders(
    system: System,
    state: State,
    bonds: list[tuple[int, int]],
    atom_types: list[str],
) -> dict[tuple[int, int], float]:
    """Infer simple bond orders by matching initial distances to UFF rest lengths."""
    inferred: dict[tuple[int, int], float] = {}
    for atom_i, atom_j in bonds:
        params_i = _params(atom_types[atom_i])
        params_j = _params(atom_types[atom_j])
        displacement = system.cell.minimum_image_displacement(state.positions[atom_i] - state.positions[atom_j])
        distance = float(np.linalg.norm(displacement))
        candidates = _bond_order_candidates(atom_types[atom_i], atom_types[atom_j])
        inferred[_normalize_pair(atom_i, atom_j)] = min(
            candidates,
            key=lambda order: abs(distance - _bond_rest_length(params_i, params_j, order)),
        )
    return inferred


def _bond_order_candidates(atom_type_i: str, atom_type_j: str) -> tuple[float, ...]:
    """Return plausible automatic bond-order candidates for the current atom types."""
    hybrid_i = _hybridization(atom_type_i)
    hybrid_j = _hybridization(atom_type_j)
    if "other" in {hybrid_i, hybrid_j}:
        return (1.0,)
    if hybrid_i == "sp" and hybrid_j == "sp":
        return (1.0, 2.0, 3.0)
    if hybrid_i in {"sp", "sp2"} and hybrid_j in {"sp", "sp2"}:
        return (1.0, 1.5, 2.0)
    return (1.0,)


def _normalize_pair(atom_i: int, atom_j: int) -> tuple[int, int]:
    """Return a deterministic unordered atom pair."""
    return (atom_i, atom_j) if atom_i <= atom_j else (atom_j, atom_i)


def _validate_atom_index(atom: int, natoms: int, interaction: str) -> None:
    """Validate an atom index used by a UFF topology interaction."""
    if atom < 0 or atom >= natoms:
        raise ValueError(f"UFF {interaction} atom index {atom} is out of range for {natoms} atoms")


def _params(atom_type: str) -> UFFAtomParameters:
    """Look up UFF parameters for an atom type."""
    return get_uff_parameters(atom_type)


def _bond_rest_length(params_i: UFFAtomParameters, params_j: UFFAtomParameters, bond_order: float) -> float:
    """Return UFF equilibrium bond distance with bond-order and EN corrections."""
    radius_sum = params_i.r1 + params_j.r1
    bond_order_correction = -UFF_BOND_ORDER_SCALE * radius_sum * log(bond_order)
    en_correction = _electronegativity_correction(params_i, params_j)
    return max(radius_sum + bond_order_correction - en_correction, 1.0e-6)


def _electronegativity_correction(params_i: UFFAtomParameters, params_j: UFFAtomParameters) -> float:
    """Return the O'Keeffe-Brese electronegativity bond-length correction."""
    chi_i = params_i.electronegativity
    chi_j = params_j.electronegativity
    if chi_i <= 0.0 or chi_j <= 0.0:
        return 0.0
    denominator = chi_i * params_i.r1 + chi_j * params_j.r1
    if denominator <= 0.0:
        return 0.0
    return params_i.r1 * params_j.r1 * (sqrt(chi_i) - sqrt(chi_j)) ** 2 / denominator


def _bond_force_constant(params_i: UFFAtomParameters, params_j: UFFAtomParameters, r0: float) -> float:
    """Estimate a UFF bond force constant in eV/angstrom^2."""
    k_kcal_per_mol_angstrom2 = 664.12 * params_i.z1 * params_j.z1 / (r0**3)
    return k_kcal_per_mol_angstrom2 * KCAL_PER_MOL_TO_EV


def _angle_force_constant(
    params_i: UFFAtomParameters,
    params_j: UFFAtomParameters,
    params_k: UFFAtomParameters,
    bond_order_ij: float = 1.0,
    bond_order_jk: float = 1.0,
) -> float:
    """Estimate a UFF angle force constant in eV/radian^2."""
    r_ij = _bond_rest_length(params_i, params_j, bond_order_ij)
    r_jk = _bond_rest_length(params_j, params_k, bond_order_jk)
    theta0 = radians(params_j.theta0_degrees)
    r_ik2 = r_ij * r_ij + r_jk * r_jk - 2.0 * r_ij * r_jk * cos(theta0)
    r_ik = sqrt(max(r_ik2, 1.0e-12))
    bracket = 3.0 * r_ij * r_jk * (1.0 - cos(theta0) ** 2) - r_ik2 * cos(theta0)
    k_kcal_per_mol_rad2 = 664.12 * params_i.z1 * params_k.z1 * bracket / (r_ij * r_jk * r_ik**5)
    return max(abs(k_kcal_per_mol_rad2) * KCAL_PER_MOL_TO_EV, 1.0e-6)


def _angle_rad(system: System, positions: np.ndarray, atom_i: int, atom_j: int, atom_k: int) -> float:
    """Return the minimum-image angle in radians for three atoms."""
    vector_ji = system.cell.minimum_image_displacement(positions[atom_i] - positions[atom_j])
    vector_jk = system.cell.minimum_image_displacement(positions[atom_k] - positions[atom_j])
    norm_ji = float(np.linalg.norm(vector_ji))
    norm_jk = float(np.linalg.norm(vector_jk))
    if norm_ji == 0.0 or norm_jk == 0.0:
        raise ValueError("UFF angle has zero atom-atom distance")
    cosine = float(np.dot(vector_ji, vector_jk) / (norm_ji * norm_jk))
    return acos(max(-1.0, min(1.0, cosine)))


def _dihedral_rad(
    system: System,
    positions: np.ndarray,
    torsion: tuple[int, int, int, int],
) -> float:
    """Return a minimum-image dihedral angle in radians."""
    atom_i, atom_j, atom_k, atom_l = torsion
    p_i = positions[atom_j] + system.cell.minimum_image_displacement(positions[atom_i] - positions[atom_j])
    p_j = positions[atom_j]
    p_k = positions[atom_j] + system.cell.minimum_image_displacement(positions[atom_k] - positions[atom_j])
    p_l = p_k + system.cell.minimum_image_displacement(positions[atom_l] - positions[atom_k])
    b0 = p_i - p_j
    b1 = p_k - p_j
    b2 = p_l - p_k
    norm_b1 = float(np.linalg.norm(b1))
    if norm_b1 == 0.0:
        raise ValueError("UFF torsion has zero central-bond distance")
    b1_unit = b1 / norm_b1
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    norm_v = float(np.linalg.norm(v))
    norm_w = float(np.linalg.norm(w))
    if norm_v == 0.0 or norm_w == 0.0:
        return 0.0
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_unit, v), w))
    return atan2(y, x)


def _out_of_plane_angle_rad(
    system: System,
    positions: np.ndarray,
    inversion: tuple[int, int, int, int],
) -> float:
    """Return the out-of-plane angle of the center atom from the neighbor plane."""
    center, atom_i, atom_j, atom_k = inversion
    p_center = positions[center]
    vector_ci = system.cell.minimum_image_displacement(positions[atom_i] - p_center)
    vector_cj = system.cell.minimum_image_displacement(positions[atom_j] - p_center)
    vector_ck = system.cell.minimum_image_displacement(positions[atom_k] - p_center)
    normal = np.cross(vector_cj, vector_ck)
    normal_norm = float(np.linalg.norm(normal))
    vector_norm = float(np.linalg.norm(vector_ci))
    if normal_norm == 0.0 or vector_norm == 0.0:
        return 0.0
    sine = float(np.dot(vector_ci, normal) / (vector_norm * normal_norm))
    sine = max(-1.0, min(1.0, sine))
    return asin_safe(sine)


def asin_safe(value: float) -> float:
    """Return asin(value) using atan2 to keep imports minimal and clipping explicit."""
    return atan2(value, sqrt(max(1.0 - value * value, 0.0)))


def _hybridization(atom_type: str) -> str:
    """Return a coarse hybridization label from a UFF atom type."""
    if "_1" in atom_type:
        return "sp"
    if "_2" in atom_type or "_R" in atom_type or atom_type.endswith("R"):
        return "sp2"
    if "_3" in atom_type or "3+" in atom_type:
        return "sp3"
    return "other"


def _sp2_torsion_barrier(params_j: UFFAtomParameters, params_k: UFFAtomParameters, bond_order: float) -> float:
    """Return a UFF-like sp2-sp2 torsion barrier in kcal/mol."""
    barrier = 5.0 * sqrt(max(params_j.vsp2_kcal_per_mol * params_k.vsp2_kcal_per_mol, 0.0)) * (1.0 + 4.18 * log(bond_order))
    return max(barrier, 0.0)


def _add_finite_difference_forces(
    forces: np.ndarray,
    positions: np.ndarray,
    atoms: tuple[int, ...],
    energy_function,
    step: float = FINITE_DIFFERENCE_ANGSTROM,
) -> None:
    """Add central finite-difference forces for a local bonded term."""
    for atom in sorted(set(atoms)):
        for axis in range(3):
            plus = np.array(positions, dtype=float, copy=True)
            minus = np.array(positions, dtype=float, copy=True)
            plus[atom, axis] += step
            minus[atom, axis] -= step
            derivative = (energy_function(plus) - energy_function(minus)) / (2.0 * step)
            forces[atom, axis] -= derivative
