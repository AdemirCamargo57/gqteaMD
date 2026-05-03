"""Universal Force Field parameter data and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass


KCAL_PER_MOL_TO_EV = 0.0433641153087705


@dataclass(frozen=True)
class UFFAtomParameters:
    """Subset of UFF atom parameters used by the implemented force terms."""

    r1: float
    theta0_degrees: float
    x1: float
    d1_kcal_per_mol: float
    z1: float
    zeta: float = 12.0
    vsp3_kcal_per_mol: float = 0.0
    vsp2_kcal_per_mol: float = 0.0
    electronegativity: float = 0.0
    coordination: int = 0


UFF_PARAMETERS: dict[str, UFFAtomParameters] = {
    "H_": UFFAtomParameters(0.354, 180.0, 2.886, 0.044, 0.712, 12.0, 0.0, 0.0, 4.5280, 0),
    "C_3": UFFAtomParameters(0.757, 109.47, 3.851, 0.105, 1.912, 12.73, 2.119, 2.0, 5.343, 4),
    "C_2": UFFAtomParameters(0.732, 120.0, 3.851, 0.105, 1.912, 12.73, 0.0, 2.0, 5.343, 3),
    "C_R": UFFAtomParameters(0.729, 120.0, 3.851, 0.105, 1.912, 12.73, 0.0, 2.0, 5.343, 3),
    "N_3": UFFAtomParameters(0.700, 106.7, 3.660, 0.069, 2.544, 13.407, 0.450, 2.0, 6.899, 3),
    "N_2": UFFAtomParameters(0.685, 111.2, 3.660, 0.069, 2.544, 13.407, 0.0, 2.0, 6.899, 2),
    "O_3": UFFAtomParameters(0.658, 104.51, 3.500, 0.060, 2.300, 14.085, 0.018, 2.0, 8.741, 2),
    "O_2": UFFAtomParameters(0.634, 120.0, 3.500, 0.060, 2.300, 14.085, 0.0, 2.0, 8.741, 1),
    "F_": UFFAtomParameters(0.668, 180.0, 3.364, 0.050, 1.735, 14.762, 0.0, 2.0, 10.874, 0),
    "P_3+3": UFFAtomParameters(1.101, 93.8, 4.147, 0.305, 2.863, 13.072, 2.400, 1.25, 5.463, 3),
    "S_3+2": UFFAtomParameters(1.064, 92.1, 4.035, 0.274, 2.703, 13.969, 0.484, 1.25, 6.928, 2),
    "Cl": UFFAtomParameters(1.044, 180.0, 3.947, 0.227, 2.348, 14.866, 0.0, 1.25, 8.564, 0),
    "Br": UFFAtomParameters(1.192, 180.0, 4.189, 0.251, 2.519, 15.0, 0.0, 0.7, 7.790, 0),
    "I_": UFFAtomParameters(1.382, 180.0, 4.500, 0.339, 2.739, 15.0, 0.0, 0.2, 6.822, 0),
}


COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def get_uff_parameters(atom_type: str) -> UFFAtomParameters:
    """Look up parameters for a UFF atom type."""
    try:
        return UFF_PARAMETERS[atom_type]
    except KeyError as exc:
        raise ValueError(f"UFF parameters are not available for atom type {atom_type!r}") from exc


def validate_uff_parameter_table(parameters: dict[str, UFFAtomParameters] | None = None) -> None:
    """Validate that every stored parameter has physically meaningful values."""
    table = UFF_PARAMETERS if parameters is None else parameters
    if not table:
        raise ValueError("UFF parameter table cannot be empty")
    for atom_type, params in table.items():
        if not atom_type:
            raise ValueError("UFF atom type names cannot be empty")
        if params.r1 <= 0:
            raise ValueError(f"UFF atom type {atom_type!r} has non-positive r1")
        if not 0.0 < params.theta0_degrees <= 180.0:
            raise ValueError(f"UFF atom type {atom_type!r} has invalid theta0_degrees")
        if params.x1 <= 0:
            raise ValueError(f"UFF atom type {atom_type!r} has non-positive x1")
        if params.d1_kcal_per_mol < 0:
            raise ValueError(f"UFF atom type {atom_type!r} has negative d1_kcal_per_mol")
        if params.z1 <= 0:
            raise ValueError(f"UFF atom type {atom_type!r} has non-positive z1")
        if params.zeta <= 0:
            raise ValueError(f"UFF atom type {atom_type!r} has non-positive zeta")
        if params.vsp3_kcal_per_mol < 0:
            raise ValueError(f"UFF atom type {atom_type!r} has negative vsp3_kcal_per_mol")
        if params.vsp2_kcal_per_mol < 0:
            raise ValueError(f"UFF atom type {atom_type!r} has negative vsp2_kcal_per_mol")
        if params.electronegativity < 0:
            raise ValueError(f"UFF atom type {atom_type!r} has negative electronegativity")
        if params.coordination < 0:
            raise ValueError(f"UFF atom type {atom_type!r} has negative coordination")


def supported_uff_atom_types() -> tuple[str, ...]:
    """Return the built-in UFF atom types in deterministic order."""
    return tuple(sorted(UFF_PARAMETERS))
