"""Restart file serialization for resumable MD runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System


def write_restart(path: str | Path, system: System, state: State) -> None:
    """Write a restart snapshot containing system and state arrays."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    potential_energy = np.nan if state.energy is None else state.energy
    total_energy = state.total_energy
    if total_energy is None and state.energy is not None:
        total_energy = state.kinetic_energy(system.masses) + state.energy
    with target.open("wb") as handle:
        np.savez(
            handle,
            symbols=np.array(system.symbols, dtype=str),
            masses=system.masses,
            cell_lengths=system.cell.lengths,
            cell_periodic=np.array(system.cell.periodic, dtype=bool),
            positions=state.positions,
            velocities=state.velocities,
            forces=state.forces,
            image_flags=state.image_flags,
            energy=np.array(potential_energy, dtype=float),
            total_energy=np.array(np.nan if total_energy is None else total_energy, dtype=float),
            step=np.array(state.step, dtype=int),
            time_fs=np.array(state.time_fs, dtype=float),
        )


def read_restart(path: str | Path) -> tuple[System, State]:
    """Read a restart snapshot and reconstruct the system and state."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Restart file not found: {source}")

    with np.load(source, allow_pickle=False) as data:
        symbols = [str(symbol) for symbol in data["symbols"].tolist()]
        cell_lengths = np.asarray(data["cell_lengths"], dtype=float)
        cell_periodic = tuple(bool(value) for value in np.asarray(data["cell_periodic"], dtype=bool).tolist())
        system = System(
            symbols=symbols,
            masses=np.asarray(data["masses"], dtype=float),
            cell=Cell(float(cell_lengths[0]), float(cell_lengths[1]), float(cell_lengths[2]), cell_periodic),
        )
        energy_value = float(np.asarray(data["energy"], dtype=float))
        total_energy_value = float(np.asarray(data["total_energy"], dtype=float)) if "total_energy" in data else np.nan
        state = State(
            positions=np.asarray(data["positions"], dtype=float),
            velocities=np.asarray(data["velocities"], dtype=float),
            forces=np.asarray(data["forces"], dtype=float),
            image_flags=np.asarray(data["image_flags"], dtype=int),
            energy=None if np.isnan(energy_value) else energy_value,
            total_energy=None if np.isnan(total_energy_value) else total_energy_value,
            step=int(np.asarray(data["step"], dtype=int)),
            time_fs=float(np.asarray(data["time_fs"], dtype=float)),
        )

    return system, state
