"""XYZ geometry reading and trajectory frame writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray, str]:
    """Read symbols, Cartesian coordinates, and the comment from an XYZ file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ file must contain atom count and comment lines")
    try:
        natoms = int(lines[0].strip().split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("First XYZ line must contain the number of atoms") from exc

    if len(lines) < natoms + 2:
        raise ValueError(f"XYZ file declares {natoms} atoms but has too few atom lines")

    symbols: list[str] = []
    positions = np.zeros((natoms, 3), dtype=float)
    for index, line in enumerate(lines[2 : 2 + natoms]):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line {index + 3}: {line!r}")
        symbols.append(parts[0])
        positions[index] = [float(parts[1]), float(parts[2]), float(parts[3])]

    return symbols, positions, lines[1]


def read_geometry(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, str]:
    """Read a GEOMETRY file with positions, velocities, and forces."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("GEOMETRY file must contain atom count and comment lines")
    try:
        natoms = int(lines[0].strip().split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("First GEOMETRY line must contain the number of atoms") from exc

    if len(lines) < natoms + 2:
        raise ValueError(f"GEOMETRY file declares {natoms} atoms but has too few atom lines")

    symbols: list[str] = []
    positions = np.zeros((natoms, 3), dtype=float)
    velocities = np.zeros((natoms, 3), dtype=float)
    forces = np.zeros((natoms, 3), dtype=float)
    for index, line in enumerate(lines[2 : 2 + natoms]):
        parts = line.split()
        if len(parts) != 10:
            raise ValueError(f"Invalid GEOMETRY atom line {index + 3}: expected 10 columns, got {len(parts)}")
        symbols.append(parts[0])
        positions[index] = [float(parts[1]), float(parts[2]), float(parts[3])]
        velocities[index] = [float(parts[4]), float(parts[5]), float(parts[6])]
        forces[index] = [float(parts[7]), float(parts[8]), float(parts[9])]

    return symbols, positions, velocities, forces, lines[1]


def format_xyz_frame(
    symbols: list[str],
    positions: np.ndarray,
    comment: str = "",
    velocities: np.ndarray | None = None,
    forces: np.ndarray | None = None,
) -> str:
    """Format one XYZ frame as text."""
    lines = [str(len(symbols)), comment]
    if velocities is None and forces is None:
        for symbol, xyz in zip(symbols, positions, strict=True):
            lines.append(f"{symbol:>2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}")
    elif velocities is not None and forces is None:
        for symbol, xyz, velocity in zip(symbols, positions, velocities, strict=True):
            lines.append(
                f"{symbol:>2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}"
                f" {velocity[0]:16.8f} {velocity[1]:16.8f} {velocity[2]:16.8f}"
            )
    elif velocities is not None and forces is not None:
        for symbol, xyz, velocity, force in zip(symbols, positions, velocities, forces, strict=True):
            lines.append(
                f"{symbol:>2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}"
                f" {velocity[0]:16.8f} {velocity[1]:16.8f} {velocity[2]:16.8f}"
                f" {force[0]:16.8f} {force[1]:16.8f} {force[2]:16.8f}"
            )
    else:
        raise ValueError("Forces can only be written with matching velocities")
    return "\n".join(lines) + "\n"


def append_xyz_frame(
    path: str | Path,
    symbols: list[str],
    positions: np.ndarray,
    comment: str = "",
    velocities: np.ndarray | None = None,
    forces: np.ndarray | None = None,
) -> None:
    """Append one XYZ frame to an existing or new file."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(format_xyz_frame(symbols, positions, comment, velocities, forces))


def write_xyz_frame(
    path: str | Path,
    symbols: list[str],
    positions: np.ndarray,
    comment: str = "",
    velocities: np.ndarray | None = None,
    forces: np.ndarray | None = None,
) -> None:
    """Write one XYZ frame, replacing any existing file."""
    Path(path).write_text(format_xyz_frame(symbols, positions, comment, velocities, forces), encoding="utf-8")
