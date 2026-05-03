"""Gaussian force-provider integration and output parsing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gqteaMD.core.state import State, System
from gqteaMD.core.units import HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, HARTREE_TO_EV
from gqteaMD.forces.base import ForceResult


SCF_DONE_RE = re.compile(r"SCF Done:\s+E\([^)]+\)\s+=\s+([-+]?\d+\.\d+)")
MEMORY_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)
PE_MACHINE_I386 = 0x014C
MAX_32BIT_GAUSSIAN_MEMORY_MB = 1800.0


@dataclass
class GaussianForceProvider:
    """Force provider that runs Gaussian single-point force calculations."""

    route: str
    charge: int
    multiplicity: int
    workdir: Path
    command: str = "g16"
    nproc: int | None = None
    memory: str | None = None
    chk: str | None = None

    def __post_init__(self) -> None:
        """Create the work directory and validate Gaussian options."""
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self.nproc is not None and self.nproc <= 0:
            raise ValueError("Gaussian nproc must be a positive integer")
        self.memory = (self.memory or "").strip() or None
        self.chk = (self.chk or "").strip() or None

    def compute(self, system: System, state: State) -> ForceResult:
        """Write one Gaussian input, run it, and parse the resulting forces."""
        command = self._resolve_command()
        self._validate_memory_for_command(command)
        stem = f"step_{state.step:06d}"
        input_path = self.workdir / f"{stem}.gjf"
        output_path = self.workdir / f"{stem}.log"

        input_path.write_text(self._render_input(system, state), encoding="utf-8")
        completed = self._run_gaussian(command, input_path, output_path)
        if completed.returncode != 0:
            raise RuntimeError(f"Gaussian command failed with exit code {completed.returncode}: {output_path}")

        return parse_gaussian_output(output_path)

    def _run_gaussian(self, command: str, input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
        """Run Gaussian with an isolated scratch directory for this step."""
        scratch_prefix = f"{input_path.stem}_scratch_"
        with tempfile.TemporaryDirectory(
            dir=self.workdir,
            prefix=scratch_prefix,
            ignore_cleanup_errors=True,
        ) as scratch_dir:
            env = os.environ.copy()
            env["GAUSS_SCRDIR"] = str(Path(scratch_dir).resolve())

            if _uses_windows_output_argument(command):
                return subprocess.run(
                    [command, str(input_path.resolve()), str(output_path.resolve())],
                    cwd=scratch_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                )

            with output_path.open("w", encoding="utf-8") as output:
                return subprocess.run(
                    [command, input_path.name],
                    cwd=self.workdir,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                )

    def _resolve_command(self) -> str:
        """Resolve the configured Gaussian command to an executable path."""
        command_path = Path(self.command)
        if command_path.parent != Path(".") or command_path.is_absolute():
            if not command_path.exists():
                raise FileNotFoundError(
                    f"Gaussian command not found: {self.command}. "
                    "Check the path in water.toml, or use --gaussian-home with the folder that contains g09.exe."
                )
            return str(command_path)

        resolved = shutil.which(self.command)
        if resolved is None:
            raise FileNotFoundError(
                f"Gaussian command not found on PATH: {self.command}. "
                "Use a full path such as C:/G09W/g09.exe, or run with --gaussian-home C:\\G09W."
            )
        return resolved

    def _validate_memory_for_command(self, command: str) -> None:
        """Fail early for memory requests that are unsafe for 32-bit Gaussian."""
        if self.memory is None or not _is_windows_32bit_executable(Path(command)):
            return
        memory_mb = _memory_to_megabytes(self.memory)
        if memory_mb is not None and memory_mb > MAX_32BIT_GAUSSIAN_MEMORY_MB:
            raise ValueError(
                "Gaussian memory is too high for the 32-bit Gaussian executable "
                f"({self.memory}). Use a value at or below 1800MB, or run a 64-bit Gaussian build."
            )

    def _render_input(self, system: System, state: State) -> str:
        """Render the Gaussian input text for the current MD state."""
        route = self.route
        if "force" not in route.lower():
            route = route.rstrip() + " Force"
        if "nosymm" not in route.lower() and "symmetry=none" not in route.lower():
            route = route.rstrip() + " NoSymm"

        chk_path = _resolve_link0_path(self.workdir, self.chk) if self.chk is not None else self.workdir / f"step_{state.step:06d}.chk"
        checkpoint_path = _gaussian_link0_path(chk_path)
        lines = [f"%chk={checkpoint_path}"]
        if self.memory is not None:
            lines.append(f"%Mem={self.memory}")
        if self.nproc is not None:
            lines.append(f"%nprocshared={self.nproc}")
        lines.extend(
            [
                route,
                "",
                f"gqteaMD step {state.step}",
                "",
                f"{self.charge} {self.multiplicity}",
            ]
        )
        positions = _gaussian_positions(system, state)
        for symbol, xyz in zip(system.symbols, positions, strict=True):
            lines.append(f"{symbol:>2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}")
        lines.extend(["", ""])
        return "\n".join(lines)


def _gaussian_positions(system: System, state: State) -> np.ndarray:
    """Return coordinates suitable for nonperiodic Gaussian calculations."""
    # Gaussian does not know about this MD cell, so write continuous molecular
    # coordinates rather than the wrapped positions stored for periodic MD.
    return state.unwrapped_positions(system.cell)


def _gaussian_link0_path(path: Path) -> str:
    """Format a path for Gaussian Link 0 directives."""
    return str(path.resolve()).replace("\\", "/")


def _resolve_link0_path(workdir: Path, value: str) -> Path:
    """Resolve a Link 0 file path relative to the Gaussian work directory."""
    path = Path(value)
    if path.is_absolute():
        return path
    return workdir / path


def _memory_to_megabytes(value: str) -> float | None:
    """Convert simple Gaussian memory strings to megabytes when possible."""
    match = MEMORY_RE.match(value)
    if match is None:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    if unit in ("", "M"):
        return amount
    if unit == "K":
        return amount / 1024.0
    if unit == "G":
        return amount * 1024.0
    if unit == "T":
        return amount * 1024.0 * 1024.0
    return None


def _is_windows_32bit_executable(path: Path) -> bool:
    """Return true when a Windows PE executable is marked as 32-bit x86."""
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"MZ":
                return False
            handle.seek(0x3C)
            pe_offset = struct.unpack("<I", handle.read(4))[0]
            handle.seek(pe_offset)
            if handle.read(4) != b"PE\0\0":
                return False
            machine = struct.unpack("<H", handle.read(2))[0]
            return machine == PE_MACHINE_I386
    except OSError:
        return False


def _uses_windows_output_argument(command: str) -> bool:
    """Detect Gaussian Windows launchers that accept an explicit log path."""
    if sys.platform != "win32":
        return False
    return Path(command).name.lower() in {"g03.exe", "g09.exe", "g09w.exe", "g16.exe"}


def parse_gaussian_output(path: str | Path) -> ForceResult:
    """Parse Gaussian energy and force tables into a ForceResult."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    energy_hartree = _parse_energy_hartree(text)
    forces_hartree_bohr = _parse_forces_hartree_bohr(text)
    return ForceResult(
        energy=energy_hartree * HARTREE_TO_EV,
        forces=forces_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        metadata={"provider": "gaussian", "source": str(path)},
    )


def _parse_energy_hartree(text: str) -> float:
    """Extract the last SCF energy from Gaussian output text."""
    matches = SCF_DONE_RE.findall(text)
    if not matches:
        raise ValueError("Could not find Gaussian SCF energy in output")
    return float(matches[-1])


def _parse_forces_hartree_bohr(text: str) -> np.ndarray:
    """Extract the last Cartesian force block from Gaussian output text."""
    lines = text.splitlines()
    blocks: list[np.ndarray] = []
    for index, line in enumerate(lines):
        if "Forces (Hartrees/Bohr)" not in line:
            continue
        values: list[list[float]] = []
        cursor = index + 3
        while cursor < len(lines):
            row = lines[cursor].strip()
            if not row or row.startswith("-"):
                if values:
                    break
                cursor += 1
                continue
            parts = row.split()
            if len(parts) >= 5 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                values.append([float(parts[2]), float(parts[3]), float(parts[4])])
            elif values:
                break
            cursor += 1
        if values:
            blocks.append(np.array(values, dtype=float))

    if not blocks:
        raise ValueError("Could not find Gaussian force table in output")
    return blocks[-1]
