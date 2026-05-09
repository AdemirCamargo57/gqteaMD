"""xTB force providers built on xtb-python or the xTB executable."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from gqteaMD.core.units import HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, HARTREE_TO_EV
from gqteaMD.core.state import State, System
from gqteaMD.forces.base import ForceResult


XTB_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class XTBForceProvider:
    """Compute energies and forces with xTB through ASE."""

    method: str = "GFN2-xTB"
    charge: float | None = 0.0
    multiplicity: int | None = 1
    accuracy: float = 1.0
    electronic_temperature: float = 300.0
    max_iterations: int = 250
    omp_num_threads: int | None = None
    solvent: str = "none"
    cache_api: bool = True
    use_unwrapped_positions: bool = True

    def __post_init__(self) -> None:
        """Validate user-facing xTB settings."""
        if not self.method:
            raise ValueError("xTB method must not be empty")
        if self.multiplicity is not None and self.multiplicity < 1:
            raise ValueError("xTB multiplicity must be 1 or greater")
        if self.accuracy <= 0.0:
            raise ValueError("xTB accuracy must be positive")
        if self.electronic_temperature <= 0.0:
            raise ValueError("xTB electronic_temperature must be positive")
        if self.max_iterations < 1:
            raise ValueError("xTB max_iterations must be 1 or greater")
        if self.omp_num_threads is not None and self.omp_num_threads < 1:
            raise ValueError("xTB omp_num_threads must be 1 or greater")

    def compute(self, system: System, state: State) -> ForceResult:
        """Return xTB potential energy and Cartesian forces for the current state."""
        _set_xtb_thread_environment(self.omp_num_threads)
        Atoms, XTB = _load_ase_xtb()
        positions = state.unwrapped_positions(system.cell) if self.use_unwrapped_positions else state.positions
        atoms = Atoms(
            symbols=system.symbols,
            positions=positions,
            cell=system.cell.matrix,
            pbc=system.cell.periodic,
        )
        self._apply_charge_and_spin(atoms)
        atoms.calc = XTB(**self._calculator_kwargs())

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if forces.shape != state.positions.shape:
            raise ValueError(f"xTB returned force shape {forces.shape}, expected {state.positions.shape}")

        return ForceResult(
            energy=energy,
            forces=forces,
            metadata={
                "provider": "xtb",
                "method": self.method,
                "charge": self.charge,
                "multiplicity": self.multiplicity,
                "accuracy": self.accuracy,
                "electronic_temperature": self.electronic_temperature,
                "max_iterations": self.max_iterations,
                "omp_num_threads": self.omp_num_threads,
                "solvent": self.solvent,
                "cache_api": self.cache_api,
            },
        )

    def _calculator_kwargs(self) -> dict[str, object]:
        """Build keyword arguments for the xtb-python ASE calculator."""
        return {
            "method": self.method,
            "accuracy": self.accuracy,
            "electronic_temperature": self.electronic_temperature,
            "max_iterations": self.max_iterations,
            "solvent": self.solvent,
            "cache_api": self.cache_api,
        }

    def _apply_charge_and_spin(self, atoms: Any) -> None:
        """Store total charge and unpaired-electron count on the ASE Atoms object."""
        natoms = len(atoms)
        if natoms == 0:
            return
        if self.charge is not None and hasattr(atoms, "set_initial_charges"):
            charges = np.zeros(natoms, dtype=float)
            charges[0] = float(self.charge)
            atoms.set_initial_charges(charges)
        if self.multiplicity is not None and hasattr(atoms, "set_initial_magnetic_moments"):
            moments = np.zeros(natoms, dtype=float)
            moments[0] = float(self.multiplicity - 1)
            atoms.set_initial_magnetic_moments(moments)


def _load_ase_xtb() -> tuple[type[Any], type[Any]]:
    """Import optional ASE/xTB dependencies with a clear installation message."""
    try:
        from ase import Atoms
        from xtb.ase.calculator import XTB
    except ImportError as exc:
        raise ImportError(
            "The xTB force provider requires optional dependencies. "
            "Install them with `python -m pip install -e .[xtb]`, or install ASE "
            "and xtb-python in the active environment."
        ) from exc
    return Atoms, XTB


def _set_xtb_thread_environment(omp_num_threads: int | None) -> None:
    """Set xTB thread controls in the current process before xtb-python is called."""
    if omp_num_threads is not None:
        value = str(omp_num_threads)
        for name in XTB_THREAD_ENV_VARS:
            os.environ[name] = value


def _environment_with_omp_num_threads(omp_num_threads: int | None) -> dict[str, str] | None:
    """Build a subprocess environment with xTB thread controls set when configured."""
    if omp_num_threads is None:
        return None
    env = os.environ.copy()
    value = str(omp_num_threads)
    for name in XTB_THREAD_ENV_VARS:
        env[name] = value
    return env


@dataclass(frozen=True)
class XTBCommandForceProvider:
    """Compute energies and forces by running the xTB executable."""

    command: str = "xtb"
    method: str = "GFN2-xTB"
    charge: float | None = 0.0
    multiplicity: int | None = 1
    accuracy: float = 1.0
    electronic_temperature: float = 300.0
    omp_num_threads: int | None = None
    solvent: str = "none"
    use_unwrapped_positions: bool = True
    workdir: Path | str = "xtb_steps"

    def __post_init__(self) -> None:
        """Validate command-line xTB settings."""
        if not self.command:
            raise ValueError("xTB command must not be empty")
        if not self.method:
            raise ValueError("xTB method must not be empty")
        if self.multiplicity is not None and self.multiplicity < 1:
            raise ValueError("xTB multiplicity must be 1 or greater")
        if self.accuracy <= 0.0:
            raise ValueError("xTB accuracy must be positive")
        if self.electronic_temperature <= 0.0:
            raise ValueError("xTB electronic_temperature must be positive")
        if self.omp_num_threads is not None and self.omp_num_threads < 1:
            raise ValueError("xTB omp_num_threads must be 1 or greater")
        object.__setattr__(self, "workdir", Path(self.workdir))

    def compute(self, system: System, state: State) -> ForceResult:
        """Run xTB in gradient mode and convert energy/gradient to gqteaMD units."""
        command = self._resolve_command()
        self.workdir.mkdir(parents=True, exist_ok=True)
        scratch_path = self.workdir / f"step_{state.step:06d}"
        suffix = 1
        while scratch_path.exists():
            scratch_path = self.workdir / f"step_{state.step:06d}_{suffix}"
            suffix += 1
        scratch_path.mkdir()
        xyz_path = scratch_path / f"step_{state.step:06d}.xyz"
        self._write_xyz(xyz_path, system, state)
        args = self._build_command(command, xyz_path.name)
        env = _environment_with_omp_num_threads(self.omp_num_threads)
        completed = subprocess.run(
            args,
            cwd=scratch_path,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        (scratch_path / "xtb.stdout").write_text(completed.stdout, encoding="utf-8")
        (scratch_path / "xtb.stderr").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                "xTB command failed with exit code "
                f"{completed.returncode}. See {scratch_path / 'xtb.stdout'} and "
                f"{scratch_path / 'xtb.stderr'}."
            )
        engrad_path = scratch_path / f"{xyz_path.stem}.engrad"
        if not engrad_path.exists():
            engrad_path = scratch_path / "gradient"
        energy, forces = parse_xtb_gradient(engrad_path, system.natoms)
        if forces.shape != state.positions.shape:
            raise ValueError(f"xTB returned force shape {forces.shape}, expected {state.positions.shape}")
        return ForceResult(
            energy=energy,
            forces=forces,
            metadata={
                "provider": "xtb-command",
                "command": command,
                "method": self.method,
                "charge": self.charge,
                "multiplicity": self.multiplicity,
                "accuracy": self.accuracy,
                "electronic_temperature": self.electronic_temperature,
                "omp_num_threads": self.omp_num_threads,
                "solvent": self.solvent,
            },
        )

    def _resolve_command(self) -> str:
        """Return an executable path for the configured xTB command."""
        command_path = Path(self.command)
        if command_path.exists():
            return str(command_path)
        resolved = shutil.which(self.command)
        if resolved:
            return resolved
        raise FileNotFoundError(f"xTB command not found: {self.command}")

    def _build_command(self, command: str, xyz_name: str) -> list[str]:
        """Build the xTB command-line arguments for a gradient calculation."""
        args = [command, xyz_name, "--grad", "--norestart", "--acc", str(self.accuracy), "--etemp", str(self.electronic_temperature)]
        method_args = _method_args(self.method)
        args.extend(method_args)
        if self.charge is not None:
            args.extend(["--chrg", _format_integer_value(self.charge, "xTB charge")])
        if self.multiplicity is not None:
            args.extend(["--uhf", str(self.multiplicity - 1)])
        solvent = (self.solvent or "none").strip()
        if solvent and solvent.lower() != "none":
            args.extend(["--gbsa", solvent])
        return args

    def _write_xyz(self, path: Path, system: System, state: State) -> None:
        """Write an XYZ geometry for xTB."""
        positions = state.unwrapped_positions(system.cell) if self.use_unwrapped_positions else state.positions
        lines = [str(system.natoms), f"gqteaMD step={state.step} time_fs={state.time_fs:.8f}"]
        for symbol, xyz in zip(system.symbols, positions, strict=True):
            lines.append(f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_xtb_gradient(path: str | Path, natoms: int) -> tuple[float, np.ndarray]:
    """Parse xTB .engrad or Turbomole gradient output into eV and eV/angstrom."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("$grad"):
        return _parse_turbomole_gradient(text, natoms)
    return _parse_engrad(text, natoms)


def _parse_engrad(text: str, natoms: int) -> tuple[float, np.ndarray]:
    """Parse an xTB .engrad file."""
    tokens = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not tokens:
        raise ValueError("xTB engrad file is empty")
    parsed_natoms = int(tokens[0])
    if parsed_natoms != natoms:
        raise ValueError(f"xTB engrad atom count {parsed_natoms}, expected {natoms}")
    energy_hartree = float(tokens[1].replace("D", "E"))
    gradient_values = [float(item.replace("D", "E")) for item in tokens[2 : 2 + 3 * natoms]]
    if len(gradient_values) != 3 * natoms:
        raise ValueError("xTB engrad file does not contain a complete gradient")
    gradient = np.array(gradient_values, dtype=float).reshape((natoms, 3))
    return energy_hartree * HARTREE_TO_EV, -gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM


def _parse_turbomole_gradient(text: str, natoms: int) -> tuple[float, np.ndarray]:
    """Parse xTB's gradient file."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = next((line for line in lines if "SCF energy" in line), "")
    if not header:
        raise ValueError("Could not find xTB SCF energy in gradient file")
    energy_hartree = float(header.split("SCF energy =", 1)[1].split()[0].replace("D", "E"))
    gradient_lines = lines[2 + natoms : 2 + 2 * natoms]
    if len(gradient_lines) != natoms:
        raise ValueError("xTB gradient file does not contain a complete gradient")
    gradient = np.array(
        [[float(value.replace("D", "E")) for value in line.split()[:3]] for line in gradient_lines],
        dtype=float,
    )
    return energy_hartree * HARTREE_TO_EV, -gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM


def _method_args(method: str) -> list[str]:
    """Translate common gqteaMD xTB method names to executable arguments."""
    normalized = method.strip().lower().replace("_", "-")
    if normalized in {"gfn2-xtb", "gfn2", "2"}:
        return ["--gfn", "2"]
    if normalized in {"gfn1-xtb", "gfn1", "1"}:
        return ["--gfn", "1"]
    if normalized in {"gfn0-xtb", "gfn0", "0"}:
        return ["--gfn", "0"]
    if normalized in {"gfn-ff", "gfnff", "gff"}:
        return ["--gfnff"]
    raise ValueError(f"Unsupported xTB executable method: {method}")


def _format_integer_value(value: float, label: str) -> str:
    """Format a numeric setting that xTB requires as an integer."""
    parsed = float(value)
    rounded = round(parsed)
    if not np.isclose(parsed, rounded):
        raise ValueError(f"{label} must be an integer for the xTB executable")
    return str(int(rounded))
