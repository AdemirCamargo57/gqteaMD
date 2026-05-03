"""Command-line entry points and configuration builders for gqteaMD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from gqteaMD.config import load_config
from gqteaMD.core.cell import Cell
from gqteaMD.core.masses import masses_for_symbols
from gqteaMD.core.state import State, System
from gqteaMD.forces.classical import ClassicalForceProvider, HarmonicBond, LennardJonesType
from gqteaMD.forces.gaussian import GaussianForceProvider
from gqteaMD.forces.mock import HarmonicForceProvider
from gqteaMD.forces.uff import UFFForceProvider
from gqteaMD.integrators.velocity_verlet import VelocityVerletIntegrator
from gqteaMD.io.restart import read_restart
from gqteaMD.io.xyz import read_geometry, read_xyz
from gqteaMD.simulation.engine import Simulation


DEFAULT_GAUSSIAN_ROUTE = "# B3LYP/6-31G(d) SCF=Tight"


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and dispatch the requested subcommand."""
    parser = argparse.ArgumentParser(prog="gqteaMD")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run a molecular dynamics simulation")
    run_parser.add_argument("input", help="Path to a TOML configuration file or an XYZ geometry file")
    run_parser.add_argument("--time-fs", type=float, help="Total simulation time in femtoseconds for XYZ runs")
    run_parser.add_argument("--dt-fs", "--timestep-fs", dest="timestep_fs", type=float, help="MD timestep in femtoseconds")
    run_parser.add_argument("--steps", type=int, help="Number of MD integration steps. Overrides --time-fs when both are given")
    run_parser.add_argument("--cell", nargs=3, type=float, metavar=("A", "B", "C"), help="Orthorhombic cell lengths in angstrom")
    run_parser.add_argument("--box-size", type=float, default=20.0, help="Cubic box length in angstrom when --cell is omitted")
    run_parser.add_argument(
        "--force-provider",
        choices=("gaussian", "harmonic", "uff"),
        default="gaussian",
        help="Force provider for XYZ runs",
    )
    run_parser.add_argument("--gaussian-home", help=r"Gaussian installation folder, for example C:\G09")
    run_parser.add_argument("--gaussian-command", help="Gaussian executable or full path. Overrides --gaussian-home")
    run_parser.add_argument("--route", default=DEFAULT_GAUSSIAN_ROUTE, help="Gaussian route section")
    run_parser.add_argument("--charge", type=int, default=0, help="Molecular charge for Gaussian runs")
    run_parser.add_argument("--multiplicity", type=int, default=1, help="Spin multiplicity for Gaussian runs")
    run_parser.add_argument("--memory", help="Gaussian memory Link 0 value, for example 4GB or 4000MB")
    run_parser.add_argument("--chk", help="Gaussian checkpoint file. Defaults to a per-step file when not provided")
    run_parser.add_argument("--workdir", default="gaussian_steps", help="Folder for Gaussian input/output files")
    run_parser.add_argument("--trajectory", default="TRAJEC.xyz", help="Output XYZ trajectory path")
    run_parser.add_argument("--log", help="Output MD log path")
    run_parser.add_argument("--velocities", help="Optional output velocity frame path, usually ending in .vel")
    run_parser.add_argument("--forces", help="Optional output force frame path, usually ending in .for")
    run_parser.add_argument("--log-interval", type=int, default=1, help="Write output every N steps")
    run_parser.add_argument("--k", type=float, default=0.1, help="Harmonic force constant in eV/angstrom^2")
    run_parser.add_argument("--bond-detection-scale", type=float, default=1.2, help="UFF bond detection scale")
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return run_command(args)
    except Exception as exc:
        parser.exit(1, f"gqteaMD: error: {exc}\n")
    return 2


def run_command(args: argparse.Namespace | str) -> int:
    """Run either a TOML-configured or direct-XYZ simulation."""
    if isinstance(args, str):
        return run_config_command(args)

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".toml":
        return run_config_command(input_path)
    if input_path.suffix.lower() == ".xyz":
        return run_xyz_command(args)
    raise ValueError("input must be a .toml configuration file or a .xyz geometry file")


def run_config_command(config_path: str | Path) -> int:
    """Build and run a simulation from a TOML configuration file."""
    config_file = Path(config_path)
    config = load_config(config_file)
    base_dir = config_file.parent
    output_config = config.get("output", {})

    restart_config = config.get("restart", {})
    restart_path = _resolve(base_dir, restart_config.get("path", "RESTART"))
    if _restart_bool(restart_config, "resume_from_GEOMETRY", "resume_from_outputs"):
        system, state = _read_output_restart(config, base_dir, restart_path)
        symbols = system.symbols
    elif _restart_bool(restart_config, "resume_from_RESTART", "resume"):
        system, state = read_restart(restart_path)
        symbols = system.symbols
    else:
        xyz_path = _resolve(base_dir, config["input"]["xyz"])
        symbols, positions, _comment = read_xyz(xyz_path)
        cell_config = config["cell"]
        cell = Cell(float(cell_config["a"]), float(cell_config["b"]), float(cell_config["c"]))
        system = System(symbols=symbols, masses=np.array(masses_for_symbols(symbols)), cell=cell)

        velocities = np.zeros_like(positions)
        if "velocities" in config:
            velocities = np.array(config["velocities"], dtype=float)

        state = State(positions=positions, velocities=velocities, forces=np.zeros_like(positions))
    dynamics_config = config["dynamics"]
    integrator = VelocityVerletIntegrator(float(dynamics_config["timestep_fs"]))
    force_provider = _build_force_provider(config, base_dir, symbols)

    trajectory = output_config.get("trajectory", "TRAJEC.xyz")
    log = output_config.get("log", _default_log_name_from_config(config))
    log_interval = int(output_config.get("log_interval", 1))
    restart_interval = restart_config.get("interval")

    simulation = Simulation(system, state, integrator, force_provider)
    simulation.run(
        int(dynamics_config["steps"]),
        trajectory_path=_resolve(base_dir, trajectory),
        log_path=_resolve(base_dir, log),
        geometry_path=_resolve(base_dir, "GEOMETRY"),
        log_interval=log_interval,
        restart_path=restart_path,
        restart_interval=None if restart_interval is None else int(restart_interval),
    )
    return 0


def run_xyz_command(args: argparse.Namespace) -> int:
    """Build and run a simulation directly from an XYZ file and CLI options."""
    xyz_path = Path(args.input)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")
    if args.timestep_fs is None:
        raise ValueError("XYZ runs need --dt-fs, for example --dt-fs 0.5")

    steps = _steps_from_args(args.steps, args.time_fs, args.timestep_fs)
    base_dir = xyz_path.parent if xyz_path.parent != Path("") else Path.cwd()
    symbols, positions, _comment = read_xyz(xyz_path)

    if args.cell is None:
        cell = Cell(args.box_size, args.box_size, args.box_size)
    else:
        cell = Cell(args.cell[0], args.cell[1], args.cell[2])

    system = System(symbols=symbols, masses=np.array(masses_for_symbols(symbols)), cell=cell)
    state = State(positions=positions, velocities=np.zeros_like(positions), forces=np.zeros_like(positions))
    integrator = VelocityVerletIntegrator(args.timestep_fs)
    force_provider = _build_quick_force_provider(args, base_dir)

    simulation = Simulation(system, state, integrator, force_provider)
    simulation.run(
        steps,
        trajectory_path=_resolve(base_dir, args.trajectory),
        log_path=_resolve(base_dir, args.log or _default_log_name_for_xyz(xyz_path)),
        geometry_path=_resolve(base_dir, "GEOMETRY"),
        velocity_path=None if args.velocities is None else _resolve(base_dir, args.velocities),
        force_path=None if args.forces is None else _resolve(base_dir, args.forces),
        log_interval=args.log_interval,
    )
    return 0


def _steps_from_args(steps: int | None, time_fs: float | None, timestep_fs: float) -> int:
    """Resolve explicit step count or total time into an integer step count."""
    if steps is not None:
        if steps < 0:
            raise ValueError("--steps must be zero or greater")
        return steps
    if time_fs is None:
        raise ValueError("XYZ runs need --time-fs or --steps")
    if time_fs < 0:
        raise ValueError("--time-fs must be zero or greater")
    raw_steps = time_fs / timestep_fs
    rounded_steps = round(raw_steps)
    if not math.isclose(raw_steps, rounded_steps, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("--time-fs must be an exact multiple of --dt-fs, or pass --steps explicitly")
    return int(rounded_steps)


def _build_quick_force_provider(args: argparse.Namespace, base_dir: Path):
    """Create the force provider used by direct XYZ runs."""
    if args.force_provider == "harmonic":
        return HarmonicForceProvider(args.k)
    if args.force_provider == "uff":
        return UFFForceProvider(bond_detection_scale=args.bond_detection_scale)
    command = args.gaussian_command or _gaussian_command_from_home(args.gaussian_home) or "g16"
    return GaussianForceProvider(
        route=args.route,
        charge=args.charge,
        multiplicity=args.multiplicity,
        workdir=_resolve(base_dir, args.workdir),
        command=command,
        memory=args.memory,
        chk=args.chk,
    )


def _gaussian_command_from_home(gaussian_home: str | None) -> str | None:
    """Choose a Gaussian executable from a Gaussian installation directory."""
    if not gaussian_home:
        return None
    home = Path(gaussian_home)
    if not home.exists():
        raise FileNotFoundError(f"Gaussian folder not found: {home}")
    candidates = ("g09.exe", "g09", "g16.exe", "g16", "g03.exe", "g03")
    for name in candidates:
        candidate = home / name
        if candidate.exists():
            return str(candidate)
    return str(home / "g09.exe")


def _build_force_provider(config: dict, base_dir: Path, symbols: list[str]):
    """Create the force provider described by the TOML configuration."""
    provider_config = config["force_provider"]
    provider_type = provider_config["type"].lower()
    if provider_type == "harmonic":
        return HarmonicForceProvider(float(provider_config.get("k_ev_per_angstrom2", 1.0)))
    if provider_type == "classical":
        return _build_classical_force_provider(config, symbols)
    if provider_type == "uff":
        return _build_uff_force_provider(provider_config)
    if provider_type == "gaussian":
        nproc = provider_config.get("nproc", provider_config.get("nprocshared"))
        return GaussianForceProvider(
            route=provider_config["route"],
            charge=int(provider_config.get("charge", 0)),
            multiplicity=int(provider_config.get("multiplicity", 1)),
            workdir=_resolve(base_dir, provider_config.get("workdir", "gaussian_steps")),
            command=provider_config.get("command", "g16"),
            nproc=None if nproc is None else int(nproc),
            memory=provider_config.get("memory", provider_config.get("mem")),
            chk=provider_config.get("chk", provider_config.get("checkpoint")),
        )
    raise ValueError(f"Unknown force provider type: {provider_type}")


def _read_output_restart(config: dict, base_dir: Path, restart_path: str | Path) -> tuple[System, State]:
    """Create a restart state from GEOMETRY and RESTART output files."""
    geometry_path = _resolve(base_dir, "GEOMETRY")
    symbols, positions, velocities, forces, comment = read_geometry(geometry_path)
    restart_system, restart_state = read_restart(restart_path)
    if restart_system.symbols != symbols:
        raise ValueError("GEOMETRY and RESTART files must contain the same atoms in the same order")

    cell_config = config["cell"]
    cell = Cell(float(cell_config["a"]), float(cell_config["b"]), float(cell_config["c"]))
    system = System(symbols=symbols, masses=np.array(masses_for_symbols(symbols)), cell=cell)
    step, time_fs = _parse_output_restart_comment(comment)
    state = State(
        positions=positions,
        velocities=velocities,
        forces=forces,
        energy=restart_state.energy,
        total_energy=restart_state.total_energy,
        step=step,
        time_fs=time_fs,
    )
    return system, state


def _default_log_name_from_config(config: dict) -> str:
    """Build the default log filename from the configured XYZ input stem."""
    xyz = config.get("input", {}).get("xyz")
    if xyz is None:
        return "gqteaMD.log"
    return _default_log_name_for_xyz(Path(str(xyz)))


def _default_log_name_for_xyz(xyz_path: str | Path) -> str:
    """Build the default log filename from an XYZ path stem."""
    return f"{Path(xyz_path).stem}_gqteaMD.log"


def _parse_output_restart_comment(comment: str) -> tuple[int, float]:
    """Extract step and time metadata from a final Cartesian output comment."""
    values: dict[str, str] = {}
    for item in comment.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key] = value
    return int(values.get("step", 0)), float(values.get("time_fs", 0.0))


def _build_uff_force_provider(provider_config: dict) -> UFFForceProvider:
    """Create a UFF provider from TOML force-provider settings."""
    cutoff = provider_config.get("cutoff_angstrom")
    atom_types = provider_config.get("atom_types")
    charges = provider_config.get("charges")
    electrostatics = provider_config.get("electrostatics")
    bond_orders = {}
    bonds = None
    if "bonds" in provider_config:
        bonds = []
        for bond_config in provider_config.get("bonds", []):
            atoms = bond_config["atoms"]
            if len(atoms) != 2:
                raise ValueError("UFF bonds entries must list exactly two atom indices")
            pair = (int(atoms[0]), int(atoms[1]))
            bonds.append(pair)
            if "order" in bond_config:
                bond_orders[pair] = float(bond_config["order"])
    for bond_order_config in provider_config.get("bond_orders", []):
        atoms = bond_order_config["atoms"]
        if len(atoms) != 2:
            raise ValueError("UFF bond_orders entries must list exactly two atom indices")
        bond_orders[(int(atoms[0]), int(atoms[1]))] = float(bond_order_config["order"])
    angles = _parse_uff_interactions(provider_config, "angles", 3)
    torsions = _parse_uff_interactions(provider_config, "torsions", 4)
    inversions = _parse_uff_interactions(provider_config, "inversions", 4)
    electrostatics_value = None if electrostatics is None else _config_bool(electrostatics)
    return UFFForceProvider(
        bond_detection_scale=float(provider_config.get("bond_detection_scale", 1.2)),
        cutoff_angstrom=None if cutoff is None else float(cutoff),
        atom_types=None if atom_types is None else list(atom_types),
        bonds=bonds,
        bond_orders=None if not bond_orders else bond_orders,
        angles=angles,
        torsions=torsions,
        inversions=inversions,
        charges=None if charges is None else [float(charge) for charge in charges],
        electrostatics=electrostatics_value,
        nonbonded_exclusions=str(provider_config.get("nonbonded_exclusions", "exclude_12_13")),
        lj_14_scale=float(provider_config.get("lj_14_scale", 1.0)),
        electrostatic_14_scale=float(provider_config.get("electrostatic_14_scale", 1.0)),
        lj_cutoff_mode=str(provider_config.get("lj_cutoff_mode", "plain")),
        use_neighbor_list=_config_bool(provider_config.get("use_neighbor_list", True)),
        neighbor_skin_angstrom=float(provider_config.get("neighbor_skin_angstrom", 2.0)),
    )


def _parse_uff_interactions(provider_config: dict, key: str, atom_count: int) -> list[tuple[int, ...]] | None:
    """Parse optional UFF interaction arrays from TOML."""
    if key not in provider_config:
        return None
    interactions: list[tuple[int, ...]] = []
    for interaction_config in provider_config.get(key, []):
        atoms = interaction_config["atoms"]
        if len(atoms) != atom_count:
            raise ValueError(f"UFF {key} entries must list exactly {atom_count} atom indices")
        interactions.append(tuple(int(atom) for atom in atoms))
    return interactions


def _config_bool(value: object) -> bool:
    """Parse booleans from TOML-native bools or simple string values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _restart_bool(restart_config: dict, key: str, legacy_key: str) -> bool:
    """Read a restart boolean, preferring the current key over its legacy alias."""
    if key in restart_config:
        return _config_bool(restart_config[key])
    return _config_bool(restart_config.get(legacy_key, False))


def _build_classical_force_provider(config: dict, symbols: list[str]) -> ClassicalForceProvider:
    """Create a classical provider with harmonic bonds and Lennard-Jones terms."""
    provider_config = config["force_provider"]
    cutoff = provider_config.get("cutoff_angstrom")
    exclude_bonded = bool(provider_config.get("exclude_bonded", True))
    atom_types = list(provider_config.get("atom_types", symbols))

    bonds = []
    for bond_config in provider_config.get("bonds", []):
        atoms = bond_config["atoms"]
        if len(atoms) != 2:
            raise ValueError("Classical bonds must list exactly two atom indices")
        bonds.append(
            HarmonicBond(
                atom_i=int(atoms[0]),
                atom_j=int(atoms[1]),
                k_ev_per_angstrom2=float(bond_config["k_ev_per_angstrom2"]),
                r0_angstrom=float(bond_config["r0_angstrom"]),
            )
        )

    lj_config = provider_config.get("lennard_jones", {})
    lennard_jones = {
        atom_type: LennardJonesType(
            epsilon_ev=float(params["epsilon_ev"]),
            sigma_angstrom=float(params["sigma_angstrom"]),
        )
        for atom_type, params in lj_config.items()
    }

    return ClassicalForceProvider(
        atom_types=atom_types,
        bonds=bonds,
        lennard_jones=lennard_jones,
        cutoff_angstrom=None if cutoff is None else float(cutoff),
        exclude_bonded=exclude_bonded,
    )


def _resolve(base_dir: Path, path: str | Path) -> Path:
    """Resolve relative config paths against the config file directory."""
    path = Path(path)
    return path if path.is_absolute() else base_dir / path


if __name__ == "__main__":
    raise SystemExit(main())
