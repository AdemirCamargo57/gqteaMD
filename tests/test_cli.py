"""Tests for CLI configuration helpers and provider builders."""

import argparse
from pathlib import Path

import pytest

from gqteaMD.cli import (
    _build_classical_force_provider,
    _build_force_provider,
    _build_quick_force_provider,
    _build_uff_force_provider,
    _build_xtb_force_provider,
    _default_log_name_for_xyz,
    _gaussian_command_from_home,
    _read_output_restart,
    _restart_bool,
    run_config_command,
    _steps_from_args,
)
from gqteaMD.config import load_config
from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.forces.classical import ClassicalForceProvider
from gqteaMD.forces.gaussian import GaussianForceProvider
from gqteaMD.forces.uff import UFFForceProvider
from gqteaMD.forces.xtb import XTBCommandForceProvider, XTBForceProvider
from gqteaMD.io.restart import write_restart
from gqteaMD.io.xyz import read_xyz


def test_steps_from_total_time_and_timestep():
    """Total time and timestep should resolve to an integer step count."""
    assert _steps_from_args(None, 10.0, 0.5) == 20


def test_steps_override_total_time():
    """An explicit step count should take precedence over total time."""
    assert _steps_from_args(7, 10.0, 0.5) == 7


def test_gaussian_home_prefers_g09(tmp_path):
    """Gaussian home lookup should prefer an existing g09 executable."""
    exe = tmp_path / "g09.exe"
    exe.write_text("", encoding="utf-8")

    assert _gaussian_command_from_home(str(tmp_path)) == str(exe)


def test_gaussian_home_falls_back_to_g09_exe_name(tmp_path):
    """Missing executables should fall back to the expected g09 path."""
    assert _gaussian_command_from_home(str(tmp_path)) == str(tmp_path / "g09.exe")


def test_gaussian_home_missing_folder_is_clear(tmp_path):
    """A missing Gaussian home should raise a clear error."""
    with pytest.raises(FileNotFoundError, match="Gaussian folder not found"):
        _gaussian_command_from_home(str(tmp_path / "missing"))


def test_default_log_name_uses_xyz_stem():
    """Default log names should be tied to the input XYZ molecule name."""
    assert _default_log_name_for_xyz(Path("water.xyz")) == "water_gqteaMD.log"


def test_restart_boolean_keys_prefer_new_names():
    """Restart parsing should accept renamed keys while preserving old aliases."""
    assert _restart_bool({"resume_from_RESTART": True, "resume": False}, "resume_from_RESTART", "resume") is True
    assert _restart_bool({"resume": True}, "resume_from_RESTART", "resume") is True
    assert _restart_bool({"resume_from_GEOMETRY": True, "resume_from_outputs": False}, "resume_from_GEOMETRY", "resume_from_outputs") is True
    assert _restart_bool({"resume_from_outputs": True}, "resume_from_GEOMETRY", "resume_from_outputs") is True


def test_build_gaussian_force_provider_uses_toml_nproc(tmp_path):
    """The Gaussian TOML builder should pass nproc to the provider."""
    provider = _build_force_provider(
        {
            "force_provider": {
                "type": "gaussian",
                "route": "# AM1 Force",
                "charge": 0,
                "multiplicity": 1,
                "workdir": "gaussian_steps",
                "command": "g09",
                "nproc": 4,
            }
        },
        Path(tmp_path),
        ["H"],
    )

    assert isinstance(provider, GaussianForceProvider)
    assert provider.nproc == 4


def test_toml_does_not_write_unused_cartesian_files(tmp_path):
    """TOML runs should not create legacy final position, velocity, or force XYZ files."""
    (tmp_path / "h.xyz").write_text("1\nh\nH 1 0 0\n", encoding="utf-8")
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[input]
xyz = "h.xyz"

[cell]
a = 10.0
b = 10.0
c = 10.0

[dynamics]
timestep_fs = 0.1
steps = 1

[force_provider]
type = "harmonic"
k_ev_per_angstrom2 = 1.0

[output]
trajectory = "traj.xyz"
positions = "pos.xyz"
velocities = "vel.xyz"
forces = "forces.xyz"
save_positions = true
save_velocities = true
save_forces = true
""",
        encoding="utf-8",
    )

    run_config_command(config_path)

    assert not (tmp_path / "pos.xyz").exists()
    assert not (tmp_path / "vel.xyz").exists()
    assert not (tmp_path / "forces.xyz").exists()
    assert (tmp_path / "GEOMETRY").exists()
    assert (tmp_path / "h_gqteaMD.log").exists()


def test_toml_writes_ten_column_geometry_each_run(tmp_path):
    """TOML runs should write restartable GEOMETRY with forces."""
    (tmp_path / "h.xyz").write_text("1\nh\nH 1 0 0\n", encoding="utf-8")
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[input]
xyz = "h.xyz"

[cell]
a = 10.0
b = 10.0
c = 10.0

[dynamics]
timestep_fs = 0.1
steps = 1

[force_provider]
type = "harmonic"
k_ev_per_angstrom2 = 1.0

[output]
trajectory = "traj.xyz"
log = "md.csv"
""",
        encoding="utf-8",
    )

    run_config_command(config_path)

    atom_line = (tmp_path / "GEOMETRY").read_text(encoding="utf-8").splitlines()[2]
    trajectory_atom_line = (tmp_path / "traj.xyz").read_text(encoding="utf-8").splitlines()[2]
    assert len(atom_line.split()) == 10
    assert len(trajectory_atom_line.split()) == 7


def test_read_output_restart_uses_geometry_and_restart_energies(tmp_path):
    """Output restart loading should rebuild arrays from GEOMETRY and energies from RESTART."""
    (tmp_path / "GEOMETRY").write_text(
        "1\nstep=7 time_fs=1.25000000 units=angstrom,angstrom/fs,eV/angstrom\n"
        "H 1 2 3 0.1 0.2 0.3 -1 -2 -3\n",
        encoding="utf-8",
    )
    restart_system = System(["H"], masses=[1.0], cell=Cell(10.0, 11.0, 12.0))
    restart_state = State(
        positions=[[9.0, 9.0, 9.0]],
        velocities=[[9.0, 9.0, 9.0]],
        forces=[[9.0, 9.0, 9.0]],
        energy=0.5,
        total_energy=0.75,
        step=7,
        time_fs=1.25,
    )
    restart_path = tmp_path / "RESTART"
    write_restart(restart_path, restart_system, restart_state)

    system, state = _read_output_restart(
        {
            "cell": {"a": 10.0, "b": 11.0, "c": 12.0},
        },
        Path(tmp_path),
        restart_path,
    )

    assert system.symbols == ["H"]
    assert state.step == 7
    assert state.time_fs == pytest.approx(1.25)
    assert state.positions.tolist() == [[1.0, 2.0, 3.0]]
    assert state.velocities.tolist() == [[0.1, 0.2, 0.3]]
    assert state.forces.tolist() == [[-1.0, -2.0, -3.0]]
    assert state.energy == pytest.approx(0.5)
    assert state.total_energy == pytest.approx(0.75)


def test_build_classical_force_provider_uses_symbols_as_default_atom_types():
    """Classical provider construction should default atom types to symbols."""
    provider = _build_classical_force_provider(
        {
            "force_provider": {
                "type": "classical",
                "bonds": [{"atoms": [0, 1], "k_ev_per_angstrom2": 45.0, "r0_angstrom": 0.9572}],
                "lennard_jones": {"O": {"epsilon_ev": 0.0067, "sigma_angstrom": 3.1507}},
            }
        },
        ["O", "H"],
    )

    assert isinstance(provider, ClassicalForceProvider)
    assert provider.atom_types == ["O", "H"]


def test_build_uff_force_provider():
    """UFF provider construction should read cutoff and atom type settings."""
    provider = _build_uff_force_provider(
        {
            "type": "uff",
            "bond_detection_scale": 1.15,
            "cutoff_angstrom": 8.0,
            "atom_types": ["O_3", "H_", "H_"],
            "charges": [-0.8, 0.4, 0.4],
            "bonds": [{"atoms": [0, 1], "order": 1.0}, {"atoms": [0, 2], "order": 1.0}],
            "bond_orders": [{"atoms": [0, 1], "order": 1.0}, {"atoms": [0, 2], "order": 1.0}],
            "angles": [{"atoms": [1, 0, 2]}],
            "torsions": [],
            "inversions": [],
            "electrostatics": True,
            "nonbonded_exclusions": "exclude_12",
            "lj_14_scale": 0.5,
            "electrostatic_14_scale": 0.833333,
            "lj_cutoff_mode": "shift",
            "use_neighbor_list": False,
            "neighbor_skin_angstrom": 1.25,
        }
    )

    assert isinstance(provider, UFFForceProvider)
    assert provider.bond_detection_scale == 1.15
    assert provider.cutoff_angstrom == 8.0
    assert provider.atom_types_override == ["O_3", "H_", "H_"]
    assert provider.bonds_override == [(0, 1), (0, 2)]
    assert provider.charges_override == [-0.8, 0.4, 0.4]
    assert provider.bond_orders_override == {(0, 1): 1.0, (0, 2): 1.0}
    assert provider.angles_override == [(1, 0, 2)]
    assert provider.torsions_override == []
    assert provider.inversions_override == []
    assert provider.electrostatics is True
    assert provider.nonbonded_exclusions == "exclude_12"
    assert provider.lj_14_scale == 0.5
    assert provider.electrostatic_14_scale == 0.833333
    assert provider.lj_cutoff_mode == "shift"
    assert provider.use_neighbor_list is False
    assert provider.neighbor_skin_angstrom == 1.25


def test_build_xtb_force_provider():
    """xTB provider construction should read TOML settings."""
    provider = _build_xtb_force_provider(
        {
            "type": "xtb",
            "method": "GFN1-xTB",
            "charge": -1,
            "multiplicity": 2,
            "accuracy": 0.75,
            "electronic_temperature": 400.0,
            "max_iterations": 123,
            "omp_num_threads": 6,
            "solvent": "water",
            "cache_api": False,
            "use_unwrapped_positions": False,
        }
    )

    assert isinstance(provider, XTBForceProvider)
    assert provider.method == "GFN1-xTB"
    assert provider.charge == -1.0
    assert provider.multiplicity == 2
    assert provider.accuracy == 0.75
    assert provider.electronic_temperature == 400.0
    assert provider.max_iterations == 123
    assert provider.omp_num_threads == 6
    assert provider.solvent == "water"
    assert provider.cache_api is False
    assert provider.use_unwrapped_positions is False


def test_build_force_provider_accepts_xtb_toml(tmp_path):
    """The general TOML builder should accept type = xtb."""
    provider = _build_force_provider(
        {
            "force_provider": {
                "type": "xtb",
                "method": "GFN2-xTB",
            }
        },
        Path(tmp_path),
        ["H"],
    )

    assert isinstance(provider, XTBForceProvider)
    assert provider.method == "GFN2-xTB"


def test_build_force_provider_accepts_xtb_command_toml(tmp_path):
    """TOML xTB provider should use the executable provider when command is set."""
    provider = _build_force_provider(
        {
            "force_provider": {
                "type": "xtb",
                "command": "C:/xTB/xtb-6.7.1/bin/xtb.exe",
                "method": "GFN2-xTB",
                "omp_num_threads": 4,
                "workdir": "xtb_steps",
            }
        },
        Path(tmp_path),
        ["H"],
    )

    assert isinstance(provider, XTBCommandForceProvider)
    assert provider.command == "C:/xTB/xtb-6.7.1/bin/xtb.exe"
    assert provider.omp_num_threads == 4
    assert provider.workdir == Path(tmp_path) / "xtb_steps"


def test_build_quick_force_provider_accepts_xtb(tmp_path):
    """Direct XYZ runs should accept --force-provider xtb settings."""
    args = argparse.Namespace(
        force_provider="xtb",
        xtb_method="GFN1-xTB",
        charge=1,
        multiplicity=2,
    )

    provider = _build_quick_force_provider(args, Path(tmp_path))

    assert isinstance(provider, XTBForceProvider)
    assert provider.method == "GFN1-xTB"
    assert provider.charge == 1.0
    assert provider.multiplicity == 2


def test_uff_example_configs_build_providers():
    """UFF example TOML files should parse into usable force providers."""
    examples = [
        Path("examples/uff_water.toml"),
        Path("examples/uff_methane.toml"),
        Path("examples/uff_water_charged.toml"),
        Path("examples/uff_ethene_explicit_topology.toml"),
    ]
    for example in examples:
        config = load_config(example)
        provider = _build_uff_force_provider(config["force_provider"])

        assert isinstance(provider, UFFForceProvider)
