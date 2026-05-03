"""Tests for Gaussian input rendering, launching, and output parsing."""

from pathlib import Path
import subprocess

import numpy as np
import pytest

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.core.units import HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, HARTREE_TO_EV
from gqteaMD.forces.gaussian import GaussianForceProvider, _uses_windows_output_argument, parse_gaussian_output


def test_parse_gaussian_force_output(tmp_path):
    """Gaussian force output should parse into eV and eV/angstrom units."""
    path = tmp_path / "sample.log"
    path.write_text(
        """
 SCF Done:  E(RB3LYP) =  -76.123456789     A.U. after   10 cycles
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1          8          -0.049849321    0.000000000   -0.028780519
      2          1           0.046711997    0.000000000   -0.023346514
      3          1           0.003137324    0.000000000    0.052127033
 -------------------------------------------------------------------
""",
        encoding="utf-8",
    )
    result = parse_gaussian_output(path)
    assert result.energy == -76.123456789 * HARTREE_TO_EV
    assert result.forces.shape == (3, 3)
    np.testing.assert_allclose(
        result.forces[0, 0],
        -0.049849321 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    )


def test_gaussian_provider_reports_missing_command(tmp_path):
    """A missing configured Gaussian executable should raise clearly."""
    provider = GaussianForceProvider(
        route="# HF/3-21G",
        charge=0,
        multiplicity=1,
        workdir=tmp_path,
        command=str(tmp_path / "missing-g09.exe"),
    )

    with pytest.raises(FileNotFoundError, match="Gaussian command not found"):
        provider._resolve_command()


def test_gaussian_provider_rejects_invalid_nproc(tmp_path):
    """The provider should reject nonpositive Gaussian process counts."""
    with pytest.raises(ValueError, match="Gaussian nproc"):
        GaussianForceProvider(
            route="# HF/3-21G",
            charge=0,
            multiplicity=1,
            workdir=tmp_path,
            nproc=0,
        )


def test_gaussian_input_uses_unwrapped_positions(tmp_path):
    """Gaussian input should contain continuous coordinates and nproc."""
    provider = GaussianForceProvider(
        route="# AM1 Force",
        charge=0,
        multiplicity=1,
        workdir=tmp_path,
        nproc=4,
    )
    system = System(["C", "H"], np.array([12.0, 1.0]), Cell(20.0, 20.0, 20.0))
    state = State(
        positions=np.array([[19.7, 0.1, 16.3], [0.1, 19.1, 16.3]]),
        velocities=np.zeros((2, 3)),
        forces=np.zeros((2, 3)),
        step=2,
        image_flags=np.array([[-1, 0, -1], [0, -1, -1]]),
    )

    lines = provider._render_input(system, state).splitlines()
    coordinate_lines = [line.split() for line in lines if line.startswith((" C", " H"))]

    assert lines[0] == f"%chk={str((tmp_path / 'step_000002.chk').resolve()).replace('\\', '/')}"
    assert lines[1] == "%nprocshared=4"
    assert coordinate_lines == [
        ["C", "-0.30000000", "0.10000000", "-3.70000000"],
        ["H", "0.10000000", "-0.90000000", "-3.70000000"],
    ]


def test_gaussian_run_uses_isolated_scratch_directory(tmp_path, monkeypatch):
    """Each Gaussian run should receive a temporary scratch directory."""
    provider = GaussianForceProvider(
        route="# AM1 Force",
        charge=0,
        multiplicity=1,
        workdir=tmp_path,
    )
    input_path = tmp_path / "step_000007.gjf"
    output_path = tmp_path / "step_000007.log"
    input_path.write_text("", encoding="utf-8")
    captured: dict[str, str] = {}

    def fake_run(*args, **kwargs):
        """Capture subprocess environment for the scratch-directory assertion."""
        scratch_dir = Path(kwargs["env"]["GAUSS_SCRDIR"])
        assert scratch_dir.is_dir()
        assert scratch_dir.parent == tmp_path
        assert scratch_dir.name.startswith("step_000007_scratch_")
        captured["scratch_dir"] = str(scratch_dir)
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider._run_gaussian("g16", input_path, output_path)

    assert not Path(captured["scratch_dir"]).exists()


def test_windows_gaussian_run_uses_scratch_directory_as_cwd(tmp_path, monkeypatch):
    """Windows Gaussian launchers should run with scratch as the cwd."""
    provider = GaussianForceProvider(
        route="# AM1 Force",
        charge=0,
        multiplicity=1,
        workdir=tmp_path,
    )
    input_path = tmp_path / "step_000006.gjf"
    output_path = tmp_path / "step_000006.log"
    input_path.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr("gqteaMD.forces.gaussian._uses_windows_output_argument", lambda _command: True)

    def fake_run(args, **kwargs):
        """Capture subprocess arguments and cwd for the Windows launcher."""
        captured["args"] = args
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider._run_gaussian("C:/G09W/g09.exe", input_path, output_path)

    args = captured["args"]
    assert args == [
        "C:/G09W/g09.exe",
        str(input_path.resolve()),
        str(output_path.resolve()),
    ]
    cwd = Path(str(captured["cwd"]))
    assert cwd.parent == tmp_path
    assert cwd.name.startswith("step_000006_scratch_")


def test_windows_gaussian_executables_use_output_argument():
    """Known Windows Gaussian executables should use explicit log arguments."""
    assert _uses_windows_output_argument("C:/G09W/g09.exe")
