"""Tests for the optional xTB force provider."""

import subprocess

import numpy as np
import pytest

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.core.units import HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, HARTREE_TO_EV
from gqteaMD.forces.xtb import XTBCommandForceProvider, XTBForceProvider, parse_xtb_gradient


class FakeAtoms:
    """Small ASE Atoms stand-in for provider unit tests."""

    instances = []

    def __init__(self, symbols, positions, cell, pbc):
        self.symbols = symbols
        self.positions = np.asarray(positions, dtype=float)
        self.cell = np.asarray(cell, dtype=float)
        self.pbc = pbc
        self.calc = None
        self.initial_charges = None
        self.initial_magnetic_moments = None
        FakeAtoms.instances.append(self)

    def __len__(self):
        return len(self.symbols)

    def set_initial_charges(self, charges):
        self.initial_charges = np.asarray(charges, dtype=float)

    def set_initial_magnetic_moments(self, moments):
        self.initial_magnetic_moments = np.asarray(moments, dtype=float)

    def get_potential_energy(self):
        return self.calc.energy

    def get_forces(self):
        return self.calc.forces


class FakeXTB:
    """Small xtb-python ASE calculator stand-in."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.energy = -137.5
        self.forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        FakeXTB.instances.append(self)


def test_xtb_provider_returns_energy_and_forces(monkeypatch):
    """xTB provider should pass ASE energy and forces through unchanged."""
    FakeAtoms.instances.clear()
    FakeXTB.instances.clear()
    monkeypatch.setattr("gqteaMD.forces.xtb._load_ase_xtb", lambda: (FakeAtoms, FakeXTB))
    system = System(["H", "H"], masses=[1.0, 1.0], cell=Cell(10.0, 11.0, 12.0))
    state = State(
        positions=[[0.5, 0.0, 0.0], [9.8, 0.0, 0.0]],
        velocities=np.zeros((2, 3)),
        forces=np.zeros((2, 3)),
        image_flags=[[0, 0, 0], [-1, 0, 0]],
    )
    provider = XTBForceProvider(
        method="GFN1-xTB",
        charge=1.0,
        multiplicity=3,
        accuracy=0.5,
        electronic_temperature=500.0,
        max_iterations=99,
        solvent="water",
        cache_api=False,
    )

    result = provider.compute(system, state)

    assert result.energy == pytest.approx(-137.5)
    np.testing.assert_allclose(result.forces, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    assert result.metadata["provider"] == "xtb"
    assert result.metadata["method"] == "GFN1-xTB"
    atoms = FakeAtoms.instances[0]
    np.testing.assert_allclose(atoms.positions, [[0.5, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    np.testing.assert_allclose(atoms.cell, np.diag([10.0, 11.0, 12.0]))
    assert atoms.pbc == (True, True, True)
    np.testing.assert_allclose(atoms.initial_charges, [1.0, 0.0])
    np.testing.assert_allclose(atoms.initial_magnetic_moments, [2.0, 0.0])
    assert FakeXTB.instances[0].kwargs == {
        "method": "GFN1-xTB",
        "accuracy": 0.5,
        "electronic_temperature": 500.0,
        "max_iterations": 99,
        "solvent": "water",
        "cache_api": False,
    }


def test_xtb_provider_validates_force_shape(monkeypatch):
    """A malformed calculator force array should fail before integration."""
    class BadForcesXTB(FakeXTB):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.forces = np.zeros((1, 3))

    monkeypatch.setattr("gqteaMD.forces.xtb._load_ase_xtb", lambda: (FakeAtoms, BadForcesXTB))
    system = System(["H", "H"], masses=[1.0, 1.0], cell=Cell(10.0, 10.0, 10.0))
    state = State(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))

    with pytest.raises(ValueError, match="xTB returned force shape"):
        XTBForceProvider().compute(system, state)


def test_xtb_provider_rejects_invalid_settings():
    """Invalid xTB settings should raise clear configuration errors."""
    with pytest.raises(ValueError, match="multiplicity"):
        XTBForceProvider(multiplicity=0)
    with pytest.raises(ValueError, match="accuracy"):
        XTBForceProvider(accuracy=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        XTBForceProvider(max_iterations=0)


def test_parse_xtb_engrad_converts_gradient_to_forces(tmp_path):
    """xTB .engrad files should convert Hartree and Hartree/Bohr to gqteaMD units."""
    engrad = tmp_path / "step.engrad"
    engrad.write_text(
        """
#
# Number of atoms
#
 2
#
# The current total energy in Eh
#
 -1.5
#
# The current gradient in Eh/bohr
#
 0.1
 0.2
 0.3
 -0.1
 -0.2
 -0.3
""",
        encoding="utf-8",
    )

    energy, forces = parse_xtb_gradient(engrad, 2)

    assert energy == pytest.approx(-1.5 * HARTREE_TO_EV)
    np.testing.assert_allclose(
        forces,
        -np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    )


def test_xtb_command_provider_runs_executable_and_parses_engrad(tmp_path, monkeypatch):
    """Command provider should run xTB gradient mode and parse the generated engrad file."""
    command = tmp_path / "xtb.exe"
    command.write_text("", encoding="utf-8")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["cwd"] = kwargs["cwd"]
        Path = type(tmp_path)
        cwd = Path(kwargs["cwd"])
        (cwd / "step_000003.engrad").write_text(
            """
#
# Number of atoms
#
 1
#
# The current total energy in Eh
#
 -2.0
#
# The current gradient in Eh/bohr
#
 0.01
 0.02
 0.03
""",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    system = System(["H"], masses=[1.0], cell=Cell(10.0, 10.0, 10.0))
    state = State(
        positions=[[9.5, 0.0, 0.0]],
        velocities=np.zeros((1, 3)),
        forces=np.zeros((1, 3)),
        image_flags=[[-1, 0, 0]],
        step=3,
    )
    provider = XTBCommandForceProvider(
        command=str(command),
        method="GFN1-xTB",
        charge=0,
        multiplicity=2,
        workdir=tmp_path / "xtb_steps",
    )

    result = provider.compute(system, state)

    assert result.energy == pytest.approx(-2.0 * HARTREE_TO_EV)
    np.testing.assert_allclose(
        result.forces,
        -np.array([[0.01, 0.02, 0.03]]) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    )
    assert captured["args"][:3] == [str(command), "step_000003.xyz", "--grad"]
    assert "--gfn" in captured["args"]
    assert "1" in captured["args"]
    assert "--uhf" in captured["args"]
    assert (tmp_path / "xtb_steps" / "step_000003" / "step_000003.xyz").read_text(encoding="utf-8").splitlines()[2].split()[1:] == [
        "-0.5000000000",
        "0.0000000000",
        "0.0000000000",
    ]
