"""Tests for trajectory, log, velocity, force, and restart outputs."""

import numpy as np

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.forces.mock import HarmonicForceProvider
from gqteaMD.integrators.velocity_verlet import VelocityVerletIntegrator
from gqteaMD.io.xyz import read_xyz
from gqteaMD.simulation.engine import Simulation


def _simulation():
    """Create a one-atom harmonic simulation for output tests."""
    system = System(["H"], np.array([1.0]), Cell(10.0, 10.0, 10.0))
    state = State(
        positions=np.array([[1.0, 0.0, 0.0]]),
        velocities=np.zeros((1, 3)),
        forces=np.zeros((1, 3)),
    )
    return Simulation(system, state, VelocityVerletIntegrator(0.1), HarmonicForceProvider())


def test_existing_trajectory_and_log_are_appended(tmp_path):
    """Existing trajectory and log files should be appended safely."""
    trajectory_path = tmp_path / "trajectory.csv"
    log_path = tmp_path / "log.csv"
    trajectory_path.write_text("existing trajectory\n", encoding="utf-8")
    log_path.write_text(
        "    step          time_fs     potential_eV       kinetic_eV         total_eV    temperature_K            CPU_s\n"
        "      99       9.90000000       1.00000000       2.00000000       3.00000000       4.00000000       0.50000000\n",
        encoding="utf-8",
    )

    _simulation().run(steps=1, trajectory_path=trajectory_path, log_path=log_path)

    trajectory_text = trajectory_path.read_text(encoding="utf-8")
    log_text = log_path.read_text(encoding="utf-8")
    assert trajectory_text.startswith("existing trajectory\n")
    assert trajectory_text.count("\n1\n") == 2
    assert log_text.count(
        "    step          time_fs     potential_eV       kinetic_eV         total_eV    temperature_K            CPU_s"
    ) == 1
    assert (
        "      99       9.90000000       1.00000000       2.00000000       3.00000000       4.00000000       0.50000000"
        in log_text
    )
    assert "       0       0.00000000" in log_text
    assert "       1       0.10000000" in log_text
    assert all(len(line.split()) == 7 for line in log_text.splitlines()[1:])


def test_new_log_gets_header(tmp_path):
    """A new log file should start with the whitespace-separated header."""
    log_path = tmp_path / "log.csv"

    _simulation().run(steps=0, log_path=log_path)

    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.startswith(
        "    step          time_fs     potential_eV       kinetic_eV         total_eV    temperature_K            CPU_s\n"
    )
    assert "       0       0.00000000" in log_text
    assert len(log_text.splitlines()[1].split()) == 7


def test_velocity_and_force_files_are_written(tmp_path):
    """Velocity and force outputs should be written as frame files."""
    velocity_path = tmp_path / "trajectory.vel"
    force_path = tmp_path / "trajectory.for"

    _simulation().run(steps=1, velocity_path=velocity_path, force_path=force_path)

    velocity_text = velocity_path.read_text(encoding="utf-8")
    force_text = force_path.read_text(encoding="utf-8")
    assert "units=angstrom/fs" in velocity_text
    assert "units=eV/angstrom" in force_text
    assert velocity_text.splitlines().count("1") == 2
    assert force_text.splitlines().count("1") == 2


def test_trajectory_uses_unwrapped_positions(tmp_path):
    """Trajectory frames should use unwrapped positions plus velocities."""
    trajectory_path = tmp_path / "trajectory.xyz"
    system = System(["C", "H"], np.array([12.0, 1.0]), Cell(20.0, 20.0, 20.0))
    state = State(
        positions=np.array([[19.7, 0.1, 16.3], [0.1, 19.1, 16.3]]),
        velocities=np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
        forces=np.zeros((2, 3)),
        energy=0.0,
        image_flags=np.array([[-1, 0, -1], [0, -1, -1]]),
    )

    Simulation(system, state, VelocityVerletIntegrator(0.1), HarmonicForceProvider()).run(
        steps=0,
        trajectory_path=trajectory_path,
    )

    symbols, positions, _comment = read_xyz(trajectory_path)
    assert symbols == ["C", "H"]
    np.testing.assert_allclose(positions, [[-0.3, 0.1, -3.7], [0.1, -0.9, -3.7]])
    atom_lines = trajectory_path.read_text(encoding="utf-8").splitlines()[2:4]
    assert [len(line.split()) for line in atom_lines] == [7, 7]
    assert atom_lines[0].split()[4:] == ["0.01000000", "0.02000000", "0.03000000"]


def test_final_cartesian_outputs_replace_existing_files(tmp_path):
    """Final Cartesian outputs should contain only the last state frame."""
    position_path = tmp_path / "pos.xyz"
    velocity_path = tmp_path / "vel.xyz"
    force_path = tmp_path / "force.xyz"
    geometry_path = tmp_path / "GEOMETRY"
    position_path.write_text("old\n", encoding="utf-8")

    final_state = _simulation().run(
        steps=1,
        final_position_path=position_path,
        final_velocity_path=velocity_path,
        final_force_path=force_path,
        final_geometry_path=geometry_path,
    )

    for path in (position_path, velocity_path, force_path, geometry_path):
        assert path.read_text(encoding="utf-8").splitlines()[0] == "1"
    _symbols, positions, position_comment = read_xyz(position_path)
    assert position_comment.startswith(f"step={final_state.step}")
    assert positions.shape == (1, 3)
    assert len(geometry_path.read_text(encoding="utf-8").splitlines()[2].split()) == 10


def test_geometry_path_is_rewritten_each_step_with_forces(tmp_path):
    """GEOMETRY should contain the latest step as ten restartable atom columns."""
    geometry_path = tmp_path / "GEOMETRY"

    final_state = _simulation().run(steps=2, geometry_path=geometry_path)

    lines = geometry_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1"
    assert lines[1].startswith(f"step={final_state.step}")
    assert len(lines[2].split()) == 10
