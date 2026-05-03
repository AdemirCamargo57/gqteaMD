"""Tests for restart serialization and scheduled restart writes."""

import numpy as np
import pytest

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.forces.mock import HarmonicForceProvider
from gqteaMD.integrators.velocity_verlet import VelocityVerletIntegrator
from gqteaMD.io.restart import read_restart, write_restart
from gqteaMD.simulation.engine import Simulation


def _system_and_state():
    """Create a representative system/state pair for restart tests."""
    system = System(["H"], np.array([1.0]), Cell(10.0, 11.0, 12.0, periodic=(True, False, True)))
    state = State(
        positions=np.array([[1.0, 2.0, 3.0]]),
        velocities=np.array([[0.1, 0.2, 0.3]]),
        forces=np.array([[-1.0, -2.0, -3.0]]),
        image_flags=np.array([[1, 0, -1]]),
        energy=0.5,
        total_energy=0.75,
        step=7,
        time_fs=1.75,
    )
    return system, state


def test_restart_round_trip_preserves_system_and_state(tmp_path):
    """Writing and reading a restart should preserve all state fields."""
    system, state = _system_and_state()
    restart_path = tmp_path / "RESTART"

    write_restart(restart_path, system, state)
    loaded_system, loaded_state = read_restart(restart_path)

    assert loaded_system.symbols == system.symbols
    assert loaded_system.cell.lengths == pytest.approx(system.cell.lengths)
    assert loaded_system.cell.periodic == system.cell.periodic
    assert loaded_state.positions == pytest.approx(state.positions)
    assert loaded_state.velocities == pytest.approx(state.velocities)
    assert loaded_state.forces == pytest.approx(state.forces)
    assert loaded_state.image_flags == pytest.approx(state.image_flags)
    assert loaded_state.energy == pytest.approx(state.energy)
    assert loaded_state.total_energy == pytest.approx(state.total_energy)
    assert loaded_state.step == state.step
    assert loaded_state.time_fs == pytest.approx(state.time_fs)


def test_simulation_writes_restart_at_interval(tmp_path):
    """The simulation should write restarts only at the requested interval."""
    system = System(["H"], np.array([1.0]), Cell(10.0, 10.0, 10.0))
    state = State(
        positions=np.array([[1.0, 0.0, 0.0]]),
        velocities=np.zeros((1, 3)),
        forces=np.zeros((1, 3)),
    )
    simulation = Simulation(system, state, VelocityVerletIntegrator(0.1), HarmonicForceProvider())
    restart_path = tmp_path / "RESTART"

    final_state = simulation.run(steps=3, restart_path=restart_path, restart_interval=2)
    _loaded_system, loaded_state = read_restart(restart_path)

    assert final_state.step == 3
    assert loaded_state.step == 2


def test_simulation_rejects_invalid_restart_interval(tmp_path):
    """Invalid restart intervals should fail before integration begins."""
    system, state = _system_and_state()
    simulation = Simulation(system, state, VelocityVerletIntegrator(0.1), HarmonicForceProvider())

    with pytest.raises(ValueError, match="Restart interval must be positive"):
        simulation.run(steps=1, restart_path=tmp_path / "RESTART", restart_interval=0)
