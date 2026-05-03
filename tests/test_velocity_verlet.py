"""Tests for the velocity Verlet integrator."""

import numpy as np

from gqteaMD.core.cell import Cell
from gqteaMD.core.state import State, System
from gqteaMD.forces.mock import HarmonicForceProvider
from gqteaMD.integrators.velocity_verlet import VelocityVerletIntegrator


def test_velocity_verlet_advances_state():
    """One integration step should advance time, step, positions, and forces."""
    system = System(["H"], np.array([1.0]), Cell(10.0, 10.0, 10.0))
    state = State(
        positions=np.array([[1.0, 0.0, 0.0]]),
        velocities=np.array([[0.0, 0.0, 0.0]]),
        forces=np.array([[-1.0, 0.0, 0.0]]),
        energy=0.5,
    )
    provider = HarmonicForceProvider(k_ev_per_angstrom2=1.0)
    next_state = VelocityVerletIntegrator(0.1).step(system, state, provider)
    assert next_state.step == 1
    assert next_state.time_fs == 0.1
    assert next_state.positions.shape == (1, 3)
    assert next_state.forces.shape == (1, 3)
