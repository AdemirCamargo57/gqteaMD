"""Velocity Verlet time integration for molecular dynamics."""

from __future__ import annotations

import numpy as np

from gqteaMD.core.state import State, System
from gqteaMD.core.units import EV_PER_ANGSTROM_PER_AMU_TO_ANGSTROM_PER_FS2
from gqteaMD.forces.base import ForceProvider


class VelocityVerletIntegrator:
    """Advance MD states with the velocity Verlet algorithm."""

    def __init__(self, timestep_fs: float):
        """Store and validate the integration timestep."""
        if timestep_fs <= 0:
            raise ValueError("Timestep must be positive")
        self.timestep_fs = float(timestep_fs)

    def step(self, system: System, state: State, force_provider: ForceProvider) -> State:
        """Advance the state by one timestep and refresh forces."""
        dt = self.timestep_fs
        masses = system.masses[:, None]
        acceleration = state.forces / masses * EV_PER_ANGSTROM_PER_AMU_TO_ANGSTROM_PER_FS2

        velocities_half = state.velocities + 0.5 * dt * acceleration
        positions_unwrapped = state.positions + dt * velocities_half
        positions_wrapped, image_delta = system.cell.wrap(positions_unwrapped)
        image_flags = np.asarray(state.image_flags, dtype=int) + image_delta

        trial_state = state.with_updates(
            positions=positions_wrapped,
            velocities=velocities_half,
            step=state.step + 1,
            time_fs=state.time_fs + dt,
            image_flags=image_flags,
        )
        result = force_provider.compute(system, trial_state)
        acceleration_new = result.forces / masses * EV_PER_ANGSTROM_PER_AMU_TO_ANGSTROM_PER_FS2
        velocities_new = velocities_half + 0.5 * dt * acceleration_new

        updated_state = trial_state.with_updates(
            velocities=velocities_new,
            forces=result.forces,
            energy=result.energy,
        )
        return updated_state.with_updates(total_energy=updated_state.kinetic_energy(system.masses) + result.energy)
