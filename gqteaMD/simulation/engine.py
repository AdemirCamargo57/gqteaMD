"""Simulation loop that connects integrators, forces, and outputs."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from gqteaMD.core.state import State, System
from gqteaMD.forces.base import ForceProvider
from gqteaMD.integrators.velocity_verlet import VelocityVerletIntegrator
from gqteaMD.io.restart import write_restart
from gqteaMD.io.xyz import append_xyz_frame, write_xyz_frame


class Simulation:
    """Coordinate force evaluation, integration, output, and restarts."""

    def __init__(
        self,
        system: System,
        state: State,
        integrator: VelocityVerletIntegrator,
        force_provider: ForceProvider,
    ):
        """Store the objects that define one MD simulation."""
        self.system = system
        self.state = state
        self.integrator = integrator
        self.force_provider = force_provider

    def initialize_forces(self) -> None:
        """Compute forces for the initial state before integration begins."""
        result = self.force_provider.compute(self.system, self.state)
        self.state = self.state.with_updates(
            forces=result.forces,
            energy=result.energy,
            total_energy=self.state.kinetic_energy(self.system.masses) + result.energy,
        )

    def run(
        self,
        steps: int,
        trajectory_path: str | Path | None = None,
        log_path: str | Path | None = None,
        velocity_path: str | Path | None = None,
        force_path: str | Path | None = None,
        geometry_path: str | Path | None = None,
        final_position_path: str | Path | None = None,
        final_velocity_path: str | Path | None = None,
        final_force_path: str | Path | None = None,
        final_geometry_path: str | Path | None = None,
        log_interval: int = 1,
        restart_path: str | Path | None = None,
        restart_interval: int | None = None,
    ) -> State:
        """Run the requested number of MD steps and write configured outputs."""
        if steps < 0:
            raise ValueError("Number of steps must be non-negative")
        if log_interval <= 0:
            raise ValueError("Log interval must be positive")
        if restart_interval is not None and restart_interval <= 0:
            raise ValueError("Restart interval must be positive")
        calculation_time_s = 0.0
        if self.state.energy is None:
            calculation_started = perf_counter()
            self.initialize_forces()
            calculation_time_s = perf_counter() - calculation_started

        if trajectory_path is not None:
            self._write_frame(trajectory_path)
        if geometry_path is not None:
            self._write_geometry(geometry_path)
        if velocity_path is not None:
            self._write_velocities(velocity_path)
        if force_path is not None:
            self._write_forces(force_path)
        if log_path is not None:
            self._write_log_header_if_needed(log_path)
            self._write_log(log_path, calculation_time_s)
        self._write_restart_if_needed(restart_path, restart_interval)

        for _ in range(steps):
            calculation_started = perf_counter()
            self.state = self.integrator.step(self.system, self.state, self.force_provider)
            calculation_time_s = perf_counter() - calculation_started
            if geometry_path is not None:
                self._write_geometry(geometry_path)
            if self.state.step % log_interval == 0:
                if trajectory_path is not None:
                    self._write_frame(trajectory_path)
                if velocity_path is not None:
                    self._write_velocities(velocity_path)
                if force_path is not None:
                    self._write_forces(force_path)
                if log_path is not None:
                    self._write_log(log_path, calculation_time_s)
            self._write_restart_if_needed(restart_path, restart_interval)
        self._write_final_outputs(final_position_path, final_velocity_path, final_force_path, final_geometry_path)
        return self.state

    def _write_frame(self, trajectory_path: str | Path) -> None:
        """Append the current geometry and velocity frame to the trajectory file."""
        comment = f"step={self.state.step} time_fs={self.state.time_fs:.8f} energy_eV={self.state.energy}"
        append_xyz_frame(
            trajectory_path,
            self.system.symbols,
            self.state.unwrapped_positions(self.system.cell),
            comment,
            self.state.velocities,
        )

    def _write_velocities(self, velocity_path: str | Path) -> None:
        """Append the current velocities to an XYZ-like frame file."""
        comment = f"step={self.state.step} time_fs={self.state.time_fs:.8f} units=angstrom/fs"
        append_xyz_frame(velocity_path, self.system.symbols, self.state.velocities, comment)

    def _write_forces(self, force_path: str | Path) -> None:
        """Append the current forces to an XYZ-like frame file."""
        comment = f"step={self.state.step} time_fs={self.state.time_fs:.8f} units=eV/angstrom"
        append_xyz_frame(force_path, self.system.symbols, self.state.forces, comment)

    def _write_geometry(self, geometry_path: str | Path) -> None:
        """Write the latest restartable positions, velocities, and forces."""
        positions = self.state.unwrapped_positions(self.system.cell)
        comment = f"step={self.state.step} time_fs={self.state.time_fs:.8f} units=angstrom,angstrom/fs,eV/angstrom"
        write_xyz_frame(
            geometry_path,
            self.system.symbols,
            positions,
            comment,
            self.state.velocities,
            self.state.forces,
        )

    def _write_final_outputs(
        self,
        position_path: str | Path | None,
        velocity_path: str | Path | None,
        force_path: str | Path | None,
        geometry_path: str | Path | None,
    ) -> None:
        """Write final single-frame Cartesian outputs requested by TOML."""
        positions = self.state.unwrapped_positions(self.system.cell)
        if position_path is not None:
            comment = f"step={self.state.step} time_fs={self.state.time_fs:.8f} units=angstrom"
            write_xyz_frame(position_path, self.system.symbols, positions, comment)
        if velocity_path is not None:
            comment = f"step={self.state.step} time_fs={self.state.time_fs:.8f} units=angstrom/fs"
            write_xyz_frame(velocity_path, self.system.symbols, self.state.velocities, comment)
        if force_path is not None:
            comment = f"step={self.state.step} time_fs={self.state.time_fs:.8f} units=eV/angstrom"
            write_xyz_frame(force_path, self.system.symbols, self.state.forces, comment)
        if geometry_path is not None:
            self._write_geometry(geometry_path)

    def _write_log_header_if_needed(self, log_path: str | Path) -> None:
        """Create the whitespace-separated log header unless the log already has content."""
        path = Path(log_path)
        if path.exists() and path.stat().st_size > 0:
            return
        header = (
            f"{'step':>8s} {'time_fs':>16s} {'potential_eV':>16s} {'kinetic_eV':>16s}"
            f" {'total_eV':>16s} {'temperature_K':>16s} {'CPU_s':>16s}\n"
        )
        path.write_text(header, encoding="utf-8")

    def _write_log(self, log_path: str | Path, calculation_time_s: float) -> None:
        """Append one whitespace-separated energy, temperature, and timing row to the log."""
        kinetic = self.state.kinetic_energy(self.system.masses)
        potential = float(np.nan if self.state.energy is None else self.state.energy)
        total = kinetic + potential if self.state.total_energy is None else self.state.total_energy
        temperature = self.state.temperature(self.system.masses)
        line = (
            f"{self.state.step:8d} {self.state.time_fs:16.8f} {potential:16.8f}"
            f" {kinetic:16.8f} {total:16.8f} {temperature:16.8f} {calculation_time_s:16.8f}\n"
        )
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _write_restart_if_needed(
        self,
        restart_path: str | Path | None,
        restart_interval: int | None,
    ) -> None:
        """Write an atomic restart file when the configured interval matches."""
        if restart_path is None or restart_interval is None:
            return
        if self.state.step % restart_interval != 0:
            return
        target = Path(restart_path)
        temporary = target.with_name(f"{target.name}.tmp")
        write_restart(temporary, self.system, self.state)
        temporary.replace(target)
