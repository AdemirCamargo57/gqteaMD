"""Orthorhombic cell utilities for periodic molecular dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cell:
    """Orthorhombic simulation cell with lengths in angstrom."""

    a: float
    b: float
    c: float
    periodic: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        """Validate that all cell lengths are physically meaningful."""
        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            raise ValueError("Cell lengths a, b, and c must be positive")

    @property
    def lengths(self) -> np.ndarray:
        """Return the three orthorhombic side lengths as an array."""
        return np.array([self.a, self.b, self.c], dtype=float)

    @property
    def matrix(self) -> np.ndarray:
        """Return the diagonal cell matrix in angstrom."""
        return np.diag(self.lengths)

    @property
    def volume(self) -> float:
        """Return the cell volume in cubic angstrom."""
        return self.a * self.b * self.c

    def wrap(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return wrapped positions and image shifts for periodic dimensions."""

        wrapped = np.array(positions, dtype=float, copy=True)
        images = np.zeros_like(wrapped, dtype=int)
        lengths = self.lengths

        for axis, is_periodic in enumerate(self.periodic):
            if is_periodic:
                shifts = np.floor(wrapped[:, axis] / lengths[axis]).astype(int)
                wrapped[:, axis] -= shifts * lengths[axis]
                images[:, axis] = shifts

        return wrapped, images

    def minimum_image_displacement(self, displacement: np.ndarray) -> np.ndarray:
        """Fold displacement vectors through the nearest periodic image."""
        result = np.array(displacement, dtype=float, copy=True)
        lengths = self.lengths

        for axis, is_periodic in enumerate(self.periodic):
            if is_periodic:
                result[..., axis] -= lengths[axis] * np.rint(result[..., axis] / lengths[axis])

        return result
