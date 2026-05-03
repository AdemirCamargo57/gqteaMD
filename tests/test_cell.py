"""Tests for periodic cell wrapping and minimum-image distances."""

import numpy as np

from gqteaMD.core.cell import Cell


def test_wrap_tracks_image_shifts():
    """Wrapping should return both wrapped coordinates and image shifts."""
    cell = Cell(10.0, 20.0, 30.0)
    positions = np.array([[11.0, -1.0, 31.0]])
    wrapped, images = cell.wrap(positions)
    np.testing.assert_allclose(wrapped, [[1.0, 19.0, 1.0]])
    np.testing.assert_array_equal(images, [[1, -1, 1]])


def test_minimum_image_displacement():
    """Displacements should fold to the nearest periodic image."""
    cell = Cell(10.0, 10.0, 10.0)
    displacement = np.array([6.0, -6.0, 4.0])
    np.testing.assert_allclose(cell.minimum_image_displacement(displacement), [-4.0, 4.0, 4.0])
