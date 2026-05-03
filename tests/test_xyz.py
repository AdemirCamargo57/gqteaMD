"""Tests for XYZ file parsing."""

from gqteaMD.io.xyz import read_xyz


def test_read_xyz(tmp_path):
    """XYZ reader should return symbols, coordinates, and comment text."""
    path = tmp_path / "h2.xyz"
    path.write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    symbols, positions, comment = read_xyz(path)
    assert symbols == ["H", "H"]
    assert comment == "h2"
    assert positions.shape == (2, 3)
