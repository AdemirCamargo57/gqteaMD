"""Small built-in atomic mass table in unified atomic mass units."""

ATOMIC_MASSES = {
    "H": 1.00784,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Fe": 55.845,
    "Cu": 63.546,
    "Zn": 65.38,
    "Br": 79.904,
    "I": 126.90447,
}


def masses_for_symbols(symbols: list[str]) -> list[float]:
    """Return atomic masses for an ordered list of element symbols."""
    masses = []
    for symbol in symbols:
        try:
            masses.append(ATOMIC_MASSES[symbol])
        except KeyError as exc:
            raise ValueError(f"No built-in atomic mass for element {symbol!r}") from exc
    return masses
