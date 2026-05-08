# gqteaMD

gqteaMD is a first-version molecular dynamics program built around velocity Verlet integration and modular force providers.

The initial implementation supports:

- XYZ input and XYZ trajectory output.
- Orthorhombic simulation cells with periodic wrapping.
- Velocity Verlet NVE propagation.
- Swappable force providers.
- A simple classical force provider with harmonic bonds and Lennard-Jones nonbonded interactions.
- A first UFF provider with automatic bond detection, atom typing, harmonic bonds, harmonic angles, and Lennard-Jones interactions.
- An optional xTB provider through the xtb-python ASE calculator.
- A Gaussian single-point force provider.
- A deterministic harmonic force provider for tests and dry runs.

## Install

From this folder:

```powershell
python -m pip install -e .[dev]
```

To use the optional xTB provider, install the extra dependencies:

```powershell
python -m pip install -e .[xtb]
```

On platforms where the `xtb` Python wheels are unavailable, install
`xtb-python` separately, for example with conda-forge.

See [USER_MANUAL.md](USER_MANUAL.md) for complete usage instructions.

## Minimal Run

```powershell
gqteaMD run examples/harmonic.toml
```

## Classical Run

The simple classical molecular dynamics provider supports harmonic bonds and
Lennard-Jones nonbonded interactions:

```powershell
gqteaMD run examples/classical_water.toml
```

It does not include standard force-field topology reading, angles, dihedrals,
Coulomb electrostatics, constraints, or force-field file support.

## UFF Run

The UFF provider assigns a small built-in Universal Force Field subset directly
from XYZ elements and coordinates. It currently supports H, C, N, O, F, P, S,
Cl, Br, and I with automatic bond detection, harmonic bonds, harmonic angles,
and Lennard-Jones nonbonded interactions. Directly bonded pairs and
angle-neighbor pairs are excluded from Lennard-Jones interactions:

```powershell
gqteaMD run examples/uff_water.toml
gqteaMD run examples/uff_methane.toml
```

You can also run UFF directly from an XYZ file:

```powershell
gqteaMD run examples/water.xyz --force-provider uff --time-fs 10 --dt-fs 0.5
```

This first milestone does not include torsions, inversion/out-of-plane terms,
aromatic/ring typing, bond-order correction, or electrostatics.

## xTB Run

The optional xTB provider uses the xtb-python ASE calculator to obtain single
point energies and forces during gqteaMD propagation. Install the optional
dependencies first:

```powershell
python -m pip install -e .[xtb]
```

On Windows or newer Python versions, install `xtb-python` separately if pip
cannot provide the compiled `xtb` package.

Then run the included example:

```powershell
gqteaMD run examples/xtb_water.toml
```

You can also use xTB directly from an XYZ file:

```powershell
gqteaMD run examples/water.xyz --force-provider xtb --xtb-method GFN2-xTB --time-fs 10 --dt-fs 0.5
```

xTB energies are used in eV and forces in eV/angstrom, matching gqteaMD's
internal units.

On Windows, where the Python `xtb` package may be unavailable, set the xTB
executable path in the TOML file:

```toml
[force_provider]
type = "xtb"
command = "C:/xTB/xtb-6.7.1/bin/xtb.exe"
method = "GFN2-xTB"
omp_num_threads = 4
```

## Restart Files

TOML runs can periodically save a restart file named `RESTART`:

```toml
[restart]
path = "RESTART"
interval = 10
```

To resume after a crash, keep the same TOML file and add:

```toml
[restart]
path = "RESTART"
interval = 10
resume_from_RESTART = true
```

The resumed run loads positions, velocities, forces, energy, step, time, image
flags, symbols, masses, and cell from the restart file. The `steps` value is
the number of additional steps to run after restart.

## Output Files

Existing trajectory and log files are appended instead of overwritten. Log
headers are only written when the log file is new or empty.

Trajectory atom lines contain seven columns: symbol, x, y, z, vx, vy, vz.

```toml
[output]
trajectory = "TRAJEC.xyz"
log = "log.csv"
log_interval = 1
```

TOML and direct XYZ runs also write `GEOMETRY` at every calculation step. Each
atom line has ten columns: symbol, x, y, z, vx, vy, vz, Fx, Fy, Fz.

To restart from `GEOMETRY` plus the energies in `RESTART`, set:

```toml
[restart]
resume_from_GEOMETRY = true
```

Direct XYZ runs can use the matching CLI flags:

```powershell
gqteaMD run water.xyz --time-fs 10 --dt-fs 0.5 --velocities trajectory.vel --forces trajectory.for
```

## Easy Gaussian Run From An XYZ File

If you are in a folder that contains `water.xyz` and Gaussian 09 is installed in
`C:\G09`, run:

```powershell
cd C:\AJCamargo\testeMD
gqteaMD run water.xyz --time-fs 10 --dt-fs 0.5 --gaussian-home C:\G09
```

This runs 20 MD steps because `10 fs / 0.5 fs = 20 steps`. By default, gqteaMD
uses a 20 angstrom cubic box and writes:

```text
TRAJEC.xyz
water_gqteaMD.log
GEOMETRY
gaussian_steps\
```

You can choose the Gaussian method/basis with `--route`:

```powershell
gqteaMD run water.xyz --time-fs 10 --dt-fs 0.5 --gaussian-home C:\G09 --route "# B3LYP/6-31G(d) SCF=Tight"
```

## Gaussian Run

Create a TOML file like:

```toml
[input]
xyz = "initial.xyz"

[cell]
a = 20.0
b = 20.0
c = 20.0

[dynamics]
timestep_fs = 0.5
steps = 100

[force_provider]
type = "gaussian"
command = "g16"
route = "# B3LYP/6-31G(d) Force NoSymm SCF=Tight"
charge = 0
multiplicity = 1
nproc = 4
workdir = "gaussian_steps"

[output]
trajectory = "TRAJEC.xyz"
log = "initial_gqteaMD.log"
```

Then run:

```powershell
gqteaMD run gaussian.toml
```

`nproc` is optional. When set, gqteaMD writes `%nprocshared=<nproc>` into each
Gaussian input file. Gaussian forces are converted from Hartree/Bohr to
eV/angstrom before integration.
