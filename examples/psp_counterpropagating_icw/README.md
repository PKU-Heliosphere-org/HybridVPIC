# PSP counterpropagating ICW HybridVPIC example

This example implements the lecture-note simulation case:

`第十一讲讲义 (2026).md -> 第二部分 -> 模拟案例-1：近日太阳风中双向传播离子回旋波（ICWs）与质子散射的动力学混合模拟`

It is a 1D kinetic-hybrid PIC setup for the PSP 2023-06-22 Alfvén-surface event. The domain is aligned with the background magnetic field:

```text
B0 = B0 zhat
Lz = 160 di
nz = 512
dz = 0.3125 di
dt = 0.05 Omega_ci^-1
periodic boundaries
```

The ions are loaded as two drifting bi-Maxwellian proton species:

- `ion_c`: core protons.
- `ion_b`: beam protons.

The density fractions, drift speeds, and parallel betas are derived from the lecture-note table:

```text
n_pc = 2530 cm^-3, v_pc = 234 km/s, T_pc_parallel = 20 eV, T_pc_perp = 52 eV
n_pb = 2169 cm^-3, v_pb = 420 km/s, T_pb_parallel = 48 eV, T_pb_perp = 107 eV
B0 = 780 nT, Vsw = 319.9 km/s
```

Following the note, `T_perp` is multiplied by 1.4 in the simulation. The resulting normalized parameters are saved in:

```text
config/psp_case1_parameters.json
```

## Files

- `psp_counterpropagating_icw.cxx`: HybridVPIC input deck.
- `HOMEWORK.md`: assignment text, required analysis, suggested grading rubric.
- `Makefile`: deck compilation entry, following the existing examples.
- `run_example.sh`: convenience build/run script for local or cluster use.
- `scripts/derive_parameters.py`: recompute normalized parameters from the lecture-note physical values.
- `scripts/visualize_counterpropagating_icw.py`: post-processing and figure-generation script.
- `scripts/convert_particle_dump_to_csv.py`: convert `particle/T.*/ion_c` and `particle/T.*/ion_b` dumps to CSV.
- `scripts/plot_particle_vdf.py`: helper for true VDF plots after particle snapshots are translated.
- `translate_psp_icw.f90`: Fortran/MPI translator from VPIC dumps to `.gda` arrays.
- `config/psp_case1_parameters.json`: physical and normalized parameters.

## Compile

Check the parameter conversion first:

```bash
python3 scripts/derive_parameters.py
```

Set `PROJECTDIR` in `Makefile` to the directory containing your HybridVPIC `vpic` deck compiler, then run:

```bash
make
```

Or use:

```bash
./run_example.sh build
```

The current repository examples use a NERSC path by default:

```text
/project/projectdirs/ntrain7/hybridVPIC/build-haswell/bin
```

Adjust it for your machine.

## Run

For a short single-rank test:

```bash
./run_example.sh run 1
```

For MPI:

```bash
./run_example.sh run 8
```

The production particle count in the paper-scale setup is very large. This deck uses `nppc = 2048` as a practical default for testing and teaching. Increase it toward `2^18` particles per cell for low-noise production runs.

## Visualize

Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

After translating field output into simple arrays, place files in `data/` with names such as:

```text
data/Bx.gda
data/By.gda
data/Bz.gda
```

Each `.gda` file is expected to contain `float32` data shaped as `(nt, nz)`.

For this example, compile and run the included translator with:

```bash
make translate
./run_example.sh translate
```

The translator reads:

```text
fields/T.*/fields.*.*
hydro/T.*/Hhydro_c.*.*
```

and writes:

```text
data/Bx.gda
data/By.gda
data/Bz.gda
data/aniso_core.gda
```

Then run:

```bash
python3 scripts/visualize_counterpropagating_icw.py --data data --out figures
```

If no translated data are available yet, the script can generate a synthetic demonstration data set that follows the lecture figures:

```bash
python3 scripts/visualize_counterpropagating_icw.py --demo --out figures
```

It writes:

- `fig2_wave_evolution.png`: wave energy, wave packet, z-t map, omega-k spectrum, k spectrum.
- `fig3_virtual_spacecraft.png`: virtual spacecraft waveform, spectrogram, polarization proxy.
- `fig4_resonant_scattering.png`: initial and relaxed core+beam VDF sketch with resonant bands.

The main plotting script is intentionally independent of a specific translator so it can be reused with either translated GDA arrays or demonstration arrays.

## Particle/VDF Diagnostics

The deck also writes sparse particle snapshots for `ion_c` and `ion_b` at:

```text
t = 0
t = 500 Omega_ci^-1
t = 700 Omega_ci^-1
```

under:

```text
particle/T.<step>/ion_c
particle/T.<step>/ion_b
```

These snapshots are large but allow a true Figure-4 style VDF. Convert the core and beam particle dumps to CSV columns `x,y,z,ux,uy,uz,w` first:

```bash
python3 scripts/convert_particle_dump_to_csv.py particle/T.0/ion_c --out data/particles/core_t0.csv --max-particles 200000
python3 scripts/convert_particle_dump_to_csv.py particle/T.0/ion_b --out data/particles/beam_t0.csv --max-particles 200000
```

The convenience wrapper does the same for a chosen snapshot step:

```bash
./run_example.sh particle-csv 0 200000
```

Then run:

```bash
python3 scripts/plot_particle_vdf.py --core-csv data/particles/core_t0.csv --beam-csv data/particles/beam_t0.csv --out figures/vdf_from_particles.png
```

Without CSV input, the script lists available particle snapshot directories and prints the required conversion format.
