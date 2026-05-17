#!/usr/bin/env python3
"""Convert HybridVPIC particle dumps to CSV columns x,y,z,ux,uy,uz,w.

The converter reads files produced by:

    dump_particles("ion_c", "particle/T.<step>/ion_c")
    dump_particles("ion_b", "particle/T.<step>/ion_b")

For a single-rank run the input file is usually:

    particle/T.0/ion_c.0.0

For MPI runs pass the base path without the rank suffix:

    python3 scripts/convert_particle_dump_to_csv.py particle/T.0/ion_c --out core_t0.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import struct
from pathlib import Path

import numpy as np


PARTICLE_DTYPE = np.dtype(
    [
        ("dx", "<f4"),
        ("dy", "<f4"),
        ("dz", "<f4"),
        ("i", "<i4"),
        ("ux", "<f4"),
        ("uy", "<f4"),
        ("uz", "<f4"),
        ("w", "<f4"),
    ]
)


def read_header(fp):
    # Binary compatibility boilerplate written by WRITE_HEADER_V0.
    compat_fmt = "<5Bhi fd".replace(" ", "")
    compat_size = struct.calcsize(compat_fmt)
    compat = struct.unpack(compat_fmt, fp.read(compat_size))
    if (compat[5] & 0xFFFF) != 0xCAFE or (compat[6] & 0xFFFFFFFF) != 0xDEADBEEF:
        raise ValueError("Unrecognized VPIC binary boilerplate")

    # V0 header: version,type,step,nx,ny,nz, dt,dx,dy,dz,x0,y0,z0,cvac,eps0,damp, rank,ndom,spid,spqm.
    header_fmt = "<6i10f3if"
    header = struct.unpack(header_fmt, fp.read(struct.calcsize(header_fmt)))
    keys = [
        "version",
        "dump_type",
        "step",
        "nx",
        "ny",
        "nz",
        "dt",
        "dx",
        "dy",
        "dz",
        "x0",
        "y0",
        "z0",
        "cvac",
        "eps0",
        "damp",
        "rank",
        "ndom",
        "spid",
        "spqm",
    ]
    meta = dict(zip(keys, header))

    elem_size, ndim = struct.unpack("<2i", fp.read(8))
    dims = struct.unpack(f"<{ndim}i", fp.read(4 * ndim))
    if elem_size != PARTICLE_DTYPE.itemsize:
        raise ValueError(f"Unexpected particle size {elem_size}, expected {PARTICLE_DTYPE.itemsize}")
    meta["particle_count"] = dims[0]
    return meta


def decode_positions(particles: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    i = particles["i"].astype(np.int64)
    nxg = nx + 2
    nyg = ny + 2
    ix = i % nxg
    iy = (i // nxg) % nyg
    iz = i // (nxg * nyg)
    x = meta["x0"] + ((ix - 1) + 0.5 * (particles["dx"] + 1.0)) * meta["dx"]
    y = meta["y0"] + ((iy - 1) + 0.5 * (particles["dy"] + 1.0)) * meta["dy"]
    z = meta["z0"] + ((iz - 1) + 0.5 * (particles["dz"] + 1.0)) * meta["dz"]
    return x, y, z


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    matches = sorted(Path(p) for p in glob.glob(str(path) + ".*"))
    if not matches:
        raise FileNotFoundError(f"No particle dump files found for {path}")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Particle dump file, or base path without rank suffix")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV file")
    parser.add_argument("--max-particles", type=int, default=0, help="Optional deterministic subsample")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["x", "y", "z", "ux", "uy", "uz", "w"])
        for fname in input_files(args.input):
            with fname.open("rb") as fp:
                meta = read_header(fp)
                particles = np.fromfile(fp, dtype=PARTICLE_DTYPE, count=int(meta["particle_count"]))
            if args.max_particles and particles.size > args.max_particles:
                stride = max(1, particles.size // args.max_particles)
                particles = particles[::stride][: args.max_particles]
            x, y, z = decode_positions(particles, meta)
            for row in zip(x, y, z, particles["ux"], particles["uy"], particles["uz"], particles["w"]):
                writer.writerow(row)
            written += particles.size
    print(f"Wrote {written} particles to {args.out}")


if __name__ == "__main__":
    main()
