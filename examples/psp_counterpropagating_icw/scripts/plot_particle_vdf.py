#!/usr/bin/env python3
"""Plot VDF diagnostics from translated particle CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"{path} contains no particle rows")
    if data.ndim == 0:
        data = np.array([data])
    names = data.dtype.names or ()
    if not {"ux", "uy", "uz"}.issubset(names):
        data = np.loadtxt(path, delimiter=",", comments="#")
        if data.ndim != 2 or data.shape[1] < 6:
            raise ValueError(f"{path} must have columns x,y,z,ux,uy,uz[,w]")
        ux, uy, uz = data[:, 3], data[:, 4], data[:, 5]
        weights = data[:, 6] if data.shape[1] > 6 else np.ones_like(uz)
    else:
        ux, uy, uz = data["ux"], data["uy"], data["uz"]
        weights = data["w"] if "w" in names else np.ones_like(uz)
    vpar = uz
    vperp = np.sqrt(ux**2 + uy**2)
    return vpar, vperp, weights


def combined_histogram(
    core_csv: Path,
    beam_csv: Path,
    vpar_edges: np.ndarray,
    vperp_edges: np.ndarray,
) -> np.ndarray:
    hist = np.zeros((vpar_edges.size - 1, vperp_edges.size - 1))
    for path in (core_csv, beam_csv):
        vpar, vperp, weights = load_csv(path)
        part, _, _ = np.histogram2d(vpar, vperp, bins=(vpar_edges, vperp_edges), weights=weights)
        hist += part
    total = hist.sum()
    return hist / total if total > 0 else hist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particle-dir", type=Path, default=Path("particle"))
    parser.add_argument("--core-csv", type=Path)
    parser.add_argument("--beam-csv", type=Path)
    parser.add_argument("--core-initial-csv", type=Path)
    parser.add_argument("--beam-initial-csv", type=Path)
    parser.add_argument("--core-final-csv", type=Path)
    parser.add_argument("--beam-final-csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures/vdf_from_particles.png"))
    args = parser.parse_args()

    delta_inputs = [args.core_initial_csv, args.beam_initial_csv, args.core_final_csv, args.beam_final_csv]
    if all(path is not None for path in delta_inputs):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        vpar_edges = np.linspace(-1.3, 1.3, 161)
        vperp_edges = np.linspace(0.0, 1.4, 141)
        h0 = combined_histogram(args.core_initial_csv, args.beam_initial_csv, vpar_edges, vperp_edges)
        h1 = combined_histogram(args.core_final_csv, args.beam_final_csv, vpar_edges, vperp_edges)
        delta = h1 - h0
        vpar = 0.5 * (vpar_edges[:-1] + vpar_edges[1:])
        vperp = 0.5 * (vperp_edges[:-1] + vperp_edges[1:])
        vmax = max(abs(delta.min()), abs(delta.max()), 1e-16)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        im0 = axes[0].pcolormesh(vpar, vperp, np.log10(h0.T + 1e-12), shading="auto", cmap="viridis")
        axes[0].set_title("initial core+beam particle VDF")
        fig.colorbar(im0, ax=axes[0], label=r"$\log_{10} f$")
        im1 = axes[1].pcolormesh(vpar, vperp, delta.T, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("final minus initial particle VDF")
        fig.colorbar(im1, ax=axes[1], label=r"$\Delta f$")
        for ax in axes:
            ax.set_xlabel(r"$v_\parallel/v_A$")
            ax.set_ylabel(r"$v_\perp/v_A$")
            ax.axvspan(-0.99, -0.59, color="gray", alpha=0.18)
            ax.axvspan(0.71, 1.3, color="gray", alpha=0.18)
        fig.suptitle("Figure 4 style: particle VDF scattering")
        fig.savefig(args.out, dpi=220)
        print(f"Wrote {args.out}")
        return

    if args.core_csv and args.beam_csv:
        vc_par, vc_perp, vc_w = load_csv(args.core_csv)
        vb_par, vb_perp, vb_w = load_csv(args.beam_csv)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].hist2d(vc_par, vc_perp, bins=120, weights=vc_w, cmap="viridis")
        axes[0].set_title("core protons")
        axes[1].hist2d(vb_par, vb_perp, bins=120, weights=vb_w, cmap="magma")
        axes[1].set_title("beam protons")
        for ax in axes:
            ax.set_xlabel(r"$v_\parallel/v_A$")
            ax.set_ylabel(r"$v_\perp/v_A$")
        fig.savefig(args.out, dpi=220)
        print(f"Wrote {args.out}")
        return

    print("Particle snapshot directories:")
    for path in sorted(args.particle_dir.glob("T.*/*")):
        print(f"  {path}")
    print("")
    print("To make a true Figure-4 VDF, translate snapshots to CSV first:")
    print("  python3 scripts/convert_particle_dump_to_csv.py particle/T.0/ion_c --out data/particles/core_t0.csv")
    print("  python3 scripts/convert_particle_dump_to_csv.py particle/T.0/ion_b --out data/particles/beam_t0.csv")
    print("Then run, for example:")
    print("  python3 scripts/plot_particle_vdf.py --core-csv data/particles/core_t0.csv --beam-csv data/particles/beam_t0.csv")
    print("For a Figure-4 style delta-f plot, pass initial and final core/beam CSV files:")
    print("  python3 scripts/plot_particle_vdf.py --core-initial-csv data/particles/core_t0.csv --beam-initial-csv data/particles/beam_t0.csv --core-final-csv data/particles/core_t14000.csv --beam-final-csv data/particles/beam_t14000.csv --out figures/fig4_resonant_scattering.png")


if __name__ == "__main__":
    main()
