#!/usr/bin/env python3
"""Visualize PSP counterpropagating ICWs from HybridVPIC output.

The script expects translated float32 arrays named Bx.gda, By.gda, Bz.gda
with shape (nt, nz). Use --demo to generate a synthetic data set with the
same analysis path before translated simulation output is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NZ = 512
LZ_DI = 160.0
DT_OMEGA = 1.0
VSW_VA = 319.9 / 248.19215828208905
CORE_U = (234.0 - 319.9) / 248.19215828208905
BEAM_U = (420.0 - 319.9) / 248.19215828208905
B0_Z = 1.0


def load_gda(path: Path, nz: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size % nz != 0:
        raise ValueError(f"{path} has {data.size} values, not divisible by nz={nz}")
    return data.reshape((-1, nz))


def demo_fields(nt: int = 700, nz: int = NZ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.linspace(-0.5 * LZ_DI, 0.5 * LZ_DI, nz, endpoint=False)
    t = np.arange(nt) * DT_OMEGA
    zz, tt = np.meshgrid(z, t)

    growth_sun = 1.0 / (1.0 + np.exp(-(tt - 260.0) / 45.0))
    growth_anti = 0.75 / (1.0 + np.exp(-(tt - 330.0) / 55.0))
    env1 = np.exp(-((zz + 0.08 * (tt - 330.0)) / 36.0) ** 2)
    env2 = np.exp(-((zz - 0.06 * (tt - 360.0)) / 42.0) ** 2)

    k1, w1 = -0.72, 0.58
    k2, w2 = 0.48, 0.42
    phase1 = k1 * zz - w1 * tt
    phase2 = k2 * zz - w2 * tt

    by = 0.055 * growth_sun * env1 * np.cos(phase1)
    bz = 0.055 * growth_sun * env1 * np.sin(phase1)
    by += 0.038 * growth_anti * env2 * np.cos(phase2)
    bz += -0.038 * growth_anti * env2 * np.sin(phase2)
    bx = 0.008 * (by**2 + bz**2) / max(np.max(by**2 + bz**2), 1e-12)

    rng = np.random.default_rng(7)
    by += rng.normal(scale=8e-4, size=by.shape)
    bz += rng.normal(scale=8e-4, size=bz.shape)
    return t, z, bx, by, bz


def load_fields(data_dir: Path, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx = load_gda(data_dir / "Bx.gda", nz)
    by = load_gda(data_dir / "By.gda", nz)
    bz = load_gda(data_dir / "Bz.gda", nz)
    nt = min(bx.shape[0], by.shape[0], bz.shape[0])
    bx, by, bz = bx[:nt], by[:nt], bz[:nt]
    t = np.arange(nt) * DT_OMEGA
    z = np.linspace(-0.5 * LZ_DI, 0.5 * LZ_DI, nz, endpoint=False)
    return t, z, bx, by, bz


def omega_k_power(field: np.ndarray, dt: float, dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_t = np.hanning(field.shape[0])[:, None]
    window_z = np.hanning(field.shape[1])[None, :]
    spec = np.fft.fftshift(np.fft.fft2((field - field.mean()) * window_t * window_z))
    omega = np.fft.fftshift(np.fft.fftfreq(field.shape[0], d=dt)) * 2 * np.pi
    k = np.fft.fftshift(np.fft.fftfreq(field.shape[1], d=dz)) * 2 * np.pi
    return omega, k, np.abs(spec) ** 2


def circular_spectra(bx: np.ndarray, by: np.ndarray, dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bl = np.fft.fftshift(np.fft.fft(bx + 1j * by, axis=1), axes=1)
    br = np.fft.fftshift(np.fft.fft(bx - 1j * by, axis=1), axes=1)
    k = np.fft.fftshift(np.fft.fftfreq(bx.shape[1], d=dz)) * 2 * np.pi
    return k, np.mean(np.abs(bl) ** 2, axis=0), np.mean(np.abs(br) ** 2, axis=0)


def plot_fig2(t: np.ndarray, z: np.ndarray, bx: np.ndarray, by: np.ndarray, bz: np.ndarray, out: Path) -> None:
    dz = z[1] - z[0]
    bx_perp = bx - np.mean(bx, axis=1, keepdims=True)
    by_perp = by - np.mean(by, axis=1, keepdims=True)
    bpar = bz - B0_Z
    bperp = bx_perp + 1j * by_perp

    db2 = bx_perp**2 + by_perp**2
    k, left_power, right_power = circular_spectra(bx_perp, by_perp, dz)
    sun_power = np.sum(left_power[k < 0]) if np.any(k < 0) else 0.0
    anti_power = np.sum(right_power[k > 0]) if np.any(k > 0) else 0.0

    spec_z = np.fft.fftshift(np.fft.fft(bperp, axis=1), axes=1) / bperp.shape[1]
    power_z = np.abs(spec_z) ** 2
    energy_total = np.mean(db2, axis=1)
    energy_sun = np.sum(power_z[:, k < 0], axis=1) if np.any(k < 0) else np.zeros_like(t)
    energy_anti = np.sum(power_z[:, k > 0], axis=1) if np.any(k > 0) else np.zeros_like(t)

    omega, kk, okw = omega_k_power(bperp, t[1] - t[0], dz)
    kspec = np.mean(power_z, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(t, energy_total + 1e-12, label="total")
    ax.semilogy(t, energy_sun + 1e-12, label="sunward proxy")
    ax.semilogy(t, energy_anti + 1e-12, label="antisunward proxy")
    ax.set_xlabel(r"$t\Omega_p$")
    ax.set_ylabel(r"$\langle\delta B_\perp^2\rangle$")
    ax.legend(frameon=False)

    axes[0, 1].plot(t, bx_perp[:, 0], label=r"$\delta B_x$")
    axes[0, 1].plot(t, by_perp[:, 0], label=r"$\delta B_y$", alpha=0.8)
    axes[0, 1].set_xlabel(r"$t\Omega_p$")
    axes[0, 1].set_ylabel("virtual point")
    axes[0, 1].legend(frameon=False)

    im = axes[0, 2].pcolormesh(t, z, by_perp.T, shading="auto", cmap="RdBu_r")
    axes[0, 2].set_xlabel(r"$t\Omega_p$")
    axes[0, 2].set_ylabel(r"$z/d_i$")
    axes[0, 2].set_title(r"$\delta B_y$ z-t")
    fig.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].pcolormesh(kk, omega, np.log10(okw + 1e-20), shading="auto", cmap="magma")
    axes[1, 0].set_xlim(-1.2, 1.2)
    axes[1, 0].set_ylim(-1.2, 1.2)
    axes[1, 0].set_xlabel(r"$k_\parallel d_i$")
    axes[1, 0].set_ylabel(r"$\omega/\Omega_p$")
    axes[1, 0].set_title(r"$\omega-k$ power")
    fig.colorbar(im, ax=axes[1, 0])

    axes[1, 1].plot(k, kspec / max(kspec.max(), 1e-30), color="black")
    axes[1, 1].set_xlim(-1.2, 1.2)
    axes[1, 1].set_xlabel(r"$k_\parallel d_i$")
    axes[1, 1].set_ylabel("normalized power")

    axes[1, 2].plot(k, left_power / max(left_power.max(), 1e-30), label=r"$B_x+iB_y$")
    axes[1, 2].plot(k, right_power / max(right_power.max(), 1e-30), label=r"$B_x-iB_y$")
    axes[1, 2].set_xlim(-1.2, 1.2)
    axes[1, 2].set_xlabel(r"$k_\parallel d_i$")
    axes[1, 2].set_ylabel("circular power")
    axes[1, 2].legend(frameon=False)
    axes[1, 2].text(0.04, 0.08, f"P(k<0)={sun_power:.2e}\nP(k>0)={anti_power:.2e}", transform=axes[1, 2].transAxes)

    fig.suptitle("Figure 2 style: counterpropagating ICW growth and spectra")
    fig.savefig(out / "fig2_wave_evolution.png", dpi=220)
    plt.close(fig)


def plot_fig3(t: np.ndarray, z: np.ndarray, bx: np.ndarray, by: np.ndarray, out: Path) -> None:
    dz = z[1] - z[0]
    bx_perp = bx - np.mean(bx, axis=1, keepdims=True)
    by_perp = by - np.mean(by, axis=1, keepdims=True)
    z_sc = (-VSW_VA * t) % (z[-1] - z[0])
    z_sc += z[0]
    idx = np.mod(np.rint((z_sc - z[0]) / dz).astype(int), z.size)
    sig_x = bx_perp[np.arange(t.size), idx]
    sig_y = by_perp[np.arange(t.size), idx]
    signal = sig_x + 1j * sig_y

    nper = min(128, max(32, t.size // 8))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(t, sig_x, label=r"$\delta B_x$")
    axes[0].plot(t, sig_y, label=r"$\delta B_y$", alpha=0.8)
    axes[0].set_ylabel("waveform")
    axes[0].legend(frameon=False)

    pxx, freq, bins, im = axes[1].specgram(np.real(signal), NFFT=nper, Fs=1.0 / (t[1] - t[0]), noverlap=nper // 2, cmap="magma")
    axes[1].set_ylabel(r"$f/\Omega_p$")
    axes[1].set_title("virtual spacecraft spectrogram")
    fig.colorbar(im, ax=axes[1], label="power")

    analytic_phase = np.unwrap(np.angle(signal + 1e-12))
    helicity_proxy = np.gradient(analytic_phase, t)
    axes[2].plot(t, np.tanh(helicity_proxy), color="purple")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xlabel(r"$t\Omega_p$")
    axes[2].set_ylabel("polarization proxy")

    fig.suptitle("Figure 3 style: virtual spacecraft observation")
    fig.savefig(out / "fig3_virtual_spacecraft.png", dpi=220)
    plt.close(fig)


def bi_maxwellian(vpar: np.ndarray, vperp: np.ndarray, density: float, drift: float, vth_par: float, vth_perp: float) -> np.ndarray:
    return density * np.exp(-((vpar - drift) / vth_par) ** 2 - (vperp / vth_perp) ** 2) / (np.pi ** 1.5 * vth_par * vth_perp**2)


def plot_fig4_schematic(out: Path) -> None:
    vpar = np.linspace(-1.3, 1.3, 320)
    vperp = np.linspace(0.0, 1.2, 240)
    vvpar, vvperp = np.meshgrid(vpar, vperp)

    ncore, nbeam = 0.5384, 0.4616
    beta_c, beta_b = 0.0334897, 0.0689068
    tcpar, tbpar = beta_c / (2 * ncore), beta_b / (2 * nbeam)
    vcpar, vbpar = np.sqrt(tcpar), np.sqrt(tbpar)
    vcperp, vbperp = vcpar * np.sqrt(3.64), vbpar * np.sqrt(3.1208333333)

    f0 = bi_maxwellian(vvpar, vvperp, ncore, CORE_U, vcpar, vcperp)
    f0 += bi_maxwellian(vvpar, vvperp, nbeam, BEAM_U, vbpar, vbperp)
    f1 = bi_maxwellian(vvpar, vvperp, ncore, CORE_U + 0.08, vcpar * 1.18, vcperp * 0.72)
    f1 += bi_maxwellian(vvpar, vvperp, nbeam, BEAM_U - 0.08, vbpar * 1.15, vbperp * 0.75)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].contourf(vpar, vperp, np.log10(f0 + 1e-8), levels=32, cmap="viridis")
    axes[0].axvline(CORE_U, color="white", lw=1, ls="--")
    axes[0].axvline(BEAM_U, color="white", lw=1, ls=":")
    axes[0].axvspan(-0.55, -0.25, color="gray", alpha=0.25)
    axes[0].axvspan(0.20, 0.55, color="gray", alpha=0.25)
    axes[0].set_title("initial core+beam VDF")
    axes[0].set_xlabel(r"$v_\parallel/v_A$")
    axes[0].set_ylabel(r"$v_\perp/v_A$")
    fig.colorbar(im0, ax=axes[0], label=r"$\log_{10} f$")

    im1 = axes[1].contourf(vpar, vperp, f1 - f0, levels=33, cmap="RdBu_r")
    theta = np.linspace(0, np.pi, 200)
    for center, radius in [(-0.42, 0.55), (0.34, 0.48)]:
        axes[1].plot(center + radius * np.cos(theta), radius * np.sin(theta), color="black", lw=1.2)
    axes[1].set_title("relaxed minus initial VDF")
    axes[1].set_xlabel(r"$v_\parallel/v_A$")
    axes[1].set_ylabel(r"$v_\perp/v_A$")
    fig.colorbar(im1, ax=axes[1], label=r"$\Delta f$")

    fig.suptitle("Figure 4 style: schematic resonant proton scattering")
    fig.savefig(out / "fig4_resonant_scattering.png", dpi=220)
    plt.close(fig)


def load_particle_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"{path} contains no particle rows")
    if data.ndim == 0:
        data = np.array([data])
    names = data.dtype.names or ()
    if not {"ux", "uy", "uz"}.issubset(names):
        raw = np.loadtxt(path, delimiter=",", comments="#")
        if raw.ndim != 2 or raw.shape[1] < 6:
            raise ValueError(f"{path} must have columns x,y,z,ux,uy,uz[,w]")
        ux, uy, uz = raw[:, 3], raw[:, 4], raw[:, 5]
        weights = raw[:, 6] if raw.shape[1] > 6 else np.ones_like(uz)
    else:
        ux, uy, uz = data["ux"], data["uy"], data["uz"]
        weights = data["w"] if "w" in names else np.ones_like(uz)
    return uz, np.sqrt(ux**2 + uy**2), weights


def combined_histogram(
    core_csv: Path,
    beam_csv: Path,
    vpar_edges: np.ndarray,
    vperp_edges: np.ndarray,
) -> np.ndarray:
    hist = np.zeros((vpar_edges.size - 1, vperp_edges.size - 1))
    for csv in (core_csv, beam_csv):
        vpar, vperp, weights = load_particle_csv(csv)
        part, _, _ = np.histogram2d(vpar, vperp, bins=(vpar_edges, vperp_edges), weights=weights)
        hist += part
    total = hist.sum()
    return hist / total if total > 0 else hist


def plot_fig4_from_particles(
    out: Path,
    core_initial_csv: Path,
    beam_initial_csv: Path,
    core_final_csv: Path,
    beam_final_csv: Path,
) -> None:
    vpar_edges = np.linspace(-1.3, 1.3, 161)
    vperp_edges = np.linspace(0.0, 1.4, 141)
    h0 = combined_histogram(core_initial_csv, beam_initial_csv, vpar_edges, vperp_edges)
    h1 = combined_histogram(core_final_csv, beam_final_csv, vpar_edges, vperp_edges)
    delta = h1 - h0
    vpar = 0.5 * (vpar_edges[:-1] + vpar_edges[1:])
    vperp = 0.5 * (vperp_edges[:-1] + vperp_edges[1:])

    vmax = max(abs(delta.min()), abs(delta.max()), 1e-16)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].pcolormesh(vpar, vperp, np.log10(h0.T + 1e-12), shading="auto", cmap="viridis")
    axes[0].axvline(CORE_U, color="white", lw=1, ls="--")
    axes[0].axvline(BEAM_U, color="white", lw=1, ls=":")
    axes[0].axvspan(-0.99, -0.59, color="gray", alpha=0.22)
    axes[0].axvspan(0.71, 1.59, color="gray", alpha=0.22)
    axes[0].set_title("initial particle VDF")
    axes[0].set_xlabel(r"$v_\parallel/v_A$")
    axes[0].set_ylabel(r"$v_\perp/v_A$")
    fig.colorbar(im0, ax=axes[0], label=r"$\log_{10} f$")

    im1 = axes[1].pcolormesh(vpar, vperp, delta.T, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    theta = np.linspace(0, np.pi, 240)
    for center, radius in [(-0.42, 0.55), (0.34, 0.48)]:
        axes[1].plot(center + radius * np.cos(theta), radius * np.sin(theta), color="black", lw=1.2)
    axes[1].axvspan(-0.99, -0.59, color="gray", alpha=0.18)
    axes[1].axvspan(0.71, 1.3, color="gray", alpha=0.18)
    axes[1].set_title("final minus initial particle VDF")
    axes[1].set_xlabel(r"$v_\parallel/v_A$")
    axes[1].set_ylabel(r"$v_\perp/v_A$")
    fig.colorbar(im1, ax=axes[1], label=r"$\Delta f$")

    fig.suptitle("Figure 4 style: particle VDF scattering")
    fig.savefig(out / "fig4_resonant_scattering.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"), help="Directory with Bx.gda, By.gda, Bz.gda")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="Output figure directory")
    parser.add_argument("--nz", type=int, default=NZ)
    parser.add_argument("--demo", action="store_true", help="Use synthetic demonstration data")
    parser.add_argument("--core-initial-csv", type=Path)
    parser.add_argument("--beam-initial-csv", type=Path)
    parser.add_argument("--core-final-csv", type=Path)
    parser.add_argument("--beam-final-csv", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        t, z, bx, by, bz = demo_fields(nz=args.nz)
    else:
        t, z, bx, by, bz = load_fields(args.data, args.nz)

    plot_fig2(t, z, bx, by, bz, args.out)
    plot_fig3(t, z, bx, by, args.out)
    particle_csvs = [args.core_initial_csv, args.beam_initial_csv, args.core_final_csv, args.beam_final_csv]
    if all(p is not None for p in particle_csvs):
        plot_fig4_from_particles(
            args.out,
            args.core_initial_csv,
            args.beam_initial_csv,
            args.core_final_csv,
            args.beam_final_csv,
        )
    else:
        plot_fig4_schematic(args.out)
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
