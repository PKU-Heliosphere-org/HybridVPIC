#!/usr/bin/env python3
"""Derive normalized HybridVPIC parameters from the lecture-note PSP inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path


QE = 1.602176634e-19
MP = 1.67262192369e-27
MU0 = 4.0 * math.pi * 1.0e-7
CM3_TO_M3 = 1.0e6


def beta_parallel(n_cm3: float, t_ev: float, b_nt: float) -> float:
    n_m3 = n_cm3 * CM3_TO_M3
    b_t = b_nt * 1.0e-9
    return 2.0 * MU0 * n_m3 * t_ev * QE / (b_t * b_t)


def alfven_speed_km_s(n_total_cm3: float, b_nt: float) -> float:
    n_m3 = n_total_cm3 * CM3_TO_M3
    b_t = b_nt * 1.0e-9
    return b_t / math.sqrt(MU0 * n_m3 * MP) / 1000.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config" / "psp_case1_parameters.json"
    data = json.loads(config_path.read_text())
    phys = data["physical_parameters"]
    core = phys["core"]
    beam = phys["beam"]

    n_total = core["density_cm3"] + beam["density_cm3"]
    v_a = alfven_speed_km_s(n_total, phys["B0_nT"])

    rows = []
    for label, pop in [("core", core), ("beam", beam)]:
        density_fraction = pop["density_cm3"] / n_total
        drift_va = (pop["bulk_speed_km_s"] - phys["Vsw_km_s"]) / v_a
        beta_par = beta_parallel(pop["density_cm3"], pop["T_parallel_eV"], phys["B0_nT"])
        anisotropy = pop["T_perp_eV_observed"] * pop["T_perp_multiplier_for_simulation"] / pop["T_parallel_eV"]
        rows.append((label, density_fraction, drift_va, beta_par, anisotropy))

    print("Derived normalized parameters for HybridVPIC")
    print(f"v_A = {v_a:.6f} km/s")
    print("")
    print("species density_fraction drift/v_A beta_parallel Tperp/Tpar")
    for label, density_fraction, drift_va, beta_par, anisotropy in rows:
        print(f"{label:>6s} {density_fraction:16.10f} {drift_va:9.6f} {beta_par:13.8f} {anisotropy:10.6f}")


if __name__ == "__main__":
    main()
