# 1D core+beam ICW anisotropy instability

This input deck extends the `pcai` example from one anisotropic proton population to two proton populations:

- `ion_c`: core protons, density fraction `n_core = 0.90`, weak perpendicular anisotropy.
- `ion_b`: beam protons, density fraction `n_beam = 0.10`, drifting along the background field with stronger perpendicular anisotropy.

The domain is 1D along `x`, with `B0 = B0 xhat` and periodic boundaries. The core drift is chosen as

```text
Uc = -n_beam * Ub / n_core
```

so that the initial total ion current vanishes in the simulation frame. The main diagnostics are transverse magnetic fluctuations `By`, `Bz`, their Fourier spectra in `k_x`, and the separate core/beam stress tensors in `hydro/Hhydro_c*` and `hydro/Hhydro_b*`.

In post-processing, use the circular components

```text
B_L(k) = By(k) + i Bz(k)
B_R(k) = By(k) - i Bz(k)
```

to separate the two ion-cyclotron branches and compare the power at positive and negative `k_x`.
