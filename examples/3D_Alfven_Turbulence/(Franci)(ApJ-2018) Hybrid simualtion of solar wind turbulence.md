
# Solar Wind Turbulent Cascade from MHD to Sub-ion Scales: Large-size 3D Hybrid Particle-in-cell Simulations  

Luca Franci<sup>1,2</sup>, Simone Landi<sup>1</sup>, Andrea Verdini<sup>1</sup>, Lorenzo Matteini<sup>3,4</sup>, and Petr Hellinger<sup>5</sup>  

<sup>1</sup> Dipartimento di Fisica e Astronomia, Università di Firenze, Firenze, Italy  
<sup>2</sup> School of Physics and Astronomy, Queen Mary University of London, London, UK  
<sup>3</sup> Department of Physics, Imperial College London, UK  
<sup>4</sup> LESIA, Observatoire de Paris, Meudon, France  
<sup>5</sup> Astronomical Institute, CAS, Prague, Czech Republic  

**Received** 2017 November 7; **revised** 2017 December 12; **accepted** 2017 December 18; **published** 2018 January 19  

**Abstract**  
We investigate properties of the turbulent cascade from fluid to kinetic scales in collisionless plasmas using large-size 3D hybrid (fluid electrons, kinetic protons) particle-in-cell simulations. Initially isotropic Alfvénic fluctuations rapidly develop a strongly anisotropic turbulent cascade, primarily in the direction perpendicular to the ambient magnetic field. The omnidirectional magnetic field spectrum exhibits a double power-law behavior over nearly two decades in wavenumber, featuring a Kolmogorov-like index at large scales, a spectral break around ion scales, and steepening at sub-ion scales. Power laws are also observed in the spectra of ion bulk velocity, density, and electric field at both magnetohydrodynamic (MHD) and kinetic scales. Despite the complex structure, the omnidirectional spectra of all fields at ion and sub-ion scales show remarkable quantitative agreement with those of a 2D simulation with similar physical parameters, providing partial a posteriori validation of the 2D approximation at kinetic scales. Conversely, at MHD scales, spectra of density, velocity, and electric field differ between 2D and 3D cases, likely due to more significant compressible effects in 3D geometry. Our findings are in striking quantitative agreement with solar wind observations.  

**Key words**: magnetohydrodynamics (MHD) – plasmas – solar wind – turbulence  


## 1. Introduction  
In situ observations from solar and heliospheric missions reveal solar wind plasma and electromagnetic fluctuations in the frequency range \(10^{-5} ~\text{Hz} < f < 10^{2} ~\text{Hz}\), with power spectra showing power-law behavior across multiple frequency decades. Different power-law indices appear at scales larger or smaller than ~1 Hz (proton spatial scales) . Third-order structure function measurements confirm that power-law spectra at scales well above proton scales (MHD scales) arise from turbulent cascades . Recent extensions of third-order structure function laws to homogeneous incompressible Hall-MHD turbulence in 2D HPIC simulations suggest the cascade continues to sub-proton scales via the Hall term .  

At MHD scales, solar wind fluctuations are predominantly Alfvénic, with magnetic fields and ion bulk velocities dominated by transverse components. Magnetic spectra exhibit a Kolmogorov-like slope (\(\sim -5/3\)), while velocity spectra are flatter (\(\sim -3/2\)) . The electric field couples strongly to ion velocity .  

At proton kinetic scales, both magnetic and velocity spectra steepen. Magnetic spectral indices range from \(-4\) to \(-2\) at sub-ion scales (typically \(\sim -2.8\) between ion and electron scales), while velocity spectra decouple from magnetic fields and show steeper, more variable slopes . The electric field spectrum flattens (\(\sim -0.8\)), dominating magnetic fluctuations, consistent with the generalized Ohm’s law . Density fluctuations exhibit a unique triple-power-law behavior with two breaks: \(\sim -5/3\) at MHD scales, \(\sim -1\) near ion scales, and \(\sim -2.8\) at sub-ion scales .  

The ion-scale break in magnetic spectra relates to ion inertial length (\(d_i\)) and gyroradius (\(\rho_i\)), with observations suggesting a transition at the larger scale when separated or a combination in intermediate-beta plasmas . Coherent structures and magnetic reconnection likely dominate spectral shaping at kinetic scales .  

While 2D simulations reproduce kinetic-scale spectral features, they lack 3D dynamics like solar wind expansion and proton velocity instabilities . 3D simulations have struggled to resolve extended power laws across MHD and kinetic scales, and quantitative 2D-3D spectral comparisons remain limited . This work extends 2D studies to 3D, validating spectral properties and turbulent anisotropy.  


## 2. Numerical Setup and Initial Conditions  
We use the CAMELIA hybrid PIC code, treating electrons as a massless fluid and protons as kinetic particles advanced via the Boris scheme . Characteristic units are \(d_i = v_A / \Omega_i\) (ion inertial length) and \(\Omega_i^{-1}\) (ion gyrofrequency inverse) .  

### 2.1 Simulation Parameters  
- **Grid**: Periodic cubic, \(512^3\) points, \(L_{\text{box}} = 128 d_i\), resolution \(0.25 d_i\).  
- **Particles**: 2048 protons per cell (ppc).  
- **Resistivity**: \(\eta = 1.5 \times 10^{-3} \cdot 4\pi v_A c^{-1} \Omega_i^{-1}\) to suppress small-scale energy accumulation .  
- **Time steps**: \(\Delta t = 0.05 \Omega_i^{-1}\) (protons), \(\Delta t_B = \Delta t / 10\) (magnetic field) .  

### 2.2 Initial Conditions  
- **Background field**: Uniform \(B_0 = \hat{z}\) (z-direction).  
- **Plasma state**: Uniform density (\(n_i = n_e\)), isotropic temperatures, \(\beta_i = \beta_e = 0.5\) .  
- **Fluctuations**: Linearly polarized shear Alfvénic fluctuations with random phases, perpendicular to the wavevector-mean field plane. Initial kinetic/magnetic energies are equipartitioned (within 10%), with divergenceless velocities and negligible density fluctuations . Fourier modes are excited in \(k_0 < k < k_{\text{inj}}\) (\(k_0 \approx 0.05 d_i^{-1}\), \(k_{\text{inj}} \approx 0.25 d_i^{-1}\)) .  

### 2.3 Spectral Definitions  
- **3D axisymmetric spectrum**: Averaged over perpendicular wavenumber rings:  
  \[
  P_{3D}(k_{\perp}, k_{\parallel}) = \frac{1}{k_{\perp}} \sum_{\sqrt{k_x^2 + k_y^2} = k_{\perp}} \hat{\Psi}_{3D}^2(k_x, k_y, k_z) 
  \]  
- **2D/1D spectra**: Obtained via integration over parallel/perpendicular wavenumbers .  
- **Omnidirectional spectrum**: Integrated over spherical shells:  
  \[
  P_{1D}(k) = \sum_{\sqrt{k_x^2 + k_y^2 + k_z^2} = k} \hat{\Psi}_{3D}^2(k_x, k_y, k_z) 
  \]  
- **RMS value**: \( \Psi^{\text{rms}} = \sqrt{\langle \Psi^2 \rangle - \langle \Psi \rangle^2} \) .  

Initial conditions yield \(P_B^{1D} \sim P_u^{1D} \propto k^2\) with \(B^{\text{rms}} / B_0 \sim 0.4\), and energy concentrated at \(k d_i \sim 0.25\) .  


## 3. Results  

### 3.1 Turbulent Cascade Development  
The root-mean-square (rms) current density (\(J^{\text{rms}}\)) peaks at \(t_{\text{max}} = 160 \Omega_i^{-1}\) (15 nonlinear times), marking maximum turbulent activity . Magnetic (\(B^{\text{rms}}\)) and velocity (\(u^{\text{rms}}\)) energies decline steadily, with magnetic energy exceeding kinetic energy by 10–15% .  

Early-time current sheets are quasi-2D, elongated along the mean field, and evolve into complex, uniformly distributed structures at \(t_{\text{max}}\) . Magnetic field lines develop strong perpendicular gradients and filamentary shapes, indicating spectral anisotropy from isotropic initial conditions .  

### 3.2 Quasi-stationary State Spectral Properties  
#### 3.2.1 Magnetic Fluctuations  
The 3D magnetic spectrum shows strong perpendicular energy cascading, with a white isocontour at \(k_{\perp} d_i \sim 2.5\) separating MHD (\(k_{\perp} d_i < 2.5\)) and kinetic (\(k_{\perp} d_i > 2.5\)) ranges . The 1D reduced perpendicular spectrum (\(P_{1D,\perp}^B\)) exhibits a double power law: \(\sim -5/3\) at MHD scales, \(\sim -3\) at sub-ion scales, with a break near ion scales . Parallel spectra (\(P_{1D,\parallel}^B\)) steepen at small \(k_{\parallel}\) and flatten at \(k_{\parallel} d_i \gtrsim 1\) due to numerical noise .  

#### 3.2.2 Ion Bulk Velocity, Electric Field, and Density  
- **Velocity**: Perpendicular spectra show a short-lived power law at large scales (\(\sim -5/3\) or \(\sim -3/2\)) and steepen rapidly (\(\sim -4.5\)) at \(k_{\perp} d_i \gtrsim 1\) .  
- **Electric field**: Follows velocity spectra at MHD scales, flattens to \(\sim -0.8\) at sub-ion scales, dominated by Hall and electron pressure terms .  
- **Density**: Exhibits a triple-power-law trend (\(\sim -1.0\), \(\sim -0.7\), \(\sim -2.8\)) across scales, with equipartition with magnetic energy at kinetic scales .  

### 3.3 2D vs. 3D Comparison  
Magnetic spectra agree closely between 2D and 3D at kinetic scales, with identical power laws and break scales, validating 2D approximations for kinetic physics . At MHD scales, 3D density and velocity spectra differ due to stronger compressible effects and parallel energy transfer . Real-space magnetic structures in 3D, when averaged along the mean field, resemble 2D vortices, suggesting flux tube geometries .  


## 4. Discussion and Conclusions  
Our 3D hybrid PIC simulations reproduce solar wind spectral features, including double-power-law magnetic spectra, Alfvénic turbulence at MHD scales, and kinetic-scale steepening. The close agreement between 2D and 3D kinetic-scale spectra supports 2D models for ion/sub-ion physics, while MHD-scale differences highlight the role of 3D compressibility .  

Key findings include:  
1. Turbulent cascades develop strong perpendicular anisotropy, with energy transfer dominated by perpendicular wavenumbers.  
2. Kinetic-scale spectral properties are robust across 2D/3D geometries, linked to magnetic reconnection and coherent structures.  
3. Solar wind observations are well-reproduced, validating the hybrid PIC approach for collisionless plasma turbulence studies.  

Future work will explore spectral anisotropy in local mean field frames and theoretical comparisons with slab/2D turbulence models .  

**Acknowledgments**: Supported by Fondazione Cassa di Risparmio di Firenze, GACR grant 15-10057S, PRACE, and CINECA.  

**ORCID iDs**:  
Luca Franci [0000-0002-7419-0527](https://orcid.org/0000-0002-7419-0527)  
Simone Landi [0000-0002-1322-8712](https://orcid.org/0000-0002-1322-8712)  
Andrea Verdini [0000-0003-4380-4837](https://orcid.org/0000-0003-4380-4837)  
Lorenzo Matteini [0000-0002-6276-7771](https://orcid.org/0000-0002-6276-7771)  
Petr Hellinger [0000-0002-5608-0834](https://orcid.org/0000-0002-5608-0834)  

**References** (abbreviated, see full text for details):  
Alexandrova et al. 2009, 2012; Bale et al. 2005; Boldyrev et al. 2013; Chen et al. 2011, 2012; Franci et al. 2015a, 2016; Hellinger et al. 2017b; Salem et al. 2009, 2012; Šafránková et al. 2015, 2016; Wan et al. 2012, 2016; Zank et al. 2017.