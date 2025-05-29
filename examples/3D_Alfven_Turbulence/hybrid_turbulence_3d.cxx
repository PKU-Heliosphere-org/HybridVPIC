//////////////////////////////////////////////////////
// 3D Hybrid Simulation of Solar Wind Turbulence
// Based on Franci et al. (ApJ 2018)
//////////////////////////////////////////////////////

#include "hybrid.h"

begin_globals {
  int restart_interval;
  int energies_interval;
  int fields_interval;
  int hydro_interval;
  int particle_interval;
  int quota_check_interval;

  // Simulation parameters
  double Lx, Ly, Lz;    // Domain size (128di)
  double dx, dy, dz;     // Grid spacing (0.25di)
  double dt;             // Time step (0.05Ωi⁻¹)
  double eta;            // Resistivity (1.5e-3)
  
  // Plasma parameters
  double beta_i;         // Ion beta (0.5)
  double beta_e;         // Electron beta (0.5)
  double B0;             // Background field (z-direction)
  
  // Turbulence parameters
  double k0;             // Minimum wavenumber (0.05di⁻¹)
  double k_inj;          // Injection wavenumber (0.25di⁻¹)
  double B_rms;          // Initial fluctuation amplitude (0.4B0)
};

begin_initialization {
  // Set simulation parameters
  global->Lx = global->Ly = global->Lz = 128.0; // in di
  global->dx = global->dy = global->dz = 0.25;  // in di
  global->dt = 0.05;                           // in Ωi⁻¹
  global->eta = 1.5e-3;                        // dimensionless
  
  // Set plasma parameters
  global->beta_i = global->beta_e = 0.5;
  global->B0 = 1.0;  // in B0
  
  // Set turbulence parameters
  global->k0 = 0.05;    // in di⁻¹
  global->k_inj = 0.25; // in di⁻¹
  global->B_rms = 0.4;  // relative to B0

  // Set output intervals
  global->restart_interval = 100;
  global->energies_interval = 10;
  global->fields_interval = 20;
  global->hydro_interval = 20;
  global->particle_interval = 100;
  global->quota_check_interval = 100;

  // Define units
  define_units(1.0, 1.0); // c, eps0
  define_timestep(global->dt);

  // Initialize grid with periodic boundary conditions in all directions
  define_periodic_grid(0, 0, 0, 
                       global->Lx, global->Ly, global->Lz,
                       (int)(global->Lx/global->dx),
                       (int)(global->Ly/global->dy),
                       (int)(global->Lz/global->dz),
                       1, 1, 1); // Topology (fully periodic)

  // Explicitly set periodic boundary conditions for fields and particles
  set_domain_field_bc(BOUNDARY(-1,0,0), periodic_fields);
  set_domain_field_bc(BOUNDARY(1,0,0), periodic_fields);
  set_domain_field_bc(BOUNDARY(0,-1,0), periodic_fields);
  set_domain_field_bc(BOUNDARY(0,1,0), periodic_fields);
  set_domain_field_bc(BOUNDARY(0,0,-1), periodic_fields);
  set_domain_field_bc(BOUNDARY(0,0,1), periodic_fields);

  set_domain_particle_bc(BOUNDARY(-1,0,0), periodic_particles);
  set_domain_particle_bc(BOUNDARY(1,0,0), periodic_particles);
  set_domain_particle_bc(BOUNDARY(0,-1,0), periodic_particles);
  set_domain_particle_bc(BOUNDARY(0,1,0), periodic_particles);
  set_domain_particle_bc(BOUNDARY(0,0,-1), periodic_particles);
  set_domain_particle_bc(BOUNDARY(0,0,1), periodic_particles);

  // Set background field and resistivity
  grid->B0x = 0;
  grid->B0y = 0;
  grid->B0z = global->B0;
  grid->eta = global->eta;

  // Initialize species
  species_t *ion = define_species("ion", 1.0, 1.0, 2048, 0, 1, 1);

  // Initialize Alfvenic turbulence
  seed_entropy(rank());
  double kx, ky, kz;
  double phase;
  double Bx, By, Bz;
  double ux, uy, uz;
  
  // Initialize Alfvenic fluctuations with random phases
  for (int i=1; i<=grid->nx; i++) {
    for (int j=1; j<=grid->ny; j++) {
      for (int k=1; k<=grid->nz; k++) {
        kx = 2*M_PI*i/global->Lx;
        ky = 2*M_PI*j/global->Ly;
        kz = 2*M_PI*k/global->Lz;
        
        // Only initialize modes in injection range
        if (sqrt(kx*kx + ky*ky + kz*kz) > global->k0 && 
            sqrt(kx*kx + ky*ky + kz*kz) < global->k_inj) {
          
          phase = 2*M_PI*drand();
          
          // Alfvenic fluctuations (divergence-free)
          Bx = global->B_rms*cos(phase);
          By = global->B_rms*sin(phase);
          Bz = 0;
          
          // Corresponding velocity fluctuations
          ux = -By/sqrt(4*M_PI*global->rho0);
          uy = Bx/sqrt(4*M_PI*global->rho0);
          uz = 0;
          
          // Apply to fields
          field(i,j,k).cbx += Bx;
          field(i,j,k).cby += By;
          field(i,j,k).cbz += Bz;
          
          // Apply to particles
          // (Particle initialization code here)
        }
      }
    }
  }
}

begin_diagnostics {
  // Output routines
  if (step() == 0) {
    dump_materials("rundata/materials");
    dump_species("rundata/species");
    dump_grid("rundata/grid");
  }

  if (should_dump(energies)) {
    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
  }

  if (should_dump(fields)) {
    dump_fields("fields/fields", step());
  }

  if (should_dump(hydro)) {
    dump_hydro("hydro/hydro", step());
  }
}

begin_particle_injection {
  // Particle boundary conditions
}

begin_current_injection {
  // Current injection if needed
}

begin_field_injection {
  // No field driving - decay turbulence simulation
}

begin_particle_collisions {
  // Collision operator if needed
}
