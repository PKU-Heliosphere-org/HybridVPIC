#pragma once
#include "math.h"
// Number of wave modes (refer to Giacalone 2021)
#define NUM_WAVES 600       
// Spectral index
#define SPECTRAL_INDEX (-5.0/3.0)
// #define COHERENCE_SCALE (0.17*AU)  // Need to convert to code units
// Normalization factor for the total amplitude of turbulence (needs calibration)
#define variation (1)   

// Define the Alfven wave structure (should be placed in a globally accessible area)
typedef struct {
  // Wavenumber along the background magnetic field direction [1/m]
  double kz;       
  // Transverse wavenumber component (can be extended to 3D)
  double ky;       
  // Transverse wavenumber component (can be extended to 3D)
  double kx;       
  // Angular frequency [rad/s]
  double omega;    
  // Amplitude [T]
  double A;        
  // Random phase [rad]
  double phi;      
} AlfvenWave;

// Initialize turbulence parameters (called during the initialization phase)
AlfvenWave* init_turbulence_params(double v_A, double Lz) {
  // double vA = get_alfven_speed();          // Get the Alfven speed
  // double Lz = get_sim_length_z();          // Length of the simulation region in the z direction
  AlfvenWave* waves = (AlfvenWave*)malloc(NUM_WAVES * sizeof(AlfvenWave));
  // Minimum wavenumber
  double kz_min = 2.0*M_PI/Lz; 
  double Lz_min = 1;            
  // Maximum wavenumber (considering the Nyquist limit)
  double kz_max = 2.0*M_PI/Lz_min;       

  // Generate logarithmically spaced wavenumbers
  double log_kz_min = log(kz_min);
  double log_kz_max = log(kz_max);
  double dlogkz = (log_kz_max - log_kz_min)/(NUM_WAVES-1);

  // Pre-calculate the total power for normalization
  double total_power = 0.0;
  for(int i=0; i<NUM_WAVES; ++i) {
    double kz = exp(log_kz_min + i*dlogkz);
    waves[i].kz = kz;
    // Can be extended to 3D wave vector
    waves[i].kx = 0.0;  
    waves[i].ky = 0.0;
    waves[i].omega = v_A*kz;
    // Random phase
    waves[i].phi = 2.0*M_PI*rand()/RAND_MAX;  

    // Allocate amplitude according to the power spectrum
    double Pk = pow(kz, SPECTRAL_INDEX);
    waves[i].A = sqrt(Pk * dlogkz);
    total_power += waves[i].A*waves[i].A*kz;
  }

  // Normalize the amplitude (to match the coherence scale parameter)
  double norm_factor = variation / sqrt(total_power);
  for(int i=0; i<NUM_WAVES; ++i) {
    waves[i].A *= norm_factor;
  }
  return waves;
}



// Modified field initialization macro
#define BX_PERT ({\
  double sum = 0.0;\
  for(int i=0; i<NUM_WAVES; ++i) {\
    double phase = waves[i].kz*z + waves[i].phi;\
    sum += waves[i].A * cos(phase);\
  }\
  sum;})

#define BY_PERT ({\
  double sum = 0.0;\
  for(int i=0; i<NUM_WAVES; ++i) {\
    double phase = waves[i].kz*z + waves[i].phi;\
    sum += waves[i].A * sin(phase);\
  }\
  sum;})

// Electric field perturbation calculation macro
#define EX_PERT ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi;\
  double sign = (waves[i].kz > 0) ? 1.0 : -1.0;\
  sum += sign * waves[i].A * sin(phase) * v_A;\
}\
sum;})

#define EY_PERT ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi;\
  double sign = (waves[i].kz > 0) ? -1.0 : 1.0; /* Note the sign direction */\
  sum += sign * waves[i].A * cos(phase) * v_A;\
}\
sum;})

// Velocity perturbation calculation macro
#define VX_PERT ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi;\
  double sign = (waves[i].kz > 0) ? -1.0 : 1.0; /* Propagation direction correction */\
  sum += sign * waves[i].A * cos(phase) *v_A / b0;\
}\
sum;})

#define VY_PERT ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi;\
  double sign = (waves[i].kz > 0) ? -1.0 : 1.0;\
  sum += sign * waves[i].A * sin(phase) *v_A / b0;\
}\
sum;})

// Modified field right injection macro
#define BX_PERT_time(t) ({\
  double sum = 0.0;\
  for(int i=0; i<NUM_WAVES; ++i) {\
    double phase = waves[i].kz*z + waves[i].phi-waves[i].omega*t;\
    sum += waves[i].A * cos(phase);\
  }\
  sum;})

#define BY_PERT_time(t) ({\
  double sum = 0.0;\
  for(int i=0; i<NUM_WAVES; ++i) {\
    double phase = waves[i].kz*z + waves[i].phi-waves[i].omega*t;\
    sum += waves[i].A * sin(phase);\
  }\
  sum;})

// Electric field perturbation calculation macro
#define EX_PERT_time(t) ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi-waves[i].omega*t;\
  double sign = (waves[i].kz > 0) ? 1.0 : -1.0;\
  sum += sign * waves[i].A * sin(phase) * v_A;\
}\
sum;})

#define EY_PERT_time(t) ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi-waves[i].omega*t;\
  double sign = (waves[i].kz > 0) ? -1.0 : 1.0; /* Note the sign direction */\
  sum += sign * waves[i].A * cos(phase) * v_A;\
}\
sum;})

// Velocity perturbation calculation macro
#define VX_PERT_time(t) ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi-waves[i].omega*t;\
  double sign = (waves[i].kz > 0) ? -1.0 : 1.0; /* Propagation direction correction */\
  sum += sign * waves[i].A * cos(phase) *v_A / b0;\
}\
sum;})

#define VY_PERT_time(t) ({\
double sum = 0.0;\
for(int i=0; i<NUM_WAVES; ++i) {\
  double phase = waves[i].kz*z + waves[i].phi-waves[i].omega*t;\
  double sign = (waves[i].kz > 0) ? -1.0 : 1.0;\
  sum += sign * waves[i].A * sin(phase) *v_A / b0;\
}\
sum;})