#ifndef _sf_interface_h_
#define _sf_interface_h_

// FIXME: THE HOST PROCESSED FIELD KERNELS SHOULD BE UPDATED TO USE
// SCALAR FMA INSTRUCTIONS WITH COMMENSURATE ROUND-OFF PROPERTIES TO
// THE FMA INSTRUCTIONS USED ON THE PIPELINE PROCESSED FIELDS!

// FIXME: (nx>1) ? (1/dx) : 0 TYPE LOGIC SHOULD BE FIXED SO THAT NX
// REFERS TO THE GLOBAL NUMBER OF CELLS IN THE X-DIRECTION (NOT THE
// _LOCAL_ NUMBER OF CELLS).  THIS LATENT BUG IS NOT EXPECTED TO
// AFFECT ANY PRACTICAL SIMULATIONS.

#include "../field_advance/field_advance.h"
// FIXME: SHOULD INCLUDE SPECIES_ADVANCE TOO ONCE READY

/*****************************************************************************/

// Interpolator arrays shall be a (nx+2) x (ny+2) x (nz+2) allocation
// indexed FORTRAN style from (0:nx+1,0:ny+1,0:nz+1). Interpolators
// for voxels on the surface of the local domain (for example
// fi(0,:,:) or fi(nx+1,:,:)) are not used.

/*
typedef struct interpolator {
  float ex, dexdy, dexdz, d2exdydz;
  float ey, deydz, deydx, d2eydzdx;
  float ez, dezdx, dezdy, d2ezdxdy;
  float cbx, dcbxdx;
  float cby, dcbydy;
  float cbz, dcbzdz;
  float _pad[2];  // 16-byte align
} interpolator_t;

typedef struct interpolator_array {
  interpolator_t * ALIGNED(128) i;
  grid_t * g;
} interpolator_array_t;
*/
//ARI:
typedef struct interpolator {
  float ex, dexdx, dexdy, dexdz, d2exdx, d2exdy, d2exdz,d2exdydz;
  float ey, deydx, deydy, deydz, d2eydx, d2eydy, d2eydz, d2eydzdx;
  float ez, dezdx, dezdy, dezdz, d2ezdx, d2ezdy, d2ezdz, d2ezdxdy;
  float cbx, dcbxdx, dcbxdy, dcbxdz, d2cbxdx, d2cbxdy, d2cbxdz;
  float cby, dcbydx, dcbydy, dcbydz, d2cbydx, d2cbydy, d2cbydz;
  float cbz, dcbzdx, dcbzdy, dcbzdz, d2cbzdx, d2cbzdy, d2cbzdz;
  float _pad[3];                       // 16-byte align    
} interpolator_t;

typedef struct interpolator_array {
  interpolator_t * ALIGNED(128) i;
  grid_t * g;
} interpolator_array_t;


BEGIN_C_DECLS

// In interpolator_array.cxx

interpolator_array_t *
new_interpolator_array( grid_t * g );

void
delete_interpolator_array( interpolator_array_t * ALIGNED(128) ia );

// Going into load_interpolator, the field array f contains the
// current information such that the fields can be interpolated to
// particles within the local domain.  Load interpolate computes the
// field array into a set of interpolation coefficients for each voxel
// inside the local domain suitable for use by the particle update
// functions.

void
load_interpolator_array( /**/  interpolator_array_t * RESTRICT ia,
                         const field_array_t        * RESTRICT fa );

//ARI:
void
hyb_load_interpolator_array( /**/  interpolator_array_t * RESTRICT ia,
                         const field_array_t        * RESTRICT fa );

END_C_DECLS

/*****************************************************************************/

// Accumulator arrays shall be a
//   POW2_CEIL((nx+2)x(ny+2)x(nz+2),2)x(1+n_pipeline)
// allocation indexed FORTRAN style.  That is, the accumulator array
// is a 4d array.  a(:,:,:,0) is the accumulator used by the host
// processor.  a(:,:,:,1:n_pipeline) are the accumulators used by
// pipelines during operations.  Like the interpolator, accumualtors
// on the surface of the local domain are not used.

/*
typedef struct accumulator {
  float jx[4];   // jx0@(0,-1,-1),jx1@(0,1,-1),jx2@(0,-1,1),jx3@(0,1,1)
  float jy[4];   // jy0@(-1,0,-1),jy1@(-1,0,1),jy2@(1,0,-1),jy3@(1,0,1)
  float jz[4];   // jz0@(-1,-1,0),jz1@(1,-1,0),jz2@(-1,1,0),jz3@(1,1,0)
} accumulator_t;

typedef struct accumulator_array {
  accumulator_t * ALIGNED(128) a;
  int n_pipeline; // Number of pipelines supported by this accumulator
  int stride;     // Stride be each pipeline's accumulator array
  grid_t * g;
} accumulator_array_t;
*/
//ARI:
typedef struct accumulator {
  float jx[7];  // rho0@(0,0,0),rho1@(-1,0,0)...,rho4@(1,0,0)...
  float jy[7];   // 
  float jz[7];   // 
  float rho[7];   //
} accumulator_t;

typedef struct accumulator_array {
  accumulator_t * ALIGNED(128) a;
  int n_pipeline; // Number of pipelines supported by this accumulator
  int stride;     // Stride be each pipeline's accumulator array
  grid_t * g;
} accumulator_array_t;

BEGIN_C_DECLS

// In sf_structors.c

accumulator_array_t *
new_accumulator_array( grid_t * g );

void
delete_accumulator_array( accumulator_array_t * a );

// In clear_accumulators.c

// This zeros out all the accumulator arrays in a pipelined fashion.

void
clear_accumulator_array( accumulator_array_t * RESTRICT a );

//ARI:
void
hyb_clear_accumulator_array( accumulator_array_t * RESTRICT a );

// In reduce_accumulators.c

// Going into reduce_accumulators, the host cores and the pipeline
// cores have each accumulated values to their personal
// accumulators.  This reduces the pipeline accumulators into the host
// accumulator with a pipelined horizontal reduction (a deterministic
// reduction).

void
reduce_accumulator_array( accumulator_array_t * RESTRICT a );

void
hyb_reduce_accumulator_array( accumulator_array_t * RESTRICT a );

// In unload_accumulator.c

// Going into unload_accumulator, the accumulator contains 4 times the
// net amount of charge that crossed the quarter face associated with
// each accumulator component (has units of physical charge, i.e. C)
// computed by the advance_p functions.  unload_accumulator computes
// the physical current density (A/m^2 in MKS units) associated with
// all local quarter faces and accumulates the local quarter faces to
// local field array jf.  unload_accumulator assumes all the pipeline
// accumulators have been reduced into the host accumulator.

void
unload_accumulator_array( /**/  field_array_t       * RESTRICT fa, 
                          const accumulator_array_t * RESTRICT aa );
//ARI:
void
hyb_unload_accumulator_array( /**/  field_array_t       * RESTRICT fa, 
                          const accumulator_array_t * RESTRICT aa );
END_C_DECLS

/*****************************************************************************/

// Hydro arrays shall be a (nx+2) x (ny+2) x (nz+2) allocation indexed
// FORTRAN style from (0:nx+1,0:ny+1,0:nz+1).  Hydros for voxels on
// the surface of the local domain (for example h(0,:,:) or
// h(nx+1,:,:)) are not used.

typedef struct hydro {
  float jx, jy, jz, rho; // Current and charge density => <q v_i f>, <q f>
  float px, py, pz, ke;  // Momentum and K.E. density  => <p_i f>, <m c^2 (gamma-1) f>
  float txx, tyy, tzz;   // Stress diagonal            => <p_i v_j f>, i==j
  float tyz, tzx, txy;   // Stress off-diagonal        => <p_i v_j f>, i!=j
  float _pad[2];         // 16-byte align
} hydro_t;

typedef struct hydro_array {
  hydro_t * ALIGNED(128) h;
  grid_t * g;
} hydro_array_t;

BEGIN_C_DECLS

// In hydro_array.c

// Construct a hydro array suitable for the grid

hydro_array_t *
new_hydro_array( grid_t * g );

// Destruct a hydro array

void
delete_hydro_array( hydro_array_t * ha );

// Zero out the hydro array.  Use before accumulating species to
// a hydro array.

void
clear_hydro_array( hydro_array_t * ha );

// Synchronize the hydro array with local boundary conditions and
// neighboring processes.  Use after all species have been
// accumulated to the hydro array.

void
synchronize_hydro_array( hydro_array_t * ha );

void
hyb_synchronize_hydro_array( hydro_array_t * ha );

END_C_DECLS

#endif // _sf_interface_h_
