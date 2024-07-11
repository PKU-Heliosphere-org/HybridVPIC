#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>
#include <iostream>

void advance_b_kokkos(k_field_t k_field, const size_t nx, const size_t ny, const size_t nz, const size_t nv,
                      const float px, const float py, const float pz) {

  #define f0_cbx k_field(f0_index, field_var::cbx)
  #define f0_cby k_field(f0_index, field_var::cby)
  #define f0_cbz k_field(f0_index, field_var::cbz)

  #define f0_ex k_field(f0_index,   field_var::ex)
  #define f0_ey k_field(f0_index,   field_var::ey)
  #define f0_ez k_field(f0_index,   field_var::ez)

  #define fx_ex k_field(fx_index,   field_var::ex)
  #define fx_ey k_field(fx_index,   field_var::ey)
  #define fx_ez k_field(fx_index,   field_var::ez)

  #define fy_ex k_field(fy_index,   field_var::ex)
  #define fy_ey k_field(fy_index,   field_var::ey)
  #define fy_ez k_field(fy_index,   field_var::ez)

  #define fz_ex k_field(fz_index,   field_var::ex)
  #define fz_ey k_field(fz_index,   field_var::ey)
  #define fz_ez k_field(fz_index,   field_var::ez)

  // WTF!  Under -ffast-math, gcc-4.1.1 thinks it is okay to treat the
  // below as
  //   f0->cbx = ( f0->cbx + py*( blah ) ) - pz*( blah )
  // even with explicit parenthesis are in there!  Oh my ...
  // -fno-unsafe-math-optimizations must be used

  #define UPDATE_CBX() f0_cbx -= ( py*( fy_ez-f0_ez ) - pz*( fz_ey-f0_ey ) );
  #define UPDATE_CBY() f0_cby -= ( pz*( fz_ex-f0_ex ) - px*( fx_ez-f0_ez ) );
  #define UPDATE_CBZ() f0_cbz -= ( px*( fx_ey-f0_ey ) - py*( fy_ex-f0_ex ) );

  // Do the bulk of the magnetic fields in the pipelines.  The host
  // handles stragglers.
  // While the pipelines are busy, do surface fields

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nz+1,ny+1,nx+1});
    Kokkos::parallel_for("advance_b main chunk", xyz_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
        size_t f0_index = VOXEL(x,   y,   z,    nx,ny,nz);
        size_t fx_index = VOXEL(x+1, y,   z,    nx,ny,nz);
        size_t fy_index = VOXEL(x,   y+1, z,    nx,ny,nz);
        size_t fz_index = VOXEL(x,   y,   z+1,  nx,ny,nz);
        UPDATE_CBX();
        UPDATE_CBY();
        UPDATE_CBZ();
    });

  // Do left over bx
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_policy({1,1},{nz+1,ny+1});
    Kokkos::parallel_for("advance_b::bx", zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_index = VOXEL(nx+1,y,  z,  nx,ny,nz);
        const size_t fy_index = VOXEL(nx+1,y+1,z,  nx,ny,nz);
        const size_t fz_index = VOXEL(nx+1,y,  z+1,nx,ny,nz);
        UPDATE_CBX();
    });

  // Do left over by
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_policy({1,1},{nz+1,nx+1});
    Kokkos::parallel_for("advance_b::by", zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_index = VOXEL(1,ny+1, z,  nx,ny,nz) + (x-1);
        const size_t fx_index = VOXEL(2,ny+1, z,  nx,ny,nz) + (x-1);
        const size_t fz_index = VOXEL(1,ny+1, z+1,nx,ny,nz) + (x-1);
        UPDATE_CBY();
    });

  // Do left over bz
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_policy({1,1},{ny+1,nx+1});
    Kokkos::parallel_for("advance_b::bz", yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_index = VOXEL(1,y,   nz+1,  nx,ny,nz) + (x-1);
        const size_t fx_index = VOXEL(2,y,   nz+1,  nx,ny,nz) + (x-1);
        const size_t fy_index = VOXEL(1,y+1, nz+1,  nx,ny,nz) + (x-1);
        UPDATE_CBZ();
    });

}

void
advance_b(field_array_t * RESTRICT fa,
          float       frac) {

  k_field_t k_field = fa->k_f_d;

  grid_t *g   = fa->g;
  size_t nx   = g->nx;
  size_t ny   = g->ny;
  size_t nz   = g->nz;
  size_t nv   = g->nv;
  float  px   = (nx>1) ? frac*g->cvac*g->dt*g->rdx : 0;
  float  py   = (ny>1) ? frac*g->cvac*g->dt*g->rdy : 0;
  float  pz   = (nz>1) ? frac*g->cvac*g->dt*g->rdz : 0;
//printf("Advance_B kernel\n");

  advance_b_kokkos(k_field, nx, ny, nz, nv, px, py, pz);

  k_local_adjust_norm_b( fa, g );
}
