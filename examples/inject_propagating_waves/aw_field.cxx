//! @file aw_field.cxx For direct inclusion into VPIC-hybrid deck file. 
//! It should be included to Line 927 of Card file. 

if (global->right) {
  double t_current = grid->t0 + grid->dt * (double)(grid->step);

  double theta_aw_rad = 20.0 * M_PI / 180.0;
  double k_aw = 2 * M_PI / (grid->x1 - grid->x0) * 5.0; // Since the module is extracted, there is no need to encapsule function here.
  double k_z = k_aw * cos(theta_aw_rad);
  double k_x = k_aw * sin(theta_aw_rad);

  // omega = k va cos theta = k_z va
  double omega = fabs(k_z) * global->v_A;
  double b_amp = 0.1 * b0;

  auto f_coord_x = [this](int i) {return grid->x0 + grid->dx * (i - 1);};
  auto f_coord_y = [this](int j) {return grid->y0 + grid->dy * (j - 1);};
  auto f_coord_z = [this](int k) {return grid->z0 + grid->dz * (k - 1);};

  double phase_0_rad = 0.0;
  auto phase_factor = [omega, t_current, k_x, k_z, f_coord_x, f_coord_y, f_coord_z](int i, int j, int k) {
    return cos(k_x * f_coord_x(i) + k_z * f_coord_z(k) - omega * t_current); };
  // double v_amp = -global->v_A * b_amp / b0;
  double e_amp = global->v_A * b_amp; // -v_amp * b0

  XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).cbx = 0;
  XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).cby = b_amp * phase_factor(x, y, z);
  XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).cbz = b0; // Do not forget the background value.
  XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).ex = e_amp * phase_factor(x, y, z);
  XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).ey = 0;
  XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).ez = 0;
  
} // If right boundary