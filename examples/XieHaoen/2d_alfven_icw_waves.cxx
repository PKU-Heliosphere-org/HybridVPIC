//////////////////////////////////////////////////////
//
//   Landau-damped Ion Acoustic Wave
//
//////////////////////////////////////////////////////

//#define NUM_TURNSTILES 16384

//////////////////////////////////////////////////////
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

struct PerturbationParams {
    std::vector<double> amplitude_ratio;
    std::vector<double> kx_random;
    std::vector<double> ky_random;
    std::vector<double> kz_random;
    std::vector<double> phi_random;
    int n_modes;
    double n0;
    double va;
    double b0x;
    double b0y;
    double b0z;
    double waveamp;
};

// 计算向量的模
double vector_norm(const std::vector<double>& v) {
    if (v.size() != 3) {
        throw std::invalid_argument("vector_norm requires a vector of size 3");
    }
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// 计算两个向量的叉积
std::vector<double> cross_product(const std::vector<double>& a, const std::vector<double>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}
std::vector<double> get_k_corss_B0(int i, const PerturbationParams& params) {
    std::vector<double> k = {params.kx_random[i], params.ky_random[i], params.kz_random[i]};
    std::vector<double> B0 = {params.b0x, params.b0y, params.b0z};
    std::vector<double> cross = cross_product(k, B0);
    double cross_norm = vector_norm(cross);
    
    if (cross_norm == 0.0) {
        // 处理叉积为零的情况：返回默认单位向量（如z轴方向）或根据业务调整
        return {-0.259, 0.0, 0.966}; // 示例默认方向，可改为抛出异常或其他处理方式
    }
    
    std::vector<double> e_k_b0 = {
        cross[0] / cross_norm,
        cross[1] / cross_norm,
        cross[2] / cross_norm
    };
    return e_k_b0;
}

double BX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double bx_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                             params.ky_random[i] * params.ky_random[i] +
                             params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double E1y = Bz * v1x - Bx * v1z;
        double cos_theta_kb = (params.kx_random[i] * Bx + params.ky_random[i] * By+ params.kz_random[i] * Bz)/norm_k;
        double b1x = params.ky_random[i] * E1z - params.kz_random[i] * E1y;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::fabs(norm_k)>1e-6)bx_pert += params.amplitude_ratio[i] * b1x / norm_k / params.va / cos_theta_kb * cos(phi);
    }
    return bx_pert;
}

double BY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double by_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                            params.ky_random[i] * params.ky_random[i] +
                            params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double cos_theta_kb = (params.kx_random[i] * Bx + params.ky_random[i] * By+ params.kz_random[i] * Bz)/norm_k;
        double E1z = Bx * v1y - By * v1x;
        double E1x = By * v1z - Bz * v1y;
        double b1y = params.kz_random[i] * E1x - params.kx_random[i] * E1z;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::fabs(norm_k)>1e-6)by_pert += params.amplitude_ratio[i] * b1y  / norm_k / params.va / cos_theta_kb * cos(phi);
    }
    return by_pert;
}
double BZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double bz_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                             params.ky_random[i] * params.ky_random[i] +
                             params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double cos_theta_kb = (params.kx_random[i] * Bx + params.ky_random[i] * By+ params.kz_random[i] * Bz)/norm_k;
        double E1x = By * v1z - Bz * v1y;
        double E1y = Bz * v1x - Bx * v1z;
        double b1z = params.kx_random[i] * E1y - params.ky_random[i] * E1x;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::fabs(norm_k)>1e-6){
        bz_pert += params.amplitude_ratio[i] * b1z  / norm_k / params.va / cos_theta_kb * cos(phi);
        }
    }
    return bz_pert;
}

double UX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ux_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        ux_pert += params.amplitude_ratio[i] * v1x * cos(psi); 
    }
    return ux_pert;
}

double UY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uy_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1y = e_k_b0[1] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        uy_pert += params.amplitude_ratio[i] * v1y * cos(psi);
    }
    return uy_pert;
}

double UZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uz_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1z = e_k_b0[2] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        uz_pert += params.amplitude_ratio[i] * v1z * cos(psi); 
    }
    return uz_pert;
}

double N_PERT(double x, double y, double z, const PerturbationParams& params) {
    double n_pert = 0.0;
    return n_pert;
}
double EX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ex_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1x = By * v1z - Bz * v1y;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        ex_pert += params.amplitude_ratio[i] * E1x * cos(phi);
    }
    return ex_pert;
}

double EY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ey_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1y = Bz * v1x - Bx * v1z;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        ey_pert += params.amplitude_ratio[i] * E1y * cos(phi);
    }
    return ey_pert;
}

double EZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ez_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        ez_pert += params.amplitude_ratio[i] * E1z * cos(phi);
    }
    return ez_pert;
}

std::vector<double> linspace(double start, double end, int num) {
  std::vector<double> result;
  if (num <= 1) {
      result.push_back(start);
      return result;
  }

  double step = (end - start) / (num - 1); // 计算步长
  for (int i = 0; i < num; ++i) {
      result.push_back(start + i * step);
  }
  
  return result;
}

begin_globals {

  int restart_interval;
  int energies_interval;
  int fields_interval;
  int ehydro_interval;
  int Hhydro_interval;
  int eparticle_interval;
  int Hparticle_interval;
  int quota_check_interval;  // How frequently to check if quota exceeded

  int rtoggle;               // enables save of last 2 restart dumps for safety
  int write_restart;     // global flag for all to write restart files
  int write_end_restart; // global flag for all to write restart files
  
  double quota_sec;          // Run quota in seconds
  double b0;                 // B0
  double v_A;
  double topology_x;       // domain topology
  double topology_y;
  double topology_z;

  double nfac;      //  Normalization factor to convert particles per cell into density
  double sn;
  double cn;

  int left,right,down,up; // 1 if this domain is along the boundary in that direction

  // Output variables
  DumpParameters fdParams;
  DumpParameters hedParams;
  DumpParameters hHdParams;
  std::vector<DumpParameters *> outputParams;

};

begin_initialization {
  
  // Use natural hybrid-PIC units:
  double ec   = 1.0;  // Charge normalization
  double mi   = 1.0;  // Mass normalization
  double mu0  = 1.0;  // Magnetic constanst
  double b0 = 1.0;    // Magnetic field. // Note for this problem B=0 (but we can still pick a reference field/units).
  double n0 = 1.0;    // Density
  
  // Derived normalization parameters:
  double v_A = b0/sqrt(mu0*n0*mi); // Alfven velocity
  double wci = ec*b0/mi;          // Cyclotron freq.
  double di = v_A/wci;            // Ion skin-depth
  
  // Initial conditions for model:
  double Ti = 1.0/5.0;      // Ion temperature
  double gamma = 5.0/3.0;   // Ratio of specific heats.
  double c_s = 1.0;         // Electron sound speed.
  double pert = 0.0;       // Size of density perturbation.
  double Lx = 384.0*di;           // Size of domain.
  // double kx = 0.5*M_PI/Lx;  // Wavenumber of perturbation.
  
  double eta = 0.03;         // Plasma resistivity.
  double hypereta = 0.01;    // Plasma hyper-resistivity.

  
  // Derived quantities for model:
  double Te = c_s/(gamma);  // Electron temperature.
  double vthi = sqrt(Ti/mi);// Ion thermal velocity
  double theta = 15.0*M_PI/180.0;  // backgrond B field angle
  double cn       = cos(theta);
  double sn       = sin(theta);
  double b0x = b0*cn;
  double b0y = b0*0.0;
  double b0z = b0*sn;
  double v1x = 0.0;
  double waveamp = 0.01;

  // Numerical parameters
  double taui    = 2000;      // Simulation run time in wci^-1.
  double amp = 0.01; // amplitude of injected wave relative to Bo
  double imf = 0.1; // the ratio of amplitudes of the two groups of waves propagating to the opposite direction
  double quota   = 23.5;     // run quota in hours
  double quota_sec = quota*3600;  // Run quota in seconds
  
  double Ly    = 1.0*di;    // size of box in y dimension
  double Lz    = 64.0*di;    // size of box in z dimension

  double nx = 256;
  double ny = 1;
  double nz = 256;

  double nppc  = 512;    // Average number of macro particle per cell per species 
  
  double topology_x = 4; // Number of domains in x, y, and z
  double topology_y = 1;
  double topology_z = 4;


  // Derived numerical parameters
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;


  double Ni  = nppc*nx*ny*nz;       // Total macroparticle ions in box
  double Np  = n0*Lx*Ly*Lz;         // Total number of physical background ions
  Ni = trunc_granular(Ni,nproc());  // Make it divisible by number of processors
  double qi = ec*Np/Ni;             // Charge per macro ion

  
  // Determine the time step
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);  // courant length
  double dt = 0.05;                               // time step

  double sort_interval = 10;  // How often to sort particles
  
  // Intervals for output
  num_step = int(taui/(wci*dt));
  int restart_interval = 1000;
  int energies_interval = 200;
  int interval = int(num_step/1000);//0.4/(wci*dt));
  int fields_interval = interval;
  int ehydro_interval = interval;
  int Hhydro_interval = interval;
  int eparticle_interval = interval;
  int Hparticle_interval = interval;
  int quota_check_interval     = 100;

    // Determine which domains area along the boundaries - Use macro from
  // grid/partition.c.

# define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE {                 \
  int _ix, _iy, _iz;                                                  \
  _ix  = (rank);                /* ix = ix+gpx*( iy+gpy*iz ) */       \
  _iy  = _ix/int(topology_x);   /* iy = iy+gpy*iz */                  \
  _ix -= _iy*int(topology_x);   /* ix = ix */                         \
  _iz  = _iy/int(topology_y);   /* iz = iz */                         \
  _iy -= _iz*int(topology_y);   /* iy = iy */                         \
  (ix) = _ix;                                                         \
  (iy) = _iy;                                                         \
  (iz) = _iz;                                                         \
} END_PRIMITIVE

int ix, iy, iz, left=0,right=0,down=0,up=0;
RANK_TO_INDEX( int(rank()), ix, iy, iz );
if ( ix ==0 ) left=1;
if ( ix ==topology_x-1 ) right=1;
if ( iz ==0 ) down=1;
if ( iz ==topology_z-1 ) up=1;

  ///////////////////////////////////////////////
  // Setup high level simulation parameters
  status_interval      = num_step/1000;
  sync_shared_interval = status_interval;
  clean_div_e_interval = status_interval;
  clean_div_b_interval = status_interval;

  global->restart_interval     = restart_interval;
  global->energies_interval    = energies_interval;
  global->fields_interval      = fields_interval;
  global->ehydro_interval      = ehydro_interval;
  global->Hhydro_interval      = Hhydro_interval;
  global->eparticle_interval   = eparticle_interval;
  global->Hparticle_interval   = Hparticle_interval;
  global->quota_check_interval = quota_check_interval;
  global->quota_sec            = quota_sec;
  global->rtoggle              = 0;
  global->left = left;
  global->right = right;
  global->down = down;
  global->up = up;
  global->sn = sn;
  global->cn = cn; 

  global->b0  = b0;
  global->v_A  = v_A;

  global->topology_x  = topology_x;
  global->topology_y  = topology_y;
  global->topology_z  = topology_z;


 
  //////////////////////////////////////////////////////////////////////////////
  // Setup the grid

  // Setup basic grid parameters
  define_units(1.0, 1.0);//c, eps0 );
  define_timestep( dt );

  // Define the grid
  define_periodic_grid(  -0.5*Lx, -0.5*Ly, -0.5*Lz,    // Low corner
                          0.5*Lx,  0.5*Ly, 0.5*Lz,     // High corner
                         nx, ny, nz,             // Resolution
                         topology_x, topology_y, topology_z); // Topology

  grid->te = Te;
  grid->den = 1.0;
  grid->eta = eta;
  grid->hypereta = hypereta;
  grid->gamma = gamma;

  grid->nsub = 1; // Number of substeps for field solve.
  grid->nsm = 2;  // Number of binomial smoothing passes (to fields & moments).
  grid->nsmb = 0; // Timesteps between additional smooths of magnetic field (0 is off).

  // ***** Set Field Boundary Conditions *****
  sim_log("Periodic boundaries");
  // Do nothing - periodic is default.

  // ***** Set Particle Boundary Conditions *****
  // Do nothing - periodic is default.
 
  //////////////////////////////////////////////////////////////////////////////
  // Setup materials
  sim_log("Setting up materials. ");
  define_material( "vacuum", 1 );

  
  //////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                       // Finalize Field Advance
  define_field_array(NULL); // second argument is damp, default to 0
  sim_log("Finalized Field Advance");

  
  //////////////////////////////////////////////////////////////////////////////
  // Setup the species
  sim_log("Setting up species. ");
  double nmax = 5*Ni/nproc();
  double nmovers = 0.1*nmax;
  double sort_method = 1;   // 0=in place and 1=out of place
  species_t *ion = define_species("ion", ec, mi, nmax, nmovers, sort_interval, sort_method);

  
  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  sim_log( "***********************************************" );
  sim_log("* Topology:                       " << topology_x
    << " " << topology_y << " " << topology_z);
  sim_log ( "taui = " << taui );
  sim_log ( "num_step = " << num_step );
  sim_log ( "Lx = " << Lx/di );
  sim_log ( "Ly = " << Ly/di );
  sim_log ( "Lz = " << Lz/di );
  sim_log ( "pert = " << pert );
  sim_log ( "Ti = " << Ti );
  sim_log ( "gamma = " << gamma );
  sim_log ( "Te = " << Te );
  sim_log ( "nx = " << nx );
  sim_log ( "ny = " << ny );
  sim_log ( "nz = " << nz );
  sim_log ( "nproc = " << nproc ()  );
  sim_log ( "nppc = " << nppc );
  sim_log ( "b0 = " << b0 );
  sim_log ( "v_A = " << v_A );
  sim_log ( "di = " << di );
  sim_log ( "Ni = " << Ni );
  sim_log ( "total # of particles = " << Ni );
  sim_log ( "dt*wci = " << wci*dt );
  sim_log ( "energies_interval: " << energies_interval );
  sim_log ( "dx = " << Lx/(di*nx) );
  sim_log ( "dy = " << Ly/(di*ny) );
  sim_log ( "dz = " << Lz/(di*nz) );
  sim_log ( "n0 = " << n0 );

 // Dump simulation information to file "info.bin" for translate script
  if (rank() == 0 ) {

    FileIO fp_info;

    // write binary info file

    if ( ! (fp_info.open("info.bin", io_write)==ok) ) ERROR(("Cannot open file."));
    
    fp_info.write(&topology_x, 1 );
    fp_info.write(&topology_y, 1 );
    fp_info.write(&topology_z, 1 );

    fp_info.write(&Lx, 1 );
    fp_info.write(&Ly, 1 );
    fp_info.write(&Lz, 1 );

    fp_info.write(&nx, 1 );
    fp_info.write(&ny, 1 );
    fp_info.write(&nz, 1 );

    fp_info.write(&dt, 1 );

    fp_info.close();

}


  ////////////////////////////
  // Load fields
sim_log( "Loading fields" );

// Note: everywhere is a region that encompasses the entire simulation                                                                                                                   
// In general, regions are specied as logical equations (i.e. x>0 && x+y<2) 
//11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
  // Load fields

  // Define wavevectors

  double kx0 = 2.0*M_PI/16/di;
  double ky0 = 2.0*M_PI/16/di;
  double kz0 = 2.0*M_PI/16/di;

    const int re_num = 1;
  double kxmin = 0; double kxmax = 0;  //floor(2.0*M_PI/4.0/hx/kx0);
  double kymin = 0; double kymax = 0;  //floor(2.0*M_PI/4.0/hy/ky0);
  double kzmin = 1; double kzmax = 1;  //floor(2.0*M_PI/4.0/hz/kz0);
  double phi_min = 0; double phi_max = 6.28;

  const int num_samples_x = (kxmax-kxmin+1);
  const int num_samples_y = (kymax-kymin+1);
  const int num_samples_z = (kzmax-kzmin+1);
  const int num_samples = num_samples_x * num_samples_y * num_samples_z * re_num;
  
  std::vector<double> kx_random_temp(num_samples_x);
  std::vector<double> ky_random_temp(num_samples_y);
  std::vector<double> kz_random_temp(num_samples_z);
  std::vector<double> kx_random(num_samples);
  std::vector<double> ky_random(num_samples);
  std::vector<double> kz_random(num_samples);
  std::vector<double> phi_random(num_samples);
  std::vector<double> amplitude_ratio(num_samples);
  // double amplitude_ratio_sum = 0;  


if (rank() == 0) {
  kx_random_temp = linspace(kxmin, kxmax, num_samples_x);
  ky_random_temp = linspace(kymin, kymax, num_samples_y);
  kz_random_temp = linspace(kzmin, kzmax, num_samples_z);
  int idx = 0;

  for  (int repeat = 0; repeat < re_num; repeat++){
  for (int i = 0; i < num_samples_x; i++) {
    for (int j = 0; j < num_samples_y; j++) {
      for (int k = 0; k < num_samples_z; k++) {
        kx_random[idx]  = kx_random_temp[i]*kx0;
        ky_random[idx]  = ky_random_temp[j]*ky0;
        kz_random[idx]  = kz_random_temp[k]*kz0;
        phi_random[idx] = uniform(rng(0), phi_min, phi_max);
        if ((std::abs(kx_random[idx]) < 1e-6) && (std::abs(ky_random[idx]) < 1e-6) && (std::abs(kz_random[idx]) < 1e-6)) {
                amplitude_ratio[idx] = 0.0;
        } else {
                amplitude_ratio[idx] = 1.0;//* std::pow(k_total_temp/k_total_0, -3.0/4.0);          
	}
        idx = idx + 1;
      }
    }
  }
  }
}
  MPI_Bcast(amplitude_ratio.data(), num_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(kx_random.data(), num_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(ky_random.data(), num_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(kz_random.data(), num_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(phi_random.data(), num_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);


sim_log( "Loading fields" );

PerturbationParams params = {amplitude_ratio, kx_random, ky_random, kz_random, phi_random, num_samples, n0, global->v_A, b0x, b0y, b0z, waveamp};
// Set initial conditions for fields
set_region_field( everywhere, EX_PERT(x,y,z,params), EY_PERT(x,y,z,params), EZ_PERT(x,y,z,params), b0x+BX_PERT(x,y,z,params), b0y+BY_PERT(x,y,z,params), b0z+BZ_PERT(x,y,z,params));

// LOAD PARTICLES
sim_log( "Loading particles" );

// Do a fast load of the particles
int rng_seed = 1;     // Random number seed increment 
seed_entropy( rank() );  //Generators desynchronized
// seed_rand( rng_seed*nproc() + rank() );  //Generators desynchronized
double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);


sim_log( "-> uniform plasma + specified waves" );

repeat (Ni/nproc()) {
  double x,y,z, ux, uy, uz, r;
  x = uniform( rng(0), xmin, xmax );
  y = uniform( rng(0), ymin, ymax );
  z = uniform( rng(0), zmin, zmax );
  ux = normal( rng(0), 0, vthi) + UX_PERT(x,y,z,params);// + JX_PERT(x,y,z,params)/(n0+N_PERT(x,y,z,params));
  uy = normal( rng(0), 0, vthi) + UY_PERT(x,y,z,params);// + JY_PERT(x,y,z,params)/(n0+N_PERT(x,y,z,params));
  uz = normal( rng(0), 0, vthi) + UZ_PERT(x,y,z,params);// + JZ_PERT(x,y,z,params)/(n0+N_PERT(x,y,z,params));

  inject_particle(ion, x, y, z, ux, uy, uz, qi, 0, 0 );
}
sim_log( "Finished loading particles" );

//   // n --> mode number in X
//   // m --> mode number in Y
//   // l --> mode number in Z


// //  Alfvenic perturbation with deltaB in the x direction
// //  works only for a pair plasma
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // case 1 only inject waves and no existing waves
// #define DBX_1(l,m,phi) amp*b0*cos(l*kz0*z + m*ky0*y + phi)
// #define DEY_1(l,m,phi) -amp*(l/abs(l))*v_A*b0*cos(l*kz0*z + m*ky0*y + phi)
// // These give velocity & current consistent with Alfven wave
// #define DUX_1(l,m,phi) -amp*(l/abs(l))*v_A*cos(l*kz0*z + m*ky0*y + phi)
// #define DJY_1(l,m,phi) -amp*b0*(l*kz0)*sin(l*kz0*z + m*ky0*y + phi)
// #define DJZ_1(l,m,phi) amp*b0*(m*ky0)*sin(l*kz0*z + m*ky0*y + phi)
// // Single wave propating to the right
// #define BX_PERT_1 DBX_1(1,0,0)
// #define EY_PERT_1 DEY_1(1,0,0)
// #define UX_PERT_1 DUX_1(1,0,0)
// #define JY_PERT_1 DJY_1(1,0,0)
// #define JZ_PERT_1 DJZ_1(1,0,0)
// #define DBY_2(l,m,phi) amp*b0*cos(l*kz0*z + m*kx0*x + phi)
// #define DEX_2(l,m,phi) amp*(l/abs(l))*v_A*b0*cos(l*kz0*z + m*kx0*x + phi)
// // These give velocity & current consistent with Alfven wave
// #define DUY_2(l,m,phi) +amp*(l/abs(l))*v_A*cos(l*kz0*z + m*kx0*x + phi)
// #define DJX_2(l,m,phi) amp*b0*(l*kz0)*sin(l*kz0*z + m*kx0*x + phi)
// #define DJZ_2(l,m,phi) -amp*b0*(m*kx0)*sin(l*kz0*z + m*kx0*x + phi)
// Single wave propating to the left
// #define BY_PERT_2 DBY_2(-1,0,0.4) + DBY_2(-2,0,2.56) + DBY_2(-3,0,4.19)   
// #define EX_PERT_2 DEX_2(-1,0,0.4) + DEX_2(-2,0,2.56) + DEX_2(-3,0,4.19) 
// #define UY_PERT_2 DUY_2(-1,0,0.4) + DUY_2(-2,0,2.56) + DUY_2(-3,0,4.19)
// #define JX_PERT_2 DJX_2(-1,0,0.4) + DJX_2(-2,0,2.56) + DJX_2(-3,0,4.19)
// #define JZ_PERT_2 DJZ_2(-1,0,0.4) + DJZ_2(-2,0,2.56) + DJZ_2(-3,0,4.19)
// #define BY_PERT_2 DBY_2(1,0,0.4) 
// #define EX_PERT_2 DEX_2(1,0,0.4)
// #define UY_PERT_2 DUY_2(1,0,0.4)
// #define JX_PERT_2 DJX_2(1,0,0.4)
// #define JZ_PERT_2 DJZ_2(1,0,0.4)


//   sim_log( "Loading fields" );

//   set_region_field( everywhere, EX_PERT_2, 0.0, 0.0, b0*cn, BY_PERT_2, b0*sn);
// #define BX_PERT_1 DBX_1(1,0,0) + DBX_1(2,0,1.5) + DBX_1(3,0,3.9)   
// #define EY_PERT_1 DEY_1(1,0,0) + DEY_1(2,0,1.5) + DEY_1(3,0,3.9)  
// #define UX_PERT_1 DUX_1(1,0,0) + DUX_1(2,0,1.5) + DUX_1(3,0,3.9)
// #define JY_PERT_1 DJY_1(1,0,0) + DJY_1(2,0,1.5) + DJY_1(3,0,3.9)
// #define JZ_PERT_1 DJZ_1(1,0,0) + DJZ_1(2,0,1.5) + DJZ_1(3,0,3.9)

// //  Alfvenic perturbation with deltaB in the y direction
// //  works only for a pair plasma
// #define DBY_2(l,m,phi) amp*imf*b0*cos(l*kz0*z + m*kx0*x + phi)
// #define DEX_2(l,m,phi) amp*imf*(l/abs(l))*v_A*b0*cos(l*kz0*z + m*kx0*x + phi)
// // These give velocity & current consistent with Alfven wave
// #define DUY_2(l,m,phi) +amp*imf*(l/abs(l))*v_A*cos(l*kz0*z + m*kx0*x + phi)
// #define DJX_2(l,m,phi) amp*b0*imf*(l*kz0)*sin(l*kz0*z + m*kx0*x + phi)
// #define DJZ_2(l,m,phi) -amp*b0*imf*(m*kx0)*sin(l*kz0*z + m*kx0*x + phi)
// // Single wave propating to the left
// #define BY_PERT_2 DBY_2(-1,0,0.4) + DBY_2(-2,0,2.56) + DBY_2(-3,0,4.19)   
// #define EX_PERT_2 DEX_2(-1,0,0.4) + DEX_2(-2,0,2.56) + DEX_2(-3,0,4.19) 
// #define UY_PERT_2 DUY_2(-1,0,0.4) + DUY_2(-2,0,2.56) + DUY_2(-3,0,4.19)
// #define JX_PERT_2 DJX_2(-1,0,0.4) + DJX_2(-2,0,2.56) + DJX_2(-3,0,4.19)
// #define JZ_PERT_2 DJZ_2(-1,0,0.4) + DJZ_2(-2,0,2.56) + DJZ_2(-3,0,4.19)
 
  // sim_log( "Loading fields" );

  // set_region_field( everywhere, EX_PERT_2, EY_PERT_1, 0.0, BX_PERT_1 + b0*cn, BY_PERT_2, b0*sn);

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//case2 only inject wave and no existing wave
  //  sim_log( "Loading fields" );

  //  set_region_field( everywhere, 0.0, 0.0, 0.0, b0*cn , 0.0, b0 *sn);



 // LOAD PARTICLES
//  sim_log( "Loading particles" );

//  // Do a fast load of the particles
// //  int rng_seed     = 1;     // Random number seed increment 
//  seed_entropy( rank() );  //Generators desynchronized
//  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
//  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
//  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);

//  repeat ( Ni/nproc() ) {
//     double x, y, z, ux, uy, uz, d0 ;

//     x = uniform(rng(0),xmin,xmax);
//     y = uniform(rng(0),ymin,ymax);
//     z = uniform(rng(0),zmin,zmax);
    
//     ux = normal( rng(0), 0, vthi) + JX_PERT_2*0.5;
//     uy = normal( rng(0), 0, vthi) + ;
//     uz = normal( rng(0), 0, vthi);

//     inject_particle( ion, x, y, z, ux, uy, uz, qi, 0, 0);

//   }

//  sim_log( "Finished loading particles" );

  /*--------------------------------------------------------------------------
   * New dump definition
   *------------------------------------------------------------------------*/

  /*--------------------------------------------------------------------------
   * Set data output format
   *
   * This option allows the user to specify the data format for an output
   * dump.  Legal settings are 'band' and 'band_interleave'.  Band-interleave
   * format is the native storage format for data in VPIC.  For field data,
   * this looks something like:
   *
   *   ex0 ey0 ez0 div_e_err0 cbx0 ... ex1 ey1 ez1 div_e_err1 cbx1 ...
   *
   * Banded data format stores all data of a particular state variable as a
   * contiguous array, and is easier for ParaView to process efficiently.
   * Banded data looks like:
   *
   *   ex0 ex1 ex2 ... exN ey0 ey1 ey2 ...
   *
   *------------------------------------------------------------------------*/

  global->fdParams.format = band;
  sim_log ( "Fields output format = band" );

  global->hedParams.format = band;
  sim_log ( "Electron species output format = band" );

  global->hHdParams.format = band;
  sim_log ( "Ion species output format = band" );

  /*--------------------------------------------------------------------------
   * Set stride
   *
   * This option allows data down-sampling at output.  Data are down-sampled
   * in each dimension by the stride specified for that dimension.  For
   * example, to down-sample the x-dimension of the field data by a factor
   * of 2, i.e., half as many data will be output, select:
   *
   *   global->fdParams.stride_x = 2;
   *
   * The following 2-D example shows down-sampling of a 7x7 grid (nx = 7,
   * ny = 7.  With ghost-cell padding the actual extents of the grid are 9x9.
   * Setting the strides in x and y to equal 2 results in an output grid of
   * nx = 4, ny = 4, with actual extents 6x6.
   *
   * G G G G G G G G G
   * G X X X X X X X G
   * G X X X X X X X G         G G G G G G
   * G X X X X X X X G         G X X X X G
   * G X X X X X X X G   ==>   G X X X X G
   * G X X X X X X X G         G X X X X G
   * G X X X X X X X G         G X X X X G
   * G X X X X X X X G         G G G G G G
   * G G G G G G G G G
   *
   * Note that grid extents in each dimension must be evenly divisible by
   * the stride for that dimension:
   *
   *   nx = 150;
   *   global->fdParams.stride_x = 10; // legal -> 150/10 = 15
   *
   *   global->fdParams.stride_x = 8; // illegal!!! -> 150/8 = 18.75
   *------------------------------------------------------------------------*/

  //  // relative path to fields data from global header
  //  sprintf(global->fdParams.baseDir, "fields");
  //  // base file name for fields output
  //  sprintf(global->fdParams.baseFileName, "fields");
   sprintf(global->fdParams.baseDir, "fields/");
   dump_mkdir("fields");
   dump_mkdir(global->fdParams.baseDir);
   // base file name for fields output
   sprintf(global->fdParams.baseFileName, "fields");

  global->fdParams.stride_x = 1;
  global->fdParams.stride_y = 1;
  global->fdParams.stride_z = 1;

  // add field parameters to list
  global->outputParams.push_back(&global->fdParams);

  sim_log ( "Fields x-stride " << global->fdParams.stride_x );
  sim_log ( "Fields y-stride " << global->fdParams.stride_y );
  sim_log ( "Fields z-stride " << global->fdParams.stride_z );

  //  // relative path to electron species data from global header
  //sprintf(global->hedParams.baseDir, "hydro");
  //

  //  // relative path to electron species data from global header
  sprintf(global->hHdParams.baseDir, "hydro");
  //sprintf(global->hHdParams.baseDir, "hydro/%d",NUMFOLD);
  dump_mkdir("hydro");
  dump_mkdir(global->hHdParams.baseDir);

  //// base file name for fields output
  //sprintf(global->hHdParams.baseFileName, "Hhydro");

  // base file name for fields output
  sprintf(global->hHdParams.baseFileName, "Hhydro");

  global->hHdParams.stride_x = 1;
  global->hHdParams.stride_y = 1;
  global->hHdParams.stride_z = 1;

  sim_log ( "Ion species x-stride " << global->hHdParams.stride_x );
  sim_log ( "Ion species y-stride " << global->hHdParams.stride_y );
  sim_log ( "Ion species z-stride " << global->hHdParams.stride_z );

  // add electron species parameters to list
  global->outputParams.push_back(&global->hHdParams);

  /*--------------------------------------------------------------------------
   * Set output fields
   *
   * It is now possible to select which state-variables are output on a
   * per-dump basis.  Variables are selected by passing an or-list of
   * state-variables by name.  For example, to only output the x-component
   * of the electric field and the y-component of the magnetic field, the
   * user would call output_variables like:
   *
   *   global->fdParams.output_variables( ex | cby );
   *
   * NOTE: OUTPUT VARIABLES ARE ONLY USED FOR THE BANDED FORMAT.  IF THE
   * FORMAT IS BAND-INTERLEAVE, ALL VARIABLES ARE OUTPUT AND CALLS TO
   * 'output_variables' WILL HAVE NO EFFECT.
   *
   * ALSO: DEFAULT OUTPUT IS NONE!  THIS IS DUE TO THE WAY THAT VPIC
   * HANDLES GLOBAL VARIABLES IN THE INPUT DECK AND IS UNAVOIDABLE.
   *
   * For convenience, the output variable 'all' is defined:
   *
   *   global->fdParams.output_variables( all );
   *------------------------------------------------------------------------*/
  /* CUT AND PASTE AS A STARTING POINT
   * REMEMBER TO ADD APPROPRIATE GLOBAL DUMPPARAMETERS VARIABLE

   output_variables( all );

   output_variables( electric | div_e_err | magnetic | div_b_err |
                     tca      | rhob      | current  | rhof |
                     emat     | nmat      | fmat     | cmat );

   output_variables( current_density  | charge_density |
                     momentum_density | ke_density     | stress_tensor );
   */

  //global->fdParams.output_variables( electric | magnetic );
  global->hedParams.output_variables( current_density | charge_density | stress_tensor );
  global->hHdParams.output_variables( current_density | charge_density | stress_tensor );


  global->fdParams.output_variables( all );
// global->hedParams.output_variables( all );
// global->hHdParams.output_variables( all );

  /*--------------------------------------------------------------------------
   * Convenience functions for simlog output
   *------------------------------------------------------------------------*/

  char varlist[512];
  create_field_list(varlist, global->fdParams);

  sim_log ( "Fields variable list: " << varlist );

  //create_hydro_list(varlist, global->hedParams);

  //sim_log ( "Electron species variable list: " << varlist );

  create_hydro_list(varlist, global->hHdParams);

  sim_log ( "Ion species variable list: " << varlist );

  sim_log("*** Finished with user-specified initialization ***");

  // Upon completion of the initialization, the following occurs:
  // - The synchronization error (tang E, norm B) is computed between domains
  //   and tang E / norm B are synchronized by averaging where discrepancies
  //   are encountered.
  // - The initial divergence error of the magnetic field is computed and
  //   one pass of cleaning is done (for good measure)
  // - The bound charge density necessary to give the simulation an initially
  //   clean divergence e is computed.
  // - The particle momentum is uncentered from u_0 to u_{-1/2}
  // - The user diagnostics are called on the initial state
  // - The physics loop is started
  //
  // The physics loop consists of:
  // - Advance particles from x_0,u_{-1/2} to x_1,u_{1/2}
  // - User particle injection at x_{1-age}, u_{1/2} (use inject_particles)
  // - User current injection (adjust field(x,y,z).jfx, jfy, jfz)
  // - Advance B from B_0 to B_{1/2}
  // - Advance E from E_0 to E_1
  // - User field injection to E_1 (adjust field(x,y,z).ex,ey,ez,cbx,cby,cbz)
  // - Advance B from B_{1/2} to B_1
  // - (periodically) Divergence clean electric field
  // - (periodically) Divergence clean magnetic field
  // - (periodically) Synchronize shared tang e and norm b
  // - Increment the time step
  // - Call user diagnostics
  // - (periodically) Print a status message

} //begin_initialization

#define should_dump(x)                                                  \
  (global->x##_interval>0 && remainder(step(), global->x##_interval) == 0)

begin_diagnostics {

  /*--------------------------------------------------------------------------
   * NOTE: YOU CANNOT DIRECTLY USE C FILE DESCRIPTORS OR SYSTEM CALLS ANYMORE
   *
   * To create a new directory, use:
   *
   *   dump_mkdir("full-path-to-directory/directoryname")
   *
   * To open a file, use: FileIO class
   *
   * Example for file creation and use:
   *
   *   // declare file and open for writing
   *   // possible modes are: io_write, io_read, io_append,
   *   // io_read_write, io_write_read, io_append_read
   *   FileIO fileIO;
   *   FileIOStatus status;
   *   status= fileIO.open("full-path-to-file/filename", io_write);
   *
   *   // formatted ASCII  output
   *   fileIO.print("format string", varg1, varg2, ...);
   *
   *   // binary output
   *   // Write n elements from array data to file.
   *   // T is the type, e.g., if T=double
   *   // fileIO.write(double * data, size_t n);
   *   // All basic types are supported.
   *   fileIO.write(T * data, size_t n);
   *
   *   // close file
   *   fileIO.close();
   *------------------------------------------------------------------------*/

  /*--------------------------------------------------------------------------
   * Data output directories
   * WARNING: The directory list passed to "global_header" must be
   * consistent with the actual directories where fields and species are
   * output using "field_dump" and "hydro_dump".
   *
   * DIRECTORY PATHES SHOULD BE RELATIVE TO
   * THE LOCATION OF THE GLOBAL HEADER!!!
   *------------------------------------------------------------------------*/

  // Adam: Can override some global params here
  // num_step = 171000;
  //global->fields_interval = 1358;
  //global->ehydro_interval = 1358;
  //global->Hhydro_interval = 1358;

  global->restart_interval = 1000;
  global->quota_sec = 23.5*3600.0;

  //  const int nsp=global->nsp;
  const int nx=grid->nx;
  const int ny=grid->ny;
  const int nz=grid->nz;

  /*--------------------------------------------------------------------------
   * Normal rundata dump
   *------------------------------------------------------------------------*/
  if(step()==0) {
    dump_mkdir("fields");
    dump_mkdir("hydro");
    dump_mkdir("rundata");
    dump_mkdir("restore0");
    dump_mkdir("restore1");  // 1st backup
    dump_mkdir("particle");
    dump_mkdir("rundata");

    
    // Make subfolders for restart
    //    char restorefold[128];
    //sprintf(restorefold, "restore0/%i", NUMFOLD);
    //    sprintf(restorefold, "restore0");
    //    dump_mkdir(restorefold);
    //    sprintf(restorefold, "restore1/%i", NUMFOLD);
    //    sprintf(restorefold, "restore1");
    //    dump_mkdir(restorefold);
    //    sprintf(restorefold, "restore2/%i", NUMFOLD);
    //    dump_mkdir(restorefold);

    // And rundata 
    //    char rundatafold[128];
    //    char rundatafile[128];
    //    sprintf(rundatafold, "rundata/%i", NUMFOLD);
    ///    sprintf(rundatafold, "rundata");
    //    dump_mkdir(rundatafold);

    dump_grid("rundata/grid");
    //    sprintf(rundatafile, "rundata/%i/grid", NUMFOLD);
    //    dump_grid(rundatafile);

    dump_materials("rundata/materials");
    dump_species("rundata/species");
    global_header("global", global->outputParams);
  } // if

  /*--------------------------------------------------------------------------
   * Normal rundata energies dump
   *------------------------------------------------------------------------*/
  if(should_dump(energies)) {
    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
  } // if

  /*--------------------------------------------------------------------------
   * Field data output
   *------------------------------------------------------------------------*/

  if(step() == 1 || should_dump(fields)) field_dump(global->fdParams);

  /*--------------------------------------------------------------------------
   * Electron species output
   *------------------------------------------------------------------------*/

  //if(should_dump(ehydro)) hydro_dump("electron", global->hedParams);

  /*--------------------------------------------------------------------------
   * Ion species output
   *------------------------------------------------------------------------*/

  if(should_dump(Hhydro)) hydro_dump("ion", global->hHdParams);


  /*--------------------------------------------------------------------------
  * Time averaging
  *------------------------------------------------------------------------*/

  //  #include "time_average.cxx"
   //#include "time_average_cori.cxx"

  /*--------------------------------------------------------------------------
   * Restart dump
   *------------------------------------------------------------------------*/

  if(step() && !(step()%global->restart_interval)) {
    global->write_restart = 1; // set restart flag. the actual restart files are written during the next step
  } else {
    if (global->write_restart) {

      global->write_restart = 0; // reset restart flag
      double dumpstart = uptime();
      if(!global->rtoggle) {
        global->rtoggle = 1;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
	checkpt("restore1/restore", 0);
	//	DUMP_INJECTORS(1);
	//    } END_TURNSTILE;
      } else {
        global->rtoggle = 0;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
	checkpt("restore0/restore", 0);
	//	DUMP_INJECTORS(0);
	//    } END_TURNSTILE;
      } // if

      //    mp_barrier();
      sim_log( "Restart dump completed");
      double dumpelapsed = uptime() - dumpstart;
      sim_log("Restart duration "<<dumpelapsed);
    } // if global->write_restart
  }


    // Dump particle data

    // char subdir[36];
    // char subdir2[36];
    // if ( should_dump(Hparticle) && step() !=0
    //      && step() > 45*(global->fields_interval)  ) {
    //   sprintf(subdir2, "particle/T.%ld", step());
    //   dump_mkdir(subdir2);
    //   sprintf(subdir,"particle/T.%ld/Hparticle",step());
    //   dump_particles("ion", subdir);
    //   }

  // Shut down simulation when wall clock time exceeds global->quota_sec.
  // Note that the mp_elapsed() is guaranteed to return the same value for all
  // processors (i.e., elapsed time on proc #0), and therefore the abort will
  // be synchronized across processors. Note that this is only checked every
  // few timesteps to eliminate the expensive mp_elapsed call from every
  // timestep. mp_elapsed has an ALL_REDUCE in it!


  if ( (step()>0 && global->quota_check_interval>0
        && (step() & global->quota_check_interval)==0 ) || (global->write_end_restart) ) {

    if ( (global->write_end_restart) ) {
      global->write_end_restart = 0; // reset restart flag

      //   if( uptime() > global->quota_sec ) {
      sim_log( "Allowed runtime exceeded for this job.  Terminating....\n");
      double dumpstart = uptime();

      if(!global->rtoggle) {
        global->rtoggle = 1;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
        checkpt("restore1", 0);
        //    } END_TURNSTILE;
      } else {
        global->rtoggle = 0;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
        checkpt("restore0", 0);
        //    } END_TURNSTILE;
      } // if

      mp_barrier(  ); // Just to be safe
      sim_log( "Restart dump restart completed." );
      double dumpelapsed = uptime() - dumpstart;
      sim_log("Restart duration "<< dumpelapsed);
      exit(0); // Exit or abort?                                                                                
    }
    //    } 
    if( uptime() > global->quota_sec ) global->write_end_restart = 1;
  }


} // end diagnostics

// ***********  PARTICLE INJECTION  - OPEN BOUNDARY *********
begin_particle_injection {
} // end particle injection

//*******************  CURRENT INJECTION ********************
begin_current_injection {
} // end current injection

//*******************  FIELD INJECTION **********************
begin_field_injection{

}// end field injection

// begin_field_injection {
//   const int nx=grid->nx;
//   const int ny=grid->ny;
//   const int nz=grid->nz;
//   int x,y,z;
//   double dt = grid->dt;
//   double v_A = global->v_A;

//   // There macros are from local.c to apply boundary conditions
// #define XYZ_LOOP(xl,xh,yl,yh,zl,zh)             \
//   for( z=zl; z<=zh; z++ )                       \
//     for( y=yl; y<=yh; y++ )                     \
//       for( x=xl; x<=xh; x++ )

// #define yz_EDGE_LOOP(x) XYZ_LOOP(x,x,0,ny+1,0,1+nz)
//   double w_inj = 0.7*M_PI;          // inject alfven wave freq
//   double current_tt = step()*dt;
//   double ampp = 0.01;
//   double cbyy = ampp*cos(w_inj*current_tt);
//   double exx = ampp*v_A*cos(w_inj*current_tt);
//   // std::cout<<" step() "<<step()<<" dt "<<dt<<" cbzz "<<cbzz<<std::endl;
//   // Right Boundary
//   if (step() > 20000){
//     if (global->down) {
//       XYZ_LOOP(0,nx+1,0,ny+1,0,1) field(x,y,z).cby = field(x,y,z).cby + cbyy;
//       XYZ_LOOP(0,nx+1,0,ny+1,0,1) field(x,y,z).ex = field(x,y,z).ex + exx;
//     }
//   }
// }  // end field injection


//*******************  COLLISIONS ***************************
begin_particle_collisions {
} // end collisions

