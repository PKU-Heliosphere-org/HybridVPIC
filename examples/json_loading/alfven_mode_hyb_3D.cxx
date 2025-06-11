//////////////////////////////////////////////////////
//    Alfven-mode-hyb-3D.cxx
//    code for 3D hybrid simulation
//    Author:  Chuanpeng Hou
//    Date:    2025.05.28
//    Version: 1.0
//////////////////////////////////////////////////////

// #define NUM_TURNSTILES 16384
#include <mpi.h>
#include <iostream>
#include <ifstream>
#include <vector>
#include <cmath>  // for round()
#include "alfven_mode_polarization_3D.cxx"
#include "nlohmann/json.hpp"
//////////////////////////////////////////////////////

using json = nlohmann::json;

namespace settings {
  struct PlasmaConstSetting {
    double b0 = 1.0;
    double n0 = 1.0;
    double eb0_x = 0.0, eb0_y = 0.0, eb0_z = 1.0;
    double mu0 = 1.0;
    double mi = 1.0;
    double ec = 1.0;

    inline double get_b0x() {return b0 * eb0_x; }
    inline double get_b0y() {return b0 * eb0_y; }
    inline double get_b0z() {return b0 * eb0_z; }
    inline double get_va() {return b0 / sqrt(mu0 * n0 * mi); }
    /*
      Chresimopoioume to sosto gramma ellenico. -- We use the correct Greek letter.
      Auto einai ena omega. -- That is an omega.
      Auto den einai ena gramma latinico 'w'. -- That is not a Latin letter w.
    */
    inline double get_omega_ci() {return ec * b0 / mi; }
    //! @deprecated
    inline double get_wci() {return get_omega_ci(); }
    inline double get_di() {return get_va() / get_omega_ci(); }
  };

  struct PlasmaFeatureSetting {
    double Vd_Va = 0.0;  // Alfven Mach number.
    double Ti_Te = 0.0;
    double beta_i = 0.5;
    double gamma_plasma = 5.0 / 3.0;  // Ratio of specific heats, but avoid the same name from special functions
    double eta = 0.001;
    double hyper_eta = 0.005;
  };

  struct WaveSetting {
    double waveamp;
  };

  struct SimuSetting {
    double taui = 200;  // Simulation run time in 1 / omega_ci.
    double Lx_di = 50;  // Length of regions in di.
    double Ly_di = 50;
    double Lz_di = 50;

    double quota_h = 23.5;

    double nx = 256;
    double ny = 256;
    double nz = 256;
    double nppc = 2'000;

    double topology_x = 32;
    double topology_y = 16;
    double topology_z = 16;

    double sort_interval = 20;

    int restart_interval = 2'000;
    int energies_interval = 100;

    int fields_interval_rel = 1;
    int shydro_interval_rel = 1;
    int Hhydro_interval_rel = 1;
    int eparticle_interval_rel = 10;
    int Hparticle_interval_rel = 0;
    int quota_check_interval_rel = 10;

    inline double get_quota_sec() {return quota_h * 3'600.0; }
    inline double get_Ni() {return trunc_granular(nppc * nx * ny * nz, nproc()); }
  };

  struct SettingTerms {
    PlasmaConstSetting consts;
    WaveSetting wave;
    PlasmaFeatureSetting plasma;
    SimuSetting rc;

    // Derived numerical parameters
    inline double get_Lx() {return rc.Lx_di * consts.get_di(); }
    inline double get_Ly() {return rc.Ly_di * consts.get_di(); }
    inline double get_Lz() {return rc.Lz_di * consts.get_di(); }
    inline double get_hx() {return rc.Lx_di * consts.get_di() / rc.nx; }
    inline double get_hy() {return rc.Ly_di * consts.get_di() / rc.ny; }
    inline double get_hz() {return rc.Lz_di * consts.get_di() / rc.nz; }
    inline double get_Np() {return consts.n0 * rc.Lx_di * rc.Ly_di * rc.Lz_di * pow(consts.get_di(), 3.0); }
    inline double get_qi() {return consts.ec * get_Np() / rc.get_Ni(); }
    inline double get_nfac() {return get_qi() / (get_hx() * get_hy() * get_hz()); }

    inline double get_dg() {return courant_length(get_Lx(), get_Ly(), get_Lz(), rc.nx, rc.ny, rc.nz);}
    inline double get_dt() {return 0.5 * get_dg() / 16.0; }
    inline int get_interval() {return int(20.0 / (consts.get_omega_ci() * get_dt())); }

#define JSTEIN_INTERVAL_DEFINE_MACHINE(name_prefix) \
    inline int get_##name_prefix##_interval() {return get_interval() * rc.name_prefix##_interval_rel; }

    JSTEIN_INTERVAL_DEFINE_MACHINE(fields)
    JSTEIN_INTERVAL_DEFINE_MACHINE(shydro)
    JSTEIN_INTERVAL_DEFINE_MACHINE(Hhydro)
    JSTEIN_INTERVAL_DEFINE_MACHINE(eparticle)
    JSTEIN_INTERVAL_DEFINE_MACHINE(Hparticle)
    JSTEIN_INTERVAL_DEFINE_MACHINE(quota_check)

#undef JSTEIN_INTERVAL_DEFINE_MACHINE
  };
} // namespace settings

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

# define NUM_SPECS (1)

//  Variables for Open BC Model
  int nsp;          //  Number of Species
  double npleft[NUM_SPECS];  // Left Densities
  double npright[NUM_SPECS];  // Right Densities
  double vth[NUM_SPECS];    // Thermal velocity
  double q[NUM_SPECS];      // Species charge
  double ur;       //  Fluid velociy on right
  double ul;       //  Fluid velociy on left
  double nfac;      //  Normalization factor to convert particles per cell into density
  double sn;       //sin(theta)

  int left,right;  // Keep track of boundary domains
  double *nleft, *uleft, *pleft, *bleft, *fleft;     // Moments for left injectors
  double *nright, *uright, *pright, *bright, *fright; // Moments for right injectors

  // Output variables
  DumpParameters fdParams;
  DumpParameters hHdParams;
  std::vector<DumpParameters *> outputParams;

};


begin_initialization {

  // Load the JSON file.
  std::ifstream f_json("simu_setup.json");
  json j = json::parse(f_json);
  auto params = j.template get<settings::SettingTerms>();

  // Use natural hybrid-PIC units:
  double ec   = 1.0;  // Charge normalization
  double mi   = 1.0;  // Mass normalization
  double mu0  = 1.0;  // Magnetic constanst
  double b0 = 1.0;    // Magnetic field
  double n0 = 1.0;    // Density
  double b0x = b0 * 0.0;
  double b0y = b0 * 0.0;
  double b0z = b0 * 1.0;
  double waveamp = 0.02;
  // Derived normalization parameters:
  double v_A = b0/sqrt(mu0*n0*mi); // Alfven velocity
  double wci = ec*b0/mi;           // Cyclotron freq
  double di = v_A/wci;             // Ion skin-depth


  // Initial conditions for model:
  double Vd_Va    = 0.0;             // Alfven Mach number
  double Ti_Te    = 1.0/1.0;         // Ion temperature / electron temperature
  double beta_i   = 0.5;             // Background ion beta
  double gamma    = 5.0/3.0;         // Ratio of specific heats
  double eta = 0.001;                // Plasma resistivity.
  double hypereta = 0.005;           // Plasma hyper-resistivity.

  // Derived quantities for model:
  double Ti = beta_i*b0*b0/2.0/n0;
  double vthi = sqrt(Ti/mi);
  double Te = Ti/Ti_Te;
  double Vd = Vd_Va*v_A;
  double Cs = sqrt(gamma*(1+1/Ti_Te)*beta_i/2)*v_A;  // reference: Shestov et al., A&A 2022 https://doi.org/10.1051/0004-6361/202142362

  // Numerical parameters
  double taui = 200;    // Simulation run time in wci^-1.

  double Lx = 50*di;    // size of box in x dimension
  double Ly = 50*di;    // size of box in y dimension
  double Lz = 50*di;    // size of box in z dimension

  double quota = 23.5;   // run quota in hours
  double quota_sec = quota*3600;  // Run quota in seconds

  double nx = 256;
  double ny = 256;
  double nz = 256;
  double nppc = 2000;             // Average number of macro particle per cell per species

  double topology_x = 32;         // Number of domains in x, y, and z
  double topology_y = 16;
  double topology_z = 16;

  // Derived numerical parameters
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;

  double Ni = nppc*nx*ny*nz;          // Total macroparticle ions in box
  double Np = n0*Lx*Ly*Lz;            // Total physical ions.

  Ni = trunc_granular(Ni,nproc());    // Make it divisible by number of processors

  double qi = ec*Np/Ni;               // Charge per macro ion

  double nfac = qi/(hx*hy*hz);        // Convert density to particles per cell

  // Determine the time step
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);  // courant length
  double dt = 0.5*dg/16.0;                      // courant limited time step
  double sort_interval = 20;  // How often to sort particles

  // Intervals for outputstd::cout<<from_wci_to_wc<<std::endl;
  int restart_interval = 2000;
  int energies_interval = 100;
  int interval = int(20/(wci*dt)); // interval for saving data.
  // int interval = int(1.0/(wci*dt));
  int fields_interval = interval;
  int ehydro_interval = interval;
  int Hhydro_interval = interval;
  int eparticle_interval = 10*interval;
  int Hparticle_interval = 0*interval;
  int quota_check_interval = 10*interval;

  ///////////////////////////////////////////////
  // Setup high level simulation parameters
  num_step             = int(taui/(wci*dt));
  status_interval      = interval;
  sync_shared_interval = status_interval/2;
  clean_div_e_interval = status_interval/2;
  clean_div_b_interval = status_interval/2;

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

  global->b0  = b0;
  global->v_A = v_A;

  global->topology_x  = topology_x;
  global->topology_y  = topology_y;
  global->topology_z  = topology_z;


  //////////////////////////////////////////////////////////////////////////////
  // Setup the grid

  // Setup basic grid parameters
  define_units(1.0, 1.0);//c, eps0 );
  define_timestep( dt );

  // Define the grid
  define_periodic_grid(  -0.5*Lx,  -0.5*Ly, -0.5*Lz,    // Low corner
                          0.5*Lx,   0.5*Ly,  0.5*Lz,     // High corner
                         nx, ny, nz,             // Resolution
                         topology_x, topology_y, topology_z); // Topology

  grid->te = Te;
  grid->den = n0*mi;
  grid->eta = eta;
  grid->hypereta = hypereta;
  // if (right) grid->hypereta=0.02;
  grid->gamma = gamma;

  grid->nsub = 1;
  grid->nsm = 2;
  // grid->nsmb = 200;

  //////////////////////////////////////////////////////////////////////////////
  // Setup materials
  sim_log("Setting up materials. ");
  define_material( "vacuum", 1 );
  define_field_array(NULL); // second argument is damp, default to 0

  sim_log("Finalized Field Advance");


  //////////////////////////////////////////////////////////////////////////////
  // Setup the species
  sim_log("Setting up species. ");
  double nmax = 20.0*Ni/nproc();
  double nmovers = 0.1*nmax;
  double sort_method = 1;   // 0=in place and 1=out of place
  species_t *ion = define_species("ion", ec, mi, nmax, nmovers, sort_interval, sort_method);

  ////////////////////////////////////cd///////////////
  // Log diagnostic information about this simulation

  sim_log( "***********************************************" );
  sim_log("* Topology:                       " << topology_x
    << " " << topology_y << " " << topology_z);
  sim_log ( "Vd/Va = " << Vd_Va ) ;
  sim_log ( "beta_i = " << beta_i );
  // sim_log ( "theta = " << theta );
  sim_log ( "Ti/Te = " << Ti_Te );
  sim_log ( "taui = " << taui );
  sim_log ( "num_step = " << num_step );
  sim_log ( "Lx/di = " << Lx/di );
  sim_log ( "Ly/di = " << Ly/di );
  sim_log ( "Lz/di = " << Lz/di );
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
  sim_log ( "dx/di = " << Lx/(di*nx) );
  sim_log ( "dy/di = " << Ly/(di*ny) );
  sim_log ( "dz/di = " << Lz/(di*nz) );
  sim_log ( "dx/rhoi = " << (Lx/nx)/(vthi/wci)  );
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

    fp_info.write(&status_interval, 1 );
    fp_info.close();

}


  ////////////////////////////
  // Load fields
  // Define wavevectors
  double x,y,z;

  double kx0 = 2.0*M_PI/Lx;
  double ky0 = 2.0*M_PI/Ly;
  double kz0 = 2.0*M_PI/Lz;

  const int re_num = 1; // 重复注入波动次数，相位随机。
  double kxmin = -2; double kxmax = 2;  //floor(2.0*M_PI/4.0/hx/kx0);
  double kymin = -2; double kymax = 2;  //floor(2.0*M_PI/4.0/hy/ky0);
  double kzmin = -2; double kzmax = 2;  //floor(2.0*M_PI/4.0/hz/kz0);
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
                  amplitude_ratio[idx] = 1.0;//* std::pow(k_total_temp/k_total_0, -3.0/4.0);//使用这个参数设置功率谱，默认注入等振幅波动。
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

// Load plasma

sim_log( "-> uniform plasma + specified waves" );

repeat (Ni/nproc()) {
  double x,y,z, ux, uy, uz, r;
  x = uniform( rng(0), xmin, xmax );
  y = uniform( rng(0), ymin, ymax );
  z = uniform( rng(0), zmin, zmax );
  ux = normal( rng(0), 0, vthi) + UX_PERT(x,y,z,params);
  uy = normal( rng(0), 0, vthi) + UY_PERT(x,y,z,params);
  uz = normal( rng(0), 0, vthi) + UZ_PERT(x,y,z,params);
  inject_particle(ion, x, y, z, ux, uy, uz, qi, 0, 0 );
}
sim_log( "Finished loading particles" );

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

  //  global->hedParams.format = band;
  //  sim_log ( "Electron species output format = band" );

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
  dump_mkdir("/fields/");
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
  sprintf(global->hHdParams.baseDir, "hydro/");
  //sprintf(global->hHdParams.baseDir, "hydro/%d",NUMFOLD);
  dump_mkdir("hydro/");
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
  //  global->hedParams.output_variables( current_density | charge_density | stress_tensor );
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
// #include <FileIO.hxx>
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

  global->restart_interval = 1400;
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
    dump_mkdir("injectors");
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

  int nsp = global->nsp;

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
	//DUMP_INJECTORS(1);
	//    } END_TURNSTILE;
      } else {
        global->rtoggle = 0;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
	checkpt("restore0/restore", 0);
	//DUMP_INJECTORS(0);
	//    } END_TURNSTILE;
      } // if

      //    mp_barrier();
      sim_log( "Restart dump completed");
      double dumpelapsed = uptime() - dumpstart;
      sim_log("Restart duration "<<dumpelapsed);
    } // if global->write_restart
  }


  /*  // Dump particle data

  char subdir[36];

  if ( should_dump(Hparticle) && step() !=0
       && step() > 56*(global->fields_interval)  ) {
    sprintf(subdir,"particle/T.%d/Hparticle",step());
    dump_particles("ion", subdir);
    }*/

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
begin_field_injection {

}  // end field injection


//*******************  COLLISIONS ***************************
begin_particle_collisions {
} // end collisions




