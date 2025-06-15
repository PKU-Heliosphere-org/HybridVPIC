//////////////////////////////////////////////////////
//
//   3D Cylindrical MHD Kink Mode Simulation
//////////////////////////////////////////////////////

//#define NUM_TURNSTILES 16384
#include <mpi.h>
#include <math.h>
#include <list>
#include <iterator>
#include <vector>
#include <cmath>
#include "vpic/dumpmacros.h"
#include "kink_mode_polarization.cxx"
//#include "injection.cxx" //  Routines to compute re-injection velocity 
//#include "f_core_plus_beam.cxx"//Inject core+beam distribution of ions
//#include "tracer.hh"
//#include "time_average_master.hh"
//#include "dumptracer_hdf5_single.cc"
#define NUMFOLD (rank()/16)
 
//////////////////////////////////////////////////////
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
  int num_i_tracer;
  int num_alpha_tracer;
  int num_pui_tracer;   // tracer index
  int i_particle;   // ion particle index
  int alpha_particle;   // alpha particle index
  int pui_particle;   // pui particle index

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
  double bg;
  double v_A;
  double topology_x;       // domain topology
  double topology_y;
  double topology_z;
  int restart_step;     // time step for restart dump
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
  double n_inject[NUM_SPECS];
  // Output variables
  DumpParameters fdParams;
  DumpParameters hHdParams;
  //DumpParameters hHdParams_b;
  std::vector<DumpParameters *> outputParams;

    // particle tracking
  int tracer_interval;         // tracer info is saved or dumped every tracer_interval steps
  int tracer_pass1_interval;   // tracer interval for the 1st run. A multiple of tracer_interval
  int tracer_pass2_interval;   // tracer interval for the re-run. A multiple of tracer_interval
  int tracer_file_interval;    // interval when multiple frames of tracers are saved in the same file
  int emf_at_tracer;           // 0 or 1, electric and magnetic fields at tracer
  int hydro_at_tracer;         // 0 or 1, hydro fields at tracer
  int ve_at_tracer;            // 0 or 1, electron bulk velocity at tracer
  int num_tracer_fields_add;   // additional number of tracer fields
  int particle_tracing;
  int particle_select;
  int i_particle_select;
  int alpha_particle_select;
  int pui_particle_select;
  int particle_tracing_start;
  int dump_traj_directly;
  //species_t *tracers_list;
  int tag;
  double mi_me;
  int Ntracer;
  double PUI_flux;
  double PUI_flux_normalized;
  double M;
  // double alpha_PUI;
  // double r;
  // double Vc; 
  int stride_particle_dump;  // stride for particle dump

};

begin_initialization {
  
  // Use natural hybrid-PIC units:
  double ec   = 1.0;  // Charge normalization
  double mi   = 1.0;  // Mass normalization
  double mi_me = 10.0; //
  double me = mi/mi_me;
  double mu0  = 1.0;  // Magnetic constanst
  double b0 = 1.0;    // Magnetic field
  double bg = 0.2;
  double n0 = 1.0;    // Density
  double b0x = b0*0.0;
  double b0y = b0*0.0;
  double b0z = b0*1.0;
  double waveamp = 0.1;
  

  
  // Derived normalization parameters:
  double v_A = b0/sqrt(mu0*n0*mi); // Alfven velocity

  double wci = ec*b0/mi;          // Cyclotron freq
  double di = v_A/wci;            // Ion skin-depth

  
  // Cylindrical simulation parameters
  double R_cylinder = 0.0;       // Cylinder radius (set below)
  double B_in = 1.0;             // Magnetic field inside cylinder
  double B_out = sqrt(1.28);            // Magnetic field outside cylinder
  double b_ratio = B_out/B_in; // Magnetic field ratio inside/outside cylinder

  double n_in = n0;             // Density inside cylinder
  double n_out = 0.2;            // Density outside cylinder (set by pressure balance)
  double beta_in = 0.3;
  double beta_out = B_in*B_in/B_out/B_out*(1+beta_in)-1;//0.01;
  double T_in = beta_in*B_in*B_in/2/n_in;//0.15;              // Ion temperature (uniform)
  double T_out = beta_out*B_out*B_out/2/n_out;//0.025;              // Electron temperature (uniform)
  double perturbation_amplitude = 0.05; // Velocity perturbation amplitude
  int m_mode = 1;                // Azimuthal mode number
  double kz = 2.0*M_PI/128.0/di; // Axial wave number
  double x0 = 0.1/kz;
  double gamma = 5.0/3.0;        // adiabatic index
  double va_in = v_A;
  double va_out = B_out/sqrt(mu0*n_out*mi);
  double Cs_in = sqrt(gamma*beta_in/2)*va_in;
  double Cs_out = sqrt(gamma*beta_out/2)*va_out;
  double vthi_in = sqrt(T_in/mi);
  double vthi_out = sqrt(T_out/mi);
  // Numerical parameters
  double taui    = 50;           // Simulation run time in wci^-1.
  double quota   = 23.5;         // run quota in hours
  double quota_sec = quota*3600; // Run quota in seconds
  
  // Domain size - cubic domain
  double Lx    = 64*di;          // size of box in x dimension
  double Ly    = 1*di;          // size of box in y dimension
  double Lz    = 512*di;          // size of box in z dimension
  
  R_cylinder = fmin(Lx, Ly)/6.0; // Cylinder radius = 1/3 of min(Lx,Ly)

  // Calculate n_out from pressure balance
  //double pressure_balance = n_in*(T_i + T_e) + B_in*B_in/(2*mu0);
  //n_out = (pressure_balance - B_out*B_out/(2*mu0)) / (T_i + T_e);
  if (n_out < 0) n_out = 0.1*n_in; // Safety check

  double nx = 64;                // Resolution in x
  double ny = 1;                // Resolution in y
  double nz = 512;                // Resolution in z

  double nppc  = 50;             // Average number of macro particle per cell per species 
  
  double topology_x = 2;         // Number of domains in x, y, and z
  double topology_y = 1;
  double topology_z = 8;

  DumpParameters fdParams;
  std::vector<DumpParameters *> outputParams;
  DumpParameters hedParams;
  DumpParameters hHdParams;

  // Derived numerical parameters
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;
  // Cylinder center
  double center_x = Lx/2.0;
  double center_y = 0;//Ly/2.0;
  #define n(r) ((r<=R_cylinder)?n_in:n_out) 
    double n_total = 0;
  
  for ( int i=0; i < nx; i++ ) {
    for ( int j=0; j < ny; j++ ) {
        double  xx = hx*(i);

        double  yy = hy*(j)-Ly/2.0;
        double rr = sqrt((xx-center_x)*(xx-center_x) + (yy-center_y)*(yy-center_y));

        n_total = n_total + hx*hy*n(rr);
    }
}
  // Total physical ions in simulation
  double Np = n_in*M_PI*R_cylinder*R_cylinder*Lz + n_out*(Lx*Ly - M_PI*R_cylinder*R_cylinder)*Lz;
  
  // Number of macroparticles
  double Ni  = nppc*nx*ny*nz;    // Total macroparticle ions in box
  Ni = trunc_granular(Ni,nproc());// Make it divisible by number of processors

  double qi = ec*Np/Ni;          // Charge per macro ion

  double nfac = qi/(hx*hy*hz);   // Convert density to particles per cell

  // Determine the time step
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);  // courant length
  double dt = 0.01/wci;          // courant limited time step
  
  int sort_interval = 10;        // How often to sort particles

  // Intervals for output
  num_step = int(taui/(wci*dt));
  int restart_interval = 3000;
  int energies_interval = 200;
  int interval = int(1.0/(wci*dt));
  int fields_interval = interval;
  int ehydro_interval = interval;
  int Hhydro_interval = interval;
  int eparticle_interval = 40*interval;
  int Hparticle_interval = interval;
  int quota_check_interval     = 100;
  int stride_particle_dump = 40; // stride for particle dump

  
 
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
  //global->restart_step         = 0;
  global->b0  = b0;
  global->bg = bg;
  global->v_A  = v_A;
  global->nsp = 1;
  global->ur  = 0.0;
  global->ul  = 0.0;
  global->q[0]  = qi;
  global->vth[0]  = sqrt(2.0*(T_in/mi)); // Thermal velocity
  global->npleft[0]  = n_in;
  global->npright[0]  = n_out;
  global->n_inject[0]=0;
  global->left = 0;
  global->right = 0;
  global->nfac = nfac;
  global->sn = 0.0;
  
  global->topology_x  = topology_x;
  global->topology_y  = topology_y;
  global->topology_z  = topology_z;
 
  //////////////////////////////////////////////////////////////////////////////
  // Setup the grid

  // Setup basic grid parameters
  define_units(1.0, 1.0);//c, eps0 );
  define_timestep( dt );

  // Define the grid - periodic in all directions
   // Define the grid
  define_periodic_grid(  -0.5*Lx,  -0.5*Ly, -0.5*Lz,    // Low corner
                         0.5*Lx,  0.5*Ly,  0.5*Lz,     // High corner
                         nx, ny, nz,             // Resolution
                         topology_x, topology_y, topology_z); // Topology

  grid->te = T_in;
  grid->den = 1.0;
  grid->eta = 0.0;       // No resistivity for ideal MHD
  grid->hypereta = 0.0;
  grid->gamma = 5.0/3.0;

  grid->nsub = 1;
  grid->nsm= 2;
  grid->nsmb=200;

  //////////////////////////////////////////////////////////////////////////////
  // Setup materials
  sim_log("Setting up materials. ");
  define_material( "vacuum", 1 );
  
  //////////////////////////////////////////////////////////////////////////////
  // Finalize Field Advance
  define_field_array(NULL);

  sim_log("Finalized Field Advance");

  
  //////////////////////////////////////////////////////////////////////////////
  // Setup the species
  sim_log("Setting up species. ");
  double nmax = 20.0*Ni/nproc();
  double nmovers = 5*nmax;
  double sort_method = 1;   // 0=in place and 1=out of place
  species_t *ion = define_species("ion", ec, mi, nmax, nmovers, sort_interval, sort_method);
  //species_t *electron = define_species("electron", -ec, me, nmax, nmovers, sort_interval, sort_method);
  
  // Log diagnostic information about this simulation
  sim_log( "***********************************************" );
  sim_log("* Cylindrical Kink Mode Simulation");
  sim_log("* Cylinder radius R/di = " << R_cylinder/di);
  sim_log("* B_in = " << B_in << ", B_out = " << B_out);
  sim_log("* n_in = " << n_in << ", n_out = " << n_out);
  sim_log("* T_in = " << T_in << ", T_out = " << T_out);
  sim_log("* Perturbation amplitude = " << perturbation_amplitude);
  sim_log("* Mode m = " << m_mode << ", kz = " << kz);
  sim_log("* Topology: " << topology_x << " " << topology_y << " " << topology_z);
  sim_log ( "Lx/di = " << Lx/di );
  sim_log ( "Ly/di = " << Ly/di );
  sim_log ( "Lz/di = " << Lz/di );
  sim_log ( "nx = " << nx );
  sim_log ( "ny = " << ny );
  sim_log ( "nz = " << nz );
  sim_log ( "nproc = " << nproc ()  );
  sim_log ( "nppc = " << nppc );
  sim_log ( "dt*wci = " << wci*dt );
  sim_log ( "taui = " << taui );
  sim_log ( "num_step = " << num_step );
  // Dump simulation information
  if (rank() == 0 ) {
    FileIO fp_info;
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
   double x,y,z;

  double kx0 = 2.0*M_PI/Lx;
  double ky0 = 2.0*M_PI/Ly;
  double kz0 = 2.0*M_PI/Lz;
 
  const int re_num = 1;
  double kxmin = 0; double kxmax = 0;  //floor(2.0*M_PI/4.0/hx/kx0);
  double kymin = 0; double kymax = 0;  //floor(2.0*M_PI/4.0/hy/ky0);
  double kzmin = 4; double kzmax = 4;  //floor(2.0*M_PI/4.0/hz/kz0);
  double phi_min = 0; double phi_max = 6.28;

  const int n_modes_x = (kxmax-kxmin+1);
  const int n_modes_y = (kymax-kymin+1);
  const int n_modes_z = (kzmax-kzmin+1);
  const int n_modes = n_modes_x * n_modes_y * n_modes_z * re_num;
  
  std::vector<double> kx_random_temp(n_modes_x);
  std::vector<double> ky_random_temp(n_modes_y);
  std::vector<double> kz_random_temp(n_modes_z);
  std::vector<double> kx_random(n_modes);
  std::vector<double> ky_random(n_modes);
  std::vector<double> kz_random(n_modes);
  std::vector<double> phi_random(n_modes);
  std::vector<double> amplitude_ratio(n_modes);

if (rank() == 0) {
  double k_total_0 = sqrt(kx0*kx0 + ky0*ky0);
  kx_random_temp = linspace(kxmin, kxmax, n_modes_x);
  ky_random_temp = linspace(kymin, kymax, n_modes_y);
  kz_random_temp = linspace(kzmin, kzmax, n_modes_z);
  int idx = 0;
  for  (int repeat = 0; repeat < re_num; repeat++){
    for (int i = 0; i < n_modes_x; i++) {
      for (int j = 0; j < n_modes_y; j++) {
        for (int k = 0; k < n_modes_z; k++) {
          kx_random[idx]  = kx_random_temp[i]*kx0;
          ky_random[idx]  = ky_random_temp[j]*ky0;
          kz_random[idx]  = kz_random_temp[k]*kz0;
          phi_random[idx] = uniform(rng(0), phi_min, phi_max);
          if ((std::abs(kx_random[idx]) < 0.0001) && (std::abs(ky_random[idx]) < 0.0001) && (std::abs(kz_random[idx]) < 0.0001)) {
                  amplitude_ratio[idx] = 0.0;
                  kx_random[idx] = 0.001;
                  ky_random[idx] = 0.001;
                  kz_random[idx] = 0.001;
          } else {
                  double k_total_temp = sqrt(kx_random[idx]*kx_random[idx] + ky_random[idx]*ky_random[idx] + kz_random[idx]*kz_random[idx]);
                  amplitude_ratio[idx] = 1.0;
          }
          idx = idx + 1;
        }
      }
    }
  }
}
MPI_Bcast(amplitude_ratio.data(), n_modes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(kx_random.data(), n_modes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(ky_random.data(), n_modes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(kz_random.data(), n_modes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(phi_random.data(), n_modes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  ////////////////////////////
  // Load fields and particles

  sim_log( "Loading fields and particles" );
  seed_entropy( rank() );  // Generators desynchronized
  double xmin = grid->x0, xmax = grid->x0 + (grid->dx)*(grid->nx);
  double ymin = grid->y0, ymax = grid->y0 + (grid->dy)*(grid->ny);
  double zmin = grid->z0, zmax = grid->z0 + (grid->dz)*(grid->nz);
//   double nnode = 0;
  
//   for ( int i=0; i < grid->nx; i++ ) {
//     for ( int j=0; j< grid->ny; j++ ) {
//         double  xx = xmin + (grid->dx)*(i);

//         double  yy = ymin + (grid->dy)*(j);
//         double rr = sqrt((xx-center_x)*(xx-center_x) + (yy-center_y)*(yy-center_y));

//         nnode = nnode + hx*hy*n(rr);
//     }
// }
  // Load magnetic field - Bz only
  double delta = 0.1*R_cylinder; // Transition width
  //#define Bz B_out + 0.5*(B_in - B_out)*(1.0 - tanh((sqrt((x - center_x)*(x - center_x) + (y - center_y)*(y - center_y)) - R_cylinder)/delta))
  PerturbationParams params = {amplitude_ratio, kx_random, ky_random, kz_random, phi_random, n_modes, n_in, n_out, Cs_in, Cs_out, va_in, va_out, b0x, b0y, b0z, waveamp, x0, b_ratio};
  set_region_field( everywhere, EX_PERT(x,y,z,params), EY_PERT(x,y,z,params), EZ_PERT(x,y,z,params), b0x+BX_PERT(x,y,z,params), b0y+BY_PERT(x,y,z,params), ((abs(x)<=x0)?b0z:b0z*b_ratio)+BZ_PERT(x,y,z,params));

  // Load particles
  // double nlocal = Ni*nnode/n_total/topology_z;
  // std::cout<<"rank="<<rank()<<" n_total="<<n_total<<" nnode="<<nnode<<std::endl;
  // int num_ion = 0;
  // int num_elec = 0;

  // // For particles
  // int num_in =0;
  // repeat ( nlocal ) {
  //   //double r = uniform(rng(0), 0, xmax);
  //   double x,y,z;
  //   // double f_max = n_back + n_0; // 提议分布的上限
  //   // double test;
  //   // // 使用拒绝方法来加载粒子位置
  //   // do {
  //   //     r = uniform(rng(0), zmin, zmax);
        
  //   //     test = uniform(rng(0), 0, 1);
  //   // } while( f_max * test > n(z) );
  //     double r_samp, density_samp;
  // double n_max = fmax(n_in, n_out); // 密度最大值
  // double accept_prob; // 接受概率

  // do {
  //   // 均匀采样位置
  //   x = uniform(rng(0), xmin, xmax);
  //   y = uniform(rng(0), ymin, ymax);
  //   z = uniform(rng(0), zmin, zmax);
    
  //   // 计算柱半径
  //   double rx_tmp = x - center_x;
  //   double ry_tmp = y - center_y;
  //   double r_tmp = sqrt(rx_tmp*rx_tmp + ry_tmp*ry_tmp);
    
  //   // 计算当前位置的密度
  //   double density = (r_tmp<=R_cylinder)? n_in : n_out;
    
  //   // 生成随机数判断是否接受
  //   accept_prob = density / n_max; // 接受概率为密度归一化值
  //   if (uniform(rng(0), 0, 1) < accept_prob) {
  //     r_samp = r_tmp;
  //     density_samp = density;
  //     break; // 接受采样点
  //   }
  // } while (1); // 循环直至接受有效点
  //    //std::cout<<"x="<<x<<" y="<<y<<" z="<<z<<" r_samp="<<r_samp<<" density_samp="<<density_samp<<std::endl;
  //   // Calculate radial position
  //   double rx = x - center_x;
  //   double ry = y - center_y;
  //   double r = sqrt(rx*rx + ry*ry);
  //   double theta = atan2(ry, rx);
    
  //   // Density profile with smooth transition
  //   //double density = n_out + 0.5*(n_in - n_out)*(1.0 - tanh((r - R_cylinder)/delta));
    
  //   // Velocity perturbation: v_r ∝ cos(mθ) sin(kz z)
  //   double v_pert = perturbation_amplitude * cos(m_mode*theta) * sin(kz*z);
  //   double vx_pert = v_pert * cos(theta);
  //   double vy_pert = v_pert * sin(theta);
    
  //   // Thermal velocities
  //   double vthi = sqrt(T_i/mi);
  //   double vthe = sqrt(T_e/me);
    
  //   // Inject ion
  //   double ux_i = normal(rng(0), 0, vthi) + 1*vx_pert;
  //   double uy_i = normal(rng(0), 0, vthi) + 1*vy_pert;
  //   double uz_i = normal(rng(0), 0, vthi);
  //   inject_particle( ion, x, y, z, ux_i, uy_i, uz_i, qi, 0, 0);
  //   num_ion++;
    
  //   // // Inject electron
  //   // double ux_e = normal(rng(0), 0, vthe) + vx_pert;
  //   // double uy_e = normal(rng(0), 0, vthe) + vy_pert;
  //   // double uz_e = normal(rng(0), 0, vthe);
  //   // inject_particle( electron, x, y, z, ux_e, uy_e, uz_e, -ec, 0, 0);
  //   // num_elec++;
  // }

  //sim_log( "Finished loading particles" );
  double nnode = 0;
double MAX_N_PERT = 0;
double temp_N;
for ( int i=0; i < grid->nx; i++ ) {
  for ( int j=0; j< grid->ny; j++ ) {
    for ( int k=0; k< grid->nz; k++ ) {
      x = xmin + (grid->dx)*(i);
      y = ymin + (grid->dy)*(j);
      z = zmin + (grid->dz)*(k);
      temp_N = ((fabs(x)<=x0)?n_in:n_out)+N_PERT(x,y,z,params);
      // if (temp_N < 0){
      //   std::cout << "temp_N: " << temp_N <<std::endl;
      // }
      nnode = nnode + hx*hy*hz*temp_N;
      if (temp_N > MAX_N_PERT) {
        MAX_N_PERT = n0+N_PERT(x,y,z,params);
      }
    }
  }
}
double nlocal = Ni*nnode/Np;
//std::cout << "MAX_N_PERT: " << MAX_N_PERT << std::endl;
sim_log( "->begin nlocal" );

repeat (static_cast<int>(std::floor(nlocal))) {
  double x,y,z, ux, uy, uz, r;
  // rejection method, sine profile
  do {
    x = uniform(rng(0), xmin, xmax); 
    y = uniform(rng(0), ymin, ymax);
    z = uniform(rng(0), zmin, zmax);  
    r = uniform(rng(0), 0, MAX_N_PERT);
  } while(r > ((abs(x)<=x0)?n_in:n_out) + N_PERT(x,y,z,params));

  ux = normal( rng(0), 0, (abs(x)<=x0)?vthi_in:vthi_out) + UX_PERT(x,y,z,params);// + JX_PERT(x,y,z,params)/(n0+N_PERT(x,y,z,params));
  uy = normal( rng(0), 0, (abs(x)<=x0)?vthi_in:vthi_out) + UY_PERT(x,y,z,params);// + JY_PERT(x,y,z,params)/(n0+N_PERT(x,y,z,params));
  uz = normal( rng(0), 0, (abs(x)<=x0)?vthi_in:vthi_out) + UZ_PERT(x,y,z,params);// + JZ_PERT(x,y,z,params)/(n0+N_PERT(x,y,z,params));

  inject_particle(ion, x, y, z, ux, uy, uz, qi, 0, 0 );
}
sim_log( "Finished loading particles" );
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

// ... [The rest of the code (diagnostics, particle injection, etc) remains similar] ...

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

  global->restart_interval = 3000;
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

  if(should_dump(Hhydro)) {
    hydro_dump("ion", global->hHdParams);
    //hydro_dump("ion_b", global->hHdParams_b);
  }

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


    // Dump particle data

  char subdir[36];
  char subdir_ion[36];
  //char subdir_b[36];
  if ( should_dump(Hparticle) ) {
    sprintf(subdir_ion,"particle/T.%ld/Hparticle",step());
    //sprintf(subdir_b,"particle/T.%ld/Hparticle_b",step());
    sprintf(subdir,"particle/T.%ld/",step());
    dump_mkdir(subdir);
    dump_particles("ion", subdir_ion);
    //dump_particles("ion_b", subdir_b);
    }
     // if global->particle_tracing
    // sprintf(subdir2, "particle/T.%ld", step());
    // dump_mkdir(subdir2);
    // //sprintf(subdir_traj,"particle/particle_traj");
    // sprintf(subdir_swi,"particle/T.%ld/Hparticle_SWI",step());
    // //sprintf(subdir_alpha,"particle/T.%ld/Hparticle_alpha",step());
    // //sprintf(subdir_pui,"particle/T.%ld/Hparticle_PUI",step());
    // dump_particles("ion", subdir_swi);
    //dump_particles("alpha", subdir_alpha);
    //dump_particles("pui", subdir_pui);
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
  //if ( global->particle_tracing > 0 ) advance_tracers(1);
  // int inject;
  // double x, y, z, age, vtherm, vd;
  // double uv[3];
  // double nfac = global->nfac;
  // const int nsp=global->nsp;
  // const int ny=grid->ny;
  // const int nz=grid->nz;
  // const double sqpi =1.772453850905516;
  // const double dt=grid->dt;
  // const double hx=grid->dx;
  // const double hy=grid->dy;
  // const double hz=grid->dz;

  // // Initialize the injectors on the first call

  //   static int initted=0;
  //   if ( !initted ) {

  //     initted=1;

  //     if (rank() == 0) MESSAGE(("----------------Initializing the Particle Injectors-----------------")); 
      
  //     // MESSAGE(("------rank=%g    right=%i     left=%i    nsp=%i",rank(),global->right,global->left,nsp)); 
  //     // Intialize injectors

  //           if (global->right) {
	//       if (rank() == 0) MESSAGE(("----------------Initializing the Right Particle Injectors-----------------")); 
	// DEFINE_INJECTOR(right,ny,nz);
	// if (step() == 0) { 
	//   for ( int n=1; n<=nsp; n++ ) { 
	//     for ( int k=1;k<=nz; k++ ) {
	//       for ( int j=1;j<=ny; j++ ) { 
	// 	bright(n,k,j) = 0;
	// 	nright(n,k,j) = global->npright[n]/nfac;
	// 	uright(1,n,k,j) = -global->ur;
	// 	uright(2,n,k,j) = 0;
	// 	uright(3,n,k,j) = 0;
	// 	pright(1,2,n,k,j)=pright(2,1,n,k,j)=pright(1,3,n,k,j)=pright(3,1,n,k,j)=pright(2,3,n,k,j)=pright(3,2,n,k,j)=0;
	// 	pright(1,1,n,k,j) = global->npright[n]*vth(n)*vth(n)/(2.0*nfac);
	// 	pright(2,2,n,k,j) = pright(1,1,n,k,j);
	// 	pright(3,3,n,k,j) = pright(1,1,n,k,j);
	//       }      
	//     }
	//   }  // end for	
	// } // endif
	// else {

  //     if (rank() == 0) MESSAGE(("----------------Reading the Particle Injectors-----------------")); 
  //     READ_INJECTOR(right, ny, nz, 0);
	// }
  //     } //end right boundary

  //     if (rank() == 0) MESSAGE(("-------------------------------------------------------------------"));

  //      }// End of Intialization

  //       if (global->right) {
  //     for ( int n=1; n<=nsp; n++ ) { 
	// species_t * species = find_species_id(n-1,species_list );  
	// for ( int k=1;k<=nz; k++ ) {
	//   for ( int j=1;j<=ny; j++ ) {
	//     vtherm = sqrt(2.0*pright(1,1,n,k,j)/nright(n,k,j));
	//     vd =  (global->ur)/vtherm;
	//     bright(n,k,j) = bright(n,k,j)+ dt*nright(n,k,j)*vtherm*(exp(-vd*vd)/sqpi+vd*(erf(vd)+1))/(2*hx);
	//     inject = (int) bright(n,k,j);
	//     bright(n,k,j) = bright(n,k,j) - (double) inject;
	//     double uflow[3] = {uright(1,n,k,j),uright(2,n,k,j),uright(3,n,k,j)};
	//     double press[9] = {pright(1,1,n,k,j),pright(1,2,n,k,j),pright(1,3,n,k,j),pright(2,1,n,k,j),pright(2,2,n,k,j),pright(2,3,n,k,j),pright(3,1,n,k,j),pright(3,2,n,k,j),pright(3,3,n,k,j)};	     

	//     //MESSAGE((" Injecting right  --> n= %i    inject=%i   nright=%e    vth=%e  vd=%e",n,inject,nright(n,k,j),vtherm,vd)); 
	//       // MESSAGE((" Injecting right  --> n= %i    inject=%i",n,inject)); 
	//     repeat(inject) {
	//       //MESSAGE((" Injecting right  --> n= %i    uvx=%e",inject,uv[0])); 

	//       compute_injection(uv,nright(n,k,j),uflow,press,-1,2,3,rng(0));
	//       x = grid->x1; 
	//       y = grid->y0 + hy*(j-1) + hy*uniform(rng(0), 0, 1); 
	//       z = grid->z0 + hz*(k-1) + hz*uniform(rng(0), 0, 1); 	    
	//       age = 0;
	//       inject_particle(species, x, y, z, uv[0], uv[1], uv[2], abs(q(n)) , age, 0 );
	//     }
	//   }
	// }
  //     }
  //   } // end right injector

} // end particle injection

//*******************  CURRENT INJECTION ********************
begin_current_injection {
} // end current injection

//*******************  FIELD INJECTION **********************
begin_field_injection {
//   const int nx=grid->nx;
//   const int ny=grid->ny;
//   const int nz=grid->nz;
//   int x,y,z;
//   double b0 = global->b0;
//   double sn = global->sn;
//   double Vflow = global->ur, r=0.005;
//   // There macros are from local.c to apply boundary conditions
// #define XYZ_LOOP(xl,xh,yl,yh,zl,zh)             \
//   for( z=zl; z<=zh; z++ )                       \
//     for( y=yl; y<=yh; y++ )                     \
//       for( x=xl; x<=xh; x++ )

// #define yz_EDGE_LOOP(x) XYZ_LOOP(x,x,0,ny+1,0,1+nz)

//   // Right Boundary
//   if (global->right) {
//     //XYZ_LOOP(nx-5,nx,0,ny+1,0,nz+1) field(x,y,z).ex  = 0;
//     //XYZ_LOOP(nx-5,nx,0,ny+1,0,nz+1) field(x,y,z).ey  = -Vflow*b0*sn;
//     //XYZ_LOOP(nx-5,nx,0,ny+1,0,nz+1) field(x,y,z).ez  = 0;
//     XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1)field(x,y,z).cbx = (1.0-r)*field(x,y,z).cbx + r*b0*sqrt(1-sn*sn);
//     XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).cby = (1.0-r)*field(x,y,z).cby;
//     XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1)field(x,y,z).cbz = (1.0-r)*field(x,y,z).cbz + r*b0*sn;
//   }
}  // end field injection


//*******************  COLLISIONS ***************************
begin_particle_collisions {
} // end collisions
