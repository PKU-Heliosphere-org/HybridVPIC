//////////////////////////////////////////////////////
//
//   Shock
//   Bounce off a wall on left. Open inject on right
//////////////////////////////////////////////////////

//#define NUM_TURNSTILES 16384
#include <math.h>
#include <list>
#include <iterator>
#include "vpic/dumpmacros.h"
#include "injection.cxx" //  Routines to compute re-injection velocity 
#include "tracer.hh" // Rountines to trace the particles
#include "hdf5.h"
#include "time_average_master.hh"
#include "injection_for_PUI.hh"
#include "turbulence_initialization.cxx"

//////////////////////////////////////////////////////
const double v_A=1;
const double b0=1;
AlfvenWave* waves = init_turbulence_params(1, 64);  // 全局波列表
#define NUMFOLD (rank()/16)
// structure to hold the data for energy diagnostics
struct edata {
  species_id sp_id;       /* species id */
  double       vth;       /* thermal energy */
  char  fname[256];       /* file to save data */
};

// electric and magnetic fields
typedef struct emf {
  float ex, ey, ez;     // Electric field and div E error
  float cbx, cby, cbz;  // Magnetic field and div B error
} emf_t;

// Whether to simulation turbulence
/* #define TURBULENCE_SIMULATION */

// Whether only electrons carry current (for forcefree sheet only)
#define ELECTRONS_CARRRY_CURRENT
// Whether to use HDF5 format for dumping fields and hydro
#define DUMP_WITH_HDF5

#ifdef DUMP_WITH_HDF5
// Deck only works if VPIC was build with HDF support. Check for that:
#ifndef VPIC_ENABLE_HDF5
#error "VPIC_ENABLE_HDF5" is required
#endif
#endif

// naming convention for the dump files
#define HYDRO_FILE_FORMAT "hydro/%d/T.%d/%s.%d.%d"
#define SPEC_FILE_FORMAT "hydro/%d/T.%d/spectrum-%s.%d.%d"

// directory on scratch space for dumping data
#define DUMP_DIR_FORMAT "./%s"

// array access
#define LOCAL_CELL_ID(x,y,z) VOXEL(x, y, z, grid->nx, grid->ny, grid->nz)
#define HYDRO(x,y,z) hi[LOCAL_CELL_ID(x,y,z)]
#define HYDRO_TOT(x,y,z) htot[LOCAL_CELL_ID(x,y,z)]

// Vadim's in-line average
#define ALLOCATE(A,LEN,TYPE)                                    \
  if ( !((A)=(TYPE *)malloc((size_t)(LEN)*sizeof(TYPE))) )      \
    ERROR(("Cannot allocate."));

void
checkpt_subdir( const char * fbase, int tag )
{
  char fname[256];
  if( !fbase ) ERROR(( "NULL filename base" ));
  sprintf( fname, "%s/%i/restore.0.%i", fbase, tag, world_rank );
  if( world_rank==0 ) log_printf( "*** Checkpointing to \"%s\"\n", fbase );
  checkpt_objects( fname );
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
  double v_A;
  double topology_x;       // domain topology
  double topology_y;
  double topology_z;
  int restart_step;     // time step for restart dump
# define NUM_SPECS (3)

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
  species_t *tracers_list;
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

// Define the PUI velocity distribution and generator the random number
/*
double stepFunction(double x) {  
    if (x < 0) {  
        return 0.0; 
    } else {  
        return 1.0;  
    }  
}  */
// double velocity_pdf(double x, double r, double alpha, double eta) {  
//     // 
//     double lambda = 3.4;
    
//     if (x <= 0) return 0.0;  
//     return pow(x,alpha-3)*exp(-lambda/r*pow(x,-alpha))*stepFunction(1-x) ; // 
// }

// double PUI_flux_to_right(double r, double alpha, double eta, double v_u, double v_b){
//   double S=0;
//   double delta_v = 0.01;
//   double delta_theta = 0.01*M_PI/2;
//   double N0 = speed_cdf(1, r, alpha, eta);
//   for (int i = 0; i<floor(sqrt(v_b*v_b-v_u*v_u)/delta_v);++i){
//     for (int j = 1; j<floor(M_PI/2/delta_theta);++j){
//       double w = sqrt(i*i*delta_v*delta_v+2*i*delta_v*v_u*cos(j*delta_theta)+v_u*v_u)/v_b;
//       //std::cout<<"i="<<i<<" "<<"w="<<w<<"\n";
//       S += 2*M_PI*velocity_pdf(w, r, alpha, eta)*w*w*w*cos(j*delta_theta)*sin(j*delta_theta)*delta_v*delta_theta/N0;
//       // if (i%20==0){
//       // std::cout<<"w="<<w<<"\n";
//       // }
//     }
    
//   }
//   return S;
// }


begin_initialization {
  
  // Use natural hybrid-PIC units:
  double ec   = 1.0;  // Charge normalization
  double mi   = 1.0;  // Mass normalization
  double mi_me = 10.0; //
  double me = mi/mi_me;
  double mu0  = 1.0;  // Magnetic constanst
  double b0 = 1.0;    // Magnetic field
  double n0 = 1.0;    // Density

  
  // Derived normalization parameters:
  double v_A = b0/sqrt(mu0*n0*mi); // Alfven velocity
  double wci = ec*b0/mi;          // Cyclotron freq
  double di = v_A/wci;            // Ion skin-depth

  
  // Initial conditions for model:
  double Vd_Va    = 1.86;           //11.4;             // Alfven Mach number(11.4 for TS, 3.0 for Interplanetary shock)
  double Ti_Te    = 1/2.6;            // Ion temperature / electron temperature
  double theta    = 90.0 * M_PI / 180.0; // 10.0*M_PI/180.0;  // Shock normal/B field angle
  double beta_i   = 0.037;     //1.0         // Background ion beta
  double beta_e = 0.5;                  //Background electron beta
  double gamma    = 5.0/3.0;              // Ratio of specific heats
  
  double eta = 0.001;      // Plasma resistivity.
  double hypereta = 0.005; // Plasma hyper-resistivity.

  
  // Derived quantities for model:
  double Ti = beta_i*b0*b0/2.0/n0;
  double vthi = sqrt(Ti/mi);
  double vth_alpha;
  double Te = Ti/Ti_Te;
  double vthe = sqrt(Te/me);
  double vtha = vthi*2;
  double Vd = Vd_Va*v_A;
  double cs       = cos(theta);
  double sn       = sin(theta);

  // Numerical parameters
  double taui    = 100;//50;    // Simulation run time in wci^-1.
  double quota   = 23.5;   // run quota in hours
  double quota_sec = quota*3600;  // Run quota in seconds
  
  double Lx    = 256*di;    // size of box in x dimension
  double Ly    = 1.0*di;    // size of box in y dimension
  double Lz    = 64*di;     // size of box in z dimension

  double nx = 256;//256;
  double ny = 1;
  double nz = 64;//64;

  double nppc  = 100;         // Average number of macro particle per cell per species 
  
  double topology_x = 8;     // Number of domains in x, y, and z
  double topology_y = 1;
  double topology_z = 2;

  DumpParameters fdParams;
  std::vector<DumpParameters *> outputParams;
  DumpParameters hedParams;
  DumpParameters hHdParams;
  edata ede;            // electron species information
  edata edH;            // ion species information

  // Derived numerical parameters
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;
  double N_i2N_pui = 5;//1/0.058;
  double N_i2N_alpha = 11.65;
  double Ni  = nppc*nx*ny*nz;         // Total macroparticle ions in box
  double Nalpha = nppc*nx*ny*nz/N_i2N_alpha;
  double Npui  = nppc*nx*ny*nz/N_i2N_pui;         // Total macroparticle PUIs in box
  double Ne  = Ni+Npui;         // Total macroparticle electrons in box
  double Np = n0*Lx*Ly*Lz;            // Total physical ions.

  Ni = trunc_granular(Ni,nproc());// Make it divisible by number of processors
  Nalpha = trunc_granular(Nalpha,nproc());// Make it divisible by number of processors
  Npui = trunc_granular(Npui,nproc());// Make it divisible by number of processors
  double qi = ec*Np/Ni; // Charge per macro ion
  double qa = 2*ec*Np/Ni;// Charge per macro alpha particle

  double nfac = qi/(hx*hy*hz);        // Convert density to particles per cell

  //  double udre = b0/L;
  //  double udri = 0.0;
  double udri = 0;//2*Ti/(ec*b0*L);   // Ion drift velocity
   
  // Determine the time step
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);  // courant length
  double dt = 0.005/wci;                      // courant limited time step
  int tracer_interval = int(0.25/(wci*dt));
  if (tracer_interval == 0) tracer_interval = 1;
  int tracer_pass1_interval = tracer_interval;
  int tracer_pass2_interval = tracer_interval;
  double sort_interval = 10;  // How often to sort particles
  
//particle tracking
  int particle_tracing = 1; // 0: notracing, 1: forward tracing 2: tracing from particle files
  int particle_select = 196; // track one every particle_select particles
  int i_particle_select = particle_select;
  int pui_particle_select = 10;
  int alpha_particle_select = 11;
  int particle_tracing_start = 0; // the time step that particle tracking is triggered
                                  // this should be set to 0 for Pass1 and 2
  int nframes_per_tracer_file = 100; // number of frames of tracers saved in one tracer file
  int dump_traj_directly = 0;     // dump particle trajectories in 1st pass
  int emf_at_tracer = 1;          // electric and magnetic fields at tracer
  int hydro_at_tracer = 0;        // hydro fields at tracer
  int ve_at_tracer = 0;           // electron bulk velocity
  int num_emf = 0;                // number of electric and magnetic field, change between passes
  int num_hydro = 0;
    if (emf_at_tracer) num_emf = 6; // Make sure the sum of these two == TRACER_NUM_ADDED_FIELDS
  if (hydro_at_tracer) {
    num_hydro = 5; // single fluid velocity, electron and ion number densities
    if (ve_at_tracer) num_hydro += 3;
  } else {
    ve_at_tracer = 0;
  }
  double Ntracer = Ne / particle_select;   // Number of particle tracers for each species
  Ntracer = trunc_granular(Ntracer, nproc());
  // Intervals for output
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
  double factor = (3-alpha_PUI)/alpha_PUI;
  double f_pui_max = pow(lambda/r_PUI/factor, -factor)*exp(-factor);
  double PUI_flux = integral_flux(Vd, 0.001, 0.01);
  double M = pow(Vc,4)*f_pui_max/PUI_flux;
  double PUI_flux_normalized = integral_flux(Vd, 0.01, 0.01)/speed_cdf(1)/pow(Vc, 3);
  // double r = 33.5;
  // double alpha_PUI = 1.4;
  // double Vc = 10.07;
  // double PUI_flux = integral_flux(Vd, 0.01, 0.01);
  
  
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

  int ix, iy, iz, left=0,right=0;
  RANK_TO_INDEX( int(rank()), ix, iy, iz );
  if ( ix ==0 ) left=1;
  if ( ix ==topology_x-1 ) right=1;

  
  ///////////////////////////////////////////////
  // Setup high level simulation parameters
  num_step             = int(taui/(wci*dt));
  status_interval      = 200;
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
  global->restart_step         = 0;
  global->b0  = b0;
  global->v_A  = v_A;

  global->nsp = 3;
  global->ur  = Vd;
  global->ul  = 0.0;
  global->q[0]  = qi;
  global->q[1]  = qa;
  global->q[2]  = qi;
  global->npleft[0]  = n0;
  global->npleft[1]  = n0/N_i2N_alpha;
  global->npleft[2]  = n0/N_i2N_pui;
  global->npright[0]  = n0;
  global->npright[1]  = n0/N_i2N_alpha;
  global->npright[2]  = n0/N_i2N_pui;
  global->vth[0]  = sqrt(2.)*vthi;
  global->vth[1]  = sqrt(2.)*vtha;
  global->vth[2]  = sqrt(2.)*vthi;
  global->n_inject[0]=0;
  global->n_inject[1]=0;
  global->n_inject[2]=0;
  global->left = left;
  global->right = right;
  global->nfac = nfac;
  global->sn = sn;
  
  global->topology_x  = topology_x;
  global->topology_y  = topology_y;
  global->topology_z  = topology_z;
  // particle tracking
  global->particle_tracing      = particle_tracing;
  global->tracer_interval       = tracer_interval;
  global->tracer_file_interval  = nframes_per_tracer_file * tracer_interval;
  global->tracer_pass1_interval = tracer_pass1_interval;
  global->tracer_pass2_interval = tracer_pass2_interval;
  global->Ntracer = int(Ntracer);
  global->dump_traj_directly = dump_traj_directly;
  global->emf_at_tracer   = emf_at_tracer;
  global->hydro_at_tracer = hydro_at_tracer;
  global->ve_at_tracer = ve_at_tracer;
  global->num_tracer_fields_add = num_emf + num_hydro;
  global->particle_select = particle_select;
  global->i_particle_select = i_particle_select;
  global->alpha_particle_select = alpha_particle_select;
  global->pui_particle_select = pui_particle_select;
  // particle dump
  global->stride_particle_dump = stride_particle_dump;
  global->M = M;
  global->PUI_flux = PUI_flux;
  global->PUI_flux_normalized = PUI_flux_normalized;
  
  // global->alpha_PUI = alpha_PUI;
  // global->r = r;
  // global->Vc = Vc;
 
  //////////////////////////////////////////////////////////////////////////////
  // Setup the grid

  // Setup basic grid parameters
  define_units(1.0, 1.0);//c, eps0 );
  define_timestep( dt );

  // Define the grid
  define_periodic_grid(  0,  -0.5*Ly, -0.5*Lz,    // Low corner
                         Lx,  0.5*Ly,  0.5*Lz,     // High corner
                         nx, ny, nz,             // Resolution
                         topology_x, topology_y, topology_z); // Topology

  grid->te = Te;
  grid->den = 1.0;
  grid->eta = eta;
  grid->hypereta = hypereta;
  if (right) grid->hypereta=0.02;
  grid->gamma = gamma;

  grid->nsub = 1;
  grid->nsm= 2;
  grid->nsmb=200;


  // ***** Set Field Boundary Conditions *****

  sim_log("Reflecting fields + particles on left X");
  if ( ix==0 )
    set_domain_field_bc( BOUNDARY(-1,0,0), pec_fields );
  if ( ix==topology_x-1 )
   set_domain_field_bc( BOUNDARY( 1,0,0), pec_fields );

  // ***** Set Particle Boundary Conditions *****
  if ( ix==0 )    set_domain_particle_bc( BOUNDARY(-1,0,0), reflect_particles );
  if ( ix==topology_x-1 ) set_domain_particle_bc( BOUNDARY(1,0,0), absorb_particles );

 
  //////////////////////////////////////////////////////////////////////////////
  // Setup materials
  sim_log("Setting up materials. ");
  define_material( "vacuum", 1 );

  material_t * layer = define_material("layer",   1.0,10.0,1.0,
				                  1.0,1.0, 1.0,
				                  0.0,0.0, 0.0);
  
  //////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                       // Finalize Field Advance
  define_field_array(NULL); // second argument is damp, default to 0

  // Define resistive layer surrounding boundary --> set thickness=0                                                                                                                
  // to eliminate this feature. Second number multiplies hypereta                                                                                                                   
  double thickness = 10;
#define INSIDE_LAYER ( (x < hx*thickness) || (x > Lx-hx*thickness))

  if (thickness > 0)  set_region_material(INSIDE_LAYER, layer, layer);

  sim_log("Finalized Field Advance");

  
  //////////////////////////////////////////////////////////////////////////////
  // Setup the species
  sim_log("Setting up species. ");
  double nmax_swi = 20.0*Ni/nproc();
  double nmax_alpha = 20.0*Nalpha/nproc();
  double nmax_pui = 20.0*Npui/nproc();
  double nmovers_swi = 0.1*nmax_swi;
  double nmovers_alpha = 0.1*nmax_alpha;
  double nmovers_pui = 0.1*nmax_pui;
  double sort_method = 1;   // 0=in place and 1=out of place
  species_t *ion = define_species("ion", ec, mi, nmax_swi, nmovers_swi, sort_interval, sort_method);
  species_t *alpha = define_species("alpha", 2*ec, 4*mi, nmax_alpha, nmovers_alpha, sort_interval, sort_method);
  species_t *pui = define_species("pui", ec, mi, nmax_pui, nmovers_pui, sort_interval, sort_method);
  //Define the trace particles
  species_t *ion_tracer = define_species("ion_tracer", ec, mi, nmax_swi, nmovers_swi, sort_interval, sort_method);
  species_t *alpha_tracer = define_species("alpha_tracer", 2*ec, 4*mi, nmax_alpha, nmovers_alpha, sort_interval, sort_method);
  species_t *pui_tracer = define_species("pui_tracer", ec, mi, nmax_pui, nmovers_pui, sort_interval, sort_method);
  hijack_tracers(3);
  // pui->max_np = ion->max_np;
  //pui->id = num_species(species_list);
 // pui->next = *species_list;
  //*species_list = pui->next;
  //species_list = append_species(pui, *species_list);
 // sim_log(pui->np<<pui->max_np);
  //sim_log(ion->np<<ion->max_np);
   sim_log(num_species(species_list));
  // species_t *electron = define_species("electron", -ec, me, nmax_pui, nmovers_pui, sort_interval, sort_method);
  
  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  sim_log( "***********************************************" );
  sim_log("* Topology:                       " << topology_x
    << " " << topology_y << " " << topology_z);
  sim_log ( "Vd/Va = " << Vd_Va ) ;
  sim_log ( "beta_i = " << beta_i );
  sim_log ( "theta = " << theta );
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
  sim_log ("Vc = " << Vc);
  sim_log ("variation = " << variation);


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

// init_turbulence_params(v_A, Lz);
sim_log( "Loading fields" );
// Note: everywhere is a region that encompasses the entire simulation                                                                                                                   
// In general, regions are specied as logical equations (i.e. x>0 && x+y<2) 
std::cout<<waves[0].A<<"\n";
  set_region_field( everywhere, EX_PERT, -Vd*b0*sn+EY_PERT, 0,
		    b0*cs+BX_PERT, BY_PERT, b0*sn ); // Magnetic field

 
 

 // LOAD PARTICLES
  int num_i_tracer    = 0;
  int num_alpha_tracer    = 0;
  int num_pui_tracer    = 0;   // tracer index
  int i_particle  = 0;   // ion particle index
  int alpha_particle  = 0;   // alpha particle index
  int pui_particle  = 0;   // pui particle index

  sim_log( "Loading particles" );

  // Do a fast load of the particles
  int rng_seed     = 1;     // Random number seed increment 
  seed_entropy( rank() );  //Generators desynchronized
  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);

  repeat ( Ni/nproc() ) {
     double x_swi, y_swi, z_swi, ux_swi, uy_swi, uz_swi, d0 ;
     //double x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui;
     // double theta_pui, phi_pui;

     x_swi = uniform(rng(0),xmin,xmax);
     y_swi = uniform(rng(0),ymin,ymax);
     z_swi = uniform(rng(0),zmin,zmax);
     double x = x_swi;
     double y = y_swi;
     double z = z_swi;
     
     ux_swi = normal( rng(0), 0, vthi)-Vd+VX_PERT;//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000)-Vd;//normal( rng(0), 0, vthi)-Vd;
     uy_swi = normal( rng(0), 0, vthi)+VY_PERT;//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
     uz_swi = normal( rng(0), 0, vthi);//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
     /*
     x_pui = uniform(rng(0),xmin,xmax);
     y_pui = uniform(rng(0),ymin,ymax);
     z_pui = uniform(rng(0),zmin,zmax);
     
     theta_pui = uniform(rng(0),0,M_PI);
     phi_pui = uniform(rng(0),0,2*M_PI);
     ux_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*sin(theta_pui)*cos(phi_pui)-Vd;
     uy_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*sin(theta_pui)*sin(phi_pui);
     uz_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*cos(theta_pui);
     */
     inject_particle( ion, x_swi, y_swi, z_swi, ux_swi, uy_swi, uz_swi, qi, 0, 0);
     ++i_particle;
       if (particle_tracing == 1) { // only tag particles in the 1st pass
      if (i_particle%i_particle_select == 0) {
        num_i_tracer++;
        /* int tag = ((((int)rank())<<19) | (itracer & 0x7ffff)); // 13 bits (8192) for rank and 19 bits (~520k) */
        int tag = ((((int)rank())<<16) | (num_i_tracer & 0x1fff)); // 19 bits (520k) for rank and 13 bits (8192)
        tag_tracer( (ion->p + ion->np-1), ion_tracer, tag );
        
      }
    }
     //inject_particle( pui, x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui, qi, 0, 0);

   }
     global->i_particle = i_particle;
     global->num_i_tracer = num_i_tracer;
     repeat ( Nalpha/nproc() ) {
     double x_alpha, y_alpha, z_alpha, ux_alpha, uy_alpha, uz_alpha, d0 ;
     //double x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui;
     // double theta_pui, phi_pui;

     x_alpha = uniform(rng(0),xmin,xmax);
     y_alpha = uniform(rng(0),ymin,ymax);
     z_alpha = uniform(rng(0),zmin,zmax);
     double x = x_alpha;
     double y = y_alpha;
     double z = z_alpha;
     
     ux_alpha = normal( rng(0), 0, vtha)-Vd+VX_PERT;//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000)-Vd;//normal( rng(0), 0, vthi)-Vd;
     uy_alpha = normal( rng(0), 0, vtha)+VY_PERT;//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
     uz_alpha = normal( rng(0), 0, vtha);//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
     /*
     x_pui = uniform(rng(0),xmin,xmax);
     y_pui = uniform(rng(0),ymin,ymax);
     z_pui = uniform(rng(0),zmin,zmax);
     
     theta_pui = uniform(rng(0),0,M_PI);
     phi_pui = uniform(rng(0),0,2*M_PI);
     ux_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*sin(theta_pui)*cos(phi_pui)-Vd;
     uy_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*sin(theta_pui)*sin(phi_pui);
     uz_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*cos(theta_pui);
     */
     inject_particle( alpha, x_alpha, y_alpha, z_alpha, ux_alpha, uy_alpha, uz_alpha, qa, 0, 0);
     ++alpha_particle;
      if (particle_tracing == 1) { // only tag particles in the 1st pass
      if (alpha_particle%alpha_particle_select == 0) {
        num_alpha_tracer++;
        /* int tag = ((((int)rank())<<19) | (itracer & 0x7ffff)); // 13 bits (8192) for rank and 19 bits (~520k) */
        int tag = ((((int)rank())<<16) | (num_alpha_tracer & 0x1fff)); // 19 bits (520k) for rank and 13 bits (8192)
        tag_tracer( (alpha->p + alpha->np-1), alpha_tracer, tag );
        
      }
    }
     //inject_particle( pui, x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui, qi, 0, 0);

   }
   /* electrons should be treated as a fluid, not particles.
   repeat ( Ni/nproc() ) {
     double x_e, y_e, z_e, ux_e, uy_e, uz_e, d0 ;
     //double x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui;
     // double theta_pui, phi_pui;

     x_e = uniform(rng(0),xmin,xmax);
     y_e = uniform(rng(0),ymin,ymax);
     z_e = uniform(rng(0),zmin,zmax);
     
     ux_e = normal( rng(0), 0, vthe)-Vd;//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000)-Vd;//normal( rng(0), 0, vthi)-Vd;
     uy_e = normal( rng(0), 0, vthe);//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
     uz_e = normal( rng(0), 0, vthe);//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
     
     x_pui = uniform(rng(0),xmin,xmax);
     y_pui = uniform(rng(0),ymin,ymax);
     z_pui = uniform(rng(0),zmin,zmax);
     
     theta_pui = uniform(rng(0),0,M_PI);
     phi_pui = uniform(rng(0),0,2*M_PI);
     ux_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*sin(theta_pui)*cos(phi_pui)-Vd;
     uy_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*sin(theta_pui)*sin(phi_pui);
     uz_pui = Vd*inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-1, 1000)*cos(theta_pui);
     
     inject_particle( electron, x_e, y_e, z_e, ux_e, uy_e, uz_e, qi, 0, 0);
     //inject_particle( pui, x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui, qi, 0, 0);

   }
   */
    global->alpha_particle = alpha_particle;
    global->num_alpha_tracer = num_alpha_tracer;
   repeat ( Npui/nproc() ) {

     // double x_swi, y_swi, z_swi, ux_swi, uy_swi, uz_swi, d0 ;
     double x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui,V;
     double theta_pui, phi_pui;
    
     //x_swi = uniform(rng(0),xmin,xmax);
     //y_swi = uniform(rng(0),ymin,ymax);
     //z_swi = uniform(rng(0),zmin,zmax);
     
    // ux_swi = normal( rng(0), 0, vthi)-Vd;//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000)-Vd;//normal( rng(0), 0, vthi)-Vd;
     //uy_swi = normal( rng(0), 0, vthi);//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
     //uz_swi = normal( rng(0), 0, vthi);//inverse_cdf(uniform(rng(0),0,1), 0.5, 1e-2, 1000);//normal( rng(0), 0, vthi);
    
     x_pui = uniform(rng(0),xmin,xmax);
     y_pui = uniform(rng(0),ymin,ymax);
     z_pui = uniform(rng(0),zmin,zmax);
     double x = x_pui;
     double y = y_pui;
     double z = z_pui;
     
     //Vc = 10.07; //PUI cutoff speed
     theta_pui = acos(uniform(rng(0),-1,1));
     phi_pui = uniform(rng(0),0,2*M_PI);
     V = Vc*inverse_cdf(uniform(rng(0),0,1), 1e-3);
     ux_pui = V*cos(theta_pui)-Vd+VX_PERT;
     uy_pui = V*sin(theta_pui)*sin(phi_pui)+VY_PERT;
     uz_pui = V*sin(theta_pui)*cos(phi_pui);

     //inject_particle( ion, x_swi, y_swi, z_swi, ux_swi, uy_swi, uz_swi, qi, 0, 0);
     inject_particle( pui, x_pui, y_pui, z_pui, ux_pui, uy_pui, uz_pui, qi, 0, 0);
     ++pui_particle;
    if (particle_tracing == 1) { // only tag particles in the 1st pass
      if (pui_particle%pui_particle_select == 0) {
        num_pui_tracer++;
        /* int tag = ((((int)rank())<<19) | (itracer & 0x7ffff)); // 13 bits (8192) for rank and 19 bits (~520k) */
        int tag = ((((int)rank())<<16) | (num_pui_tracer & 0x1fff)); // 19 bits (520k) for rank and 13 bits (8192)
        tag_tracer( (pui->p + pui->np-1), pui_tracer, tag );
        
      }
    }

   }
   global->pui_particle = pui_particle;
   global->num_pui_tracer = num_pui_tracer;
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
    dump_mkdir("tracer");
    dump_mkdir("tracer/tracer1");
    dump_mkdir("tracer/tracer2");
    dump_mkdir("tracer/traj1");
    dump_mkdir("tracer/traj2");

    
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
	DUMP_INJECTORS(1);
	//    } END_TURNSTILE;
      } else {
        global->rtoggle = 0;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
	checkpt("restore0/restore", 0);
	DUMP_INJECTORS(0);
	//    } END_TURNSTILE;
      } // if

      //    mp_barrier();
      sim_log( "Restart dump completed");
      double dumpelapsed = uptime() - dumpstart;
      sim_log("Restart duration "<<dumpelapsed);
    } // if global->write_restart
  }

    // Dump particle data

  char subdir_swi[36];
  char subdir_alpha[36];
  char subdir_pui[36];
  char subdir2[36];
  char subdir_traj[36];
  //if (step()==0){
   // std::vector<double> dx_arr = {};
  //}


  if ( should_dump(Hparticle) && step() !=0
       && step() > 0*(global->fields_interval)  ) {
    
    //if (rank()==0){
    //std::cout<<particle_mover->i<<"\n";}
    /*
    species_t * s_pui = find_species_id(2,species_list );
    int i=0;
    int trace_particle_num = 100;
    if (trace_particle_num > s_pui->np){
      trace_particle_num = s_pui->np;
    }
    for(i==0;i<trace_particle_num;i++){
      particle_t *p_pui = s_pui->p+i;
      dx_tmp = p_pui->dx;
      dy_tmp = p_pui->dy;
      dz_tmp = p_pui->dz;
      ux_tmp = p_pui->ux;
      uy_tmp = p_pui->uy;
      uz_tmp = p_pui->uz;


      //std::cout<<step()<<" "<<i<<" "<<dx_tmp<<"\n";

    }*/
    

    sprintf(subdir2, "particle/T.%ld", step());
    dump_mkdir(subdir2);
    //sprintf(subdir_traj,"particle/particle_traj");
    sprintf(subdir_swi,"particle/T.%ld/Hparticle_SWI",step());
    sprintf(subdir_alpha,"particle/T.%ld/Hparticle_alpha",step());
    sprintf(subdir_pui,"particle/T.%ld/Hparticle_PUI",step());
    dump_particles("ion", subdir_swi);
    dump_particles("alpha", subdir_alpha);
    dump_particles("pui", subdir_pui);
    //dump_traj(subdir_traj);
    }
     /*--------------------------------------------------------------------------
   * particle tracking
   * -----------------------------------------------------------------------*/
//tag{last start comment}
  // Set TRACER_ACCUM_HYDRO to 1 if we need to accumulate hydro moments before
  // writing trajectory output. Since this involves a pass through all the particles
  // in the system as well as synchronization (which hits MPI), don't do this step
  // unless we must.

#undef  TRACER_DO_ACCUM_HYDRO
#define TRACER_DO_ACCUM_HYDRO (0)       //CHANGE BETWEEN PASSES

//  // Setup data needed for hydro output
//# ifdef TRACER_DO_ACCUM_HYDRO
//    TRACER_HYDRO_SETUP( e, "electron" )
//    TRACER_HYDRO_SETUP( H, "H"       )
//# endif

  // Be careful! This number should be set correctly
#undef  TRACER_NUM_ADDED_FIELDS
#define TRACER_NUM_ADDED_FIELDS (11)       //CHANGE BETWEEN PASSES

#undef CALC_TRACER_USER_DEFINED_DATA
#define CALC_TRACER_USER_DEFINED_DATA                           \
  if ( global->emf_at_tracer ) {                                \
    CALC_EMFS_AT_TRACER;                                        \
  }                                                             \
  if ( global->hydro_at_tracer ) {                              \
      CALC_HYDRO_FIELDS_AT_TRACER;                              \
  }

  // We assume hydro fields are alway behind electric and magnetic fields
#undef TRACER_USER_DEFINED_DATA
#define TRACER_USER_DEFINED_DATA                                \
  if ( global->emf_at_tracer ) {                                \
    pout[index + 6 + 1]  = ex;                                  \
    pout[index + 6 + 2]  = ey;                                  \
    pout[index + 6 + 3]  = ez;                                  \
    pout[index + 6 + 4]  = bx;                                  \
    pout[index + 6 + 5]  = by;                                  \
    pout[index + 6 + 6]  = bz;                                  \
    if ( global->hydro_at_tracer ) {                            \
      pout[index + 12 + 1] = vx;                                \
      pout[index + 12 + 2] = vy;                                \
      pout[index + 12 + 3] = vz;                                \
      pout[index + 12 + 4] = ne;                                \
      pout[index + 12 + 5] = ni;                                \
      if ( global->ve_at_tracer ) {                             \
        pout[index + 12 + 6] = vex;                             \
        pout[index + 12 + 7] = vey;                             \
        pout[index + 12 + 8] = vez;                             \
      }                                                         \
    }                                                           \
  } else {                                                      \
    if ( global->hydro_at_tracer ) {                            \
      pout[index + 6 + 1] = vx;                                 \
      pout[index + 6 + 2] = vy;                                 \
      pout[index + 6 + 3] = vz;                                 \
      pout[index + 6 + 4] = ne;                                 \
      pout[index + 6 + 5] = ni;                                 \
      if ( global->ve_at_tracer ) {                             \
        pout[index + 6 + 6] = vex;                              \
        pout[index + 6 + 7] = vey;                              \
        pout[index + 6 + 8] = vez;                              \
      }                                                         \
    }                                                           \
  }

  // Hydro fields at tracer positions
  static hydro_array_t * hydro_tot_array;
  static hydro_t * ALIGNED(128) htot;
  static hydro_t * ALIGNED(128) hi;
  static hydro_t * RESTRICT ALIGNED(16) htot0;
  static hydro_t * RESTRICT ALIGNED(16) h0;

  int frame;

  const int tracer_ratio1 = global->tracer_pass1_interval / global->tracer_interval;
  const int tracer_ratio2 = global->tracer_pass2_interval / global->tracer_interval;
  
  // initialize buffered tracer data
  if ( step() == 0 || (step()>1 && step()==global->restart_step+1) ) {
    if ( global->particle_tracing==1 && tracer_ratio1 > 1 ) {
      init_buffered_tracers(tracer_ratio1);
    } else if ( global->particle_tracing==2 && tracer_ratio2 > 1 ){
      init_buffered_tracers(tracer_ratio2);
    }
    if ( global->particle_tracing > 0 && (step()>1 &&
         step()==global->restart_step+1 && (tracer_ratio1 > 1 || tracer_ratio2 > 1)) ) {
      read_buffered_tracer_restart(global->rtoggle);
    }
  }

  // Accumulate hydro
  if ( global->particle_tracing > 0 && global->hydro_at_tracer ) {
    if ( step() == 0 || (step()>1 && step()==global->restart_step+1) ) {
      hydro_tot_array = new_hydro_array(grid);
      UNREGISTER_OBJECT(hydro_tot_array);
    }
  }

  if( global->particle_tracing > 0 && should_dump(tracer) ) {
    if ( global->hydro_at_tracer ) {  // accumulate hydro at tracer positions
      int x, y, z;
      float rho_tot;
      clear_hydro_array(hydro_tot_array);
      
      species_t *sp = find_species_name("ion", species_list);
      clear_hydro_array(hydro_array);
      accumulate_hydro_p(hydro_array, sp, interpolator_array);
      synchronize_hydro_array(hydro_array);
      htot = hydro_tot_array->h;
      hi   = hydro_array->h;
      for (z = 1; z <= nz + 1; z++) {
        for (y = 1; y <= ny + 1; y++) {
          htot0 = &HYDRO_TOT(1, y, z);
          h0    = &HYDRO(1, y, z);
          for (x = 1; x <= nx + 1; x++) {
            // we use txx, tyy, and tzz as electron bulk velocity
            if (global->ve_at_tracer && fabs(htot0->rho) > 0) {
              htot0->txx = htot0->jx / htot0->rho;
              htot0->tyy = htot0->jy / htot0->rho;
              htot0->tzz = htot0->jz / htot0->rho;
            }
            // Assuming electron has -1 charge, ion has +1 charge
            rho_tot = fabs(htot0->rho) + h0->rho * global->mi_me;
            // jx, jy, jz are actually vx, vy, vz for single fluid now
            htot0->jx = (-htot0->jx + h0->jx*global->mi_me) / rho_tot;
            htot0->jy = (-htot0->jy + h0->jy*global->mi_me) / rho_tot;
            htot0->jz = (-htot0->jz + h0->jz*global->mi_me) / rho_tot;
            htot0->rho = fabs(htot0->rho); // Electron number density
            htot0->px = h0->rho;           // Ion number density
            htot0++;
            h0++;
          }
        }
      }
    } // if global->hydro_at_tracer

    // Buffer tracer data for latter dump
    if (global->particle_tracing==1 && tracer_ratio1 > 1) {
      frame = ((step() % global->tracer_pass1_interval)-1) / global->tracer_interval;
      if (frame < 0) frame = 0;
      buffer_tracers(tracer_ratio1, frame);
    } else if (global->particle_tracing==2 && tracer_ratio2 > 1) {
      frame = ((step() % global->tracer_pass2_interval)-1) / global->tracer_interval;
      if (frame < 0) frame = 0;
      buffer_tracers(tracer_ratio2, frame);
    }

  } // if should_dump(tracer)
  
  if ( global->particle_tracing==1 ) {           // First pass
    if ( should_dump(tracer_pass1) || step() == num_step) {
      //if ( TRACER_DO_ACCUM_HYDRO ) {
      //  // accumulate electron hydro
      //  TRACER_ACCUM_HYDRO( e );
      //  // accumulate H hydro
      //  TRACER_ACCUM_HYDRO( H );
      //} // if
      if (global->dump_traj_directly) {
        //
        dump_traj("tracer/traj1");
        
      } else {
        if (tracer_ratio1 == 1) { // tracer data is not buffered
          //dump_tracers("tracer/tracer1");

#include "dumptracer_hdf5_single.cc"
        } else {
          dump_buffered_tracer(tracer_ratio1, "tracer/tracer1");
          clear_buffered_tracers(tracer_ratio1);
        }
      }
    } // if
  } else if ( global->particle_tracing==2 ) {    // Second pass
    if ( should_dump(tracer_pass2) || step() == num_step) {
      //if ( TRACER_DO_ACCUM_HYDRO ) {
      //  // accumulate electron hydro
      //  TRACER_ACCUM_HYDRO( e );
      //  // accumulate H hydro
      //  TRACER_ACCUM_HYDRO( H );
      //}  // if
      if (global->dump_traj_directly) {
        dump_traj("tracer/traj2");
      } else {
        if (tracer_ratio2 == 1) { // tracer data is not buffered
          //dump_tracers("tracer/tracer2");

#include "dumptracer_hdf5_single.cc"
        } else {
          dump_buffered_tracer(tracer_ratio2, "tracer/tracer2");
          clear_buffered_tracers(tracer_ratio2);
        }
      }
    }  // if
  } // if global->particle_tracing
//tag{last end comment}

  /*--------------------------------------------------------------------------
   * Normal rundata energies dump
   *------------------------------------------------------------------------*/
  if(should_dump(energies)) {
    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
  } // if

  /*--------------------------------------------------------------------------
   * Field data output
   *------------------------------------------------------------------------*/
/*
#ifdef DUMP_WITH_HDF5
  field_dump_flag.disableE();
  field_dump_flag.disableCB();
  field_dump_flag.disableTCA();
  field_dump_flag.disableJF();
  field_dump_flag.disableEMAT();
  field_dump_flag.disableFMAT();
  field_dump_flag.ex = true;
  field_dump_flag.ey = true;
  field_dump_flag.ez = true;
  field_dump_flag.cbx = true;
  field_dump_flag.cby = true;
  field_dump_flag.cbz = true;

  if(step() == 1 || should_dump(fields)) {
    double time_to_dump_fields = uptime();
    dump_fields_hdf5(global->fdParams.baseDir, 0);
    time_to_dump_fields = uptime() - time_to_dump_fields;
    sim_log("Time in dumping fields: "<< time_to_dump_fields << " s");
  }

#else // #ifdef DUMP_WITH_HDF5

  if(step() == 1 || should_dump(fields)) {
    double time_to_dump_fields = uptime();
    field_dump(global->fdParams);
    time_to_dump_fields = uptime() - time_to_dump_fields;
    sim_log("Time in dumping fields: "<< time_to_dump_fields << " s");
  }

#endif // #ifdef DUMP_WITH_HDF5


#ifdef DUMP_WITH_HDF5
  hydro_dump_flag.resetToDefaults();
*/

  /*--------------------------------------------------------------------------
   * Ion species output
   *------------------------------------------------------------------------*/

/*
  if(should_dump(Hhydro)) {
    double time_to_dump_hydro = uptime();
    dump_hydro_hdf5("ion", global->hHdParams.baseDir, 0);
    time_to_dump_hydro = uptime() - time_to_dump_hydro;
    sim_log("Time in dumping hydro: "<< time_to_dump_hydro << " s");
  }

#else // #ifdef DUMP_WITH_HDF5


  if(should_dump(ehydro)) {
    double time_to_dump_hydro = uptime();
    hydro_dump("electron", global->hedParams);
    time_to_dump_hydro = uptime() - time_to_dump_hydro;
    sim_log("Time in dumping hydro: "<< time_to_dump_hydro << " s");
  }
*/

  /*--------------------------------------------------------------------------
   * Ion species output
   *------------------------------------------------------------------------*/

/*
  if(should_dump(Hhydro)) {
    double time_to_dump_hydro = uptime();
    hydro_dump("ion", global->hHdParams);
    time_to_dump_hydro = uptime() - time_to_dump_hydro;
    sim_log("Time in dumping hydro: "<< time_to_dump_hydro << " s");
  }


#endif // #ifdef DUMP_WITH_HDF5
*/
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
  int inject;
  double x, y, z, age, vtherm, vd;
  double uv[3];
  double nfac = global->nfac;
  const int nsp=global->nsp;
  const int ny=grid->ny;
  const int nz=grid->nz;
  const double sqpi =1.772453850905516;
  const double dt=grid->dt;
  const double hx=grid->dx;
  const double hy=grid->dy;
  const double hz=grid->dz;
  

  // Initialize the injectors on the first call

    static int initted=0;
    if ( !initted ) {

      initted=1;

      if (rank() == 0) MESSAGE(("----------------Initializing the Particle Injectors-----------------")); 
      
      // MESSAGE(("------rank=%g    right=%i     left=%i    nsp=%i",rank(),global->right,global->left,nsp)); 
      // Intialize injectors

            if (global->right) {
	      if (rank() == 0) MESSAGE(("----------------Initializing the Right Particle Injectors-----------------")); 
	DEFINE_INJECTOR(right,ny,nz);
  
	if (step() == 0) { 
    
	  for ( int n=1; n<=nsp; n++ ) { 
      n_inject(n)=0;
	    for ( int k=1;k<=nz; k++ ) {
	      for ( int j=1;j<=ny; j++ ) { 
		bright(n,k,j) = 0;
		nright(n,k,j) = global->npright[n-1]/nfac;
    
		uright(1,n,k,j) = -global->ur;
		uright(2,n,k,j) = 0;
		uright(3,n,k,j) = 0;
		pright(1,2,n,k,j)=pright(2,1,n,k,j)=pright(1,3,n,k,j)=pright(3,1,n,k,j)=pright(2,3,n,k,j)=pright(3,2,n,k,j)=0;
		pright(1,1,n,k,j) = global->npright[n-1]*vth(n)*vth(n)/(2.0*nfac);
		pright(2,2,n,k,j) = pright(1,1,n,k,j);
		pright(3,3,n,k,j) = pright(1,1,n,k,j);
	      }      
	    }
	  }  // end for	
	} // endif
	else {

      if (rank() == 0) MESSAGE(("----------------Reading the Particle Injectors-----------------")); 
      READ_INJECTOR(right, ny, nz, 0);
	}
      } //end right boundary

      if (rank() == 0) MESSAGE(("-------------------------------------------------------------------"));

       }// End of Intialization

        if (global->right) {
          //std::cout<<nsp<<"\n";
  //            std::cout<<"Time step= "<<step()<<"\n";
  int num_i_tracer    = global->num_i_tracer;
  int num_alpha_tracer    = global->num_alpha_tracer;
  int num_pui_tracer    = global->num_pui_tracer;   // tracer index
  int i_particle  = global->i_particle;   // ion particle index
  int alpha_particle  = global->alpha_particle;   // alpha particle index
  int pui_particle  = global->pui_particle;   // pui particle index
  int PUI_inject_number = 0;
  //double pui_flux = PUI_flux_to_right(33.5,1.4,5,global->ur,10.07);
  //const double denominator = integral_flux(global->ur, 0.01, 0.01);
  //std::cout<<denominator/speed_cdf(1,33.5,1.4,5)/pow(10.07,3)<<" "<<pui_flux<<"\n";
  // std::cout<<speed_cdf(1)<<" "<<global->PUI_flux<<"\n";
  
      for ( int n=1; n<=nsp; n++ ) { 
	species_t * species = find_species_id(n-1,species_list );  
  //std::cout<<species->np<<"\n";
  species_t *tracer = find_species_id(n+2, global->tracers_list);

  //std::cout<<n<<" "<<tracer->name<<"\n";
	for ( int k=1;k<=nz; k++ ) {
    //std::cout<<k<<"**********"<<"\n";
	  for ( int j=1;j<=ny; j++ ) {
	    vtherm = sqrt(2.0*pright(1,1,n,k,j)/nright(n,k,j));
      //std::cout <<"n="<<n<<", vth="<<vtherm <<"\n";
	    vd =  (global->ur)/vtherm;
      if (n!=nsp){
	      bright(n,k,j) = bright(n,k,j)+ dt*nright(n,k,j)*vtherm*(exp(-vd*vd)/sqpi+vd*(erf(vd)+1))/(2*hx);
        n_inject(n) += dt*nright(n,k,j)*vtherm*(exp(-vd*vd)/sqpi+vd*(erf(vd)+1))/(2*hx);
        

       
      }
      else{
        bright(n,k,j) = bright(n,k,j)+dt*nright(n,k,j)*global->PUI_flux_normalized/hx;//escape PUIs plus inject PUI flow
        n_inject(n) += dt*nright(n,k,j)*global->PUI_flux_normalized/hx;
        
        //std::cout<<global->PUI_flux_normalized*dt*nright(n,k,j)/hx;
        //bright(n,k,j) = bright(n,k,j)+ dt*nright(n,k,j)*(global->ur)/hx;//inject PUI flow
        //std::cout<<dt*nright(n,k,j)*pui_flux/hx+dt*nright(n,k,j)*(global->ur)/hx<<" "<<denominator*nright(n,k,j)*dt/hx<<"\n";
      }
	    inject = (long) bright(n,k,j);
	    bright(n,k,j) = bright(n,k,j) - (double) inject;
      // inject = (long)n_inject(n);
      // n_inject(n) = n_inject(n)-(double)inject;
      if (n==3){
        PUI_inject_number += inject;
        //std::cout<<"step="<<step()<<", n_inject="<<n_inject(n) <<", inject="<<inject<<", k="<<k<<"\n";
      }
    //   double integer_part = floor(bright(n,k,j));
    //   double fractional = bright(n,k,j)-integer_part;
    //   if (uniform(rng(0), 0, 1) < fractional) {
    //     inject = (long)(integer_part + 1);
    // } else {
    //     inject = (long)integer_part;
    // }
    // bright(n,k,j) = bright(n,k,j) - inject;
    // std::cout<<inject<<"\n";
	    double uflow[3] = {uright(1,n,k,j),uright(2,n,k,j),uright(3,n,k,j)};
	    double press[9] = {pright(1,1,n,k,j),pright(1,2,n,k,j),pright(1,3,n,k,j),pright(2,1,n,k,j),pright(2,2,n,k,j),pright(2,3,n,k,j),pright(3,1,n,k,j),pright(3,2,n,k,j),pright(3,3,n,k,j)};	     

	    //MESSAGE((" Injecting right  --> n= %i    inject=%i   nright=%e    vth=%e  vd=%e",n,inject,nright(n,k,j),vtherm,vd)); 
	      // MESSAGE((" Injecting right  --> n= %i    inject=%i",n,inject)); 
      //std::cout<<species->np<<"\n";
	    repeat(inject) {
	      //MESSAGE((" Injecting right  --> n= %i    uvx=%e",inject,uv[0])); 

	      compute_injection(uv,nright(n,k,j),uflow,press,-1,2,3,rng(0));
	      x = grid->x1; 
	      y = grid->y0 + hy*(j-1) + hy*uniform(rng(0), 0, 1); 
	      z = grid->z0 + hz*(k-1) + hz*uniform(rng(0), 0, 1); 	    
	      age = 0;
        //std::cout<<inject<<"\n";
        if (n!=nsp){
	      inject_particle(species, x, y, z, uv[0]+VX_PERT_time(0.005*step()), uv[1]+VY_PERT_time(0.005*step()), uv[2], abs(q(n)) , age, 0 );
        //std::cout<<uv[0]<<"\n";
        /*
        if (n==1){
          i_particle++;
             if (global->particle_tracing == 1) { // only tag particles in the 1st pass
      if (i_particle%global->particle_select == 0) {
        num_i_tracer++;
        // int tag = ((((int)rank())<<19) | (itracer & 0x7ffff)); // 13 bits (8192) for rank and 19 bits (~520k) 
        int tag = ((((int)rank())<<16) | (num_i_tracer & 0x1fff)); // 19 bits (520k) for rank and 13 bits (8192)
        tag_tracer( (species->p + species->np-1), tracer, tag );
        
      }
    }
        }
           if (n==2){
          alpha_particle++;
             if (global->particle_tracing == 1) { // only tag particles in the 1st pass
      if (alpha_particle%global->particle_select == 0) {
        num_alpha_tracer++;
        std::cout<<step()<<" "<<inject<<" "<<num_alpha_tracer<<"\n";
        // int tag = ((((int)rank())<<19) | (itracer & 0x7ffff)); // 13 bits (8192) for rank and 19 bits (~520k) 
        int tag = ((((int)rank())<<16) | (num_alpha_tracer & 0x1fff)); // 19 bits (520k) for rank and 13 bits (8192)
        tag_tracer( (species->p + species->np-1), tracer, tag );
        
      }
    }
        }*/
        }
        //end if
        
        else{
          double ux_pui, uy_pui, uz_pui;
          //double theta_pui, phi_pui;
          // Vc = 10.07;
         //double x_pui = uniform(rng(0), grid->x1-2*hx, grid->x1);
          // theta_pui = acos(uniform(rng(0),-1,global->ur/Vc));
          // phi_pui = uniform(rng(0),0,2*M_PI);

          //V = Vc*inverse_cdf(uniform(rng(0),0,1), 1e-3);
          
          std::vector<double> random_velocity = rejection_sampling_cylindrical(global->ur, 3*global->M, global->PUI_flux);
          //std::vector<double> random_velocity_2 = rejection_sampling_cylindrical(global->ur, global->M, global->PUI_flux);
          ux_pui = -random_velocity[0];//-inverse_F(uniform(rng(0),0,1),global->ur,1e-1,1e-1);//V*sin(theta_pui)*cos(phi_pui)-global->ur;
          uy_pui = random_velocity[1];
          uz_pui = random_velocity[2];
          //std::cout<<random_velocity[0]<<" "<<random_velocity_2[0]<<"\n";
          //std::cout<<VX_PERT<<"\n";
          inject_particle(species, x, y, z, ux_pui+VX_PERT_time(0.005*step()), uy_pui+VY_PERT_time(0.005*step()), uz_pui, abs(q(n)) , age, 0 );
             
          pui_particle++;
            /*
             if (global->particle_tracing == 1) { // only tag particles in the 1st pass
      if (pui_particle%global->particle_select == 0) {
        num_pui_tracer++;
        // int tag = ((((int)rank())<<19) | (itracer & 0x7ffff)); // 13 bits (8192) for rank and 19 bits (~520k) 
        int tag = ((((int)rank())<<16) | (num_pui_tracer & 0x1fff)); // 19 bits (520k) for rank and 13 bits (8192)
        tag_tracer( (species->p + species->np-1), tracer, tag );
        
      }
    }*/
        
          //if (rank()==0){
            //particle_t *p = species->p;
            //std::cout<<step()<<" "<<p->dx<<"\n";
          //}
        }//PUI injection
        
	    }
      global->i_particle = i_particle;
      global->alpha_particle = alpha_particle;
      global-> pui_particle = pui_particle;
      //global->num_i_tracer = num_i_tracer;
      //global->num_alpha_tracer = num_alpha_tracer;
      //global->num_pui_tracer = num_pui_tracer;

	  }
    
	}
  // if (n==3){
  //   std::cout<<"step="<<step()<<", inject="<<PUI_inject_number<<"\n";
  // }
      }
    } // end right injector
if ( global->particle_tracing > 0 ) advance_tracers(1);//advance the tracer particles
} // end particle injection

//*******************  CURRENT INJECTION ********************
begin_current_injection {
} // end current injection

//*******************  FIELD INJECTION **********************
begin_field_injection {
  const int nx=grid->nx;
  const int ny=grid->ny;
  const int nz=grid->nz;
  int x,y,z;
  double b0 = global->b0;
  double sn = global->sn;
  double Vflow = global->ur, r=0.005;
  // There macros are from local.c to apply boundary conditions
#define XYZ_LOOP(xl,xh,yl,yh,zl,zh)             \
  for( z=zl; z<=zh; z++ )                       \
    for( y=yl; y<=yh; y++ )                     \
      for( x=xl; x<=xh; x++ )

#define yz_EDGE_LOOP(x) XYZ_LOOP(x,x,0,ny+1,0,1+nz)

  // Right Boundary
#if true
  if (global->right) {
    //XYZ_LOOP(nx-5,nx,0,ny+1,0,nz+1) field(x,y,z).ex  = 0;
    //XYZ_LOOP(nx-5,nx,0,ny+1,0,nz+1) field(x,y,z).ey  = -Vflow*b0*sn;
    //XYZ_LOOP(nx-5,nx,0,ny+1,0,nz+1) field(x,y,z).ez  = 0;
    XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1)field(x,y,z).cbx = (1.0-r)*field(x,y,z).cbx + r*(b0*sqrt(1-sn*sn)+BX_PERT_time(0.005*step()));
    XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1) field(x,y,z).cby = (1.0-r)*field(x,y,z).cby + r*BY_PERT_time(0.005*step());
    XYZ_LOOP(nx-1,nx,0,ny+1,0,nz+1)field(x,y,z).cbz = (1.0-r)*field(x,y,z).cbz + r*b0*sn;
  }
#else
  #include "aw_field.cxx"
#endif
}  // end field injection


//*******************  COLLISIONS ***************************
begin_particle_collisions {
} // end collisions



