//////////////////////////////////////////////////////
//
//   Shock
//   Bounce off a wall on left. Open inject on right
//////////////////////////////////////////////////////

//#define NUM_TURNSTILES 16384


// *tracing*
#include <math.h>
#include <list>
#include <iterator>
#include "vpic/dumpmacros.h"
#include "injection.cxx" //  Routines to compute re-injection velocity 
#include "tracer.hh" // Rountines to trace the particles
#include "hdf5.h"
#include "time_average_master.hh"

//////////////////////////////////////////////////////
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
// *end_tracing*

begin_globals {
  // *tracing*
  int num_i_tracer;
  int i_particle;   // ion particle index
  // *end_tracing*

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

  // *tracing*
  int tracer_interval;         // tracer info is saved or dumped every tracer_interval steps
  int tracer_pass1_interval;   // tracer interval for the 1st run. A multiple of tracer_interval
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
  int stride_particle_dump;  // stride for particle dump

  // *end_tracing*


};

begin_initialization {
  
  // Use natural hybrid-PIC units:
  double ec   = 1.0;  // Charge normalization
  double mi   = 1.0;  // Mass normalization
  double mu0  = 1.0;  // Magnetic constanst
  double b0 = 1.0;    // Magnetic field
  double n0 = 1.0;    // Density

  
  // Derived normalization parameters:
  double v_A = b0/sqrt(mu0*n0*mi); // Alfven velocity
  double wci = ec*b0/mi;          // Cyclotron freq
  double di = v_A/wci;            // Ion skin-depth

  
  // Initial conditions for model:
  double Vd_Va    = 11.4;             // Alfven Mach number, NOT IN USE
  double Ti_Te    = 1.0;            // Ion temperature / electron temperature
  double theta    = 10.0*M_PI/180.0;  // Shock normal/B field angle, NOT IN USE
  double beta_i   = 0.5;              // Background ion beta
  double gamma    = 5.0/3.0;              // Ratio of specific heats
  
  double eta = 0.001;      // Plasma resistivity.
  double hypereta = 0.005; // Plasma hyper-resistivity.

  
  // Derived quantities for model:
  double Ti = beta_i*b0*b0/2.0/n0;
  double vthi = sqrt(Ti/mi);
  double Te = Ti/Ti_Te;
  double Vd = Vd_Va*v_A;        // NOT IN USE
  double cs       = cos(theta); // NOT IN USE
  double sn       = sin(theta); // NOT IN USE

  // Numerical parameters
  double taui    = 301;    // Simulation run time in wci^-1.
  double quota   = 23.5;   // run quota in hours
  double quota_sec = quota*3600;  // Run quota in seconds
  
  double Lx    = 24*di;    // size of box in x dimension
  double Ly    = 192*di;    // size of box in y dimension
  double Lz    = 1.0*di;     // size of box in z dimension

  double nx = 120;
  double ny = 960;
  double nz = 1;

  double nppc  = 1024;         // Average number of macro particle per cell per species 
  
  double topology_x = 4;     // Number of domains in x, y, and z
  double topology_y = 32;
  double topology_z = 1;


  // Derived numerical parameters
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;

  double Ni  = nppc*nx*ny*nz;         // Total macroparticle ions in box
  double Np = n0*Lx*Ly*Lz;            // Total physical ions.

  Ni = trunc_granular(Ni,nproc());// Make it divisible by number of processors

  double qi = ec*Np/Ni; // Charge per macro ion

  double nfac = qi/(hx*hy*hz);        // Convert density to particles per cell
  double udri = 0;//2*Ti/(ec*b0*L);   // Ion drift velocity
   
  // Determine the time step
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);  // courant length
  double dt = 0.01/wci;                      // courant limited time step
  double sort_interval = 10;  // How often to sort particles
  // *tracing*
  int tracer_interval = int(0.25/(wci*dt)); // the number of 0.25 means cyclotron period
  if (tracer_interval == 0) tracer_interval = 1; 
  int tracer_pass1_interval = tracer_interval;

  int particle_tracing = 1; // 0: notracing, 1: forward tracing 2: tracing from particle files
  int particle_select = 512; // track one every particle_select particles
  int i_particle_select = particle_select;
  int particle_tracing_start = 0; // the time step that particle tracking is triggered
                                  // this should be set to 0 for Pass1 and 2
  int nframes_per_tracer_file = 100; // number of frames of tracers saved in one tracer file
  int dump_traj_directly = 0;     // dump particle trajectories in 1st pass
  int emf_at_tracer = 1;          // electric and magnetic fields at tracer
  int hydro_at_tracer = 1;        // hydro fields at tracer
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
  double Ntracer = Ni / particle_select;   // Number of particle tracers for each species
  Ntracer = trunc_granular(Ntracer, nproc());
  // *end_tracing*


  // Intervals for output
  int restart_interval = 99999;
  int energies_interval = 200;
  int interval = int(1./(wci*dt));
  int fields_interval = interval;
  int ehydro_interval = interval;
  int Hhydro_interval = interval;
  int eparticle_interval = 9999*interval;
  int Hparticle_interval = 50*interval;
  int quota_check_interval     = 100;
  int stride_particle_dump = 40; // stride for particle dump


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
  status_interval      = 20;
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

  global->nsp = 1;
  global->ur  = Vd;
  global->ul  = 0.0;
  global->q[0]  = qi;
  global->npleft[0]  = n0;
  global->npright[0]  = n0;
  global->vth[0]  = sqrt(2.)*vthi;
  global->left = left;
  global->right = right;
  global->nfac = nfac;
  global->sn = sn;
  
  global->topology_x  = topology_x;
  global->topology_y  = topology_y;
  global->topology_z  = topology_z;
  // *tracing*
  global->particle_tracing      = particle_tracing;
  global->tracer_interval       = tracer_interval; 
  global->tracer_pass1_interval = tracer_pass1_interval;
  global->tracer_file_interval  = nframes_per_tracer_file * tracer_interval; 
  global->Ntracer = int(Ntracer);
  global->dump_traj_directly = dump_traj_directly;
  global->emf_at_tracer   = emf_at_tracer;
  global->hydro_at_tracer = hydro_at_tracer;
  global->ve_at_tracer = ve_at_tracer;
  global->num_tracer_fields_add = num_emf + num_hydro; // new variable
  global->particle_select = particle_select;
  global->i_particle_select = i_particle_select;
  global->stride_particle_dump = stride_particle_dump;
  // *end_tracing*



 
  //////////////////////////////////////////////////////////////////////////////
  // Setup the grid

  // Setup basic grid parameters
  define_units(1.0, 1.0); //c, eps0 );
  define_timestep( dt );

  // Define the grid
  define_periodic_grid(  0,  0, -0.5*Lz,    // Low corner
                         Lx, Ly,  0.5*Lz,     // High corner
                         nx, ny, nz,             // Resolution
                         topology_x, topology_y, topology_z); // Topology

  grid->te = Te;
  grid->den = 1.0; // density? TODO
  grid->eta = eta;
  grid->hypereta = hypereta;
  // if (right) grid->hypereta=0.02; (in shock.cxx)
  grid->gamma = gamma;

  grid->nsub = 1;   // DON'T KNOW WHAT
  grid->nsm= 2;     // DON'T KNOW WHAT
  grid->nsmb=200;   // DON'T KNOW WHAT


  // ***** Set Field Boundary Conditions *****
  sim_log("Periodic boundaries");
  // Do nothing - periodic is default.

  // ***** Set Particle Boundary Conditions *****
  // Do nothing - periodic is default.

  //////////////////////////////////////////////////////////////////////////////
  // Setup materials
  sim_log("Setting up materials. ");
  define_material( "vacuum", 1 );

  // RONGLIN: does not use the following
  // material_t * layer = define_material("layer",   1.0,10.0,1.0,
	// 			                  1.0,1.0, 1.0,
	// 			                  0.0,0.0, 0.0);
  
  //////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                       // Finalize Field Advance
  define_field_array(NULL); // second argument is damp, default to 0

  // Define resistive layer surrounding boundary --> set thickness=0                                                                                                                
  // to eliminate this feature. Second number multiplies hypereta                                                                                                                   
//   double thickness = 10;
// #define INSIDE_LAYER ( (x < hx*thickness) || (x > Lx-hx*thickness))

//   if (thickness > 0)  set_region_material(INSIDE_LAYER, layer, layer);

  sim_log("Finalized Field Advance");

  
  //////////////////////////////////////////////////////////////////////////////
  // Setup the species
  sim_log("Setting up species. ");
  double nmax = 20.0*Ni/nproc();
  double nmovers = 0.4*nmax;
  double sort_method = 1;   // 0=in place and 1=out of place
  species_t *ion = define_species("ion", ec, mi, nmax, nmovers, sort_interval, sort_method);
  // *tracing*
  species_t *ion_tracer = define_species("ion_tracer", ec, mi, nmax, nmovers, sort_interval, sort_method);
  hijack_tracers(1); // should it be number of kinds of tracer?
  // *end_tracing*
  
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
double PI = 3.14159265;
double width_tanh = 10.8 / 3;
double maximum_rotation_angle_deg = 120.0;
double B0X = 0;
double B0Y = 0.4 * b0;
double B0Z = std::sqrt(b0 * b0 - B0Y * B0Y);

double rho = n0;      
double mu_0 = 1.0;    

#define TRANSITION_1_Y (Ly * 3 / 8)
#define TRANSITION_2_Y (Ly * 5 / 8)
#define SHAPE_FUNC(y) ((tanh((y - TRANSITION_1_Y) / width_tanh) + tanh((TRANSITION_2_Y - y) / width_tanh)) / 2.0)
#define ANGLES_RAD(y) (SHAPE_FUNC(y) * (-120.0 * PI / 180.0))
#define BX(y) (B0X * cos(ANGLES_RAD(y)) - B0Z * sin(ANGLES_RAD(y)))
#define BY (B0Y)
#define BZ(y) (B0X * sin(ANGLES_RAD(y)) + B0Z * cos(ANGLES_RAD(y)))

#define D_BX_DY(y) ( \
    (-120.0 * PI / 180.0) * 0.5 * \
    ((1.0 - tanh((y - TRANSITION_1_Y) / width_tanh) * tanh((y - TRANSITION_1_Y) / width_tanh)) / width_tanh - \
     (1.0 - tanh((TRANSITION_2_Y - y) / width_tanh) * tanh((TRANSITION_2_Y - y) / width_tanh)) / width_tanh) * \
    (-B0X * sin(ANGLES_RAD(y)) - B0Z * cos(ANGLES_RAD(y))) \
)
#define D_BZ_DY(y) ( \
    (-120.0 * PI / 180.0) * 0.5 * \
    ((1.0 - tanh((y - TRANSITION_1_Y) / width_tanh) * tanh((y - TRANSITION_1_Y) / width_tanh)) / width_tanh - \
     (1.0 - tanh((TRANSITION_2_Y - y) / width_tanh) * tanh((TRANSITION_2_Y - y) / width_tanh)) / width_tanh) * \
    (B0X * cos(ANGLES_RAD(y)) - B0Z * sin(ANGLES_RAD(y))) \
)

#define VX(y) (-BX(y) + D_BZ_DY(y))
#define VY (-BY)
#define VZ(y) (-BZ(y) - D_BX_DY(y))

#define JX(y) (D_BZ_DY(y))
#define JY (0.0)
#define JZ(y) (-D_BX_DY(y))

#define EX(y) (JY * BZ(y) - JZ(y) * BY)
#define EY(y) 0.
#define EZ(y) (JX(y) * BY - JY * BX(y))



sim_log( "Loading fields" );
// Note: everywhere is a region that encompasses the entire simulation                                                                                                                   
// In general, regions are specied as logical equations (i.e. x>0 && x+y<2) 
 set_region_field(
    everywhere,
    EX(y),
    EY(y),
    EZ(y),
    BX(y),
    BY,
    BZ(y)
);


 // LOAD PARTICLES
  
  // *tracing*
  int num_i_tracer = 0; // tracer index
  int i_particle = 0; // ion particle index, should be renamed?
  // *end_tracing*

  sim_log( "Loading particles" );

  // Do a fast load of the particles
  int rng_seed     = 1;     // Random number seed increment 
  seed_entropy( rank() );  //Generators desynchronized
  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);

  repeat ( Ni/nproc() ) {
     double x, y, z, ux, uy, uz, d0 ;

     x = uniform(rng(0),xmin,xmax);
     y = uniform(rng(0),ymin,ymax);
     z = uniform(rng(0),zmin,zmax);
     
     ux = normal( rng(0), 0, vthi)+VX(y);
     uy = normal( rng(0), 0, vthi)+VY;
     uz = normal( rng(0), 0, vthi)+VZ(y);

     inject_particle( ion, x, y, z, ux, uy, uz, qi, 0, 0);
     // *tracing*
     // the logic seems to be: when injected a particle, mark it
     ++i_particle;
     if (particle_tracing == 1){
      if (i_particle%i_particle_select == 0){
        num_i_tracer++;
        int tag = ((((int)rank())<<16) | (num_i_tracer & 0x1fff)); // 19 bits (520k) for rank and 13 bits (8192)
        // encoding related
        tag_tracer( (ion->p + ion->np-1), ion_tracer, tag ); // idk what np is
      }
     } 
     // *end_tracing*
   }
   // *tracing*
   global->i_particle = i_particle;
   global->num_i_tracer = num_i_tracer;
   // *end_tracing*

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

  global->restart_interval = 99999;
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
    // *tracing*
    dump_mkdir("tracer");
    dump_mkdir("tracer/tracer1");
    dump_mkdir("tracer/tracer2");
    dump_mkdir("tracer/traj1");
    dump_mkdir("tracer/traj2");
   // *end_tracing*

    
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

  char subdir[256]; // for particles
  char filename[256]; 
  // *tracing*
  char subdir_traj[256];
  // *end_tracing*
  // *tracing*
  // *end_tracing*
  
  
  // if (should_dump(Hparticle) && step() != 0 && step() > 30 * (global->fields_interval)) {  
  if (should_dump(Hparticle)) {      
      sprintf(subdir, "particle/T.%d", step());
      dump_mkdir(subdir); 
      
      sprintf(filename, "%s/Hparticle.%d", subdir, step());
      dump_particles("ion", filename, 0);
  }
  // *tracing*
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
// seem to be wrong but works

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
  // initialize buffered tracer data
  if ( step() == 0 || (step()>1 && step()==global->restart_step+1) ) {
    if ( global->particle_tracing==1 && tracer_ratio1 > 1 ) {
      init_buffered_tracers(tracer_ratio1);
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
    }
  }
  if ( global->particle_tracing==1 ) {           // First pass
    if ( should_dump(tracer_pass1) || step() == num_step) {
      //if ( TRACER_DO_ACCUM_HYDRO ) {
      //  // accumulate electron hydro
      //  TRACER_ACCUM_HYDRO( e );
      //  // accumulate H hydro
      //  TRACER_ACCUM_HYDRO( H );
      //} // if
      if (global->dump_traj_directly) {
        // seems to be zero
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
  }
  /*--------------------------------------------------------------------------
   * Normal rundata energies dump
   *------------------------------------------------------------------------*/
  if(should_dump(energies)) {
    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
  } // if
  // *end_tracing*

  // Shut down simulation when wall clock time exceeds global->quota_sec.
  // Note that the mp_elapsed() is guaranteed to return the same value for all
  // processors (i.e., elapsed time on proc #0), and therefore the abort will
  // be synchronized across processors. Note that this is only checked every
  // few timesteps to eliminate the expensive mp_elapsed call from every
  // timestep. mp_elapsed has an ALL_REDUCE in it!


  if ( (step()>=0 && global->quota_check_interval>0
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
  if ( global->particle_tracing) advance_tracers(1);//advance the tracer particles
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

