#ifndef GET_CONFIGURATION_H
#define GET_CONFIGURATION_H

typedef struct {
    int key_index;              // the index sorting key of the file
    int sort_key_only;          // only sort the key
    int skew_data;              // the data is in skew shape
    int verbose;                // verbose diagnostics
    int write_result;           // whether to write the sorted data
    int collect_data;           // whether to collect sorted data
    int weak_scale_test;        // whether to do a weak scale test
    int weak_scale_test_length; // the data size for weak scale test
    int local_sort_threaded;    // parallelize local sort using threads
    int local_sort_threads_num; // number of threads in local sort
    int meta_data;              // the meta data is used determine particle position
    int tracking_traj;          // whether to track particle trajectories
    int load_tracer_meta;       // whether to load tracer meta data
    int is_recreate;            // whether to recreate a HDF5 file
    int reduced_tracer;         // whether to use reduced tracer
    int tmin;                   // the particle output minimum time step
    int tmax;                   // the particle output maximum time step
    int tstep;                  // current time step
    int tinterval;              // the particle output time interval
    int multi_tsteps;           // run sorting for multiple time steps
    int nsteps;                 // # of steps are saved in each time interval
    int ux_kindex;              // the key index of ux in the HDF5 file
    int nptl_traj;              // number of particles for trajectory tracking
    int single_h5;              // whether all tracers (electron + ion + ... + meta_data) are in a single file
    int single_group;           // whether all steps in a tracer file are saved in the same group
    int reduce_tracer_time;     // whether to reduce the time resolution
    int reduce_factor_time;     // factor for reducing the time resolution
    float ratio_emax;           // maximum energy of all particles / that of tracked ones
    char *filename;             // HDF5 particle tracer file name
    char *group_name;           // group name in the HDF5 file
    char *group_name_output;    // group name in the output HDF5 file
    char *subgroup_name;        // sub-group name for the tracer data if single_h5 == 1
    char *meta_group_name;      // group name for the metadata if single_h5 == 1
    char *filename_sorted;      // file name to store sorted results
    char *filename_attribute;   // attribute file name to store sort table
    char *filename_meta;        // file name storing the meta data
    char *filepath;             // file directory saving the particle tracing data
    char *species;              // particle species for sorting
    char *filename_traj;        // output file for particle trajectories
} config_t;

int get_configuration(int argc, char **argv, int mpi_rank, config_t *config);

void init_configuration(config_t *config);
void free_configuration(config_t *config);
void copy_configuration(config_t *destination, config_t *source);
#endif
