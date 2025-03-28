#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>
#include "constants.h"
#include "configuration.h"

void print_help();

/******************************************************************************
 * Get the analysis configuration from command line arguments
 ******************************************************************************/
int get_configuration(int argc, char **argv, int mpi_rank, config_t *config)
{
    int c;
    static const char *options="a:b:c:ef:g:hi:k:l:m:o:pqrst:u:vw";
    static struct option long_options[] =
    {
        {"tmax", required_argument, 0, 'b'},
        {"tinterval", required_argument, 0, 'i'},
        {"tstep", required_argument, 0, 11},
        {"filepath", required_argument, 0, 1},
        {"species", required_argument, 0, 2},
        {"filename_traj", required_argument, 0, 3},
        {"nptl_traj", required_argument, 0, 4},
        {"ratio_emax", required_argument, 0, 5},
        {"tmin", required_argument, 0, 6},
        {"is_recreate", required_argument, 0, 7},
        {"nsteps", required_argument, 0, 8},
        {"reduced_tracer", required_argument, 0, 9},
        {"load_tracer_meta", required_argument, 0, 'r'},
        {"single_h5", required_argument, 0, 12},
        {"single_group", required_argument, 0, 13},
        {"subgroup_name", required_argument, 0, 14},
        {"meta_group_name", required_argument, 0, 15},
        {"group_name_output", required_argument, 0, 16},
        {"reduce_tracer_time", required_argument, 0, 17},
        {"reduce_factor_time", required_argument, 0, 18},
        {0, 0, 0, 0},
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    extern char *optarg;

    /* Default values */
    config->key_index = 1;
    config->sort_key_only = 0;
    config->skew_data = 0;
    config->verbose = 0;
    config->write_result = 1;
    config->collect_data = 1;
    config->weak_scale_test = 0;
    config->weak_scale_test_length = 1000000;
    config->local_sort_threaded = 0;
    config->local_sort_threads_num = 16;
    config->meta_data = 0;
    config->multi_tsteps = 0;
    config->ux_kindex = 0;
    config->nptl_traj = 10;
    config->ratio_emax = 1;
    config->tracking_traj = 0;
    config->load_tracer_meta = 0;
    config->is_recreate = 0;    // Do not recreate a HDF5 file when it exists
    config->nsteps = 1;         // # steps are saved in each time interval
    config->reduced_tracer = 0; // original tracer
    config->single_h5 = 0;      // whether all tracers (electron + ion + ... + meta_data) are in a single file
    config->single_group = 0;   // whether all steps in a tracer file are saved in the same group
    config->reduce_tracer_time = 0; // whether to reduce the time resolution
    config->reduce_factor_time = 1; // factor for reducing the time resolution

    while ((c = getopt_long (argc, argv, options, long_options, &option_index)) != -1){
        switch (c){
            case 'a':
                strcpy(config->filename_attribute, optarg);
                break;
            case 'b':
                config->tmax = atoi(optarg);
                break;
            case 'c':
                config->collect_data = 0;
                break;
            case 'e':
                config->sort_key_only = 1;
                break;
            case 'f':
                strcpy(config->filename, optarg);
                break;
            case 'g':
                strcpy(config->group_name, optarg);
                break;
            case 'i':
                config->tinterval = atoi(optarg);
                break;
            case 'k':
                config->key_index = atoi(optarg);
                break;
            case 'l':
                config->weak_scale_test = 1;
                config->weak_scale_test_length = atoi(optarg);
                break;
            case 'm':
                config->meta_data = 1;
                strcpy(config->filename_meta, optarg);
                break;
            case 'o':
                strcpy(config->filename_sorted, optarg);
                break;
            case 'p':
                config->multi_tsteps = 1;
                break;
            case 'q':
                config->tracking_traj = 1;
                break;
            case 'r':
                config->load_tracer_meta = 1;
                break;
            case 's':
                config->skew_data = 1;
                break;
            case 't':
                config->local_sort_threaded = 1;
                config->local_sort_threads_num = atoi(optarg);
                break;
            case 'u':
                config->ux_kindex = atoi(optarg);
                break;
            case 'v':
                config->verbose = 1;
                break;
            case 'w':
                config->write_result = 0;
                break;
            case 1:
                strcpy(config->filepath, optarg);
                break;
            case 2:
                strcpy(config->species, optarg);
                break;
            case 3:
                strcpy(config->filename_traj, optarg);
                break;
            case 4:
                config->nptl_traj = atoi(optarg);
                break;
            case 5:
                config->ratio_emax = atof(optarg);
                break;
            case 6:
                config->tmin = atoi(optarg);
                break;
            case 7:
                config->is_recreate = atoi(optarg);
                break;
            case 8:
                config->nsteps = atoi(optarg);
                break;
            case 9:
                config->reduced_tracer = atoi(optarg);
                break;
            case 11:
                config->tstep = atoi(optarg);
                break;
            case 12:
                config->single_h5 = atoi(optarg);
                break;
            case 13:
                config->single_group = atoi(optarg);
                break;
            case 14:
                strcpy(config->subgroup_name, optarg);
                break;
            case 15:
                strcpy(config->meta_group_name, optarg);
                break;
            case 16:
                strcpy(config->group_name_output, optarg);
                break;
            case 17:
                config->reduce_tracer_time = atoi(optarg);
                break;
            case 18:
                config->reduce_factor_time = atoi(optarg);
                break;
            case 'h':
                if (mpi_rank == 0) {
                    print_help();
                }
                return 1;
            default:
                printf("Error option [%s]\n", optarg);
                exit(-1);
        }
    }
    return 0;
}


/******************************************************************************
 * Print help information.
 ******************************************************************************/
void print_help(){
    char *msg="Usage: %s [OPTION] \n\
               -a name of the attribute file to store sort table  \n\
               -b the particle output maximum time step \n\
               -c whether to collect sorted data \n\
               -e only sort the key  \n\
               -f name of the file to sort \n\
               -g group path within HDF5 file to data set \n\
               -h help (--help)\n\
               -i the particle output time interval \n\
               -k the index key of the file \n\
               -l whether to do weak scale test \n\
               -m whether to use meta data to determine particle position \n\
               -o name of the file to store sorted results \n\
               -p run sorting for multiple time steps \n\
               -q tracking the trajectories of particles\n\
               -r whether to load tracer meta data \n\
               -s the data is in skew shape \n\
               -t whether to use threads in local sorting \n\
               -u the key index of ux \n\
               -v verbose  \n\
               -w won't write the sorted data \n\
               --tmin the particle output minimum time step \n\
               --tstep current time step \n\
               --filepath file path saving the particle tracing data \n\
               --species particle species for sorting \n\
               --filename_traj output file for particle trajectories \n\
               --nptl_traj number of particles for trajectory tracking \n\
               --ratio_emax ratio of Emax of all particles to that of tracked ones \n\
               --is_recreate whether to recreate a HDF5 file \n\
               --nsteps # of steps are save in each time interval \n\
               --reduced_tracer whether to use reduced tracer \n\
               --single_h5 whether all tracers (electron + ion + ... + meta_data) are in a single file \n\
               --single_group whether all steps in a tracer file are saved in the same group \n\
               --subgroup_name sub-group name the tracer data if single_h5 == 1 \n\
               --meta_group_name group name for the metadata if single_h5 == 1 \n\
               --group_name_output group name in the output HDF5 file \n\
               --reduce_tracer_time whether to reduce the time resolution \n\
               --reduce_factor_time factor for reducing the time resolution \n\
               example: ./h5group-sorter -f testf.h5p  -g /testg  -o testg-sorted.h5p -a testf.attribute -k 0 \n";
    fprintf(stdout, msg, "h5group-sorter");
}

/******************************************************************************
 * Initialize configuration data
 ******************************************************************************/
void init_configuration(config_t *config)
{
    config->filename = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->group_name = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->group_name_output = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->subgroup_name = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->meta_group_name = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_sorted = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_attribute = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_meta = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_traj = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filepath = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->species = (char *)malloc(16 * sizeof(char));
}

/******************************************************************************
 * Free configuration data
 ******************************************************************************/
void free_configuration(config_t *config)
{
    free(config->filename);
    free(config->group_name);
    free(config->group_name_output);
    free(config->subgroup_name);
    free(config->meta_group_name);
    free(config->filename_sorted);
    free(config->filename_attribute);
    free(config->filename_meta);
    free(config->filename_traj);
    free(config->filepath);
    free(config->species);
    free(config);
}

/******************************************************************************
 * Deep copy of the configuration data
 ******************************************************************************/
void copy_configuration(config_t *destination, config_t *source)
{
    destination->key_index = source->key_index;
    destination->sort_key_only = source->sort_key_only;
    destination->skew_data = source->skew_data;
    destination->verbose = source->verbose;
    destination->write_result = source->write_result;
    destination->collect_data = source->collect_data;
    destination->weak_scale_test = source->weak_scale_test;
    destination->weak_scale_test_length = source->weak_scale_test_length;
    destination->local_sort_threaded = source->local_sort_threaded;
    destination->local_sort_threads_num = source->local_sort_threads_num;
    destination->meta_data = source->meta_data;
    destination->tracking_traj = source->tracking_traj;
    destination->load_tracer_meta = source->load_tracer_meta;
    destination->is_recreate = source->is_recreate;
    destination->reduced_tracer = source->reduced_tracer;
    destination->tmin = source->tmin;
    destination->tmax = source->tmax;
    destination->tstep = source->tstep;
    destination->tinterval = source->tinterval;
    destination->multi_tsteps = source->multi_tsteps;
    destination->nsteps = source->nsteps;
    destination->ux_kindex = source->ux_kindex;
    destination->nptl_traj = source->nptl_traj;
    destination->single_h5 = source->single_h5;
    destination->single_group = source->single_group;
    destination->reduce_tracer_time = source->reduce_tracer_time;
    destination->reduce_factor_time = source->reduce_factor_time;
    destination->ratio_emax = source->ratio_emax;
    strcpy(destination->filename, source->filename);
    strcpy(destination->group_name, source->group_name);
    strcpy(destination->group_name_output, source->group_name_output);
    strcpy(destination->subgroup_name, source->subgroup_name);
    strcpy(destination->meta_group_name, source->meta_group_name);
    strcpy(destination->filename_sorted, source->filename_sorted);
    strcpy(destination->filename_attribute, source->filename_attribute);
    strcpy(destination->filename_meta, source->filename_meta);
    strcpy(destination->filename_traj, source->filename_traj);
    strcpy(destination->filepath, source->filepath);
    strcpy(destination->species, source->species);
}
