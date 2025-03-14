/*
 * Generally, this file sort the dataset inside a HDF5 group
 * The original code is developed by Bin Dong at LBNL.
 * Modified by Xiaocan Li.
 *
 */

#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "constants.h"
#include "qsort-parallel.h"
#include "mpi_io.h"
#include "vpic_data.h"
#include "configuration.h"
#include "get_data.h"
#include "meta_data.h"
#include "tracked_particle.h"
#include "particle_tags.h"

void set_filenames(int tstep, config_t *config_init, config_t *config);
void set_filenames_reduced(int tstep, char *filepath, char *species,
        char *filename, char *group_name, char *filename_sorted,
        char *filename_attribute, char *filename_meta);

/******************************************************************************
 * Main of the parallel sampling sort
 ******************************************************************************/
int main(int argc, char **argv){
    int mpi_size, mpi_rank;
    double t0, t1;
    int is_help;
    char *final_buff;
    unsigned long long rsize;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    t0 = MPI_Wtime();

    config_t *config = (config_t *)malloc(sizeof(config_t));
    config_t *config_init = (config_t *)malloc(sizeof(config_t));
    init_configuration(config);
    init_configuration(config_init);

    is_help = get_configuration(argc, argv, mpi_rank, config_init);
    if (config->reduce_tracer_time == 0) {
        config->reduce_factor_time = 1;  // do not reduce time resolution
    }

    // when -h flag is set to seek help of how to use this program
    if (is_help) {
        free_configuration(config);
        free_configuration(config_init);
        MPI_Finalize();
        return 1;
    }

    copy_configuration(config, config_init);

    int ntf, mtf, tstep, nsteps_tot;
    mtf = config->tmin / (config->tinterval * config->nsteps);
    ntf = config->tmax / (config->tinterval * config->nsteps) + 1;
    nsteps_tot = config->tmax / config->tinterval + 1;
    nsteps_tot = (nsteps_tot + config->reduce_factor_time - 1) / config->reduce_factor_time;

    // Get the particle tags from sorted-by-energy data of the last time frame,
    // and then sort the tags
    int *tags, qindex;
    int row_size, dataset_num, max_type_size, key_value_type;
    char *tracked_particles, *tracked_particles_sum, *package_data;
    dset_name_item *dname_array;
    hsize_t my_data_size;
    // In case tracer has too many steps, tracer data is chopped in blocks
    size_t ndata_tracer, ndata_block;
    int nblocks, block_size = 1000;
    if (config->tracking_traj) {
        tags = (int *)malloc(config->nptl_traj * sizeof(int));
        tstep = config->tstep;
        char filename_ene[MAX_FILENAME_LEN];
        int tinterval_file = config->tinterval * config->nsteps;
        int tstep_file = (tstep / tinterval_file) * tinterval_file;
        snprintf(filename_ene, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
                config->filepath, "/T.", tstep_file, "/", config->species,
                "_tracer_energy_sorted.h5p");
        // get particle tags w.r.t some energy
        get_particle_tags(filename_ene, tstep, config->ratio_emax,
                config->nptl_traj, tags);

        // sort the tags
        qsort(tags, config->nptl_traj, sizeof(int), CompareInt32Value);

        // get the indices for Ux and q in the HDF5 file
        snprintf(config->group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
        dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
        package_data = get_vpic_pure_data_h5(mpi_rank, mpi_size, filename_ene,
            config->group_name, &row_size, &my_data_size, &dataset_num,
            &max_type_size, &key_value_type, dname_array);
        free(package_data);
        qindex = get_dataset_index("q", dname_array, dataset_num);
        config->ux_kindex = get_dataset_index("Ux", dname_array, dataset_num);

        nblocks = (nsteps_tot + block_size - 1) / block_size;
        ndata_tracer = (size_t)nsteps_tot * config->nptl_traj * row_size;
        ndata_block = (size_t)block_size * config->nptl_traj * row_size;
        tracked_particles = (char *)malloc(ndata_block);
        for (size_t j = 0; j < ndata_block; j++) {
            tracked_particles[j] = 0;
        }
        if (mpi_rank == 0) {
            tracked_particles_sum = (char *)malloc(ndata_tracer);
            for (size_t j = 0; j < ndata_tracer; j++) {
                tracked_particles_sum[j] = 0;
            }
        }
    } // if (config->tracking_traj)

    if (config->multi_tsteps) {
        int block_id = 0;
        for (int i = mtf; i < ntf; i++) {
          for (int j = 0; j < config->nsteps; j++) {
            int tframe = i * config->nsteps + j;
            tstep = tframe * config->tinterval;
            if (tstep > config->tmax) break;
            if (tframe % config->reduce_factor_time == 0) {
              if (mpi_rank == 0) printf("Time Step: %d\n", tstep);
              if (config->reduced_tracer) {
                  set_filenames_reduced(tstep, config->filepath, config->species,
                          config->filename, config->group_name, config->filename_sorted,
                          config->filename_attribute, config->filename_meta);
              } else {
                  set_filenames(tstep, config_init, config);
              }
              final_buff = sorting_single_tstep(tstep, mpi_size, mpi_rank, config, &rsize);
              if (config->tracking_traj) {
                  get_tracked_particle_info(final_buff, qindex, row_size,
                          rsize, tframe % block_size, block_size, tags,
                          config->nptl_traj, tracked_particles);
              }
              if(config->collect_data == 1) {
                  free(final_buff);
              }
              if (config->tracking_traj && tframe > mtf * config->nsteps &&
                  (((tframe + 1) % block_size) == 0 || tstep == config->tmax)) {
                int data_size;
                if (block_id == nblocks - 1) {
                  data_size = (int)(ndata_tracer - ndata_block * block_id);
                  char *tmp_particles = (char *)malloc(data_size);
                  for (int j = 0; j < data_size; j++) {
                    tmp_particles[j] = 0;
                  }
                  int nsteps_final_block = nsteps_tot - block_id * block_size;
                  for (int i = 0; i < config->nptl_traj; ++i) {
                    memcpy(tmp_particles + nsteps_final_block * row_size * i,
                           tracked_particles + block_size * row_size * i,
                           nsteps_final_block * row_size);
                  }
                  MPI_Reduce(tmp_particles,
                      tracked_particles_sum + ndata_block * block_id,
                      data_size, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                  free(tmp_particles);
                } else {
                  data_size = ndata_block;
                  MPI_Reduce(tracked_particles,
                      tracked_particles_sum + ndata_block * block_id,
                      data_size, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                  for (size_t j = 0; j < ndata_block; j++) {
                    tracked_particles[j] = 0;
                  }
                }
                block_id++;
              }
            }
          } // j
        }
    } else {
        set_filenames(config->tstep, config_init, config);
        if (mpi_rank == 0) {
            printf("Input filename: %s\n", config->filename);
            printf("Group name: %s\n", config->group_name);
            printf("Output filename: %s\n", config->filename_sorted);
            printf("Meta data filename: %s\n", config->filename_meta);
            if (config->single_h5) {
                printf("Sub groupname: %s\n", config->subgroup_name);
                printf("Meta groupname: %s\n", config->meta_group_name);
            }
        }

        final_buff = sorting_single_tstep(config->tstep, mpi_size, mpi_rank, config, &rsize);
        if(config->collect_data == 1) {
            free(final_buff);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (config->tracking_traj) { // Save the particle data
        if (mpi_rank == 0) {
            save_tracked_particles(config->filename_traj, tracked_particles_sum,
                nsteps_tot, config->nptl_traj, row_size, dataset_num,
                max_type_size, dname_array, tags, nblocks, block_size);
        }
        free(tracked_particles);
        if (mpi_rank == 0) {
            free(tracked_particles_sum);
        }
        free(tags);
        free(dname_array);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    if(mpi_rank == 0) {
        printf("Overall time is [%f]s \n", (t1 - t0));
    }

    free_configuration(config);
    free_configuration(config_init);

    MPI_Finalize();
    return 0;
}

/******************************************************************************
 * Set filenames to include file directories
 ******************************************************************************/
void set_filenames(int tstep, config_t *config_init, config_t *config)
{
    // multiples steps can be in the same file
    int tinterval_file = config->tinterval * config->nsteps;
    int tstep_file = (tstep / tinterval_file) * tinterval_file;
    char *tempname = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));

    int tstep_group;
    if (config->single_group) {
      tstep_group = tstep_file;
    } else {
      tstep_group = tstep;
    }

    if (config->single_h5) {
        snprintf(config->group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep_group);
        snprintf(config->subgroup_name, MAX_FILENAME_LEN, "%s%d%s%s%s",
                "/Step#", tstep_group, "/", config->species, "_tracer");
        snprintf(config->meta_group_name, MAX_FILENAME_LEN, "%s%d%s",
                "/Step#", tstep_group, "/grid_metadata");
    } else {
        snprintf(config->group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep_group);
    }
    snprintf(config->group_name_output, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);

    strcpy(tempname, config_init->filename);
    snprintf(config->filename, MAX_FILENAME_LEN, "%s%s%d%s%s",
            config->filepath, "/T.", tstep_file, "/", tempname);

    strcpy(tempname, config_init->filename_sorted);
    snprintf(config->filename_sorted, MAX_FILENAME_LEN, "%s%s%d%s%s",
            config->filepath, "/T.", tstep_file, "/", tempname);

    snprintf(config->filename_attribute, MAX_FILENAME_LEN, "%s", "attribute");

    strcpy(tempname, config_init->filename_meta);
    snprintf(config->filename_meta, MAX_FILENAME_LEN, "%s%s%d%s%s",
            config->filepath, "/T.", tstep_file, "/", tempname);
    free(tempname);
}

/******************************************************************************
 * Set filenames for reduced tracer.
 ******************************************************************************/
void set_filenames_reduced(int tstep, char *filepath, char *species,
        char *filename, char *group_name, char *filename_sorted,
        char *filename_attribute, char *filename_meta)
{
    snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath,
            "/T.", tstep, "/", species, "_tracer_reduced_sorted.h5p");
    snprintf(filename_sorted, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
            filepath, "/T.", tstep, "/", species, "_tracer_energy_sorted.h5p");
    snprintf(filename_attribute, MAX_FILENAME_LEN, "%s", "attribute");
    snprintf(filename_meta, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath,
            "/T.", tstep, "/grid_metadata_", species, "_tracer_reduced.h5p");
}
