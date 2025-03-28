/*
 * This reduces tracer data size
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

void set_filenames(int tstep, char *filepath, char *species, char *filename,
        char *group_name, char *filename_sorted, char *filename_attribute,
        char *filename_meta, char *filename_reduced);

/******************************************************************************
 * Main of the parallel sampling sort
 ******************************************************************************/
int main(int argc, char **argv){
    int mpi_size, mpi_rank;
    double t0, t1;
    int is_help;
    char *filename_reduced, *final_buff;
    unsigned long long rsize;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    config_t *config = (config_t *)malloc(sizeof(config_t));
    config->filename = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->group_name = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_sorted = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_attribute = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_meta = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filename_traj = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->filepath = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    config->species = (char *)malloc(16 * sizeof(char));

    t0 = MPI_Wtime();
    is_help = get_configuration(argc, argv, mpi_rank, config);

    /* when -h flag is set to seek help of how to use this program */
    if (is_help) {
        MPI_Finalize();
        return 1;
    }

    int ntf, mtf, tstep;
    mtf = config->tmin / config->tinterval;
    ntf = config->tmax / config->tinterval + 1;

    /* Get the particle tags from sorted-by-energy data of the last time frame */
    /* Then sort the tags */
    int *tags, qindex;
    int row_size, dataset_num, max_type_size, key_value_type;
    char *tracked_particles, *tracked_particles_sum, *package_data;
    dset_name_item *dname_array;
    hsize_t my_data_size;
    char filename_initial[MAX_FILENAME_LEN];
    tstep = 0;
    snprintf(filename_initial, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
            config->filepath, "/T.", tstep, "/", config->species,
            "_tracer.h5p");
    snprintf(config->group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    if (config->nsteps != 1) tstep--;
    int rank = 1;
    int particle_select = 25;
    hsize_t dims_out[rank], count;
    if (mpi_rank == 0) {
        get_hdf5_data_size(filename_initial, config->group_name, "q", dims_out);
    }
    MPI_Bcast(dims_out, rank, MPI_LONG, 0, MPI_COMM_WORLD);
    count = ceil(dims_out[0] / (particle_select + 0.0));
    tags = (int *)malloc(count * sizeof(int));
    if (mpi_rank == 0) {
        get_particle_tags_initial(filename_initial, tstep, rank,
                particle_select, tags);
    }
    MPI_Bcast(tags, count, MPI_INT, 0, MPI_COMM_WORLD);
    qsort(tags, count, sizeof(int), CompareInt32Value);
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    // Just for getting the attributes of the HDF5 file
    package_data = get_vpic_pure_data_h5(mpi_rank, mpi_size, filename_initial,
        config->group_name, &row_size, &my_data_size, &dataset_num,
        &max_type_size, &key_value_type, dname_array);
    free(package_data);
    qindex = get_dataset_index("q", dname_array, dataset_num);

    /* // Initialize meta data information */
    /* float cell_sizes[3]; */
    /* int grid_dims[3]; */
    /* int dataset_num_meta; */
    /* int *np_local; */
    /* long int *np_global; */
    /* float *x0, *y0, *z0; */
    /* int dim; */
    /* dset_name_item *dname_array_meta; */
    /* hid_t file_id, group_id; */
    /* hsize_t dims_out_meta[1]; */
    /* if (mpi_rank == 0) { */
    /*     dname_array_meta = (dset_name_item *)malloc(MAX_DATASET_NUM * */
    /*             sizeof(dset_name_item)); */
    /*     open_file_group_dset(filename_meta, group_name, &file_id, &group_id, */
    /*             dname_array_meta, dims_out_meta, &dataset_num_meta); */
    /*     dim = (int)dims_out_meta[0]; */
    /*     np_local = (int *)malloc(dim * sizeof(int)); */
    /*     x0 = (float *)malloc(dim * sizeof(float)); */
    /*     y0 = (float *)malloc(dim * sizeof(float)); */
    /*     z0 = (float *)malloc(dim * sizeof(float)); */
    /*     H5Gclose(group_id); */
    /*     H5Fclose(file_id); */
    /* } */
    /* MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD); */

    int particle_per_core = count / mpi_size;
    // 2 times larger to make sure the data size is enough
    tracked_particles = (char *)calloc(particle_per_core * row_size * 2, sizeof(char));

    filename_reduced = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    unsigned long long nptl_reduce = 0;
    for (int i = mtf; i < ntf; i++) {
        tstep = i * config->tinterval;
        if (mpi_rank == 0) printf("%d\n", tstep);
        set_filenames(tstep, config->filepath, config->species, config->filename,
                config->group_name, config->filename_sorted, config->filename_attribute,
                config->filename_meta, filename_reduced);
        final_buff = sorting_single_tstep(tstep, mpi_size, mpi_rank, config, &rsize);
        get_reduced_particle_info(final_buff, qindex, row_size, rsize, tags,
                count, &nptl_reduce, tracked_particles);
        if(config->collect_data == 1) {
            free(final_buff);
        }
        write_result_file(mpi_rank, mpi_size, tracked_particles, nptl_reduce,
                row_size, dataset_num, max_type_size, config->key_index, config->group_name,
                filename_reduced, config->filename_attribute, dname_array, config->is_recreate);

/*         /1* Read the data and broadcast to all MPI processes *1/ */
/*         if (mpi_rank == 0) { */
/*             open_file_group_dset(filename_meta, group_name, &file_id, &group_id, */
/*                     dname_array_meta, dims_out_meta, &dataset_num_meta); */
/*             read_vpic_meta_data_h5(dataset_num_meta, dims_out_meta, */
/*                     dname_array_meta, cell_sizes, grid_dims, np_local, x0, y0, z0); */
/*             H5Gclose(group_id); */
/*             H5Fclose(file_id); */
/*         } */
/*         if (mpi_rank == 0) { */
/*             for (int j = 0; j < dim; j++) { */
/*                 printf("%d\n", np_local[j]); */
/*             } */
/*         } */
/*         MPI_Gather(&nptl_reduce, 1, MPI_INT, np_local, 1, MPI_INT, 0, */
/*                 MPI_COMM_WORLD); */
/*         if (mpi_rank == 0) { */
/*             for (int j = 0; j < dim; j++) { */
/*                 printf("%d\n", np_local[j]); */
/*             } */
/*         } */
    }
    free(tracked_particles);

    /* if (mpi_rank == 0) { */
    /*     free(np_local); */
    /*     free(x0); */
    /*     free(y0); */
    /*     free(z0); */
    /*     free(dname_array_meta); */
    /* } */

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    if(mpi_rank == 0) {
        printf("Overall time is [%f]s \n", (t1 - t0));
    }

    free(filename_reduced);
    free(tags);

    free(config->filename);
    free(config->group_name);
    free(config->filename_sorted);
    free(config->filename_attribute);
    free(config->filename_meta);
    free(config->filename_traj);
    free(config->filepath);
    free(config->species);
    free(config);

    MPI_Finalize();
    return 0;
}

/******************************************************************************
 * Set filenames.
 ******************************************************************************/
void set_filenames(int tstep, char *filepath, char *species, char *filename,
        char *group_name, char *filename_sorted, char *filename_attribute,
        char *filename_meta, char *filename_reduced)
{
    snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath,
            "/T.", tstep, "/", species, "_tracer.h5p");
    snprintf(filename_sorted, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
            filepath, "/T.", tstep, "/", species, "_tracer_sorted.h5p");
    snprintf(filename_reduced, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
            filepath, "/T.", tstep, "/", species, "_tracer_reduced.h5p");
    snprintf(filename_attribute, MAX_FILENAME_LEN, "%s", "attribute");
    snprintf(filename_meta, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath,
            "/T.", tstep, "/grid_metadata_", species, "_tracer.h5p");
}
