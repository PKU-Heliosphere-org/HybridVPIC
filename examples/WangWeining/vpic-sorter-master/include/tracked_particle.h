#ifndef TRACKED_PARTICLE_H
#define TRACKED_PARTICLE_H

#include "hdf5.h"
#include "dset_name_item.h"

void track_particles(int mpi_rank, int mpi_size, int ntf, int tinterval,
        char *filepath, int *tags, int num_ptl, char *filename_out, char *particle);
void get_tracked_particle_info(char *package_data, int qindex, int row_size,
        hsize_t my_data_size, int ct, int ntf, int *tags, int num_ptl, 
        char *tracked_particles);
int CompareInt32Value (const void * a, const void * b);
void save_tracked_particles(char *filename_out, char *tracked_particles,
        int ntf, int num_ptl, int row_size, int dataset_num, int max_type_size,
        dset_name_item *dname_array, int *tags, int nblocks, int block_size);
int get_dataset_index(char *dname, dset_name_item *dname_array, int dataset_num);
void get_reduced_particle_info(char *package_data, int qindex, int row_size,
        hsize_t my_data_size, int *tags, int max_num_ptl,
        unsigned long long *nptl_reduce, char *tracked_particles);

#endif
