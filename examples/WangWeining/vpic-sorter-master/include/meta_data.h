#ifndef META_DATA_H
#define META_DATA_H

void calc_particle_positions(int mpi_rank, hsize_t my_offset, int row_size,
        int max_type_size, hsize_t my_data_size, dset_name_item *dname_array,
        int dataset_num, config_t *config, char *package_data);
void open_file_group_dset(config_t *config, hid_t *file_id, hid_t *group_id,
        dset_name_item *dname_array, hsize_t *dims_out, int *dataset_num);
void read_vpic_meta_data_h5(int dataset_num, hsize_t *dims_out,
        dset_name_item *dname_array, float *cell_sizes, int *grid_dims,
        int *np_local, float *x0, float *y0, float *z0);

// Get the number of tracer particles in each MPI rank
int* get_np_local(config_t *config, int mpi_rank, int *nframes, int *pic_mpi_size);

#endif
