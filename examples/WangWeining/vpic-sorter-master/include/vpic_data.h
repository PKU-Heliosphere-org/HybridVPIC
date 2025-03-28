#ifndef VPIC_DATA_H
#define VPIC_DATA_H

#include "dset_name_item.h"
#include "configuration.h"

//Find the type of dataset
hid_t getDataType (hid_t dtid);
int getIndexDataType(hid_t did);
char* get_vpic_data_h5(int mpi_rank, int mpi_size, config_t *config,
    int *row_size, hsize_t *my_data_size, int *dataset_num,
    int *max_type_size, int *key_value_type,
    dset_name_item *dname_array, hsize_t *my_offset);

char* get_vpic_pure_data_h5(int mpi_rank, int mpi_size, char *filename,
    char *group_name, int *row_size, hsize_t *my_data_size,
    int *dataset_num, int *max_type_size, int *key_value_type,
    dset_name_item *dname_array);

void open_file_group_h5(char *filename, char *group_name, hid_t *plist_id,
    hid_t *file_id, hid_t *gid);

void open_dataset_h5(hid_t gid, int is_all_dset, int key_index,
    dset_name_item *dname_array, int *dataset_num, int *max_type_size);

#endif
