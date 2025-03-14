/******************************************************************************
 * Check the datasets in an HDF5 file
 ******************************************************************************/
#include <stdlib.h>
#include "vpic_data.h"
#include "dset_name_item.h"
#include "constants.h"

int main(int argc, char *argv[])
{
    int mpi_size, mpi_rank;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    hid_t plist_id, file_id, gid;
    open_file_group_h5(argv[1], argv[2], &plist_id, &file_id, &gid);

    int dataset_num = 0;
    int is_all_dset = 1;
    int max_type_size;
    int key_index = 0;
    dset_name_item *dname_array = (dset_name_item *)malloc(
            MAX_DATASET_NUM * sizeof(dset_name_item));
    open_dataset_h5(gid, is_all_dset, key_index, dname_array,
            &dataset_num, &max_type_size);

    if (mpi_rank == 0) {
        printf("Number of datasets: %d\n", dataset_num);
        for (int i = 0 ; i < dataset_num; i++){
            printf("%2.2d, %3s, type id (%d), type size (%d)\n", i,
                    dname_array[i].dataset_name, dname_array[i].type_id,
                    dname_array[i].type_size);
            H5Dclose(dname_array[i].did);
        }
    }

    free(dname_array);
    H5Pclose(plist_id);
    H5Gclose(gid);
    H5Fclose(file_id);

    MPI_Finalize();
    return 0;
}
