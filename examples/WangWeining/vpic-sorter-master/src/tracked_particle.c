#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <stdint.h>
#include "hdf5.h"
#include "time_frame_info.h"
#include "constants.h"
#include "vpic_data.h"
#include "get_data.h"
#include "package_data.h"

/******************************************************************************
 * Get tracked particles information.
 ******************************************************************************/
void get_tracked_particle_info(char *package_data, int qindex, int row_size,
        hsize_t my_data_size, int ct, int ntf, int *tags, int num_ptl, 
        char *tracked_particles)
{
    hsize_t i;
    int qvalue, qvalue_tracked, iptl;
    /* Make sure tracked qvalue is not smaller than the 1st qvalue in the data */
    qvalue = getInt32Value(qindex, package_data);
    iptl = 0;
    while (tags[iptl] < qvalue && iptl < num_ptl) {
        iptl++;
    }
    if (iptl < num_ptl) {
        qvalue_tracked = tags[iptl];
    } else {
        qvalue_tracked = -1;
    }
    for (i = 0; i < my_data_size; i++) {
        qvalue = getInt32Value(qindex, package_data + i*row_size);
        if (qvalue == qvalue_tracked) {
            memcpy(tracked_particles + (iptl * ntf + ct) * row_size,
                    package_data + i*row_size, row_size);
            if (iptl >= num_ptl-1) {
                break;
            } else {
                qvalue_tracked = tags[++iptl];
            }
        } else if (qvalue > qvalue_tracked) {
            while (qvalue_tracked < qvalue && iptl < num_ptl) {
                qvalue_tracked = tags[++iptl];
            }
            if (qvalue_tracked == qvalue) {
                memcpy(tracked_particles + (iptl * ntf + ct) * row_size,
                        package_data + i*row_size, row_size);
                if (iptl >= num_ptl-1) {
                    break;
                } else {
                    qvalue_tracked = tags[++iptl];
                }
            }
        }
    }
}

/******************************************************************************
 * Get reduced particles information.
 ******************************************************************************/
void get_reduced_particle_info(char *package_data, int qindex, int row_size,
        hsize_t my_data_size, int *tags, int max_num_ptl,
        unsigned long long *nptl_reduce, char *tracked_particles)
{
    hsize_t i;
    int qvalue, qvalue_tracked, iptl;
    /* Make sure tracked qvalue is not smaller than the 1st qvalue in the data */
    qvalue = getInt32Value(qindex, package_data);
    iptl = 0;
    while (tags[iptl] < qvalue && iptl < max_num_ptl) {
        iptl++;
    }
    if (iptl < max_num_ptl) {
        qvalue_tracked = tags[iptl];
    } else {
        qvalue_tracked = -1;
    }
    *nptl_reduce = 0;
    for (i = 0; i < my_data_size; i++) {
        qvalue = getInt32Value(qindex, package_data + i*row_size);
        if (qvalue == qvalue_tracked) {
            memcpy(tracked_particles + (*nptl_reduce) * row_size,
                    package_data + i*row_size, row_size);
            if (iptl >= max_num_ptl-1) {
                break;
            } else {
                qvalue_tracked = tags[++iptl];
            }
            (*nptl_reduce)++;
        } else if (qvalue > qvalue_tracked) {
            while (qvalue_tracked < qvalue && iptl < max_num_ptl) {
                qvalue_tracked = tags[++iptl];
            }
            if (qvalue_tracked == qvalue) {
                memcpy(tracked_particles + (*nptl_reduce) * row_size,
                        package_data + i*row_size, row_size);
                if (iptl >= max_num_ptl-1) {
                    break;
                } else {
                    qvalue_tracked = tags[++iptl];
                }
                (*nptl_reduce)++;
            }
        }
    }
}

/******************************************************************************
 * Write data from HDF5 file using one process.
 ******************************************************************************/
void write_data_serial_h5(hid_t file_id, char *gname, int dataset_num, int rank,
        dset_name_item *dname_array, hsize_t *dimsf, hsize_t *count,
        hsize_t *offset, int my_data_size, int row_size, int max_type_size,
        char *data)
{
    hid_t group_id;
    hid_t filespace, memspace;
    herr_t status;
    hid_t typeid;
    int i;

    /* Create a group */
    group_id = H5Gcreate2(file_id, gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    char *temp_data;
    temp_data = (char *)malloc(max_type_size * my_data_size);
    if(temp_data == NULL){
        printf("Memory allocation fails ! \n");
        exit(-1);
    }

    for (i = 0; i < dataset_num; i++) {
        filespace = H5Screate_simple(rank, dimsf, NULL);
        dname_array[i].did = H5Dcreate2(group_id, dname_array[i].dataset_name,
                dname_array[i].type_id, filespace, H5P_DEFAULT, H5P_DEFAULT,
                H5P_DEFAULT);

        memspace = H5Screate_simple(rank, count, NULL);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
        unpackage(data, i, my_data_size, temp_data, row_size,
                dname_array[i].type_size, max_type_size);
        H5Dwrite(dname_array[i].did, dname_array[i].type_id, memspace,
                filespace, H5P_DEFAULT, data);
        typeid = H5Dget_type(dname_array[i].did);
        switch (H5Tget_class(typeid)){
            case H5T_INTEGER:
                if(H5Tequal(typeid, H5T_STD_I32LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_INT, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I64LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_LLONG, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I8LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_CHAR, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I16LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_SHORT, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }
                break;
            case H5T_FLOAT:
                if(H5Tequal(typeid, H5T_IEEE_F32LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_FLOAT, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_IEEE_F64LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_DOUBLE, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }
                break;
            default:
                break;
        }

        /* Close/release resources */
        status = H5Dclose(dname_array[i].did);
        status = H5Sclose(memspace);
        status = H5Sclose(filespace);
    }
    free(temp_data);
    status = H5Gclose(group_id);
}

/******************************************************************************
 * Save the tracked particle data.
 ******************************************************************************/
void save_tracked_particles(char *filename_out, char *tracked_particles,
        int ntf, int num_ptl, int row_size, int dataset_num, int max_type_size,
        dset_name_item *dname_array, int *tags, int nblocks, int block_size)
{
    hid_t file_id;
    herr_t status;
    hsize_t dimsf[1], count[1], offset[1];
    char *temp_data;
    char gname[MAX_FILENAME_LEN];
    int rank;
    temp_data = (char *)malloc(ntf * row_size);
    file_id = H5Fcreate(filename_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    rank = 1;
    dimsf[0] = ntf;
    count[0] = ntf;
    offset[0] = 0;
    size_t ndata_block = (size_t)block_size * num_ptl * row_size;
    int nframes, offset_tracer;
    for (int i = 0; i < num_ptl; i++) {
      for (int j = 0; j < nblocks; j++) {
        offset_tracer = block_size * j;
        if (j == nblocks - 1) {
          nframes = ntf - (block_size * j);
        } else {
          nframes = block_size;
        }
        size_t doffset = ndata_block*j + (size_t)i*row_size*nframes;
        memcpy(temp_data+offset_tracer*row_size,
            tracked_particles+doffset, row_size*nframes);
      }
      snprintf(gname, MAX_FILENAME_LEN, "%s%d", "/Particle#", tags[i]);
      write_data_serial_h5(file_id, gname, dataset_num, rank, dname_array,
              dimsf, count, offset, ntf, row_size, max_type_size, temp_data);
    }

    status = H5Fclose(file_id);
    free(temp_data);
}

/******************************************************************************
 * Get the index of one dataset.
 ******************************************************************************/
int get_dataset_index(char *dname, dset_name_item *dname_array, int dataset_num)
{
    int i = 0;
    for (i = 0; i < dataset_num; i++) {
        if (strcmp(dname, dname_array[i].dataset_name) == 0)
            break;
    }
    return i;
}

/******************************************************************************
 * Track particles
 ******************************************************************************/
void track_particles(int mpi_rank, int mpi_size, int ntf, int tinterval,
        char *filepath, int *tags, int num_ptl, char *filename_out, char *particle)
{
    int i, row_size, dataset_num, max_type_size, key_value_type;
    hsize_t my_data_size;
    dset_name_item *dname_array;
    char *package_data;
    double t0, t1;
    char *tracked_particles, *tracked_particles_sum;
    int tstep, qindex;
    hsize_t j;

    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    char filename[MAX_FILENAME_LEN];
    char group_name[MAX_FILENAME_LEN];

    t0 = MPI_Wtime();

    tstep = 0;
    snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath, "T.",
            tstep, "/", particle, "_tracer_qtag_sorted.h5p");
    snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    package_data = get_vpic_pure_data_h5(mpi_rank, mpi_size, filename,
            group_name, &row_size, &my_data_size, &dataset_num,
            &max_type_size, &key_value_type, dname_array);

    set_variable_data(max_type_size, 0, dataset_num, key_value_type, 0);
    qindex = get_dataset_index("q", dname_array, dataset_num);

    tracked_particles = (char *)malloc(ntf * num_ptl * row_size);
    for (j = 0; j < ntf*num_ptl*row_size; j++) {
        tracked_particles[j] = 0;
    }
    if (mpi_rank == 0) {
        tracked_particles_sum = (char *)malloc(ntf * num_ptl * row_size);
        for (j = 0; j < ntf*num_ptl*row_size; j++) {
            tracked_particles_sum[j] = 0;
        }
    } else {
        tracked_particles_sum = (char *)malloc(row_size);
    }
    get_tracked_particle_info(package_data, qindex, row_size,
            my_data_size, 0, ntf, tags, num_ptl, tracked_particles);
    free(package_data);

    for (i = 1; i < ntf; i++) {
        tstep = i * tinterval;
        snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath, "T.",
                tstep, "/", particle, "_tracer_qtag_sorted.h5p");
        snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
        if (mpi_rank == 0) {
            printf("Time Step %d\n", tstep);
        }
        package_data = get_vpic_pure_data_h5(mpi_rank, mpi_size, filename,
                group_name, &row_size, &my_data_size, &dataset_num,
                &max_type_size, &key_value_type, dname_array);
        get_tracked_particle_info(package_data, qindex, row_size,
                my_data_size, i, ntf, tags, num_ptl, tracked_particles);
        free(package_data);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(tracked_particles, tracked_particles_sum, ntf*num_ptl*row_size,
            MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Save the particle data. */
    if (mpi_rank == 0) {
        save_tracked_particles(filename_out, tracked_particles_sum, ntf, num_ptl,
                row_size, dataset_num, max_type_size, dname_array, tags, 1, ntf);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    if(mpi_rank == 0)
        printf("Overall time is [%f]s \n", (t1 - t0));

    free(tracked_particles);
    if (mpi_rank == 0) {
        free(tracked_particles_sum);
    }
    free(dname_array);
}

/******************************************************************************
 * Compare the value "int32" type
 ******************************************************************************/
int CompareInt32Value (const void * a, const void * b)
{
    int va = *(const int*) a;
    int vb = *(const int*) b;
    return (va > vb) - (va < vb);
}
