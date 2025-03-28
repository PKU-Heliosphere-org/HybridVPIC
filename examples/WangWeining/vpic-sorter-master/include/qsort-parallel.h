#ifndef QSORT_PARALLEL_H
#define QSORT_PARALLEL_H

#include "dset_name_item.h"
#include "configuration.h"

//Compare the key in "long long" type
int CompareLongLongKey(const void *a, const void *b);

//Phase 1 of the parallel sampling sorting
//Sort the data first and select the samples
int phase1(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        char *sample_of_rank0, int row_size);

//Phase 2 of the parallel sampling sorting
char *phase2(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        char *pivots, int row_size, int skew_data, int collect_data,
        int write_result, unsigned long long *rsize, int is_recreate);

// Master does slave's job, and also gather and sort pivots
char *master(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        int row_size, int type_size_max, int dset_num, int key_data_typ,
        dset_name_item *dataname_array, config_t *config, unsigned long long *rsize);

/*Do sort and sample*/
char *slave(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        int row_size, int type_size_max, int dset_num, int key_data_type,
        dset_name_item *dataname_array, config_t *config, unsigned long long *rsize);

//Sort the data based on the type
int qsort_type(void *data, int64_t my_data_size, size_t row_size);

//int pivots_replicated(char *pivots, int dest, int *dest_pivot_replicated_size,
//        int *dest_pivot_replicated_rank, int mpi_size, int mpi_rank, int row_size);

int pivots_replicated(char *pivots, int dest, int *dest_pivot_replicated_size,
        int *dest_pivot_replicated_rank, int mpi_size, int mpi_rank,
        int row_size, double *p_value_head);

void rank_pivot(char *pivots, char *data,  int64_t my_data_size,
        int dest_pivot, int *rank_less, int *rank_equal, int row_size,
        int mpi_size, double p_value_head);

int openmp_sort(char *data, int data_size, int threads, size_t row_size);

void create_opic_data_type(int row_size);
void free_opic_data_type();
double get_value_double(int index, char *row_data);

char* sorting_single_tstep(int tstep, int mpi_size, int mpi_rank, config_t *config,
        unsigned long long *rsize);

#endif
