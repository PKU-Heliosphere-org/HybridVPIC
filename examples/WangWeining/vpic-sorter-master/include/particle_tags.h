#ifndef PARTICLE_TAGS_H
#define PARTICLE_TAGS_H

#include <hdf5.h>
void get_particle_tags(char *filename, int tstep, float ratio_emax,
        int num_ptl, int *tags);
void get_particle_tags_initial(char *filename, int tstep, int rank,
        int particle_select, int *tags);
void get_hdf5_data_size(char *filename, char *groupname, char *dataset_name,
        hsize_t *dims_out);

#endif
