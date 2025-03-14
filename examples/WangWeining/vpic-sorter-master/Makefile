# CC = mpicc
CC = cc

# change HDF5_DIR to the HDF5 install directory in your system
HDF5_DIR = $(HDF5_ROOT)

CFLAGS += -Wall -std=gnu99 -Iinclude -I$(HDF5_DIR)/include
LDFLAGS += -Llib -L$(HDF5_DIR)/lib
LDLIBS += -lm -ldl -lhdf5 -lz

SRC_DIR = src
OBJ_DIR = obj

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJ_H5GROUP = $(addprefix $(OBJ_DIR)/, configuration.o mpi_io.o vpic_data.o \
			  package_data.o qsort-parallel.o get_data.o meta_data.o \
			  particle_tags.o tracked_particle.o h5group-sorter.o)
OBJ_TRAJ = $(addprefix $(OBJ_DIR)/, time_frame_info.o particle_tags.o \
		   vpic_data.o get_data.o package_data.o mpi_io.o tracked_particle.o \
		   particle_trajectory.o)
OBJ_BH5 = $(addprefix $(OBJ_DIR)/, get_data.o package_data.o mpi_io.o \
		  binary_to_hdf5.o)
OBJ_REDUCE = $(addprefix $(OBJ_DIR)/, configuration.o particle_tags.o \
			 tracked_particle.o vpic_data.o get_data.o package_data.o \
			 mpi_io.o qsort-parallel.o meta_data.o reduce_tracer.o)
OBJ_CHECK_HDF5 = $(addprefix $(OBJ_DIR)/, get_data.o package_data.o vpic_data.o \
				 mpi_io.o check_hdf5.o)

.PHONY: all clean

all: $(OBJ) h5group-sorter h5trajectory binary_to_hdf5 reduce_tracer \
			check_hdf5

h5group-sorter: $(OBJ_H5GROUP)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

h5trajectory: $(OBJ_TRAJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

binary_to_hdf5: $(OBJ_BH5)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

reduce_tracer: $(OBJ_REDUCE)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

check_hdf5: $(OBJ_CHECK_HDF5)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@ mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ)
