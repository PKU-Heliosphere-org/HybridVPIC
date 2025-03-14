#!/bin/bash

filepath=/net/scratch3/xiaocanli/reconnection/turbulent-sheet3D-mixing-trinity-Feb16-test
fpath_binary=$filepath/tracer
fpath_hdf5=$filepath/tracer_test
particle=electron
tstep_min=0
tstep_max=5
tinterval=5

mpi_size_pic=256
dataset_num=8

mpi_size=32

# Create the directories
if [ ! -d "$fpath_hdf5" ]; then
    mkdir $fpath_hdf5
fi
i=$tstep_min
while [ "$i" -le "$tstep_max" ]; do
    if [ ! -d "$fpath_hdf5/T.$i" ]; then
        mkdir $fpath_hdf5/T.$i
    fi
    i=$(($i+$tinterval))
done

echo "Minimum and maximum time step:" $tstep_min $tstep_max
echo "Time interval:" $tinterval
echo "Particle species:" $particle
echo "Binary file path:" $filepath
echo "Number of CPUs used in PIC simulation:" $mpi_size_pic

cd ..

mpirun -np $mpi_size \
./binary_to_hdf5 --tmin=$tstep_min \
                 --tmax=$tstep_max \
                 --tinterval=$tinterval \
                 --fpath_binary=$fpath_binary \
                 --fpath_hdf5=$fpath_hdf5 \
                 --species=${particle} \
                 --ncpus=$mpi_size_pic \
                 --dataset_num=$dataset_num

cd config
