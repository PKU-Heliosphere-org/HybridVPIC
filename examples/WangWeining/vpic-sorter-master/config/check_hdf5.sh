#!/bin/bash

# Modify as you need
# export filepath=/net/scratch3/xiaocanli/reconnection/open3d-full/tracer_test/
# export filepath=/lustre/scratch4/turquoise/xiaocanli/fan_prl/tracking_particle/tracer/tracer1
# export filepath=/lustre/scratch3/turquoise/xiaocanli/reconnection/Cori_runs/test_new_tracking/tracer/tracer1
export filepath=/net/scratch3/xiaocanli/reconnection/local_perturb/harris/tracer/
export particle=electron
tstep=0

# May not need to change
mpi_size=1
fpath=$filepath/T.$tstep
# input_file=$fpath/${particle}_tracer_reduced_sorted.h5p
# input_file=$fpath/${particle}_tracer.h5p

# input_file=$fpath/tracers.h5p
# group_name=/Step#$tstep/${particle}_tracer
# subgroup_name=${particle}_tracer

input_file=$fpath/${particle}_tracer.h5p
group_name=/Step#$tstep
subgroup_name=${particle}_tracer

cd ../

srun -n $mpi_size ./check_hdf5 $input_file $group_name $subgroup_name

cd config
