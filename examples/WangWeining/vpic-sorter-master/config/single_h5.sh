#!/bin/bash

# Modify these parameters
# -----------------------------------------------------------------------------
trace_particles_without_save_sorted_files=true
trace_particles_and_save_sorted_files=true
load_tracer_data=false
filepath=/lustre/scratch3/turquoise/xiaocanli/reconnection/Cori_runs/test_new_tracking/tracer/tracer1/
particle=electron
tstep_min=0
tstep_max=18090
tinterval=45
is_recreate=1 # recreate a file?
nsteps=1
reduced_tracer=0
mpi_size=1
ux_index=6
q_index=18       # particle tag index in the HDF5 file
energy_index=19  # >= number of datasets in the HDF5 file
echo "Maximum time step:" $tstep_max
echo "Time interval:" $tinterval

tstep=18090
fpath=$filepath/T.$tstep
input_file=tracers.h5p
meta_file=tracers.h5p
energy_sorted_file=${particle}_tracer_energy_sorted.h5p
qtag_sorted_file=${particle}_tracer_qtag_sorted.h5p
group_name=/Step#$tstep
subgroup_name=/Step#$tstep/${particle}_tracer
meta_group_name=/Step#$tstep/grid_metadata
single_h5=1  # tracers for all species + metadata are saved in a single file
# -----------------------------------------------------------------------------

# if [ "$trace_particles_without_save_sorted_files" = true ] ; then
#     additional_flags="-q -w"
# fi

if [ "$trace_particles_and_save_sorted_files" = true ] ; then
    additional_flags="-q -w"
fi

if [ "$load_tracer_data" = true ] ; then
    additional_flags="-r $additional_flags"
fi

echo $additional_flags

cd ../

# sort by particle energy
rm $fpath/$energy_sorted_file
srun -n $mpi_size \
./h5group-sorter -f $input_file \
                 -o $energy_sorted_file \
                 -g $group_name \
                 -m $meta_file \
                 -k $energy_index \
                 -a attribute \
                 -u $ux_index \
                 --tmin=$tstep_min \
                 --tmax=$tstep_max \
                 --tstep=$tstep \
                 --tinterval=$tinterval \
                 --filepath=$filepath \
                 --species=$particle \
                 --is_recreate=$is_recreate \
                 --nsteps=$nsteps \
                 --reduced_tracer=$reduced_tracer \
                 --single_h5=$single_h5 \
                 --subgroup_name=$subgroup_name \
                 --meta_group_name=$meta_group_name

# # sort by particle tag and save sorted tracers
srun -n $mpi_size \
./h5group-sorter -f $input_file \
                 -o $qtag_sorted_file \
                 -g $group_name \
                 -m $meta_file \
                 -k $q_index \
                 -a attribute \
                 -u $ux_index \
                 -p $additional_flags \
                 --tmin=$tstep_min \
                 --tmax=$tstep_max \
                 --tinterval=$tinterval \
                 --filepath=$filepath \
                 --species=${particle} \
                 --filename_traj=data/${particle}s_3.h5p \
                 --nptl_traj=10 \
                 --ratio_emax=1 \
                 --is_recreate=$is_recreate \
                 --nsteps=$nsteps \
                 --reduced_tracer=$reduced_tracer \
                 --single_h5=$single_h5 \
                 --subgroup_name=$subgroup_name \
                 --meta_group_name=$meta_group_name

cd config
