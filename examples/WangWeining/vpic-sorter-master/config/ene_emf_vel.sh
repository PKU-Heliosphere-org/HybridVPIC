#!/bin/bash

# export filepath=/net/scratch3/xiaocanli/tracer_test_avg/tracer_hdf5
export filepath=/net/scratch3/xiaocanli/open3d-full/reduced_tracer

# # Find the maximum time step
# export tstep_max=-1
# export ct=0
# for D in `find $filepath ! -path $filepath -type d`
# do
#     arr=( $(echo $D | awk -F "." '{print $2}') )
#     tstep_tmp=${arr[0]}
#     if [ $tstep_tmp -gt $tstep_max ]
#     then
#         tstep_max=$tstep_tmp
#     fi
#     tsteps[ct]=$tstep_tmp
#     ct=$ct+1
#     echo $tstep_tmp
# done
# tstep=$tstep_max

# From http://stackoverflow.com/a/11789688
IFS=$'\n' tsorted=($(sort -n <<<"${tsteps[*]}"))
# printf "[%s]\n" "${sorted[@]}"

# Time interval
tinterval=`expr ${tsorted[1]} - ${tsorted[0]}`

export particle=electron
tstep_min=0
tstep_max=16536
tinterval=104
is_recreate=1 # recreate a file?
# nsteps=$tinterval
nsteps=1
echo "Maximum time step:" $tstep_max
echo "Time interval:" $tinterval

key_index=17 # Sort by energy
tstep=$tstep_max
tstep_max1=`expr $tstep_max - 1`
mpirun -np 128 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer_reduced_sorted.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer_reduced.h5p \
-k $key_index -a attribute --tmin=$tstep_min --tmax=$tstep_max \
--tinterval=$tinterval --filepath=$filepath --species=${particle} -u 6 \
--is_recreate=$is_recreate --nsteps=$nsteps

key_index=16 # sort by particle tag
mpirun -np 128 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer_reduced_sorted.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer_reduced.h5p \
-k $key_index -a attribute --tmin=$tstep_min --tmax=$tstep_max \
--tinterval=$tinterval --filepath=$filepath --species=${particle} \
-p -q -w -r -u 6 --filename_traj=data/${particle}s_2.h5p \
--nptl_traj=1000 --ratio_emax=1 --is_recreate=$is_recreate --nsteps=$nsteps \
--reduced_tracer=1
