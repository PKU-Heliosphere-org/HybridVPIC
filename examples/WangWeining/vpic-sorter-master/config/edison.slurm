#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 2
#SBATCH -t 16:00:00
#SBATCH -J traj
#SBATCH -o traj.o%j
#SBATCH -L SCRATCH,project

#Edison has 24 cores per compute node
source module_cray.sh
export filepath=/scratch3/scratchdirs/guofan/open3d-full/tracer
export particle=electron
tstep_max=16614
tstep_min=0
tinterval=13
is_recreate=1 # recreate a file?
nsteps=1
echo "Maximum time step:" $tstep_max
echo "Time interval:" $tinterval

tstep=8307
key_index=8 # Sort by energy
srun -n 48 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer.h5p \
-k $key_index -a attribute --tmin=$tstep_min --tmax=$tstep_max \
--tinterval=$tinterval --filepath=$filepath --species=${particle} \
--is_recreate=$is_recreate --nsteps=$nsteps

key_index=7 # sort by particle tag
srun -n 48 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer.h5p \
-k $key_index -a attribute --tmin=$tstep_min --tmax=$tstep_max \
--tinterval=$tinterval --filepath=$filepath --species=${particle} \
-q -w -r --filename_traj=data/${particle}s_3.h5p \
--nptl_traj=1000 --ratio_emax=1 --is_recreate=$is_recreate --nsteps=$nsteps
