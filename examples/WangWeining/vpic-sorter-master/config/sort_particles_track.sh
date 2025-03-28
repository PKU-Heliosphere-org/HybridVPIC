#!/bin/bash

# Modify these parameters
# -----------------------------------------------------------------------------
trace_particles=true
save_sorted_files=true
load_tracer_meta=false
run_name=patch
runpath=/home/wwn/HybridVPIC-main/$run_name
# 定义 particle 数组
particles=("pui" "ion" "alpha")
# 定义对应的 ntraj 数组，元素数量和顺序要与 particles 数组对应
# ntraj_values=(32000 8000 12000)
# case_index=28
# 检查输入参数合法性
if [ $# -lt 2 ]; then
  echo "Usage: $0 <case_index> <ntraj_values_comma_separated>"
  echo "Example: $0 28 32000,8000,12000"
  exit 1
fi

# 从命令行参数获取输入
case_index=$1
IFS=',' read -ra ntraj_values <<< "$2" # 将逗号分隔的字符串转换为数组

# 检查 ntraj 数量是否与粒子种类匹配
if [ ${#ntraj_values[@]} -ne ${#particles[@]} ]; then
  echo "Error: Expected ${#particles[@]} ntraj values, got ${#ntraj_values[@]}"
  exit 1
fi
tstep_min=0
tstep_max=20000
tstep_interval=50
is_recreate=0 # recreate a file?
nsteps=100
reduced_tracer=0
mpi_size=16
ux_index=6
q_index=13       # particle tag index in the HDF5 file
energy_index=14  # >= number of datasets in the HDF5 file
ratio_emax=1    # maximum energy / starting energy
echo "Maximum time step:" $tstep_max
echo "Time interval:" $tstep_interval

tstep=20000
input_file=tracers.h5p
meta_file=tracers.h5p
group_name=/Step#$tstep
group_name_output=/Step#$tstep
meta_group_name=/Step#$tstep/grid_metadata
single_h5=1  # tracers for all species + metadata are saved in a single file
single_group=1  # nsteps of tracer data are saved in the same group
# -----------------------------------------------------------------------------

if [ "$trace_particles" = true ] ; then
    additional_flags="-q"
fi

if [ "$save_sorted_files" = false ] ; then
    additional_flags="$additional_flags -w"
fi

if [ "$load_tracer_meta" = true ] ; then
    additional_flags="-r $additional_flags"
fi

echo "Additional flags: " $additional_flags

# 循环遍历每个 particle 及其对应的 ntraj 值
for i in "${!particles[@]}"
do
    particle=${particles[$i]}
    ntraj=${ntraj_values[$i]}

    energy_sorted_file=${particle}_tracer_energy_sorted.h5p
    qtag_sorted_file=${particle}_tracer_qtag_sorted.h5p

    tinterval_file=$(($nsteps * $tstep_interval))
    tstep_file=$(( $tstep / $tinterval_file ))
    tstep_file=$(( $tstep_file * $tinterval_file))
    fpath=$runpath/tracer/tracer1/T.$tstep_file
    if [ -e $fpath/$energy_sorted_file ];then
        rm $fpath/$energy_sorted_file
    fi
    for ((j=0; j<=$(($tstep_max / $tinterval_file)); j++))
    do
        tstep_file_tmp=$((j * $tinterval_file))
        fpath_tmp=$runpath/tracer/tracer1/T.$tstep_file_tmp
        if [ -e $fpath_tmp/$qtag_sorted_file ];then
            rm $fpath_tmp/$qtag_sorted_file
        fi
    done

    # Sort the tracer particle at $tstep ($tstep_max in default) at the by particle
    # energies. This will generate a new file $energy_sorted_file in the directory
    # ($fpath below). Particles at the beginning of $energy_sorted_file have the
    # Lowest energy. Particles at the end of of $energy_sorted_file have the highest
    # energy.
    # srun -n $mpi_size \
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
                     --tinterval=$tstep_interval \
                     --filepath=$runpath/tracer/tracer1 \
                     --species=$particle \
                     --is_recreate=$is_recreate \
                     --nsteps=$nsteps \
                     --reduced_tracer=$reduced_tracer \
                     --single_h5=$single_h5 \
                     --single_group=$single_group \
                     --subgroup_name=/Step#$tstep/${particle}_tracer \
                     --meta_group_name=$meta_group_name \
                     --group_name_output=$group_name_output

    # Sort tracer particles at all time steps by particle tags (q dataset in the
    # *.h5p files) and save the sorted tracers into $qtag_sorted_file.
    # At the same time, we will get some particle trajectories.
    # 1. We will select some high-energy particles from $energy_sorted_file based on
    #    $ratio_emax. Assuming the highest-energy particle has energy emax, we will
    #    search for particles with energies closest to but smaller than emax/ratio_emax.
    # 2. We will keep tracking the these selected tracer particles as we sort through
    #    the tracer particles at all time steps.
    # 3. We will save the tracer trajectory data into $filename_traj below. Each tracer
    #    particle in $filename_traj will occupy one group.

    # filename for particle trajectory data
    filename_traj=data/${particle}s_ntraj${ntraj}_${ratio_emax}emax_${case_index}.h5p

    #srun -n $mpi_size \
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
                     --tstep=$tstep \
                     --tinterval=$tstep_interval \
                     --filepath=$runpath/tracer/tracer1 \
                     --species=$particle \
                     --filename_traj=$filename_traj \
                     --nptl_traj=$ntraj \
                     --ratio_emax=$ratio_emax \
                     --is_recreate=$is_recreate \
                     --nsteps=$nsteps \
                     --reduced_tracer=$reduced_tracer \
                     --single_h5=$single_h5 \
                     --single_group=$single_group \
                     --subgroup_name=/Step#$tstep/${particle}_tracer \
                     --meta_group_name=$meta_group_name \
                     --group_name_output=$group_name_output

    data_dir=data/power_law_index/$run_name
    mkdir -p $data_dir
    mv $filename_traj $data_dir
done

cd config