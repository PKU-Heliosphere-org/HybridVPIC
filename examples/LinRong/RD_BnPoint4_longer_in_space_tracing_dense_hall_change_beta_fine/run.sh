#!/bin/bash
#PBS -N test_Vasquez98
#PBS -l mem=100gb
#PBS -l nodes=1:ppn=128
#PBS -l walltime=6:00:00
#PBS -A 2024_095
#PBS -j n
#PBS -m abe
#PBS -M rong.lin@student.kuleuven.be

cd $PBS_O_WORKDIR

module purge
module load vsc-mympirun

module load OpenMPI/4.1.1-GCC-10.3.0
module load CMake/3.20.1-GCCcore-10.3.0
module load HDF5/1.10.7-gompi-2021a

mympirun Double_RD-hyb.Linux
