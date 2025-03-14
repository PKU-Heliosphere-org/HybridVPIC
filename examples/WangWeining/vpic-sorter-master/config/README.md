# Scripts for running code
There are many command line arguments for running `h5group-sorter`.
You can check them out by running
```sh
./h5group-sorter -h
```
after compiling the code. We provide a few scripts to run the code.
Typically, you will have tracer files for different particle species at
several time frames after running VPIC, but theses tracers are not sorted.
Tracers can be HDF5 format or binary format. If it is binary format, we need
to use `binary_hdf5.sh` to transfer the binary data to HDF5 format first. Then,
1. Use `check_hdf5.sh` to check the dataset in the HDF5 file. You are going to
the indices of `Ux` and `q` for sorting.
    - The position each variable may not be the same as what `h5dump -H` gives.
2. Use `ene_emf.sh` to sort the tracers. This will save the sorted tracers into
new files as default.
3. Use `trajectory.sh` to get particle trajectories.

`ene_emf.sh` saves the sorted tracers as default, but you may not want to do that
if your tracer data is hundres of terabytes. `ene_emf.sh` also provides a choice to
get particle directly without saving the sorted tracers. See details below.

## Script to transfer bindary data to HDF5
`binary_hdf5.sh` calls `binary_to_hdf5` to transfer binary tracer files to HDF5 format.
You need to modify some parameters in the script. It should be self-explaining.

## Tracer sorting scripts
These scripts will use `h5group-sorter` to sort particles based on different
variables, such as particles tags and energy.

## Scripts to get particle trajectories
* `trajectory.sh`: get particle trajectories from tag-sorted tracers. It requires
that the tracers at each time frames have sorted by theirs tags.
    - `ratio_emax`: starting maximum energy / maximum energy of all particles
    
## Scripts to load modules
* `module.sh`: for LANL clusters
* `module_cray.sh`: for Cray systems at NERSC
