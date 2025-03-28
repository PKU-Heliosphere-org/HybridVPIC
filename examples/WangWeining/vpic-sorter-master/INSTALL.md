# Install Guide

## Requirements
* Compiler: **GNU** or **Intel** compilers
* MPI library: **OpenMPI** or others
* Parallel **HDF5**

## Install
In the root directory, modify `HDF5_DIR` to your parallel **HDF5** installation directory and then
```sh
$ make
```
You will get four executable files:
* `h5group-sorter`: the parallel sorter to sort tracers
* `h5trajector`: get particle trajectories from the sorted tracers
* `reduce_tracer`: get a subset of the tracers for testing
* `bindar_to_hdf5`: transfer binary files to **HDF5** files if you dump tracers as binary format
