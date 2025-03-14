"""
Read particle tracer
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np

def read_var(group, dset_name, sz):
    """Read data from a HDF5 group

    Args:
        group: one HDF5 group
        var: the dataset name
        sz: the size of the data
    """
    dset = group[dset_name]
    fdata = np.zeros(sz, dtype=dset.dtype)
    dset.read_direct(fdata)
    return fdata


def read_particle_data(iptl, particle_tags, fh):
    """Read particle data for a HDF5 file

    Args:
        iptl: particle index
        particles_tags: all the particle tags
        fh: HDF5 file handler
    """
    group = fh[particle_tags[iptl]]
    dset = group['dX']
    sz, = dset.shape
    ptl = {}
    for dset in group:
        dset = str(dset)
        ptl[str(dset)] = read_var(group, dset, sz)

    return (ptl, sz)


def main():
    """
    """
    fname = "../data/electrons_3.h5p"
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    print("Total number of particles: %d" % nptl)
    pindex = 6
    ptl, sz = read_particle_data(pindex, particle_tags, fh)
    print("All fields: %s" % sorted(ptl.keys()))
    # The first two points seem to have some problem
    gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    vx = ptl["Ux"]/gamma
    vy = ptl["Uy"]/gamma
    vz = ptl["Uz"]/gamma
    x = ptl["dX"]
    y = ptl["dY"]
    z = ptl["dZ"]
    ex = ptl["Ex"]
    ey = ptl["Ey"]
    ez = ptl["Ez"]
    bx = ptl["Bx"]
    by = ptl["By"]
    bz = ptl["Bz"]
    q = ptl["q"]
    ni = ptl["ni"]
    ne = ptl["ne"]
    ntime = len(x)
    bounce = 0
    vdote_parallel = ex*bx*vx*bx + ey*by*vy*by + ez*bz*vz*bz
    vdote = ex*vx + ey*vy + ez*vz

    gamma_filter = gamma > 1.0

    # for i in range(15, ntime-15, 15):
    #   diff1 = x[i] - x[i-10]
    #   diff2 = x[i+10] - x[i]
    #   if (diff1*diff2 < 0):
    #     bounce = bounce + 1
    #     print (bounce,diff1, diff2)
    # print(gamma[ntime-1]-1)
    # print(gamma[gamma_filter])
    # plt.plot(x[gamma_filter], gamma[gamma_filter])
    plt.plot(x[gamma_filter], z[gamma_filter])
    # plt.plot(gamma)
    plt.show()


if __name__ == "__main__":
    main()
