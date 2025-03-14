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


if __name__ == "__main__":
    fname = "electrons_3.h5p"
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    bins = np.linspace(0,10,100)
    np_select = 10000
    dE = 0.1
    nbins = 100
    bounce_number = np.zeros(nbins)
    count_number = np.zeros(nbins)
    energy_gain_bin = np.zeros(nbins)
    print("Total number of particles: %d" % nptl)
    for pindex in range(0, np_select):
      ptl, sz = read_particle_data(pindex, particle_tags, fh)
      #print("All fields: %s" % sorted(ptl.keys()))
      # The first two points seem to have some problem
      gamma = np.sqrt(1 + ptl["Ux"][2:]**2 + ptl["Uy"][2:]**2 + ptl["Uz"][2:]**2)
      #vx = ptl["Ux"][2:]/gamma
      #vy = ptl["Uy"][2:]/gamma
      #vz = ptl["Uz"][2:]/gamma
      x = ptl["dX"][2:]
      #ex = ptl["Ex"][2:]
      #ey = ptl["Ey"][2:]
      #ez = ptl["Ez"][2:]
      #bx = ptl["Bx"][2:]
      #by = ptl["By"][2:]
      #bz = ptl["Bz"][2:]
      ntime = len(x)
      bounce = 0
      energy_gain = 0
      #vdote_parallel = ex*bx*vx*bx + ey*by*vy*by + ez*bz*vz*bz
      #vdote = ex*vx + ey*vy + ez*vz
      njump = 50
      for i in range(njump, ntime-njump, njump):
        diff1 = x[i] - x[i-njump]
        diff2 = x[i+njump] - x[i]
        ediff = gamma[i+njump] - gamma[i-njump]
        if (diff1*diff2 < 0):
          bounce = bounce + 1
          energy_gain = energy_gain + ediff
          #print (bounce,diff1, diff2)
      #print (bounce,gamma[ntime-1]-1)
      ienergy = gamma[ntime-1]-1
      iselect = int((gamma[ntime-1]-1)/dE)
      bounce_number[iselect] = bounce_number[iselect] + bounce
      count_number[iselect] = count_number[iselect] + 1
      energy_gain_bin[iselect] = energy_gain_bin[iselect] + energy_gain
    #plt.plot(bins,energy_gain_bin/count_number)
    #plt.xlim(0,5)
    #plt.ylim(0,5)
    plt.plot(bins,bounce_number/count_number)
    #plt.plot(bins,0.65*bins)
    #plt.plot(bins,bins)
    #plt.plot(x,gamma+vdote*10)
    plt.show()
    fh.close
