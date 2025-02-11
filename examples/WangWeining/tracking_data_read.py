import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import errno
import palettable
import pyvista as pv
#%%


class Tracer:
    def __init__(self, tracer_name, fdir, filename):
        self.name = tracer_name
        self.fdir = fdir
        self.filename = filename
        # 初始化属性字典
        self.data = {
            'nptl': None, 'nframe': None,
            'x': None, 'y': None, 'z': None,
            'ux': None, 'uy': None, 'uz': None,
            'ex': None, 'ey': None, 'ez': None,
            'bx': None, 'by': None, 'bz': None,
            'tag': None
        }
        self.read_multi_tracer_particle_data()

    def read_multi_tracer_particle_data(self):
        """
        读取所有追踪粒子的数据，并且记录追踪粒子的总数、追踪粒子的序号、三维位置、速度信息以及电磁场数据
        """
        try:
            fpath = os.path.join(self.fdir, self.filename)
            with h5py.File(fpath, 'r') as fh:
                particle_tags = list(fh.keys())
                self.data['nptl'] = len(particle_tags)
                for iptl in range(self.data['nptl']):
                    group = fh[particle_tags[iptl]]
                    dtwpe_tracer = 0.005 * 50
                    dset = group['dX']
                    nframes, = dset.shape
                    if iptl == 0:
                        for key in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'bx', 'by', 'bz', 'ex', 'ey', 'ez']:
                            self.data[key] = np.zeros((self.data['nptl'], nframes))
                        self.data["nframe"] = nframes
                    ptl = {}
                    for dset_name in group:
                        dset = group[dset_name]
                        ptl[dset_name] = np.zeros(dset.shape, dset.dtype)
                        dset.read_direct(ptl[dset_name])
                    ttracer = np.arange(0, nframes) * dtwpe_tracer
                    tmin, tmax = ttracer[0], ttracer[-1]
                    for key, attr in [('x', 'dX'), ('y', 'dY'), ('z', 'dZ'),
                                      ('ux', 'Ux'), ('uy', 'Uy'), ('uz', 'Uz'),
                                      ('bx', 'Bx'), ('by', 'By'), ('bz', 'Bz'),
                                      ('ex', 'Ex'), ('ey', 'Ey'), ('ez', 'Ez')]:
                        self.data[key][iptl, :] = ptl[attr]
        except FileNotFoundError:
            print(f"文件 {self.filename} 未找到。")
        except Exception as e:
            print(f"加载数据时发生错误: {e}")

    def __getattr__(self, attr):
        if attr in self.data:
            return self.data[attr]
        raise AttributeError(f"'Tracer' object has no attribute '{attr}'")

    @staticmethod
    def plot_electric_force_work_histogram(*tracers, title="Case for turbulence amplitude=0.36"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        labels = [tracer.name for tracer in tracers]
        for tracer in tracers:
            ex_mat = tracer.data["ex"]
            ey_mat = tracer.data["ey"]
            ez_mat = tracer.data["ez"]
            ux_mat = tracer.data["ux"]
            uy_mat = tracer.data["uy"]
            uz_mat = tracer.data["uz"]
            work_x = np.sum(ex_mat * ux_mat, axis=1)
            work_y = np.sum(ey_mat * uy_mat, axis=1)
            work_z = np.sum(ez_mat * uz_mat, axis=1)
            ax = axes[0]
            ax.hist(work_x, label=tracer.name, alpha=0.7)
            ax.set_xlabel(r"$W_x$", fontsize=15)
            ax.set_ylabel("counts", fontsize=15)
            ax.set_xlim([-300, 1000])
            ax.set_ylim([0, 1000])
            ax = axes[1]
            ax.hist(work_y, label=tracer.name, alpha=0.7)
            ax.set_xlabel(r"$W_y$", fontsize=15)
            ax.set_ylabel("counts", fontsize=15)
            ax.set_xlim([-300, 1000])
            ax.set_ylim([0, 1000])
            ax = axes[2]
            ax.hist(work_z, label=tracer.name, alpha=0.7)
            ax.set_xlabel(r"$W_z$", fontsize=15)
            ax.set_ylabel("counts", fontsize=15)
            ax.set_xlim([-300, 1000])
            ax.set_ylim([0, 1000])
        for i, ax in enumerate(axes):
            ax.legend()
            if i == 0:
                work = np.concatenate([np.sum(tracer.data["ex"] * tracer.data["ux"], axis=1) for tracer in tracers])
            elif i == 1:
                work = np.concatenate([np.sum(tracer.data["ey"] * tracer.data["uy"], axis=1) for tracer in tracers])
            else:
                work = np.concatenate([np.sum(tracer.data["ez"] * tracer.data["uz"], axis=1) for tracer in tracers])
            ax.text(500, 800, f"Mean={np.mean(work):.2f}", fontsize=15)
        plt.suptitle(title, fontsize=16)
        plt.show()

    @staticmethod
    def plot_energy_variation(*tracers, iptl):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for tracer in tracers:
            x_mat = tracer.data["x"]
            ux_mat = tracer.data["ux"]
            uy_mat = tracer.data["uy"]
            uz_mat = tracer.data["uz"]
            E_mat = ux_mat**2+uy_mat**2+uz_mat**2
            nframes = tracer.data["nframe"]
            ax.scatter(x_mat[iptl, :], E_mat[iptl, :], c=range(nframes), cmap="jet")
            ax.set_xlabel("x", fontsize=15)
            ax.set_ylabel("E", fontsize=15)
        plt.show()


#%%
if __name__ == "__main__":
    num_particle_traj = 2000
    ratio_emax = 1
    name = "pui"
    fname1 = f"{name}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot36.h5p"
    fname2 = f"{name}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot18.h5p"
    fdir = "D:/Research/Codes/Hybrid-vpic/data_ip_shock/trace_data/pui_trace_data/"
    pui_tracer_1 = Tracer(name, fdir, fname1)
    pui_tracer_2 = Tracer(name, fdir, fname2)
    Tracer.plot_electric_force_work_histogram(pui_tracer_1, pui_tracer_2, title="abc")
    Tracer.plot_energy_variation(pui_tracer_1, pui_tracer_2, iptl=1)

