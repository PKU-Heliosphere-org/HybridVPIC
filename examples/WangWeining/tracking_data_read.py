import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
# from scipy.optimize import leastsq
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
    def plot_electric_force_work_histogram(*tracers, num_bins=20, color_lst):
        """
        画出统计x,y,z三个方向粒子分别受电场力做的总功的统计直方图
        :*tracers: 长度可变的tracer类的列表
        :num_bins: 直方图的bin数量
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        labels = [tracer.name for tracer in tracers]

        all_work_x = []
        all_work_y = []
        all_work_z = []

        # 收集所有 tracer 在三个方向的功
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

            all_work_x.extend(work_x)
            all_work_y.extend(work_y)
            all_work_z.extend(work_z)

        all_works = [all_work_x, all_work_y, all_work_z]
        labels_axis = [r"$W_x$", r"$W_y$", r"$W_z$"]
        axis_lst = ["x", "y", "z"]
        legend_handles = []  # 用于存储图例句柄
        legend_labels = []  # 用于存储图例标签
        # 颜色列表，可根据需要修改
        # colors = ['#3498db', '#e74c3c']

        for i, (ax, work, label) in enumerate(zip(axes, all_works, labels_axis)):
            # 确定数据的整体范围
            min_val = np.min(work)
            max_val = np.max(work)

            # 计算统一的 bin 边界
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            text_y_spacing = 200  # 计算每个文本之间的垂直间距



            for j, tracer in enumerate(tracers):
                if i == 0:
                    work_per_tracer = np.sum(tracer.data["ex"] * tracer.data["ux"], axis=1)
                elif i == 1:
                    work_per_tracer = np.sum(tracer.data["ey"] * tracer.data["uy"], axis=1)
                else:
                    work_per_tracer = np.sum(tracer.data["ez"] * tracer.data["uz"], axis=1)
                hist = ax.hist(work_per_tracer, bins=bin_edges, label=tracer.name, alpha=0.5, color=color_lst[j],
                        edgecolor='black', linewidth=0.5)
                if i == 0:  # 只在第一个子图中收集图例句柄和标签
                    proxy_patch = Patch(facecolor=color_lst[j], edgecolor='black', label=tracer.name)
                    legend_handles.append(proxy_patch)
                    if tracer.name == "ion":
                        legend_labels.append(r"SWI($\mathrm{H}^+$)")
                    elif tracer.name == "alpha":
                        legend_labels.append(r"SWI($\mathrm{He}^{2+}$)")
                    else:
                        legend_labels.append("PUI($\mathrm{H}^+$)")

                mean_work = np.mean(work_per_tracer)
                ylim = ax.get_ylim()
                start_y = 600  # 起始垂直位置
                text_y = start_y + j * text_y_spacing  # 计算当前文本的垂直位置
                xlim = ax.get_xlim()
                ax.text(xlim[1]*0.4, text_y, fr"Mean($W_{{{axis_lst[i]},\mathrm{{{tracer.name}}}}}$)={mean_work:.2f}", fontsize=15, ha='left',
                        color=color_lst[j])

            ax.set_xlabel(label, fontsize=15)
            ax.set_ylabel("Counts", fontsize=15)
            ax.set_xlim([min_val, max_val])
            ax.set_ylim(bottom=0)
            # ax.legend(loc='upper right')

        # 修改主标题
        plt.suptitle("Electric work histogram for different particle species", fontsize=16)  # 修改为更具描述性的标题

        # 添加浅色背景
        fig.patch.set_facecolor('#f9f9f9')
        plt.subplots_adjust(hspace=0.5, bottom=0.2)  # 调整底部间距，为图例留出空间

        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=1, fontsize=15)  # 创建并放置总图例
        plt.show()

    @staticmethod
    def plot_energy_variation_DownstreamFrame(*tracers, iptl_list):
        """
        画出在激波下游参考系下的粒子能量变化
        :*tracers: 长度可变的tracer类的列表
        :iptl_list:长度和*tracers相同的粒子序号列表，用以决定对于每个tracer要花哪个序号的粒子
        """
        # 检查 tracers 和 iptl_list 的长度是否一致
        if len(tracers) != len(iptl_list):
            raise ValueError("The number of tracers must be equal to the number of iptl values.")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for tracer, iptl in zip(tracers, iptl_list):
            x_mat = tracer.data["x"]
            ux_mat = tracer.data["ux"]
            uy_mat = tracer.data["uy"]
            uz_mat = tracer.data["uz"]
            E_mat = ux_mat ** 2 + uy_mat ** 2 + uz_mat ** 2
            nframes = tracer.data["nframe"]
            ax.scatter(x_mat[iptl, :], E_mat[iptl, :], c=range(nframes), cmap="jet")
            ax.set_xlabel("x", fontsize=15)
            ax.set_ylabel("E", fontsize=15)
        plt.show()

    def x_shock_arr(self):
        nframe = self.data["nframe"]

        x_shock = np.zeros(nframe)
        for i_frame in range(1, nframe):
            v_threshold = np.max(self.data["ux"][:, i_frame])*0.75
            ey_threshold = 0.7*np.max(self.data["ey"][:, i_frame])+(1-0.7)*np.min(self.data["ey"][:, i_frame])
            print(f"step={i_frame}, Threshold velocity={v_threshold}, Threshold ey={ey_threshold}")
            ux = self.data["ux"][:, i_frame]
            ey = self.data["ey"][:, i_frame]
            x = self.data["x"][:, i_frame]
            shock_dn_idx_1 = np.where(ey > ey_threshold)
            shock_dn_idx_2 = np.where(ux > v_threshold)
            x_shock[i_frame] = np.max(x[shock_dn_idx_1])#min(np.max(x[shock_dn_idx_1]), np.max(x[shock_dn_idx_2]))
        return x_shock

    def x_shock_arr_fit(self):
        """
        用最小二乘线性拟合求出各个时刻激波面的位置
        """
        nframe = self.data["nframe"]
        epoch = range(nframe)
        x_shock_arr = self.x_shock_arr()
        slope, intercept = np.polyfit(epoch, x_shock_arr, 1)
        return slope*epoch+intercept

    @staticmethod
    def plot_energy_variation_ShockFrame(*tracers, iptl_list):
        """
        画出在激波参考系下的粒子能量变化
        :*tracers: 长度可变的tracer类的列表
        :iptl_list:长度和*tracers相同的粒子序号列表，用以决定对于每个tracer要花哪个序号的粒子
        """
        # 检查 tracers 和 iptl_list 的长度是否一致
        if len(tracers) != len(iptl_list):
            raise ValueError("The number of tracers must be equal to the number of iptl values.")

        fig, axes = plt.subplots(4, 1, figsize=(15, 18))
        for tracer, iptl in zip(tracers, iptl_list):
            x = tracer.data["x"][iptl, :]
            ux = tracer.data["ux"][iptl, :]
            uy = tracer.data["uy"][iptl, :]
            uz = tracer.data["uz"][iptl, :]
            ex = tracer.data["ex"][iptl, :]
            ey = tracer.data["ey"][iptl, :]
            ez = tracer.data["ez"][iptl, :]
            x_shock = tracer.x_shock_arr_fit()
            E = ux ** 2 + uy ** 2 + uz ** 2
            nframe = tracer.data["nframe"]
            ax = axes[0]
            sc = ax.scatter(x-x_shock, E, c=range(nframe), cmap="jet")
            # ax.scatter(x[~upstream_condition]-x_shock[~upstream_condition], E[~upstream_condition], c="r")
            ax.plot(x-x_shock, E, c="k")
            ax.axvline(0, c="k", linestyle="--")
            ax.set_xlabel("x(from shock)", fontsize=15)
            ax.set_ylabel("E", fontsize=15)
            cbar = plt.colorbar(sc, ax=ax)
            ax = axes[2]
            sc = ax.scatter(x-x_shock, np.cumsum(uy * ey), c=range(nframe), cmap="jet")
            ax.plot(x - x_shock, np.cumsum(uy * ey), c="k")
            ax.axvline(0, c="k", linestyle="--")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Epoch", fontsize=15)
            ax.set_xlabel("x(from shock)", fontsize=15)
            ax.set_ylabel(r"$W_y$", fontsize=15)
            # ax.set_title("y direction", fontsize=15)
            ax = axes[1]
            sc = ax.scatter(x - x_shock, np.cumsum(ux * ex), c=range(nframe), cmap="jet")
            ax.plot(x - x_shock, np.cumsum(ux * ex), c="k")
            ax.axvline(0, c="k", linestyle="--")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Epoch", fontsize=15)
            ax.set_xlabel("x(from shock)", fontsize=15)
            ax.set_ylabel(r"$W_x$", fontsize=15)
            ax = axes[3]
            sc = ax.scatter(x - x_shock, np.cumsum(uz * ez), c=range(nframe), cmap="jet")
            ax.plot(x - x_shock, np.cumsum(uz * ez), c="k")
            ax.axvline(0, c="k", linestyle="--")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Epoch", fontsize=15)
            ax.set_xlabel("x(from shock)", fontsize=15)
            ax.set_ylabel(r"$W_z$", fontsize=15)
        plt.show()

    def plot_electric_work_DifferentRegion(self, xc, turbulence=False, turbulence_amplitude=None):
        """
        画出在距激波面不同距离的区域中，粒子受到的加速情况
        :xc: |x-x_shock|<xc的区域被视为是靠近激波面的
        :turbulence:在模拟过程中是否引入湍动
        :turbulence_amplitude:用以激发湍动的Alfven波振幅
        """
        x_shock_arr = self.x_shock_arr_fit()
        nptl = self.data["nptl"]
        w_near_shock_arr = np.zeros(nptl)
        w_far_shock_arr = np.zeros(nptl)
        for iptl in range(nptl):
            ux = self.data["ux"][iptl, :]
            uy = self.data["uy"][iptl, :]
            uz = self.data["uz"][iptl, :]
            ex = self.data["ex"][iptl, :]
            ey = self.data["ey"][iptl, :]
            ez = self.data["ez"][iptl, :]
            w_total = ux*ex + uy * ey + uz*ez
            condition_near_shock = (np.abs(self.x[iptl, :]-x_shock_arr) < xc)
            condition_far_shock = self.x[iptl, :] - x_shock_arr <= -xc
            w_near_shock_arr[iptl] = np.sum(w_total[condition_near_shock])
            w_far_shock_arr[iptl] = np.sum(w_total[condition_far_shock])
            # print(f"iptl={iptl}, w_near_shock={np.sum(w_total[condition_near_shock])}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(w_near_shock_arr, label="near", alpha=0.5, color="r")
        ax.hist(w_far_shock_arr, label="far", alpha=0.5, color="b")
        ax.set_ylim([0, 1200])
        ax.set_ylabel("Counts", fontsize=15)
        ax.set_xlabel(r"$W_{q}$", fontsize=15)
        ax.text(w_near_shock_arr.mean(), 500, fr"$\mathrm{{Mean}}(|x-x_{{\mathrm{{shock}}}}|<{xc})={w_near_shock_arr.mean():.2f}$"
                , fontsize=15, color="r")
        ax.text(w_far_shock_arr.mean(), 700, fr"$\mathrm{{Mean}}(x-x_{{\mathrm{{shock}}}}<-{xc})={w_far_shock_arr.mean():.2f}$"
                , fontsize=15, color="b")
        if turbulence:
            if turbulence_amplitude is not None:
                title_suffix = f"\n(Turbulence, Amplitude={turbulence_amplitude})"
            else:
                title_suffix = "\n(Turbulence)"
        else:
            title_suffix = "\n(No Turbulence)"

        ax.set_title(f"Histogram of the electric work to {self.name}s in different shock regions {title_suffix}",
                     fontsize=15)
        plt.legend()
        plt.legend()
        plt.show()












#%%
if __name__ == "__main__":
    num_particle_traj = 2000
    ratio_emax = 1
    species_name_lst = ["ion", "alpha", "pui"]

    fname1 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot36.h5p"
    fname2 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot18.h5p"
    fname3 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax.h5p"
    fname4 = f"{species_name_lst[0]}_trace_data/{species_name_lst[0]}s_ntraj{num_particle_traj}_{ratio_emax}emax.h5p"
    fname5 = f"{species_name_lst[1]}_trace_data/{species_name_lst[1]}s_ntraj{num_particle_traj}_{ratio_emax}emax.h5p"
    fdir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/trace_data/"
    pui_tracer_1 = Tracer(species_name_lst[2], fdir, fname1)
    pui_tracer_2 = Tracer(species_name_lst[2], fdir, fname2)
    pui_tracer_3 = Tracer(species_name_lst[2], fdir, fname3)
    ion_tracer = Tracer(species_name_lst[0], fdir, fname4)
    alpha_tracer = Tracer(species_name_lst[1], fdir, fname5)
    Tracer.plot_electric_force_work_histogram(pui_tracer_3, ion_tracer, alpha_tracer, color_lst=['#3498db', '#e74c3c', 'g'])
    # Tracer.plot_energy_variation(pui_tracer_1, pui_tracer_2, pui_tracer_3,
    #                              iptl_list=[1, 1, 2])
    # x_shock = pui_tracer_3.x_shock_arr()
    # Tracer.plot_energy_variation_ShockFrame(pui_tracer_2, iptl_list=[1000])
    # plt.scatter(pui_tracer_2.data["x"][:, 200], pui_tracer_2.data["ey"][:, 200])
    # alpha_tracer.plot_electric_work_DifferentRegion(xc=10)
    # pui_tracer_2.plot_electric_work_DifferentRegion(xc=10)
    # pui_tracer_2.plot_electric_work_DifferentRegion(xc=10, turbulence=True, turbulence_amplitude=0.18)
    # plt.scatter(range(ion_tracer.data["nframe"]), ion_tracer.x_shock_arr_fit())
    # plt.xlabel("Epoch", fontsize=15)
    # plt.ylabel(r"$x_{\mathrm{shock}}$", fontsize=15)
    # plt.show()

