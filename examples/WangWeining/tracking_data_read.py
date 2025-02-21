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
    def __init__(self, tracer_name, tracer_fullname, fdir, filename, sample_step=1):
        self.name = tracer_name
        self.fullname = tracer_fullname
        self.fdir = fdir
        self.filename = filename
        self.sample_step = sample_step
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
                # 计算采样后的粒子数量
                sampled_particle_count = len(range(0, len(particle_tags), self.sample_step))
                self.data['nptl'] = sampled_particle_count

                for i, iptl in enumerate(range(0, len(particle_tags), self.sample_step)):
                    group = fh[particle_tags[iptl]]
                    dtwpe_tracer = 0.005 * 50
                    dset = group['dX']
                    nframes, = dset.shape
                    if i == 0:
                        for key in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'bx', 'by', 'bz', 'ex', 'ey', 'ez']:
                            self.data[key] = np.zeros((sampled_particle_count, nframes))
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
                        self.data[key][i, :] = ptl[attr]
        except FileNotFoundError:
            print(f"文件 {self.filename} 未找到。")
        except Exception as e:
            print(f"加载数据时发生错误: {e}")

    # def read_multi_tracer_particle_data(self):
    #     """
    #     读取所有追踪粒子的数据，并且记录追踪粒子的总数、追踪粒子的序号、三维位置、速度信息以及电磁场数据
    #     """
    #     try:
    #         fpath = os.path.join(self.fdir, self.filename)
    #         with h5py.File(fpath, 'r') as fh:
    #             particle_tags = list(fh.keys())
    #             self.data['nptl'] = len(particle_tags)
    #             for iptl in range(0, self.data['nptl'], self.sample_step):
    #                 group = fh[particle_tags[iptl]]
    #                 dtwpe_tracer = 0.005 * 50
    #                 dset = group['dX']
    #                 nframes, = dset.shape
    #                 if iptl == 0:
    #                     for key in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'bx', 'by', 'bz', 'ex', 'ey', 'ez']:
    #                         self.data[key] = np.zeros((self.data['nptl'], nframes))
    #                     self.data["nframe"] = nframes
    #                 ptl = {}
    #                 for dset_name in group:
    #                     dset = group[dset_name]
    #                     ptl[dset_name] = np.zeros(dset.shape, dset.dtype)
    #                     dset.read_direct(ptl[dset_name])
    #                 ttracer = np.arange(0, nframes) * dtwpe_tracer
    #                 tmin, tmax = ttracer[0], ttracer[-1]
    #                 for key, attr in [('x', 'dX'), ('y', 'dY'), ('z', 'dZ'),
    #                                   ('ux', 'Ux'), ('uy', 'Uy'), ('uz', 'Uz'),
    #                                   ('bx', 'Bx'), ('by', 'By'), ('bz', 'Bz'),
    #                                   ('ex', 'Ex'), ('ey', 'Ey'), ('ez', 'Ez')]:
    #                     self.data[key][iptl, :] = ptl[attr]
    #     except FileNotFoundError:
    #         print(f"文件 {self.filename} 未找到。")
    #     except Exception as e:
    #         print(f"加载数据时发生错误: {e}")
    #
    # def __getattr__(self, attr):
    #     if attr in self.data:
    #         return self.data[attr]
    #     raise AttributeError(f"'Tracer' object has no attribute '{attr}'")

    def get_max_energy(self):
        E = self.data["ux"] ** 2 + self.data["uy"] ** 2 + self.data["uz"] ** 2
        return np.max(E[:, -1])

    def get_min_energy(self):
        E = self.data["ux"] ** 2 + self.data["uy"] ** 2 + self.data["uz"] ** 2
        return np.min(E[:, -1])

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
            if tracer.name == "ion":
                sample_step = 2
            elif tracer.name == "alpha":
                sample_step = 3
            else:
                sample_step = 8
            ex_mat = tracer.data["ex"]
            ey_mat = tracer.data["ey"]
            ez_mat = tracer.data["ez"]
            ux_mat = tracer.data["ux"]
            uy_mat = tracer.data["uy"]
            uz_mat = tracer.data["uz"]
            work_x = np.sum(ex_mat * ux_mat, axis=1)/tracer.get_max_energy()
            work_y = np.sum(ey_mat * uy_mat, axis=1)/tracer.get_max_energy()
            work_z = np.sum(ez_mat * uz_mat, axis=1)/tracer.get_max_energy()

            all_work_x.extend(work_x)
            all_work_y.extend(work_y)
            all_work_z.extend(work_z)

        all_works = [all_work_x, all_work_y, all_work_z]
        labels_axis = [r"$W_x/W_{\mathrm{max}}$", r"$W_y/W_{\mathrm{max}}$", r"$W_z/W_{\mathrm{max}}$"]
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

            text_y_spacing = 300  # 计算每个文本之间的垂直间距

            for j, tracer in enumerate(tracers):
                if i == 0:
                    work_per_tracer = np.sum(tracer.data["ex"] * tracer.data["ux"], axis=1)/tracer.get_max_energy()
                elif i == 1:
                    work_per_tracer = np.sum(tracer.data["ey"] * tracer.data["uy"], axis=1)/tracer.get_max_energy()
                else:
                    work_per_tracer = np.sum(tracer.data["ez"] * tracer.data["uz"], axis=1)/tracer.get_max_energy()
                hist = ax.hist(work_per_tracer, bins=bin_edges, label=tracer.name, alpha=0.4, color=color_lst[j],
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
                ax.text(xlim[1]*0.35, text_y, fr"Mean($W_{{{axis_lst[i]},\mathrm{{{tracer.name}}}}}$)={mean_work:.2f}", fontsize=15, ha='left',
                        color=color_lst[j])

            ax.set_xlabel(label, fontsize=15)
            ax.set_ylabel("Counts", fontsize=15)
            ax.set_xlim([min_val, max_val])
            ax.set_ylim(bottom=0)
            # ax.legend(loc='upper right')

        # 修改主标题
        plt.suptitle(" Normalized electric work histogram for different particle species", fontsize=16)  # 修改为更具描述性的标题

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
            # print(f"step={i_frame}, Threshold velocity={v_threshold}, Threshold ey={ey_threshold}")
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
        slope, intercept = np.polyfit(epoch[0:200], x_shock_arr[0:200], 1)
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
            ax.set_ylabel(r"$W_x", fontsize=15)
            ax = axes[3]
            sc = ax.scatter(x - x_shock, np.cumsum(uz * ez), c=range(nframe), cmap="jet")
            ax.plot(x - x_shock, np.cumsum(uz * ez), c="k")
            ax.axvline(0, c="k", linestyle="--")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Epoch", fontsize=15)
            ax.set_xlabel("x(from shock)", fontsize=15)
            ax.set_ylabel(r"$W_z", fontsize=15)
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
        n1, bins, patches = ax.hist(w_near_shock_arr, label="near", alpha=0.5, color="r")
        n2, bins, patches = ax.hist(w_far_shock_arr, label="far", alpha=0.5, color="b")
        ax.set_ylim([0, max(np.max(n1), np.max(n2))+100])
        ax.set_ylabel("Counts", fontsize=15)
        ax.set_xlabel(r"$W_{q}$", fontsize=15)
        ax.text(w_near_shock_arr.mean()+5, 500, fr"$\mathrm{{Mean}}(|x-x_{{\mathrm{{shock}}}}|<{xc})={w_near_shock_arr.mean():.2f}$"
                , fontsize=15, color="r")
        ax.text(w_far_shock_arr.mean()+5, 700, fr"$\mathrm{{Mean}}(x-x_{{\mathrm{{shock}}}}<-{xc})={w_far_shock_arr.mean():.2f}$"
                , fontsize=15, color="b")
        if turbulence:
            if turbulence_amplitude is not None:
                title_suffix = f"\n(Turbulence, Amplitude={turbulence_amplitude})"
            else:
                title_suffix = "\n(Turbulence)"
        else:
            title_suffix = "\n(No Turbulence)"

        ax.set_title(f"Histogram of the electric work to {self.fullname} in different shock regions {title_suffix}",
                     fontsize=15)
        plt.legend()
        plt.show()

    def num_particles_of_shock_crossing(self):
        x_shock_fit = self.x_shock_arr_fit()
        num = 0
        nptl = self.data["nptl"]
        x = self.data["x"]
        for iptl in range(nptl):
            if x[iptl, -1] < x_shock_fit[-1]:
                num = num + 1
        return num














#%%
if __name__ == "__main__":
    num_particle_traj = 2000
    ratio_emax = 1
    species_name_lst = ["ion", "alpha", "pui"]
    sample_lst = [2, 3, 8]

    fname1 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot36.h5p"
    fname2 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot18.h5p"
    fname3 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax(step=20000).h5p"
    fname4 = f"{species_name_lst[0]}_trace_data/{species_name_lst[0]}s_ntraj8000_{ratio_emax}emax.h5p"
    fname5 = f"{species_name_lst[1]}_trace_data/{species_name_lst[1]}s_ntraj12000_{ratio_emax}emax.h5p"
    fname6 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj32000_{ratio_emax}emax.h5p"
    fdir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/trace_data/"
    pui_tracer_1 = Tracer(species_name_lst[2], "PUI", fdir, fname1, sample_step=sample_lst[2])
    pui_tracer_2 = Tracer(species_name_lst[2], "PUI", fdir, fname2, sample_step=sample_lst[2])
    pui_tracer_3 = Tracer(species_name_lst[2], "PUI", fdir, fname3, sample_step=sample_lst[2])
    pui_tracer_4 = Tracer(species_name_lst[2], "PUI", fdir, fname6, sample_step=sample_lst[2])
    ion_tracer = Tracer(species_name_lst[0], r"SWI($\mathrm{H}^+$)", fdir, fname4, sample_step=sample_lst[0])
    alpha_tracer = Tracer(species_name_lst[1], r"SWI($\mathrm{He}^{2+}$)", fdir, fname5, sample_step=sample_lst[1])
#%%
    # Wy = np.sum(ion_tracer.data["uy"]*ion_tracer.data["ey"], axis=1)/ion_tracer.get_max_energy()
    # plt.hist(Wy)
    # plt.show()
    Tracer.plot_electric_force_work_histogram(pui_tracer_4, ion_tracer, alpha_tracer, color_lst=['#3498db', '#e74c3c', 'g'])
    # Tracer.plot_energy_variation(pui_tracer_1, pui_tracer_2, pui_tracer_3,
    #                              iptl_list=[1, 1, 2])
    # x_shock = pui_tracer_3.x_shock_arr()
    # Tracer.plot_energy_variation_ShockFrame(ion_tracer, iptl_list=[1000])
    # E = ion_tracer.data["ux"]**2+ion_tracer.data["uy"]**2+ion_tracer.data["uz"]**2
    # E = pui_tracer_3.data["ux"] ** 2 + pui_tracer_3.data["uy"] ** 2 + pui_tracer_3.data["uz"] ** 2
    # print(np.max(E[:, -1]))
    # print(alpha_tracer.num_particles_of_shock_crossing())
    # plt.scatter(pui_tracer_2.data["x"][:, 200], pui_tracer_2.data["ey"][:, 200])
    print([pui_tracer_4.get_min_energy(), pui_tracer_4.get_max_energy()])
    # ion_tracer.plot_electric_work_DifferentRegion(xc=10)
    plt.hist(np.sum(pui_tracer_4.data["uy"]*pui_tracer_4.data["ey"]/pui_tracer_4.get_max_energy(), axis=1))
    print(np.mean(np.sum(pui_tracer_4.data["uy"]*pui_tracer_4.data["ey"]/pui_tracer_4.get_max_energy(), axis=1)))
    plt.show()
    # pui_tracer_2.plot_electric_work_DifferentRegion(xc=10)
    # pui_tracer_2.plot_electric_work_DifferentRegion(xc=10, turbulence=True, turbulence_amplitude=0.18)
    # plt.scatter(range(ion_tracer.data["nframe"]), alpha_tracer.x_shock_arr_fit())
    # plt.scatter(range(ion_tracer.data["nframe"]), alpha_tracer.x_shock_arr())
    # plt.xlabel("Epoch", fontsize=15)
    # plt.ylabel(r"$x_{\mathrm{shock}}$", fontsize=15)
    # plt.show()

