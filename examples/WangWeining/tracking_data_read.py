import matplotlib.colors as mcl
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
vd = 1.86
b0 = 1

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
            'E': None, 'tag': None, 'pitch_angle': None
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
                # 计算能量 E
                ux = self.data['ux']
                uy = self.data['uy']
                uz = self.data['uz']
                bx = self.data['bx']
                by = self.data['by']
                bz = self.data['bz']
                self.data['E'] = 0.5 * (ux ** 2 + uy ** 2 + uz ** 2)
                self.data["pitch_angle"] = np.arccos((ux * bx + uy * by + uz * bz) / np.sqrt((bx**2+by**2+bz**2)*self.data['E']*2))
        except FileNotFoundError:
            print(f"文件 {self.filename} 未找到。")
        except Exception as e:
            print(f"加载数据时发生错误: {e}")

    def get_max_energy(self):
        return np.max(self.data["E"][:, -1])

    def get_min_energy(self):
        return np.min(self.data["E"][:, -1])

    def x_shock_arr(self):
        nframe = self.data["nframe"]

        x_shock = np.zeros(nframe)
        for i_frame in range(1, nframe):
            v_threshold = np.max(self.data["ux"][:, i_frame]) * 0.75
            ey_threshold = 0.75 * np.max(self.data["ey"][:, i_frame]) + (1 - 0.75) * (-vd * b0)
            # print(f"step={i_frame}, Threshold velocity={v_threshold}, Threshold ey={ey_threshold}")
            ux = self.data["ux"][:, i_frame]
            ey = self.data["ey"][:, i_frame]
            x = self.data["x"][:, i_frame]
            shock_dn_idx_1 = np.where(ey > ey_threshold)
            shock_dn_idx_2 = np.where(ux > v_threshold)
            x_shock[i_frame] = np.max(x[shock_dn_idx_1])  # min(np.max(x[shock_dn_idx_1]), np.max(x[shock_dn_idx_2]))
        return x_shock

    def x_shock_arr_fit(self):
        """
        用最小二乘线性拟合求出各个时刻激波面的位置
        """
        nframe = self.data["nframe"]
        epoch = range(nframe)
        x_shock_arr = self.x_shock_arr()
        slope, intercept = np.polyfit(epoch[10:200], x_shock_arr[10:200], 1)
        return slope * epoch + intercept

    def count_particle_crossings(self):
        """
        统计每个粒子在模拟过程中来回穿越激波面的次数
        """
        nptl = self.data['nptl']
        nframe = self.data['nframe']
        x_positions = self.data['x']
        shock_positions = self.x_shock_arr_fit()

        crossing_counts = np.zeros(nptl, dtype=int)

        for i in range(nptl):
            particle_positions = x_positions[i, :]
            # 记录粒子在每帧相对于激波面的位置，大于激波面为 1，小于为 -1
            relative_positions = np.sign(particle_positions - shock_positions)
            # 计算相邻两帧相对位置的变化，不为 0 表示发生了穿越
            crossings = np.abs(np.diff(relative_positions)) > 0
            # 统计穿越次数
            crossing_counts[i] = np.sum(crossings)

        return crossing_counts

    def num_particles_of_shock_crossing(self):
       return np.sum(self.count_particle_crossings() > 0)

    @staticmethod
    def plot_electric_force_work_histogram(*tracers, num_bins=20, color_lst, turbulence=False, turbulence_variations):
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
            work_x = np.sum(ex_mat * ux_mat, axis=1)
            work_y = np.sum(ey_mat * uy_mat, axis=1)
            work_z = np.sum(ez_mat * uz_mat, axis=1)

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
                    work_per_tracer = np.sum(tracer.data["ex"] * tracer.data["ux"], axis=1)
                elif i == 1:
                    work_per_tracer = np.sum(tracer.data["ey"] * tracer.data["uy"], axis=1)
                else:
                    work_per_tracer = np.sum(tracer.data["ez"] * tracer.data["uz"], axis=1)
                hist = ax.hist(work_per_tracer, bins=bin_edges, label=tracer.name, alpha=0.4, color=color_lst[j],
                        edgecolor='black', linewidth=0.5)
                if i == 0:  # 只在第一个子图中收集图例句柄和标签
                    if turbulence:
                        proxy_patch = Patch(facecolor=color_lst[j], edgecolor='black', label=tracer.name+fr"$\sigma^2$={turbulence_variations[j]}")
                    else:
                        proxy_patch = Patch(facecolor=color_lst[j], edgecolor='black',
                                            label=tracer.name)

                    legend_handles.append(proxy_patch)
                    if turbulence:
                        if tracer.name == "ion":
                            legend_labels.append(r"SWI($\mathrm{H}^+$)"+fr"$\sigma^2$={turbulence_variations[j]}")
                        elif tracer.name == "alpha":
                            legend_labels.append(r"SWI($\mathrm{He}^{2+}$)"+fr"$\sigma^2$={turbulence_variations[j]}")
                        else:
                            legend_labels.append("PUI($\mathrm{H}^+$)"+fr"$\sigma^2$={turbulence_variations[j]}")
                    else:
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
        plt.suptitle(" Electric work histogram for different particle species", fontsize=16)  # 修改为更具描述性的标题

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
            E = tracer.data["E"][iptl, :]
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
            condition_near_shock = (np.abs(self.data["x"][iptl, :]-x_shock_arr) < xc)
            condition_far_shock = np.abs(self.data["x"][iptl, :] - x_shock_arr + 2*xc) <= xc
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
                title_suffix = "\n"+fr"(Turbulence, Amplitude={turbulence_amplitude}, E$\in ({self.get_min_energy()/self.get_max_energy():.1f}E_{{max}},E_{{max}}))$"
            else:
                title_suffix = "\n"+fr"(Turbulence, E$\in ({self.get_min_energy()/self.get_max_energy():.1f}E_{{max}},E_{{max}}))$"
        else:
            title_suffix = "\n"+fr"(No Turbulence, E$\in ({self.get_min_energy()/self.get_max_energy():.1f}E_{{max}},E_{{max}}))$"

        ax.set_title(f"Histogram of the electric work to {self.fullname} in different shock regions {title_suffix}",
                     fontsize=15)
        plt.legend()
        plt.show()

    def plot_Vg2Y_points_ColoredByEnergy(self, clim=None, xc=10):
        x_shock = self.x_shock_arr_fit()
        vg_initial = np.sqrt(self.data["ux"][:, 0]**2+self.data["uy"][:, 0]**2)
        E_final = self.data["E"][:, self.data["nframe"]-1]
        nptl = self.data["nptl"]
        y_surf_arr = np.zeros(nptl)
        for iptl in range(nptl):
            near_condition = np.abs(self.data["x"][iptl, :] - x_shock) < xc
            y_surf_arr[iptl] = np.sum(self.data["uy"][iptl, near_condition])

        y_surf = np.sum(self.data["uy"], axis=1)
        scatter = plt.scatter(vg_initial, -y_surf_arr, c=E_final, cmap="jet", edgecolors="k", norm=mcl.LogNorm())
        plt.yscale("log")
        plt.ylim(bottom=1)
        # 如果传入了 clim 参数，则设置颜色范围
        if clim is not None:
            scatter.set_clim(clim[0], clim[1])
        cbar = plt.colorbar()
        cbar.set_label(fr"$\mathrm{{E_{{{self.fullname}}}}}$", fontsize=15)
        plt.xlabel("gyro radius", fontsize=15)
        plt.ylabel("Surfing distance around the shock", fontsize=15)

        plt.show()




















#%%
if __name__ == "__main__":
    num_particle_traj = 2000
    ratio_emax = 1
    species_name_lst = ["ion", "alpha", "pui"]
    species_fullname_lst = ["SWI(proton)", "SWI(alpha)", "PUI"]
    sample_lst = [2, 3, 8]
    ntraj_lst = [8000, 12000, 32000]
    # fname1 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot36.h5p"
    # fname2 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax_turb_amp_0dot18.h5p"
    fname3 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj{num_particle_traj}_{ratio_emax}emax(step=20000).h5p"
    fname4 = f"{species_name_lst[0]}_trace_data/{species_name_lst[0]}s_ntraj8000_{ratio_emax}emax_7.h5p"
    fname5 = f"{species_name_lst[1]}_trace_data/{species_name_lst[1]}s_ntraj12000_{ratio_emax}emax_7.h5p"
    # fname7 = f"{species_name_lst[2]}_trace_data/{species_name_lst[2]}s_ntraj32000_{ratio_emax}emax_7.h5p"
    fdir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/trace_data/"
    for index in [7, 12, 13]:
        for j in range(3):
            exec(f"fname{index}_{species_name_lst[j]}='{species_name_lst[j]}_trace_data/{species_name_lst[j]}s_ntraj{ntraj_lst[j]}_{ratio_emax}emax_{index}.h5p'")
            exec(f"{species_name_lst[j]}_tracer_{index}=Tracer('{species_name_lst[j]}','{species_fullname_lst[j]}',fdir,fname{index}_{species_name_lst[j]}, sample_step={sample_lst[j]})")
    # pui_tracer_1 = Tracer(species_name_lst[2], "PUI", fdir, fname1, sample_step=sample_lst[2])
    # pui_tracer_2 = Tracer(species_name_lst[2], "PUI", fdir, fname2, sample_step=1)
    # pui_tracer_3 = Tracer(species_name_lst[2], "PUI", fdir, fname3, sample_step=1)
    # pui_tracer_7 = Tracer(species_name_lst[2], "PUI", fdir, fname7, sample_step=sample_lst[2])
    # ion_tracer = Tracer(species_name_lst[0], "SWI(proton)", fdir, fname4, sample_step=sample_lst[0])
    # alpha_tracer = Tracer(species_name_lst[1], "SWI(alpha)", fdir, fname5, sample_step=sample_lst[1])
 #%%
    # Wy = np.sum(ion_tracer.data["uy"]*ion_tracer.data["ey"], axis=1)/ion_tracer.get_max_energy()
    # plt.hist(Wy)
    # plt.show()
    # Tracer.plot_electric_force_work_histogram(ion_tracer_13, alpha_tracer_13, pui_tracer_13,
    #                                           color_lst=['#3498db', '#e74c3c', 'g'], turbulence=True,
    #                                           turbulence_variations=[0, 1, 2])
    a = pui_tracer_12.get_max_energy()
    b = np.where(pui_tracer_12.data["E"][:, 400] > 200)
    c = pui_tracer_7.data["x"]
    d = pui_tracer_12.data["x"]
    f = pui_tracer_13.data["x"]
    g = pui_tracer_7.data["ux"]
    i = 20
    dt_tracer = 0.25
    w1_13 = pui_tracer_13.data["E"][:, 400]-pui_tracer_13.data["E"][:, 0]
    w2_13 = dt_tracer*np.sum(pui_tracer_13.data["ex"]*pui_tracer_13.data["ux"] +
                 pui_tracer_13.data["ey"]*pui_tracer_13.data["uy"] +
                 pui_tracer_13.data["ez"]*pui_tracer_13.data["uz"], axis=1)
    w1_12 = pui_tracer_12.data["E"][:, 400] - pui_tracer_12.data["E"][:, 0]
    w2_12 = dt_tracer*np.sum(pui_tracer_12.data["ex"] * pui_tracer_12.data["ux"] +
                   pui_tracer_12.data["ey"] * pui_tracer_12.data["uy"] +
                   pui_tracer_12.data["ez"] * pui_tracer_12.data["uz"], axis=1)
    w1_7 = pui_tracer_7.data["E"][:, 400] - pui_tracer_7.data["E"][:, 0]
    w2_7 = dt_tracer*np.sum(pui_tracer_7.data["ex"] * pui_tracer_7.data["ux"] +
                   pui_tracer_7.data["ey"] * pui_tracer_7.data["uy"] +
                   pui_tracer_7.data["ez"] * pui_tracer_7.data["uz"], axis=1)
    plt.hist(w2_13/w1_13, bins=np.linspace(0, 5, 10), alpha=0.5, color="r")
    plt.hist(w2_12 / w1_12, bins=np.linspace(0, 5, 10), alpha=0.5, color="g")
    plt.hist(w2_7 / w1_7, bins=np.linspace(0, 5, 10), alpha=0.3, color="b")
    plt.yscale("log")
    # plt.scatter(range(401), np.var(c, axis=0) / range(1, 402), s=1)
    # plt.scatter(range(401), np.var(d, axis=0) / range(1, 402), s=1)
    # plt.scatter(range(401), np.var(f, axis=0) / range(1, 402), s=1)
    # plt.yscale("log")
    # # plt.scatter(range(401), np.var(d, axis=0))
    # # plt.scatter(range(401), np.var(f, axis=0))
    # ey = pui_tracer_12.data["ey"]
    # x = pui_tracer_12.data["x"]
    plt.show()
    # plt.scatter(x[:, 10], ey[:, 10])
    # plt.plot(a)
    # plt.show()
    Tracer.plot_energy_variation_ShockFrame(pui_tracer_12, iptl_list=[435])
    # Tracer.plot_energy_variation(pui_tracer_1, pui_tracer_2, pui_tracer_3,
    #                              iptl_list=[1, 1, 2])
    # x_shock = pui_tracer_3.x_shock_arr()
    # Tracer.plot_energy_variation_ShockFrame(ion_tracer, iptl_list=[400])
    # E = ion_tracer.data["ux"]**2+ion_tracer.data["uy"]**2+ion_tracer.data["uz"]**2
    # E = pui_tracer_3.data["ux"] ** 2 + pui_tracer_3.data["uy"] ** 2 + pui_tracer_3.data["uz"] ** 2
    # print(np.max(E[:, -1]))
    # print(pui_tracer_4.num_particles_of_shock_crossing())
    # # plt.scatter(pui_tracer_2.data["x"][:, 200], pui_tracer_2.data["ey"][:, 200])
    # print([pui_tracer_4.get_min_energy(), pui_tracer_4.get_max_energy()])
    # pui_tracer_7.plot_Vg2Y_points_ColoredByEnergy(clim=[10, pui_tracer_7.get_max_energy()])
    # i = 100
    # y_cum = np.cumsum(pui_tracer_4.data["uy"][i, :])
    # e = pui_tracer_3.data["E"]
    # plt.plot(range(401), y_cum)
    # a = np.sqrt(pui_tracer_4.data["ux"][:, 0]**2 + pui_tracer_4.data["uy"][:, 0]**2)
    # index_1 = np.where(a < 6)
    # index_2 = np.where(a > 6)
    # E = pui_tracer_4.data["ux"] ** 2 + pui_tracer_4.data["uy"] ** 2 + pui_tracer_4.data["uz"] ** 2
    # w = E[index_1]
    # u = E[index_2]
    # w1 = w[:, 400]
    # w2 = u[:, 400]
    # n = pui_tracer_3.count_particle_crossings()
    # y = np.sum(pui_tracer_4.data["uy"], axis=1)
    # plt.scatter(a, -y, c=E[:, 400], cmap="jet")
    # plt.ylim([1, 4000])
    # plt.colorbar()
    # plt.yscale("log")
    # # plt.hist(n, bins=range(8))
    # # plt.hist(w1, alpha=0.5, label=r"$v_g<5$")
    # # plt.hist(w2, alpha=0.5, label=r"$v_g>5$")
    # # plt.legend()
    # plt.show()

    # pui_tracer_2.plot_electric_work_DifferentRegion(xc=10)
    # plt.hist(np.sum(pui_tracer_4.data["uy"]*pui_tracer_4.data["ey"]/pui_tracer_4.get_max_energy(), axis=1))
    # print(np.mean(np.sum(pui_tracer_4.data["uy"]*pui_tracer_4.data["ey"]/pui_tracer_4.get_max_energy(), axis=1)))
    # plt.show()
    # pui_tracer_2.plot_electric_work_DifferentRegion(xc=10)
    # pui_tracer_2.plot_electric_work_DifferentRegion(xc=10, turbulence=True, turbulence_amplitude=0.18)
    # plt.scatter(range(ion_tracer.data["nframe"]), alpha_tracer.x_shock_arr_fit())
    # plt.scatter(range(ion_tracer.data["nframe"]), alpha_tracer.x_shock_arr())
    # plt.xlabel("Epoch", fontsize=15)
    # plt.ylabel(r"$x_{\mathrm{shock}}$", fontsize=15)
    # plt.show()

