"""
Particle phase space diagram

There are a few things you need to set and the rest are controlled by the
command line arguments.
1. topo_x, topo_y, topo_z, particle_interval for your run.
2. xcut, xwidth, nzones_z, nvbins, vmax_vth, tframe, and species through
   commandline arguments. Check out the description of the command line
   arguments using "python phase_diagram.py -h". For example,
   python phase_diagram.py --xcut 64.0 --xwidth 10.0 --nzones_z 64 \
          --nvbins 32 --vmax_vth 5.0 --tframe 10 --species e
3. You can also process multiple frames like
python phase_diagram.py --multi_frames --tstart 1 --tend 10 --species e
"""
import argparse
import collections
import errno
import math
import os
import scipy.io as sio
from scipy.optimize import leastsq
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as sio
import scipy.stats as scs

#plt.style.use("seaborn-deep")
# mpl.rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = \
        (r"\usepackage{amsmath, bm}" +
         r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}" +
         r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}" +
         r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}")

# give some PIC simulation parameters here
topo_x, topo_y, topo_z = 128, 1, 2
particle_interval = 1616


def mkdir_p(path):
    """Create a directory
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_vpic_info(pic_run_dir):
    """Get information of the VPIC simulation
    """
    with open(pic_run_dir + '/info') as f:
        content = f.readlines()
    f.close()
    vpic_info = {}
    for line in content[1:]:
        if "=" in line:
            line_splits = line.split("=")
        elif ":" in line:
            line_splits = line.split(":")

        tail = line_splits[1].split("\n")
        vpic_info[line_splits[0].strip()] = float(tail[0])
    return vpic_info


def read_particle_header(fh):
    """Read particle file header

    Args:
        fh: file handler.
    """
    offset = 23  # the size of the boilerplate is 23
    tmp1 = np.memmap(fh,
                     dtype='int32',
                     mode='r',
                     offset=offset,
                     shape=(6),
                     order='F')
    offset += 6 * 4
    tmp2 = np.memmap(fh,
                     dtype='float32',
                     mode='r',
                     offset=offset,
                     shape=(10),
                     order='F')
    offset += 10 * 4
    tmp3 = np.memmap(fh,
                     dtype='int32',
                     mode='r',
                     offset=offset,
                     shape=(4),
                     order='F')
    v0header = collections.namedtuple("v0header", [
        "version", "type", "nt", "nx", "ny", "nz", "dt", "dx", "dy", "dz",
        "x0", "y0", "z0", "cvac", "eps0", "damp", "rank", "ndom", "spid",
        "spqm"
    ])
    v0 = v0header(version=tmp1[0],
                  type=tmp1[1],
                  nt=tmp1[2],
                  nx=tmp1[3],
                  ny=tmp1[4],
                  nz=tmp1[5],
                  dt=tmp2[0],
                  dx=tmp2[1],
                  dy=tmp2[2],
                  dz=tmp2[3],
                  x0=tmp2[4],
                  y0=tmp2[5],
                  z0=tmp2[6],
                  cvac=tmp2[7],
                  eps0=tmp2[8],
                  damp=tmp2[9],
                  rank=tmp3[0],
                  ndom=tmp3[1],
                  spid=tmp3[2],
                  spqm=tmp3[3])
    header_particle = collections.namedtuple("header_particle",
                                             ["size", "ndim", "dim"])
    offset += 4 * 4
    tmp4 = np.memmap(fh,
                     dtype='int32',
                     mode='r',
                     offset=offset,
                     shape=(3),
                     order='F')
    pheader = header_particle(size=tmp4[0], ndim=tmp4[1], dim=tmp4[2])
    offset += 3 * 4
    return (v0, pheader, offset)


def read_boilerplate(fh):
    """Read boilerplate of a file

    Args:
        fh: file handler
    """
    offset = 0
    sizearr = np.memmap(fh,
                        dtype='int8',
                        mode='r',
                        offset=offset,
                        shape=(5),
                        order='F')
    offset += 5
    cafevar = np.memmap(fh,
                        dtype='int16',
                        mode='r',
                        offset=offset,
                        shape=(1),
                        order='F')
    offset += 2
    deadbeefvar = np.memmap(fh,
                            dtype='int32',
                            mode='r',
                            offset=offset,
                            shape=(1),
                            order='F')
    offset += 4
    realone = np.memmap(fh,
                        dtype='float32',
                        mode='r',
                        offset=offset,
                        shape=(1),
                        order='F')
    offset += 4
    doubleone = np.memmap(fh,
                          dtype='float64',
                          mode='r',
                          offset=offset,
                          shape=(1),
                          order='F')


def read_particle_data(fname):
    """Read particle information from a file.

    Args:
        fname: file name.
    """
    fh = open(fname, 'r')
    read_boilerplate(fh)
    v0, pheader, offset = read_particle_header(fh)
    nptl = pheader.dim
    particle_type = np.dtype([('dxyz', np.float32, 3), ('icell', np.int32),
                              ('u', np.float32, 3), ('q', np.float32)])
    fh.seek(offset, os.SEEK_SET)
    data = np.fromfile(fh, dtype=particle_type, count=nptl)
    fh.close()
    return v0, pheader, data


def read_field_data(fname):
    """Read particle information from a file.

    Args:
        fname: file name.
    """
    fh = open(fname, 'r')
    read_boilerplate(fh)
    v0, pheader, offset = read_particle_header(fh)
    nptl = pheader.dim
    particle_type = np.dtype([('dxyz', np.float32, 3), ('icell', np.int32),
                              ('u', np.float32, 3), ('q', np.float32)])
    fh.seek(offset, os.SEEK_SET)
    data = np.fromfile(fh, dtype=particle_type, count=nptl)
    fh.close()
    return (v0, pheader, data)


def plot_phase_diagram(plot_config, show_plot=True):
    """Plot particle phase space diagram
    """
    pic_run_dir = plot_config["pic_run_dir"]
    vpic_info = get_vpic_info(pic_run_dir)
    lx_pic = vpic_info["Lx/de"]
    lz_pic = vpic_info["Lz/de"]
    nx_pic = int(vpic_info["nx"])
    nz_pic = int(vpic_info["nz"])
    dx_de = lx_pic / nx_pic
    dz_de = lz_pic / nz_pic
    dx_rank = lx_pic / topo_x
    dz_rank = lz_pic / topo_z
    xmin, xmax = 0, lx_pic
    zmin, zmax = -0.5 * lz_pic, 0.5 * lz_pic

    nzones_z = plot_config["nzones_z"]
    xcut = plot_config["xcut"]
    xwidth = plot_config["xwidth"]
    nz_per_zone = nz_pic // nzones_z
    dz_zone = nz_per_zone * dz_de
    xs = xcut - xwidth * 0.5
    xe = xcut + xwidth * 0.5
    srankx = math.floor(xs / dx_rank)
    erankx = math.ceil(xe / dx_rank)

    zbins = np.linspace(zmin, zmax, nzones_z + 1)
    species = plot_config["species"]
    nvbins = plot_config["nvbins"]
    if species in ["e", "electron"]:
        vth = vpic_info["vtheb/c"]
        pname = "eparticle"
    else:
        vth = vpic_info["vthib/c"]
        pname = "hparticle"
    vmax_vth = plot_config["vmax_vth"]
    vmin, vmax = -vmax_vth * vth, vmax_vth * vth
    vmin_norm, vmax_norm = vmin / vth, vmax / vth
    vbins = np.linspace(vmin, vmax, nvbins + 1)
    pdist = np.zeros([nzones_z, nvbins])

    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    tindex = tframe * particle_interval
    dir_name = pic_run_dir + 'particle/T.' + str(tindex) + '/'
    fbase = dir_name + pname + '.' + str(tindex) + '.'

    for mpi_iz in range(topo_z):
        for mpi_ix in range(srankx, erankx + 1):
            mpi_rank = mpi_iz * topo_x + mpi_ix
            fname = fbase + str(mpi_rank)
            print(fname)
            v0, pheader, ptl = read_particle_data(fname)
            ux = ptl['u'][:, 0]
            uy = ptl['u'][:, 1]
            uz = ptl['u'][:, 2]
            gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
            vx = ux / gamma
            vy = uy / gamma
            vz = uz / gamma
            dx = ptl['dxyz'][:, 0]
            dz = ptl['dxyz'][:, 2]
            nx = v0.nx + 2
            ny = v0.ny + 2
            icell = ptl['icell']
            ix = icell % nx
            iz = icell // (nx * ny)
            x = v0.x0 + ((ix - 1.0) + (dx + 1.0) * 0.5) * v0.dx
            z = v0.z0 + ((iz - 1.0) + (dz + 1.0) * 0.5) * v0.dz
            condx = np.logical_and(x > xs, x < xe)
            hist, _, _ = np.histogram2d(z[condx],
                                        vz[condx],
                                        bins=(zbins, vbins))
            pdist += hist

    fig = plt.figure(figsize=[7, 4])
    rect = [0.1, 0.15, 0.78, 0.75]
    ax1 = fig.add_axes(rect)
    dmin, dmax = 1, 1.5E3
    im1 = ax1.imshow(pdist.T,
                     extent=[zmin, zmax, vmin_norm, vmax_norm],
                     cmap=plt.cm.jet,
                     aspect='auto',
                     origin='lower',
                     interpolation='bicubic')
    ax1.set_xlabel(r'$z/d_e$', fontsize=16)
    if species in ["e", "electron"]:
        ylabel = r"$v_z/v_\text{the}$"
    else:
        ylabel = r"$v_z/v_\text{thi}$"
    ax1.set_ylabel(ylabel, fontsize=16)
    ax1.tick_params(labelsize=12)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += 0.45
    rect_cbar[1] += 0.1
    rect_cbar[2] = 0.3
    rect_cbar[3] = 0.03
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1,
                        cax=cbar_ax,
                        extend='max',
                        orientation="horizontal")
    cbar.ax.tick_params(labelsize=10, color='w')
    cbar.ax.yaxis.set_tick_params(color='w')
    cbar.outline.set_edgecolor('w')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
    cbar.ax.tick_params(labelsize=12)

    fname = pic_run_dir + "data/bx.gda"
    bx = np.fromfile(fname,
                     offset=nx_pic * nz_pic * tframe * 4,
                     count=nx_pic * nz_pic,
                     dtype=np.float32)
    bx = bx.reshape([nz_pic, nx_pic])
    ax2 = ax1.twinx()
    ix = int(xcut / dx_de)
    zgrid = np.linspace(zmin, zmax, nz_pic)
    b0 = vpic_info["b0"]
    ax2.plot(zgrid, bx[:, ix] / b0, linewidth=1, color='w', alpha=0.7)
    ax2.set_ylabel(r"$B_x/B_0$", fontsize=16)
    ax2.tick_params(labelsize=12)

    twpe = math.ceil(tindex * vpic_info["dt*wpe"] / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=16)

    img_dir = 'img/phase_diagram/'
    img_dir += "tframe_" + str(tframe) + "/"
    mkdir_p(img_dir)
    fname = (img_dir + "phase_x" + str(xcut) + "_xw" + str(xwidth) +
             "_nzones" + str(nzones_z) + "_" + species + ".jpg")
    fig.savefig(fname, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()


def get_cmd_args():
    """Get command line arguments """
    default_run_name = "test"
    default_run_dir = ("./")
    parser = argparse.ArgumentParser(
        description="Particle phase space diagram")
    parser.add_argument("--pic_run_dir",
                        action="store",
                        default=default_run_dir,
                        help="PIC run directory")
    parser.add_argument("--pic_run",
                        action="store",
                        default=default_run_name,
                        help="PIC run name")
    parser.add_argument("--xcut",
                        action="store",
                        default=64.0,
                        type=float,
                        help="x-position of the slice in de")
    parser.add_argument("--xwidth",
                        action="store",
                        default=10.0,
                        type=float,
                        help="Width of the slice in de")
    parser.add_argument("--nzones_z",
                        action="store",
                        default=64,
                        type=int,
                        help="Number of zones along z")
    parser.add_argument("--nvbins",
                        action="store",
                        default=32,
                        type=int,
                        help="Number of velocity bins")
    parser.add_argument(
        "--vmax_vth",
        action="store",
        default=5,
        type=float,
        help="Maximum velocity in the unit of thermal velocity")
    parser.add_argument('--tframe',
                        action="store",
                        default='4',
                        type=int,
                        help='Time frame')
    parser.add_argument('--multi_frames',
                        action="store_true",
                        default=False,
                        help='whether to analyze multiple frames')
    parser.add_argument('--tstart',
                        action="store",
                        default='1',
                        type=int,
                        help='Starting time frame')
    parser.add_argument('--tend',
                        action="store",
                        default='4',
                        type=int,
                        help='Ending time frame')
    parser.add_argument("--species",
                        action="store",
                        default="electron",
                        help="particle species")
    return parser.parse_args()


def analysis_single_frame(plot_config, args):
    """Analysis for single time frame
    """
    print("Time frame: %d" % plot_config["tframe"])
    plot_phase_diagram(plot_config, show_plot=True)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(args.tstart, args.tend + 1)
    for tframe in tframes:
        plot_config["tframe"] = tframe
        plot_phase_diagram(plot_config, show_plot=False)


class Species:
    def __init__(self, name, filename, num_files):
        """
        初始化Species类
        :param name: 粒子种类的名称
        :param filename: 存储粒子数据的文件名
        """
        self.name = name
        self.filename = filename
        self.num_files = num_files
        self.x = None
        self.y = None
        self.z = None
        self.ux = None
        self.uy = None
        self.uz = None
        self.read_multiple_particle_files()

    def read_multiple_particle_files(self):
        """
        读取多个粒子文件并合并数据
        :param base_fname: 文件名模板，例如 'data_ip_shock/particle_data_{run_case_index}/T.{step}/Hparticle_SWI.{step}.{}'
        :param num_files: 文件数量
        :return: 合并后的 x, y, z, ux, uy, uz 数据
        """
        try:
            for i in range(self.num_files):
                fname = self.filename.format(i)
                v0, pheader, ptl = read_particle_data(fname)

                ux = ptl['u'][:, 0]
                uy = ptl['u'][:, 1]
                uz = ptl['u'][:, 2]
                dx = ptl['dxyz'][:, 0]
                dy = ptl['dxyz'][:, 1]
                dz = ptl['dxyz'][:, 2]
                nx = v0.nx + 2
                ny = v0.ny + 2
                nz = v0.nz + 2
                icell = ptl['icell']
                ix = icell % nx
                iy = (icell // nx) % ny
                iz = icell // (nx * ny)
                x = v0.x0 + ((ix - 1.0) + (dx + 1.0) * 0.5) * v0.dx
                y = v0.y0 + ((iy - 1.0) + (dy + 1.0) * 0.5) * v0.dy
                z = v0.z0 + ((iz - 1.0) + (dz + 1.0) * 0.5) * v0.dz

                if i == 0:
                    x_total = x
                    y_total = y
                    z_total = z
                    ux_total = ux
                    uy_total = uy
                    uz_total = uz
                else:
                    x_total = np.concatenate((x_total, x))
                    y_total = np.concatenate((y_total, y))
                    z_total = np.concatenate((z_total, z))
                    ux_total = np.concatenate((ux_total, ux))
                    uy_total = np.concatenate((uy_total, uy))
                    uz_total = np.concatenate((uz_total, uz))

            self.x = x_total
            self.y = y_total
            self.z = z_total
            self.ux = ux_total
            self.uy = uy_total
            self.uz = uz_total
        except FileNotFoundError:
            print(f"文件 {self.filename} 未找到。")
        except Exception as e:
            print(f"加载数据时发生错误: {e}")

    def plot_phase_space_2D(self, sample_step, x_plot_name, y_plot_name, color, size):
        # 使用 getattr 函数来获取属性值
        x_plot = getattr(self, x_plot_name, None)
        y_plot = getattr(self, y_plot_name, None)
        # 检查属性是否成功获取
        if x_plot is None or y_plot is None:
            print(f"属性 {x_plot_name} 或 {y_plot_name} 不存在。")
            return
        # 绘制散点图
        plt.scatter(x_plot[::sample_step], y_plot[::sample_step], c=color, s=size)
        plt.xlabel(x_plot_name, fontsize=16)
        plt.ylabel(y_plot_name, fontsize=16)
        plt.show()

    def get_counts_in_SWAP_view(self, energy_bin, up_region_start, up_region_end, dn_region_start, dn_region_end,
                                v_spacesraft):
        phi = 276 * np.pi / 180
        theta = 10 * np.pi / 180
        x_total = self.x
        y_total = self.y
        z_total = self.z
        ux_total = self.ux
        uy_total = self.uy
        uz_total = self.uz
        E = (ux_total+v_spacesraft)**2+uy_total**2+uz_total**2
        idx_up_swap = np.where((x_total > up_region_start) & (x_total < up_region_end) &
                                     (np.abs(uy_total) / np.sqrt(E) < np.sin(theta/2))
                                     & (np.abs(ux_total + v_spacesraft) / np.sqrt(
            (ux_total + v_spacesraft) ** 2 + uy_total ** 2) >= np.cos(phi / 2)))[0]
        idx_dn_swap = np.where((x_total > dn_region_start) & (x_total < dn_region_end) &
                                     (np.abs(uy_total) / np.sqrt(E) < np.sin(theta/2))
                                     & (np.abs(ux_total + v_spacesraft) / np.sqrt(
            (ux_total + v_spacesraft) ** 2 + uy_total ** 2) >= np.cos(phi / 2)))[0]
        counts_up, bins = np.histogram(E[idx_up_swap], bins=energy_bin)
        counts_dn, bins = np.histogram(E[idx_dn_swap], bins=energy_bin)
        return counts_up, counts_dn, bins

    def plot_counts_variation(self):
        n, bins = np.histogram(self.x, bins=range(257))
        plt.plot(range(256), n)
        plt.xlabel("x", fontsize=16)
        plt.ylabel(r"$n_{"+self.name+"}$", fontsize=16)
        plt.show()




# 使用示例
if __name__ == "__main__":
    mq = 1.6726e-27
    va = 40
    energy_charge_bin = sio.loadmat("Energy charge bin.mat")["energy_charge_bin"]
    energy_bin = np.squeeze((np.sqrt(2 * energy_charge_bin * 1.6e-19 / mq) / 1e3 / va) ** 2)
    step = 10000
    run_case_index = 3
    num_files = 16
    species_lst = ["SWI", "alpha", "PUI"]
    species_index = 2
    base_fname_swi = f"data_ip_shock/particle_data_{run_case_index}/T.{step}/Hparticle_{species_lst[species_index]}.{step}.{{}}"
    swi = Species(name=species_lst[species_index], filename=base_fname_swi, num_files=num_files)
    swi.plot_phase_space_2D(sample_step=1, x_plot_name="x", y_plot_name="ux", color="k", size=1)
    counts_swi_up, counts_swi_dn, bins = swi.get_counts_in_SWAP_view(energy_bin=energy_bin, up_region_start=230, up_region_end=240
                                , dn_region_start=20, dn_region_end=30, v_spacesraft=11-14/va)
    swi.plot_counts_variation()


