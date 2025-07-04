#%%
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
from scipy.optimize import curve_fit, leastsq, least_squares
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as sio
import scipy.stats as scs
from read_field_data import loadinfo, load_data_at_certain_t
from scipy.integrate import nquad
import pandas as pd
from scipy.fft import fftn, fftfreq, fft
from tracking_data_read import Tracer
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
#%%
#plt.style.use("seaborn-deep")
# mpl.rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = \
        (r"\usepackage{amsmath, bm}" +
         r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}" +
         r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}" +
         r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}")

# give some PIC simulation parameters here
topo_x, topo_y, topo_z = 32, 1, 8
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
                              ('u', np.float32, 3), ('q', np.float32), ('id', np.int32)])
    fh.seek(offset, os.SEEK_SET)
    data = np.fromfile(fh, dtype=particle_type, count=nptl)
    fh.close()
    return v0, pheader, data


# def read_field_data(fname):
#     """Read particle information from a file.
#
#     Args:
#         fname: file name.
#     """
#     fh = open(fname, 'r')
#     read_boilerplate(fh)
#     v0, pheader, offset = read_particle_header(fh)
#     nptl = pheader.dim
#     particle_type = np.dtype([('dxyz', np.float32, 3), ('icell', np.int32),
#                               ('u', np.float32, 3), ('q', np.float32)])
#     fh.seek(offset, os.SEEK_SET)
#     data = np.fromfile(fh, dtype=particle_type, count=nptl)
#     fh.close()
#     return (v0, pheader, data)


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


def f_pui(vx, vy, vz, v_drift):
    v = np.sqrt((vx + v_drift)**2 + vy**2 + vz**2)
    vc = 10.07
    alpha = 1.4
    lambda_pui = 3.4
    r = 33.5
    # 使用 np.where 处理数组输入
    result = np.where(v > vc, 0, (v / vc)**(alpha - 3) * np.exp(-lambda_pui / r * (v / vc)**(-alpha)))
    return result



def counts_pui_E(E, p):
    # vc = 10.07
    # alpha = 1.4
    vc, alpha, eta = p
    lambda_pui = 3.4
    r = 33.5
    v = np.sqrt(2*E)
    conditions = [E <= 0, (v <= vc) & (E > 0), v > vc]
    choices = [0, (v / vc) ** (alpha - 3) * np.exp(-lambda_pui / r * (v / vc) ** (-alpha))
            , (v/vc)**(-eta)*np.exp(-lambda_pui / r)]
    result = 4 * np.pi * v * np.select(condlist=conditions, choicelist=choices)
    return result

def counts_pui_E_SWAP(E, p):
    # vc = 10.07
    # alpha = 1.4
    vc, alpha, eta = p
    lambda_pui = 3.4
    r = 33.5
    v = np.sqrt(2*E)
    conditions = [E <= 0, (v <= vc) & (E > 0), v > vc]
    choices = [0, (v / vc) ** (alpha - 3) * np.exp(-lambda_pui / r * (v / vc) ** (-alpha))
            , (v/vc)**(-eta)*np.exp(-lambda_pui / r)]
    result = 4 * np.pi * v * np.select(condlist=conditions, choicelist=choices)
    return result

def sum_counts_pui_E(p, E_min, E_max, num):
    E_arr = np.logspace(np.log10(E_min), np.log10(E_max),num)
    sum = 0
    for i in range(len(E_arr)-1):
        dE = E_arr[i+1]-E_arr[i]
        E = E_arr[i]
        sum += counts_pui_E(E, p)
    return sum
# 定义误差函数

def g_pui(vx, vy, vz, v_drift):
    v = np.sqrt((vx + v_drift)**2 + vy**2 + vz**2)
    vc = 10.07
    alpha = 1.4
    lambda_pui = 3.4
    r = 33.5
    # 使用 np.where 处理数组输入
    result = np.where((v > vc) | (vx > 0), 0, np.abs(vx) * (v / vc)**(alpha - 3) * np.exp(-lambda_pui / r * (v / vc)**(-alpha)))
    return result


def g_pui_cylin(vx, vr, v_drift):
    v = np.sqrt((vx + v_drift)**2 + vr**2)
    vc = 10.07
    alpha = 1.4
    lambda_pui = 3.4
    r = 33.5
    # 使用 np.where 处理数组输入
    result = np.where((v > vc) | (vx > 0), 0, np.abs(vx) * (v / vc)**(alpha - 3) * np.exp(-lambda_pui / r * (v / vc)**(-alpha)))
    return result


# 计算某一个方向上的一维速度分布函数
def f_pui_1d(direction, v, v_drift=0, integration_range=20, singularity_threshold=1e-6, num_points=100):
    # 检查输入的方向是否合法
    if direction not in ["x", "y", "z"]:
        raise ValueError("Invalid direction. Please choose 'x', 'y', or 'z'.")

    if np.isscalar(v):
        # 确定奇点位置
        if direction == "x":
            singularity_point = -v_drift
        else:
            singularity_point = 0

        # 划分积分区间
        x1 = np.linspace(singularity_point + singularity_threshold, integration_range, num_points)
        x2 = np.linspace(0, integration_range, num_points)

        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]

        result = 0
        for i in range(num_points - 1):
            for j in range(num_points - 1):
                if direction == "x":
                    vx = v
                    vy1, vy2 = x2[j], x2[j + 1]
                    vz1, vz2 = x1[i], x1[i + 1]
                    vx1, vx2 = v, v
                elif direction == "y":
                    vy = v
                    vx1, vx2 = x1[i], x1[i + 1]
                    vz1, vz2 = x2[j], x2[j + 1]
                    vy1, vy2 = v, v
                else:
                    vz = v
                    vx1, vx2 = x1[i], x1[i + 1]
                    vy1, vy2 = x2[j], x2[j + 1]
                    vz1, vz2 = v, v

                # 计算梯形积分
                f1 = f_pui(vx1, vy1, vz1, v_drift)
                f2 = f_pui(vx1, vy1, vz2, v_drift)
                f3 = f_pui(vx1, vy2, vz1, v_drift)
                f4 = f_pui(vx1, vy2, vz2, v_drift)
                f5 = f_pui(vx2, vy1, vz1, v_drift)
                f6 = f_pui(vx2, vy1, vz2, v_drift)
                f7 = f_pui(vx2, vy2, vz1, v_drift)
                f8 = f_pui(vx2, vy2, vz2, v_drift)

                sub_integral = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8) / 8 * dx1 * dx2
                result += sub_integral

        # 考虑两个方向的对称性，结果乘以 2
        result *= 2
    else:
        result = []
        for single_v in v:
            # 确定奇点位置
            if direction == "x":
                singularity_point = -v_drift
            else:
                singularity_point = 0

            # 划分积分区间
            x1 = np.linspace(singularity_point + singularity_threshold, integration_range, num_points)
            x2 = np.linspace(0, integration_range, num_points)

            dx1 = x1[1] - x1[0]
            dx2 = x2[1] - x2[0]

            single_result = 0
            for i in range(num_points - 1):
                for j in range(num_points - 1):
                    if direction == "x":
                        vx = single_v
                        vy1, vy2 = x2[j], x2[j + 1]
                        vz1, vz2 = x1[i], x1[i + 1]
                        vx1, vx2 = single_v, single_v
                    elif direction == "y":
                        vy = single_v
                        vx1, vx2 = x1[i], x1[i + 1]
                        vz1, vz2 = x2[j], x2[j + 1]
                        vy1, vy2 = single_v, single_v
                    else:
                        vz = single_v
                        vx1, vx2 = x1[i], x1[i + 1]
                        vy1, vy2 = x2[j], x2[j + 1]
                        vz1, vz2 = single_v, single_v

                    # 计算梯形积分
                    f1 = f_pui(vx1, vy1, vz1, v_drift)
                    f2 = f_pui(vx1, vy1, vz2, v_drift)
                    f3 = f_pui(vx1, vy2, vz1, v_drift)
                    f4 = f_pui(vx1, vy2, vz2, v_drift)
                    f5 = f_pui(vx2, vy1, vz1, v_drift)
                    f6 = f_pui(vx2, vy1, vz2, v_drift)
                    f7 = f_pui(vx2, vy2, vz1, v_drift)
                    f8 = f_pui(vx2, vy2, vz2, v_drift)

                    sub_integral = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8) / 8 * dx1 * dx2
                    single_result += sub_integral

            # 考虑两个方向的对称性，结果乘以 2
            single_result *= 2
            result.append(single_result)
        result = np.array(result)

    return result


def g_pui_1d(direction, v, v_drift=0, integration_range=20, singularity_threshold=1e-6, num_points=100):
    # 检查输入的方向是否合法
    if direction not in ["x", "y", "z"]:
        raise ValueError("Invalid direction. Please choose 'x', 'y', or 'z'.")

    if np.isscalar(v):
        # 确定奇点位置
        if direction == "x":
            singularity_point = -v_drift
            # 划分积分区间
            x1 = np.linspace(singularity_point + singularity_threshold, integration_range, num_points)
        else:
            singularity_point = 0
            # 当 direction 为 y 或 z 时，x 的积分区间为负无穷到 0
            x1 = np.linspace(-integration_range, 0, num_points)

        x2 = np.linspace(0, integration_range, num_points)

        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]

        result = 0
        for i in range(num_points - 1):
            for j in range(num_points - 1):
                if direction == "x":
                    vx = v
                    vy1, vy2 = x2[j], x2[j + 1]
                    vz1, vz2 = x1[i], x1[i + 1]
                    vx1, vx2 = v, v
                elif direction == "y":
                    vy = v
                    vx1, vx2 = x1[i], x1[i + 1]
                    vz1, vz2 = x2[j], x2[j + 1]
                    vy1, vy2 = v, v
                else:
                    vz = v
                    vx1, vx2 = x1[i], x1[i + 1]
                    vy1, vy2 = x2[j], x2[j + 1]
                    vz1, vz2 = v, v

                # 计算梯形积分
                f1 = g_pui(vx1, vy1, vz1, v_drift)
                f2 = g_pui(vx1, vy1, vz2, v_drift)
                f3 = g_pui(vx1, vy2, vz1, v_drift)
                f4 = g_pui(vx1, vy2, vz2, v_drift)
                f5 = g_pui(vx2, vy1, vz1, v_drift)
                f6 = g_pui(vx2, vy1, vz2, v_drift)
                f7 = g_pui(vx2, vy2, vz1, v_drift)
                f8 = g_pui(vx2, vy2, vz2, v_drift)

                sub_integral = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8) / 8 * dx1 * dx2
                result += sub_integral

        # 考虑两个方向的对称性，结果乘以 2
        result *= 2
    else:
        result = []
        for single_v in v:
            # 确定奇点位置
            if direction == "x":
                singularity_point = -v_drift
                # 划分积分区间
                x1 = np.linspace(singularity_point + singularity_threshold, integration_range, num_points)
            else:
                singularity_point = 0
                # 当 direction 为 y 或 z 时，x 的积分区间为负无穷到 0
                x1 = np.linspace(-integration_range, 0, num_points)

            x2 = np.linspace(0, integration_range, num_points)

            dx1 = x1[1] - x1[0]
            dx2 = x2[1] - x2[0]

            single_result = 0
            for i in range(num_points - 1):
                for j in range(num_points - 1):
                    if direction == "x":
                        vx = single_v
                        vy1, vy2 = x2[j], x2[j + 1]
                        vz1, vz2 = x1[i], x1[i + 1]
                        vx1, vx2 = single_v, single_v
                    elif direction == "y":
                        vy = single_v
                        vx1, vx2 = x1[i], x1[i + 1]
                        vz1, vz2 = x2[j], x2[j + 1]
                        vy1, vy2 = single_v, single_v
                    else:
                        vz = single_v
                        vx1, vx2 = x1[i], x1[i + 1]
                        vy1, vy2 = x2[j], x2[j + 1]
                        vz1, vz2 = single_v, single_v

                    # 计算梯形积分
                    f1 = g_pui(vx1, vy1, vz1, v_drift)
                    f2 = g_pui(vx1, vy1, vz2, v_drift)
                    f3 = g_pui(vx1, vy2, vz1, v_drift)
                    f4 = g_pui(vx1, vy2, vz2, v_drift)
                    f5 = g_pui(vx2, vy1, vz1, v_drift)
                    f6 = g_pui(vx2, vy1, vz2, v_drift)
                    f7 = g_pui(vx2, vy2, vz1, v_drift)
                    f8 = g_pui(vx2, vy2, vz2, v_drift)

                    sub_integral = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8) / 8 * dx1 * dx2
                    single_result += sub_integral

            # 考虑两个方向的对称性，结果乘以 2
            single_result *= 2
            result.append(single_result)
        result = np.array(result)

    return result


def get_shock_position(field_dir, epoch, bz_threshold, nx, nz):
    bz_t = load_data_at_certain_t(f"{field_dir}"+"bz.gda", epoch, nx, nz)
    if epoch < 10:
        bz_threshold_t = 0.8*np.max(np.mean(bz_t, axis=1))+0.2
    else:
        bz_threshold_t = bz_threshold
    index_dn = np.where(np.mean(bz_t, axis=1) > bz_threshold_t)[0]
    index_shock = np.max(index_dn)
    if epoch == 0:
        return 0
    return index_shock


def plot_power_spectra_at_different_positions(*i_xs, field_dir, epoch, nx, nz, lambda_min):
    kz_min = 2 * np.pi / nz
    kz_max = 2 * np.pi / lambda_min
    bx_0 = load_data_at_certain_t(f"{field_dir}/bx.gda", 0, nx, nz)
    by_0 = load_data_at_certain_t(f"{field_dir}/by.gda", 0, nx, nz)
    bz_0 = load_data_at_certain_t(f"{field_dir}/bz.gda", 0, nx, nz)
    kz = fftfreq(bx_0.shape[1], d=2.5) * 2 * np.pi
    # kz = np.logspace(np.log10(0.01), np.log10(kz_max), nz)
    bx_k0 = fft(bx_0[20, :])
    Pk_0 = np.abs(bx_k0) ** 2
    Pk_integral_0 = np.sum(Pk_0) * (kz[1] - kz[0])
    for i_x in i_xs:
        bx_t = load_data_at_certain_t(f"{field_dir}/bx.gda", epoch, nx, nz)
        # kz_t = fftfreq(bx.shape[1], d=1) * 2 * np.pi
        # kz = np.logspace(np.log10(0.01), np.log10(kz_max), nz)
        bx_k = fft(bx_t[i_x, :])
        Pk = np.abs(bx_k) ** 2
        k_bins = np.linspace(kz_min, kz_max, 15)
        Pk_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk)
        plt.plot(edges[:-1], Pk_avg / Pk_integral_0, label=f"Epoch={epoch},x={i_x}")
    plt.plot(edges, edges**(-5/3)/(kz_min**(-2/3)-kz_max**(-2/3))/1.5,
             label=r"theoretical spectrum($k^{-\frac{5}{3}}$)", c="r")
    plt.yscale('log')
    plt.xscale('log')
    # plt.ylim([1e-3, 20])
    plt.legend(fontsize=14)
    plt.ylabel("PSD", fontsize=15)
    plt.xlabel("k", fontsize=15)
    # plt.title("PSD at different regions", fontsize=15)
    plt.title("Initial PSD", fontsize=15)
    plt.show()


class Species:
    def __init__(self, name, filename, num_files, fullname=None,
                 sample_step=1, region=None, tag_chosen=None, field_dir=None):
        """
        初始化Species类

        :param name: 粒子种类的名称
        :param filename: 存储粒子数据的文件名模板
        :param num_files: 文件数量
        :param fullname: 粒子种类的完整名称
        :param sample_step: 采样步长，用于降采样
        :param region: 几何区域范围，格式为[x_min, x_max, z_min, z_max]
        :param tag_chosen: 要选择的粒子标签列表
        """
        # 参数验证
        if region is not None:
            if len(region) != 4:
                raise ValueError("region参数必须是包含4个元素的列表: [x_min, x_max, z_min, z_max]")
            if region[0] >= region[1] or region[2] >= region[3]:
                raise ValueError("region参数中min值必须小于max值")

        self.name = name
        self.fullname = fullname
        self.filename = filename
        self.num_files = num_files
        self.sample_step = sample_step
        self.region = region
        self.tag_chosen = tag_chosen
        self.tag = None

        # 初始化数据属性
        self.dt = None
        self.nt = None
        self.it = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.x = None
        self.y = None
        self.z = None
        self.ux = None
        self.uy = None
        self.uz = None
        self.bx = None
        self.by = None
        self.bz = None
        self.ex = None
        self.ey = None
        self.ez = None
        self.E = None
        self.icell = None
        self.ix = None
        self.iy = None
        self.iz = None
        self.rank = None
        self.field_dir = field_dir
        # 读取并处理粒子数据
        self.read_multiple_particle_files()

    def read_multiple_particle_files(self):
        """
        读取多个粒子文件并合并数据，根据region和tag_chosen筛选粒子
        """
        try:
            # 用于存储最终筛选后的粒子数据
            if self.field_dir is None:
                data_buffers = {
                    'x': [], 'y': [], 'z': [],
                    'ux': [], 'uy': [], 'uz': [],
                    'ix': [], 'iy': [], 'iz': [],
                    'icell': [], 'rank': [], 'tag': []
                }
            else:
                data_buffers = {
                    'x': [], 'y': [], 'z': [],
                    'ux': [], 'uy': [], 'uz': [],
                    'bx': [], 'by': [], 'bz': [],
                    'ex': [], 'ey': [], 'ez': [],
                    'ix': [], 'iy': [], 'iz': [],
                    'icell': [], 'rank': [], 'tag': []
                }
            fname_0 = self.filename.format(0)
            try:
                v0, pheader, ptl = read_particle_data(fname_0)
                self.dt = v0.dt
                self.nt = v0.nt
                self.it = round(v0.dt * v0.nt)
                self.nx = v0.nx * topo_x
                self.ny = v0.ny * topo_y
                self.nz = v0.nz * topo_z
            except Exception as e:
                print(f"无法读取文件 {fname_0}: {e}")

            if self.field_dir is not None:
                bx_data = load_data_at_certain_t(self.field_dir + "bx.gda", i_t=self.it, num_dim1=self.nx,
                                                 num_dim2=self.ny, num_dim3=self.nz)
                by_data = load_data_at_certain_t(self.field_dir + "by.gda", i_t=self.it, num_dim1=self.nx,
                                                 num_dim2=self.ny, num_dim3=self.nz)
                bz_data = load_data_at_certain_t(self.field_dir + "bz.gda", i_t=self.it, num_dim1=self.nx,
                                                 num_dim2=self.ny, num_dim3=self.nz)
                ex_data = load_data_at_certain_t(self.field_dir + "ex.gda", i_t=self.it, num_dim1=self.nx,
                                                 num_dim2=self.ny, num_dim3=self.nz)
                ey_data = load_data_at_certain_t(self.field_dir + "ey.gda", i_t=self.it, num_dim1=self.nx,
                                                 num_dim2=self.ny, num_dim3=self.nz)
                ez_data = load_data_at_certain_t(self.field_dir + "ez.gda", i_t=self.it, num_dim1=self.nx,
                                                 num_dim2=self.ny, num_dim3=self.nz)
            for i in range(self.num_files):
                fname = self.filename.format(i)
                try:
                    v0, pheader, ptl = read_particle_data(fname)
                except Exception as e:
                    print(f"无法读取文件 {fname}: {e}")
                    continue

                # 只在读取第一个文件时设置全局网格信息




                # 提取粒子数据
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
                tag = ptl['id']

                # 计算粒子坐标
                ix = icell % nx
                iy = (icell // nx) % ny
                iz = icell // (nx * ny)
                ixx = np.array(ix + v0.x0/v0.dx, dtype=int)

                izz = np.array(iz + (v0.z0+32)/v0.dz, dtype=int)
                # print(ixx.max(), izz.max())
                if self.field_dir is not None:
                    bx = bx_data[np.maximum(ixx-1, 0), np.maximum(izz-1, 0)]
                    by = by_data[np.maximum(ixx - 1, 0), np.maximum(izz - 1, 0)]
                    bz = bz_data[np.maximum(ixx - 1, 0), np.maximum(izz - 1, 0)]
                    ex = ex_data[np.maximum(ixx - 1, 0), np.maximum(izz - 1, 0)]
                    ey = ey_data[np.maximum(ixx - 1, 0), np.maximum(izz - 1, 0)]
                    ez = ez_data[np.maximum(ixx - 1, 0), np.maximum(izz - 1, 0)]
                x = v0.x0 + ((ix - 1.0) + (dx + 1.0) * 0.5) * v0.dx
                y = v0.y0 + ((iy - 1.0) + (dy + 1.0) * 0.5) * v0.dy
                z = v0.z0 + ((iz - 1.0) + (dz + 1.0) * 0.5) * v0.dz

                # 先根据region进行初步筛选，判断该文件是否需要处理
                # 如果文件区域完全不在感兴趣区域内，则跳过该文件
                if self.region is not None:
                    if (v0.x0 + v0.dx * v0.nx < self.region[0] or
                            v0.x0 > self.region[1] or
                            v0.z0 + v0.dz * v0.nz < self.region[2] or
                            v0.z0 > self.region[3]):
                        continue

                # 创建初始筛选条件（全部为True）
                condition = np.ones_like(x, dtype=bool)

                # 如果设置了region，添加区域筛选条件
                if self.region is not None:
                    region_condition = (x >= self.region[0]) & (x < self.region[1]) & \
                                       (z >= self.region[2]) & (z < self.region[3])
                    condition &= region_condition

                # 如果设置了tag_chosen，添加标签筛选条件
                if self.tag_chosen is not None:
                    # 确保tag_chosen是一个列表
                    if not isinstance(self.tag_chosen, list):
                        tag_list = [self.tag_chosen]
                    else:
                        tag_list = self.tag_chosen
                    tag_condition = np.isin(tag, tag_list)
                    condition &= tag_condition

                # 应用筛选条件并采样
                if np.any(condition):
                    # 应用筛选条件并按sample_step采样
                    indices = np.where(condition)[0][::self.sample_step]

                    # 将筛选后的数据添加到缓冲区
                    data_buffers['x'].append(x[indices])
                    data_buffers['y'].append(y[indices])
                    data_buffers['z'].append(z[indices])
                    data_buffers['ux'].append(ux[indices])
                    data_buffers['uy'].append(uy[indices])
                    data_buffers['uz'].append(uz[indices])
                    data_buffers['ix'].append(ix[indices])
                    data_buffers['iy'].append(iy[indices])
                    data_buffers['iz'].append(iz[indices])
                    data_buffers['icell'].append(icell[indices])
                    data_buffers['tag'].append(tag[indices])
                    data_buffers['rank'].append(i * np.ones_like(indices))
                    if self.field_dir is not None:
                        data_buffers['bx'].append(bx[indices])
                        data_buffers['by'].append(by[indices])
                        data_buffers['bz'].append(bz[indices])
                        data_buffers['ex'].append(ex[indices])
                        data_buffers['ey'].append(ey[indices])
                        data_buffers['ez'].append(ez[indices])
            # 如果没有找到符合条件的粒子
            if not data_buffers['x']:
                print("警告: 没有找到符合条件的粒子数据")
                return

            # 合并所有缓冲区数据
            self.x = np.concatenate(data_buffers['x'])
            self.y = np.concatenate(data_buffers['y'])
            self.z = np.concatenate(data_buffers['z'])
            self.ux = np.concatenate(data_buffers['ux'])
            self.uy = np.concatenate(data_buffers['uy'])
            self.uz = np.concatenate(data_buffers['uz'])
            self.ix = np.concatenate(data_buffers['ix'])
            self.iy = np.concatenate(data_buffers['iy'])
            self.iz = np.concatenate(data_buffers['iz'])
            self.icell = np.concatenate(data_buffers['icell'])
            self.tag = np.concatenate(data_buffers['tag'])
            self.rank = np.concatenate(data_buffers['rank'])
            if self.field_dir is not None:
                self.bx = np.concatenate(data_buffers['bx'])
                self.by = np.concatenate(data_buffers['by'])
                self.bz = np.concatenate(data_buffers['bz'])
                self.ex = np.concatenate(data_buffers['ex'])
                self.ey = np.concatenate(data_buffers['ey'])
                self.ez = np.concatenate(data_buffers['ez'])
            # 计算粒子能量
            self.E = 0.5 * (self.ux ** 2 + self.uy ** 2 + self.uz ** 2)

        except FileNotFoundError:
            print(f"文件 {self.filename} 未找到。")
        except Exception as e:
            print(f"加载数据时发生错误: {e}")

    def get_shock_position_by_density(self):
        n, bins = np.histogram(self.x, bins=range(257))
        x_arr = np.linspace(0, 255, 256)
        x_arr_shock = x_arr[n > 0.7*np.max(n)]
        return np.argmax(x_arr_shock)

    def fit_pui_spectrum(self, x_range, p0, bounds_lst=None, bins=None):
        condition = (self.x >= x_range[0]) & (self.x < x_range[1])
        if bins is None:
            bins = np.logspace(np.log10(5), np.log10(0.8*self.E[condition].max()))
        counts, bins = np.histogram(self.E[condition], bins=bins)

        def residuals(p, y, x):
            """
            误差函数: 计算残差
            :param p: 参数列表 [a, b, c]
            :param y: 观测值
            :param x: 自变量
            :return: 残差
            """
            return y- counts_pui_E(x, p) / sum_counts_pui_E(p, bins[0], bins[-1], num=len(bins))
        if bounds_lst is None:
            plsq = least_squares(residuals, p0, args=(bins[1:], counts/counts.sum()))
        else:
            plsq = least_squares(residuals, p0, args=(bins[1:], counts/counts.sum()), bounds=bounds_lst)
        p = plsq.x
        return p, counts, bins


    def plot_phase_space_2D(self, sample_step, x_plot_name, y_plot_name, color, size,
                            ax=None, fig=None, cmap=None, vmin=None, vmax=None, set_cbar=True
                            , x_offset=None):
        # 如果没有传入 ax 对象，则使用当前的 Axes 对象
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        if cmap is None:
            cmap = "jet"
        if x_offset is None:
            x_offset = 0
        # 使用 getattr 函数来获取属性值
        x_plot = getattr(self, x_plot_name, None)
        y_plot = getattr(self, y_plot_name, None)
        # 检查属性是否成功获取
        if x_plot is None or y_plot is None:
            print(f"属性 {x_plot_name} 或 {y_plot_name} 不存在。")
            return
        # 在指定的 ax 上绘制散点图
        if vmin is None:
            ax.scatter(x_plot[::sample_step], y_plot[::sample_step], c=color, s=size, cmap=cmap)
        else:
            points = ax.scatter(x_plot[::sample_step]-x_offset, y_plot[::sample_step], c=color, s=size, cmap=cmap, vmin=vmin, vmax=vmax)
            pos = ax.get_position()
            if set_cbar:
                cax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.01, 0.02, pos.y1 - pos.y0 - 0.01])
                cbar = plt.colorbar(points, cax=cax)
                cbar.set_label("log(E)", fontsize=35)
        ax.set_xlabel(x_plot_name, fontsize=15)
        ax.set_ylabel(y_plot_name, fontsize=15)
        # plt.show()

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
        n, bins = np.histogram(self.x, bins=range(self.nx+1))

        plt.plot(range(self.nx), n/np.max(n), label=f"{self.fullname} counts")
        # plt.plot(range(256), np.mean(bz, axis=1)/np.max(np.mean(bz, axis=1)), label=r"$B_z$")
        # plt.plot(range(256), np.mean(ey, axis=1)/np.max(np.mean(ey, axis=1)), label=r"$E_y$")
        plt.xlabel("x", fontsize=16)
        # plt.ylabel(r"$n_{"+self.name+"}$", fontsize=16)
        plt.title(f"Normalized {self.fullname} counts", fontsize=13)
        plt.legend()
        plt.show()

    def plot_temperature_variation(self, field_data_dir):
        temperature = np.zeros(self.nx)
        for i in range(self.nx):
            index_tmp = np.where((self.x >= i) & (self.x < i+1))
            # temperature[i] = np.var(self.ux[index_tmp])+np.var(self.uy[index_tmp])+np.var(self.uz[index_tmp])
            temperature[i] = np.var(self.ux[index_tmp])
        plt.plot(range(self.nx), temperature)
        plt.show()

    @staticmethod
    def plot_velocity_distribution(*species_lst, x_ranges, labels=None, x_shock
                                   , energy_bins_num=20, normalize=False, swap=False):
        # 检查输入列表的长度是否一致
        if len(species_lst) != len(x_ranges) != len(labels):
            raise ValueError("The lengths of species_lst, x_ranges, and sigmas must be the same.")
        # 获取 jet 颜色图对象
        jet_colormap = cm.get_cmap('jet')

        # 采样的颜色数量
        num_colors = len(x_ranges)

        # 从低到高采样颜色
        sampled_colors = jet_colormap(np.linspace(0, 1, num_colors))

        # 提取 RGB 数组（去掉 alpha 通道）
        rgb_arrays = sampled_colors[:, :3]
        for species, x_range, label, rgb_array in zip(species_lst, x_ranges, labels, rgb_arrays):
            # counts_mean = 0
            index = np.where((species.x >= x_range[0]) & (species.x < x_range[1]))
            bin_min = 0  # np.min(species.uy[index])
            bin_max = 2.8  # np.max(species.uy[index])
            E_swap = 0.5*((species.ux+11)**2+species.uy**2+species.uz**2)
            if swap:
                counts, bins = np.histogram(E_swap[index], bins=np.logspace(bin_min, bin_max, energy_bins_num))
            else:
                counts, bins = np.histogram(species.E[index], bins=np.logspace(bin_min, bin_max, energy_bins_num))
            bin_center = 0.5 * (bins[1:] + bins[:-1])
            # print(np.logspace(bin_min, bin_max, 20))
            # print(counts_y)
            # counts_mean += np.sum(counts_y)
            if normalize:
                plt.plot(bin_center, counts/counts.sum()
                         , label=fr'$x_{{shock}}$: {x_range[0] - x_shock:.1f} ~ {x_range[1] - x_shock:.1f}' + label
                         , color=rgb_array)
            else:
                plt.plot(bin_center, counts
                         , label=fr'$x_{{shock}}$: {x_range[0]-x_shock:.1f} ~ {x_range[1]-x_shock:.1f}'+label
                         , color=rgb_array)

            # counts_mean = counts_mean / 1  # 这里因为只有一个 x_range 了，所以除以 1

            # f_pui_x = f_pui_1d("y", bin_center, v_drift=1.86)
            # g_pui_y = g_pui_1d("y", bin_center, v_drift=1.86)
            # g_pui_z = g_pui_1d("z", bin_center, v_drift=1.86)
            # print(g_pui_y)
            # plt.plot(bin_center, g_pui_y / np.sum(g_pui_y), label=r"theoretical $g_{pui, y}$")
            # plt.plot(bin_center, g_pui_z / np.sum(g_pui_z), label=r"theoretical $g_{pui, z}$")
            # plt.plot(bin_center, f_pui_x / np.sum(f_pui_x), label=r"theoretical $f_{pui}$")

        plt.xlabel('Energy')
        plt.ylabel('Counts')
        plt.yscale("log")
        plt.xscale("log")
        plt.title('Energy spectrum of PUIs in downstream')
        plt.legend()
        plt.legend(loc='lower left')

        # plt.show()

    def plot_energy_distribution_map(self):
        counts_mat = np.zeros((self.nx, 19))

        for i in range(self.nx):
            condition = (self.x >= i) & (self.x < i+1)
            counts, bins = np.histogram(self.E[condition], bins=np.logspace(np.log10(np.min(self.E)), np.log10(np.max(self.E)), 20))
            counts_mat[i, :] = counts
        bins_center = 0.5*(bins[:-1]+bins[1:])
        plt.pcolormesh(np.linspace(0, self.nx-1, self.nx), bins_center, counts_mat.T,
                       cmap="jet", norm=mpl.colors.LogNorm())
        plt.yscale("log")
        plt.xlabel("x")
        plt.ylabel("E")
        cbar = plt.colorbar()
        cbar.set_label("conuts")
        # plt.xlim([50, 150])
        plt.show()

    def plot_counts_dis_map(self, vmin, vmax):
        counts_mat, xedges, zedges = np.histogram2d(self.x, self.z, bins=[self.nx, self.nz],
                                                    range=[[0, self.x.max()], [self.z.min(), self.z.max()]])
        plt.pcolormesh(xedges[:-1], zedges[:-1], counts_mat.T, cmap="jet", vmin=vmin, vmax=vmax)
        plt.xlabel("x")
        plt.ylabel("z")
        cbar = plt.colorbar()
        cbar.set_label(fr"$N_{{\mathrm{{{self.name}}}}}$", fontsize=15)

        # plt.show()







#%%
# 使用示例
if __name__ == "__main__":
    #%%
    mq = 1.6726e-27
    va = 40
    energy_charge_bin = sio.loadmat("Energy charge bin.mat")["energy_charge_bin"]
    energy_bin = np.squeeze((np.sqrt(2 * energy_charge_bin * 1.6e-19 / mq) / 1e3 / va) ** 2)
    step = 10000
    run_case_index = 42
    num_files = 16
    species_lst = ["SWI", "alpha", "PUI"]
    fullname_lst = ["SW proton", " SW alpha", "PUI(H+)"]
    species_index = 2
    field_dir = f"data_ip_shock/field_data_{run_case_index}/"
    base_fname_swi_1 = f"data_ip_shock/particle_data/particle_data_{run_case_index}/T.10000/Hparticle_{species_lst[species_index]}.10000.{{}}"
    base_fname_swi_2 = f"data_ip_shock/particle_data/particle_data_39/T.{step}/Hparticle_{species_lst[species_index]}.{step}.{{}}"
    base_fname_swi_3 = f"data_ip_shock/particle_data/particle_data_40/T.{step}/Hparticle_{species_lst[species_index]}.{step}.{{}}"
    base_fname_swi_4 = f"data_ip_shock/particle_data/particle_data_34/T.{step}/Hparticle_{species_lst[0]}.{step}.{{}}"
    base_fname_swi_5 = f"data_ip_shock/particle_data/particle_data_35/T.{step}/Hparticle_{species_lst[0]}.{step}.{{}}"
    p_1 = Species(name=species_lst[species_index], fullname=fullname_lst[species_index],
                  filename=base_fname_swi_1, num_files=num_files)
    p_2 = Species(name=species_lst[species_index], fullname=fullname_lst[species_index],
                    filename=base_fname_swi_2, num_files=num_files)
    p_3 = Species(name=species_lst[species_index], fullname=fullname_lst[species_index],
                    filename=base_fname_swi_3, num_files=num_files)
    p_4 = Species(name=species_lst[0], fullname=fullname_lst[0],
                    filename=base_fname_swi_4, num_files=num_files)
    p_5 = Species(name=species_lst[0], fullname=fullname_lst[0],
                  filename=base_fname_swi_5, num_files=num_files)
#%%
    v0, pheader, ptl = read_particle_data(f"data_ip_shock/particle_data/particle_data_{run_case_index}/T.10000/Hparticle_{species_lst[species_index]}.10000.0")
    print(ptl)





