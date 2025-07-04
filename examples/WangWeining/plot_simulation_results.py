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
topo_x, topo_y, topo_z = 8, 1, 2
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


def calculate_current_density(Bx, By, Bz, dx, dy, dz):
    """
    通过三维磁场旋度计算电流密度

    参数:
    Bx (numpy.ndarray): 磁场的 x 分量
    By (numpy.ndarray): 磁场的 y 分量
    Bz (numpy.ndarray): 磁场的 z 分量
    dx (float): x 方向的网格间距
    dy (float): y 方向的网格间距
    dz (float): z 方向的网格间距

    返回:
    Jx (numpy.ndarray): 电流密度的 x 分量
    Jy (numpy.ndarray): 电流密度的 y 分量
    Jz (numpy.ndarray): 电流密度的 z 分量
    """
    mu_0 = 1
    # 计算磁场的旋度
    dBz_dx = np.gradient(Bz, dx, axis=0)
    dBx_dz = np.gradient(Bx, dz, axis=1)
    dBy_dz = np.gradient(By, dz, axis=1)
    dBy_dx = np.gradient(By, dx, axis=0)

    # 由于是二维平面（xz平面），这里dBz_dy和dBx_dy为0
    Jx = -dBy_dz / mu_0
    Jy = -(dBz_dx - dBx_dz) / mu_0
    Jz = dBy_dx / mu_0

    return Jx, Jy, Jz
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
    def __init__(self, name, filename, num_files, fullname=None, sample_step=1, region=None):
        """
        初始化Species类
        :param name: 粒子种类的名称
        :param filename: 存储粒子数据的文件名
        """
        self.name = name
        self.fullname = fullname
        self.filename = filename
        self.num_files = num_files
        self.sample_step = sample_step
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
        self.E = None
        self.icell = None
        self.ix = None
        self.iy = None
        self.iz = None
        self.rank = None
        self.region = region
        self.read_multiple_particle_files()

    def read_multiple_particle_files(self):
        """
        读取多个粒子文件并合并数据
        :param base_fname: 文件名模板，例如 'data_ip_shock/particle_data_{run_case_index}/T.{step}/Hparticle_SWI.{step}.{}'
        :param num_files: 文件数量
        :return: 合并后的 x, y, z, ux, uy, uz 数据
        """
        flag = 0
        try:
            for i in range(self.num_files):
                fname = self.filename.format(i)
                v0, pheader, ptl = read_particle_data(fname)
                if i == 0:
                    self.dt = v0.dt
                    self.nt = v0.nt
                    self.it = round(v0.dt * v0.nt)
                    self.nx = v0.nx * topo_x
                    self.ny = v0.ny * topo_y
                    self.nz = v0.nz * topo_z
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
                # flag = 0
                if v0.x0+v0.dx*v0.nx<self.region[0] or v0.x0>self.region[1] or v0.z0+v0.dz*v0.nz<self.region[2] or v0.z0>self.region[3]:
                    # flag += 1
                    continue
                condtion = (x >= self.region[0]) & (x < self.region[1]) & (z >= self.region[2]) & (z < self.region[3])

                if flag == 0:
                    x_total = x[condtion][::self.sample_step]
                    y_total = y[condtion][::self.sample_step]
                    z_total = z[condtion][::self.sample_step]
                    ux_total = ux[condtion][::self.sample_step]
                    uy_total = uy[condtion][::self.sample_step]
                    uz_total = uz[condtion][::self.sample_step]
                    ix_total = ix[condtion][::self.sample_step]
                    iy_total = iy[condtion][::self.sample_step]
                    iz_total = iz[condtion][::self.sample_step]
                    i_cell_total = icell[condtion][::self.sample_step]
                    rank_total = i*np.ones_like(x[condtion][::self.sample_step])
                    flag += 1
                else:
                    x_total = np.concatenate((x_total, x[condtion][::self.sample_step]))
                    y_total = np.concatenate((y_total, y[condtion][::self.sample_step]))
                    z_total = np.concatenate((z_total, z[condtion][::self.sample_step]))
                    ux_total = np.concatenate((ux_total, ux[condtion][::self.sample_step]))
                    uy_total = np.concatenate((uy_total, uy[condtion][::self.sample_step]))
                    uz_total = np.concatenate((uz_total, uz[condtion][::self.sample_step]))
                    ix_total = np.concatenate((ix_total, ix[condtion][::self.sample_step]))
                    iy_total = np.concatenate((iy_total, iy[condtion][::self.sample_step]))
                    iz_total = np.concatenate((iz_total, iz[condtion][::self.sample_step]))
                    i_cell_total = np.concatenate((i_cell_total, icell[condtion][::self.sample_step]))
                    rank_total = np.concatenate((rank_total, i*np.ones_like(x[condtion][::self.sample_step])))

            self.x = x_total
            self.y = y_total
            self.z = z_total
            self.ux = ux_total
            self.uy = uy_total
            self.uz = uz_total
            self.E = 0.5*(ux_total**2+uy_total**2+uz_total**2)
            self.icell = i_cell_total
            self.ix = ix_total
            self.iy = iy_total
            self.iz = iz_total
            self.rank = rank_total
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
    #%%
    condition = (p_2.x > 0)&(p_2.x<100)
    print(p_2.E[condition].mean())
    #%%
    bounds_lst = ((10, 1, 0.01), (25, 4, 10))
    p, counts, bins = p_3.fit_pui_spectrum(x_range=[200, 250], p0=[17, 2.5, 5]
                                                 , bins=np.logspace(0.5, np.log10(0.8*p_1.E.max()), 20),
                                           bounds_lst=bounds_lst)
    # p[3] = p[3]-1

    vc_arr = np.arange(13, 15, 0.02)
    alpha_arr = np.arange(2, 3, 0.02)
    eta_arr = np.arange(8, 10, 0.2)
    residual_mat = np.zeros((len(vc_arr), len(alpha_arr), len(eta_arr)))
    for i in range(len(vc_arr)):
        print(i)
        for j in range(len(alpha_arr)):
            for k in range(len(eta_arr)):
                p = [vc_arr[i], alpha_arr[j], eta_arr[k]]
                residual_mat[i, j, k] = np.sum((counts/counts.sum()-counts_pui_E(bins[1:], p)/sum_counts_pui_E(p, bins[0], bins[-1], len(bins)))**2/(counts/counts.sum())**2)
    #%%
    position_min = np.unravel_index(np.argmin(residual_mat), residual_mat.shape)
    vc_fit = vc_arr[position_min[0]]
    alpha_fit = alpha_arr[position_min[1]]
    eta_fit = eta_arr[position_min[2]]

    p = [vc_fit, alpha_fit, eta_fit]
    plt.plot(bins[1:], counts)
    plt.plot(bins[1:], counts_pui_E(bins[1:], p)/sum_counts_pui_E(p, bins[0], bins[-1], len(bins))*counts.sum())
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    #%%
    x_shock_34 = np.load(f"data_ip_shock/x_shock_data/x_shock_arr_34.npy")
    x_shock_39 = np.load(f"data_ip_shock/x_shock_data/x_shock_arr_39.npy")
    x_shock_38_interp = np.load(f"data_ip_shock/x_shock_data/x_shock_arr_38_interp.npy")
    x_shock_34[1] = 5
    x_shock_34[2] = 10
    x_shock_35 = np.load(f"data_ip_shock/x_shock_data/x_shock_arr_35.npy")
    # plt.plot(x_shock_35)
    x_shock_35[1] = 5
    x_shock_35[2] = 8
    epoch = np.linspace(0, 400, 101)
    f_35 = interp1d(np.linspace(0, 400, 101), x_shock_35, kind='linear')
    f_34 = interp1d(np.linspace(0, 400, 101), x_shock_34, kind='linear')
    x_shock_35_inerp = f_35(np.linspace(0, 400, 401))
    slope_35, intercept_35 = np.polyfit(epoch[4:], x_shock_35[4:], 1)
    x_shock_35_fit = slope_35*epoch+intercept_35
    x_shock_34_inerp = f_34(np.linspace(0, 400, 401))
    slope_34, intercept_34 = np.polyfit(epoch[4:], x_shock_34[4:], 1)
    x_shock_34_fit = slope_34 * epoch + intercept_34
    # plt.show()
    #%%
    """
    READ TRACER DATA
    """
    num_particle_traj = 2000
    ratio_emax = 1
    species_name_lst = ["ion", "alpha", "pui"]
    species_fullname_lst = ["SWI(proton)", "SWI(alpha)", "PUI"]
    sample_lst = [2, 3, 1]
    ntraj_lst = [8000, 12000, 32000]
    #%%
    fdir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/trace_data/"
    # 用于存储文件名的字典
    file_names = {}
    # 用于存储 Tracer 对象的字典
    tracers = {}
#%%
    # 遍历 index 列表
    for index in [28]:
        # 遍历物种名称列表
        for j in range(len(species_name_lst)):
            # 生成文件名
            fname_key = f"fname{index}_{species_name_lst[j]}"
            file_names[
                fname_key] = f"{species_name_lst[j]}_trace_data/{species_name_lst[j]}s_ntraj{ntraj_lst[j]}_{ratio_emax}emax_{index}.h5p"
            # 生成 Tracer 对象
            tracer_key = f"{species_name_lst[j]}_tracer_{index}"
            tracers[tracer_key] = Tracer(
                species_name_lst[j],
                species_fullname_lst[j],
                fdir,
                file_names[fname_key],
                sample_step=sample_lst[j]
            )
    #%%
    tracer = tracers["pui_tracer_28"]
    plt.scatter(tracer.data["x"][1000, :101], tracer.data["uy"][1000, :101], c=range(101), cmap="jet")
    # plt.plot(tracer.data["x"][1000, :], tracer.data["uy"][1000, :], c="k")
    plt.show()
    #%%
    a = tracers["pui_tracer_38"].count_particle_crossings(x_shock_38_interp)
    plt.hist(a)
    plt.yscale("log")
    plt.show()
    #%%
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    #%%
    ax = axes[2]
    set_cbar = True
    show = True
    alpha = 3
    theta_bn = 75
    tracer = tracers["pui_tracer_43"]
    E_bin = np.logspace(np.log10(10), np.log10(150), 15)
    E_bin = np.linspace(10, 300, 15)
    E0_bin = np.logspace(np.log10(1), np.log10(70), 15)
    E0_bin = np.linspace(10, 70, 20)
    jet_colormap = cm.get_cmap('jet')

    # 采样的颜色数量
    num_colors = len(E_bin)-1

    # 从低到高采样颜色
    sampled_colors = jet_colormap(np.linspace(0, 1, num_colors))

    # 提取 RGB 数组（去掉 alpha 通道）
    rgb_arrays = sampled_colors[:, :3]
    E_dis_mat = np.zeros((len(E0_bin)-1, len(E_bin)-1))
    E_init = tracer.data["E"][:, 0]
    E_final = tracer.data["E"][:, 400]
    condition_final = (E_final >= E_bin[0]) & (E_final <= E_bin[-1])
    counts_init, bins = np.histogram(E_init, bins=E0_bin)
    counts_final, bins = np.histogram(E_final, bins=E_bin)
    for j in range(len(E_bin)-1):
        for i in range(len(E0_bin)-1):
            condition = (E_init >= E0_bin[i]) & (E_init < E0_bin[i+1])

            E_dn = E_final[condition]

            counts_dn, bins = np.histogram(E_dn, bins=E_bin)
            E_dis_mat[i, j] = counts_dn[j]/counts_final[j]
        # plt.plot(bins[1:], counts_dn/counts_total, label=f"{E_bin[i]:.2f}<E0<{E_bin[i+1]:.2f}",c=rgb_arrays[i, :])
    # plt.legend()
    print(np.sum(E_dis_mat[3, :]))
    pclr = ax.pcolormesh(E_bin, E0_bin, E_dis_mat, cmap="jet", vmin=0, vmax=0.2)

    # im = ax.imshow(a_smooth[:, 0, 8, :].T, cmap='jet')
    if set_cbar:
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.01, 0.02, pos.y1 - pos.y0 - 0.01])
        cbar = plt.colorbar(pclr, cax=cax)
        # cbar.set_label(r"   $\mathrm{W}_x$", fontsize=35, rotation=0)
        cbar.set_label(r"$\log_{10}P'(E_0, E)$", fontsize=17)
    ax.set_xlabel(r"$E$", fontsize=17)
    ax.set_ylabel(r"$E_0$", fontsize=15)
    ax.set_title(fr"$\theta_{{Bn}}={theta_bn}, \alpha={alpha}$", fontsize=17)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    # plt.plot(bins[1:], counts_hi/counts_total)
    # plt.xlim([0, 75])
    if show:
        plt.show()
    #%%
    a = tracers["pui_tracer_34"].count_particle_crossings(x_shock_34_inerp)
    x_shock_36 = np.load("data_ip_shock/x_shock_data/x_shock_arr_36.npy")
    x_shock_36[1] = 4
    x_shock_36[2] = 10
    f_36 = interp1d(np.linspace(0, 400, 101), x_shock_36, kind='linear')
    x_shock_36_inerp = f_36(np.linspace(0, 400, 401))
    slope_36, intercept_36 = np.polyfit(epoch[4:], x_shock_36[4:], 1)
    x_shock_36_fit = slope_36 * np.linspace(0, 400, 401) + intercept_36
    b = tracers["pui_tracer_35"].count_particle_crossings(x_shock_35_inerp)
    c = tracers["pui_tracer_36"].count_particle_crossings(x_shock_36_inerp)
    plt.hist(a, label=r"$\theta_{Bn}=60\degree$")
    plt.hist(b, label=r"$\theta_{Bn}=75\degree$")
    plt.hist(c, label=r"$\theta_{Bn}=90\degree$")
    plt.yscale("log")
    plt.xlabel("Times of shock crossing", fontsize=15)
    plt.ylabel("Counts", fontsize=15)
    plt.legend()
    plt.show()
    #%%
    tracer = tracers["pui_tracer_35"]
    condition_dn = tracer.data["x"][:, 400] < x_shock_35_inerp[400]
    ptl_core = np.zeros(tracer.data["nptl"], dtype=bool)
    for iptl in range(tracer.data["nptl"]):
        condition_beam = (tracer.data["ux"][iptl, :] > 10) & (tracer.data["x"][iptl, :] > x_shock_35_inerp+20)
        if np.any(condition_beam):
            ptl_core[iptl] = False
        else:
            ptl_core[iptl] = True
    m = np.where(condition_dn*(~ptl_core))[0]
    counts, bins = np.histogram(tracer.data["E"][condition_dn, 400], bins=np.logspace(np.log10(tracer.data["E"][:, 400].min()), np.log10(tracer.data["E"][:, 400].max()), 20))
    counts_perp, bins_2 = np.histogram(tracers["pui_tracer_36"].data["E"][:, 400], bins=np.logspace(np.log10(tracers["pui_tracer_36"].data["E"][:, 400].min()),
                                                                           np.log10(tracers["pui_tracer_36"].data["E"][:, 400].max()),
                                                                           20))
    counts_core, bins = np.histogram(tracer.data["E"][ptl_core*condition_dn, 400], bins=np.logspace(np.log10(tracer.data["E"][:, 400].min()),
                                                                           np.log10(tracer.data["E"][:, 400].max()),
                                                                           20))
    counts_beam, bins = np.histogram(tracer.data["E"][(~ptl_core) * condition_dn, 400],
                                     bins=np.logspace(np.log10(tracer.data["E"][:, 400].min()),
                                                      np.log10(tracer.data["E"][:, 400].max()),
                                                      20))
    print(len(tracer.data["E"][ptl_core, 400]))
    plt.plot(bins[1:], counts/np.sum(counts), label="total")
    plt.plot(bins[1:], counts_core/np.sum(counts_core+counts_beam), label=f"core({np.sum(counts_core)/np.sum(counts_core+counts_beam)*100:.2f}%)")
    plt.plot(bins[1:], counts_beam / np.sum(counts_core + counts_beam), label=f"beam({np.sum(counts_beam)/np.sum(counts_core+counts_beam)*100:.2f}%)")
    plt.plot(bins_2[1:], counts_perp/counts_perp.sum(), label="perpendicular shock")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("E", fontsize=15)
    plt.ylabel("Normalized counts", fontsize=15)
    plt.title(r"$\theta_{Bn}=75\degree$", fontsize=15)
    plt.show()
    #%%
    epoch = 100
    particle_tracking_figs_dir = "particle_tracking_figs/"
    if not os.path.exists(particle_tracking_figs_dir):
        os.mkdir(particle_tracking_figs_dir)
    for epoch in range(401):
        print(epoch)
        plt.figure(figsize=(10, 7))
        plt.scatter(tracers["pui_tracer_34"].data["x"][:, epoch], tracers["pui_tracer_34"].data["E"][:, epoch], edgecolors="k", c=tracers["pui_tracer_34"].data["tag"], cmap="jet")
        plt.axvline(x_shock_34_inerp[epoch], linestyle="--", c="k")
        plt.xlabel("x", fontsize=15)
        plt.ylabel("E", fontsize=15)
        plt.title(fr"epoch={epoch}, $\theta_{{Bn}}=60\degree$", fontsize=15)
        plt.ylim([0, 300])
        plt.savefig(particle_tracking_figs_dir+f"fig_{epoch}.png")
        plt.close()


    #plt.show()
    #%%

    field_dir = f"data_ip_shock/field_data/field_data_37/"
    nx = 256
    nz = 128
    epoch = 50
    bx = load_data_at_certain_t(field_dir+"bx.gda", epoch, nx, nz)
    by = load_data_at_certain_t(field_dir + "by.gda", epoch, nx, nz)
    bz = load_data_at_certain_t(field_dir + "bz.gda", epoch, nx, nz)
    b = np.sqrt(bx**2+by**2+bz**2)
    ex = load_data_at_certain_t(field_dir + "ex.gda", epoch, nx, nz)
    ey = load_data_at_certain_t(field_dir + "ey.gda", epoch, nx, nz)
    ez = load_data_at_certain_t(field_dir + "ez.gda", epoch, nx, nz)
    ni = load_data_at_certain_t(field_dir + "ni.gda", epoch, nx, nz)
    # plt.plot(np.var(bz, axis=1)+np.var(bx, axis=1)+np.var(by, axis=1))
    # plt.plot(np.var(bx, ))
    # plt.plot(ey[:, 32])
    plt.plot(ez[:, 70])
    plt.plot(ey[:, 70])
    plt.plot(ex[:, 70])
    plt.show()
    # %%
    p_1.plot_phase_space_2D(sample_step=1, x_plot_name="x", y_plot_name="uz", color="k", size=2)
    # plt.ylim([-25, 25])
    # plt.plot(bz[:, 94]*100, c="b")
    plt.title(r"$\theta_{Bn}=60\degree, \sigma^2=0.1$, step=10000", fontsize=17)
    plt.show()
    #%%
    condition = (p_1.x > 100) & (p_1.x < 130)
    counts_mat, xedges, zedges = np.histogram2d(p_1.ux[condition], p_1.uz[condition], bins=[81, 81],
                                                range=[[-20, 20], [-20, 30]])
    plt.pcolormesh(xedges[:-1], zedges[:-1], np.log10(counts_mat.T), cmap="jet")
    plt.xlabel(r"$v_{\mathrm{pui,x}}$", fontsize=17)
    plt.ylabel(r"$v_{\mathrm{pui,z}}$", fontsize=17)
    plt.arrow(-1.86, 0, 3, 3*np.tan(80*np.pi/180), head_width=0.8, fc="k")
    plt.text(2, 8, r"$\mathbf{B_0}$", fontsize=15)
    cbar = plt.colorbar()
    cbar.set_label(r"$\mathrm{log_{10}}$(counts)", fontsize=17)
    plt.title(r"$\theta_{Bn}=60\degree$, 100<x<130", fontsize=15)
    plt.show()
    #%%
    print(p_1.get_shock_position_by_density())
    #%%
    p_3.plot_counts_dis_map(vmin=5, vmax=30)
    #%%
    p_3.plot_counts_variation()
    # %%
    v_arr = np.zeros(256)
    for i in range(256):
        condition = (p_1.x >= i) & (p_1.x < i + 1)
        v_arr[i] = p_1.ux[condition].mean()
    plt.plot(v_arr)
    plt.plot(bx[:, 32])
    # plt.plot(ez[:, 33])
    # plt.xlim([25, 125])
    plt.show()
    #%%
    plt.subplot(5, 1, 1)
    plt.pcolormesh(range(nx), range(-nz//2, nz//2), bz.T, cmap="jet", vmin=0, vmax=4)
    cbar = plt.colorbar()
    cbar.set_label("Bz", fontsize=15)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("z", fontsize=15)
    plt.suptitle(r"$\theta_{Bn}=75\degree$, $\Omega_i t=10000$", fontsize=15)
    plt.xticks([])
    plt.subplot(5, 1, 2)

    plt.pcolormesh(range(nx), range(-nz//2, nz//2), bx.T, cmap="jet", vmin=-1, vmax=2)
    cbar = plt.colorbar()
    cbar.set_label("Bx", fontsize=15)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("z", fontsize=15)
    plt.xticks([])
    plt.subplot(5, 1, 3)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("z", fontsize=15)
    plt.pcolormesh(range(nx), range(-nz // 2, nz // 2), by.T, cmap="jet", vmin=-2, vmax=2)
    cbar = plt.colorbar()
    cbar.set_label("By", fontsize=15)
    plt.xticks([])
    plt.subplot(5, 1, 4)
    p_1.plot_counts_dis_map(vmin=5, vmax=25)
    plt.xticks([])
    # plt.title(r"$N_{PUI}$", fontsize=15)
    plt.subplot(5, 1, 5)
    p_5.plot_counts_dis_map(vmin=5, vmax=150)
    plt.subplots_adjust(hspace=0)
    plt.show()
    # p_2.plot_temperature_variation("abc")
    # p_1.plot_energy_distribution_map()
    #%%
    case = 32
    p_plot = p_1
    labels = ["", "", "", "", "", "", "", ""]
    field_dir = f"data_ip_shock/field_data/field_data_{case}/"
    # x_shock_fit = tracers["pui_tracer_28"].x_shock_arr_fit()
    x_shock = get_shock_position(field_dir, 50, bz_threshold=1.6, nx=256, nz=128)
    x_shock = x_shock_39[50]
    # Species.plot_velocity_distribution(p_plot, p_plot, p_plot, p_plot, p_plot,
    #                                    x_ranges=[[x_shock+100, x_shock+120], [x_shock+10, x_shock+20],[x_shock,x_shock+10], [x_shock-10, x_shock],[x_shock-30, x_shock-20]],
    #                                    labels=labels, x_shock=x_shock, energy_bins_num=25)
    Species.plot_velocity_distribution(p_plot, p_plot, p_plot, p_plot, p_plot, p_plot,
                                       x_ranges=[[x_shock + 90, x_shock + 120], [x_shock + 60, x_shock + 90],
                                                 [x_shock + 30, x_shock + 60], [x_shock-5, x_shock + 25],
                                                 [x_shock - 35, x_shock-5], [x_shock - 65, x_shock - 35]],
                                       labels=labels, x_shock=x_shock, energy_bins_num=25)
    plt.ylim(bottom=1e1)
    plt.show()
    #%%
    labels = [r"($\sigma^2=0,\theta=90\degree$)", r"($\sigma^2=0,\theta=90\degree$)"
        , r"($\sigma^2=0,\theta=75\degree$)",  r"($\sigma^2=0,\theta=60\degree$)"]
    Species.plot_velocity_distribution(p_2, p_2, p_1, p_3, x_ranges=[[200, 230], [30, 60], [30, 60], [30, 60]], labels=labels, x_shock=100)
    # plt.show()
    #%%
    labels = ["", ""]
    Species.plot_velocity_distribution(p_1, p_3, x_ranges=[[0, 100], [0, 100]], labels=labels,
                                       x_shock=100, energy_bins_num=35, normalize=True, swap=False)
    # plt.ylim([1e1, 1e5])
    plt.show()
    #%%
    """
    CALCULATE ACCELERATION IN DIFFERENT REGIONS
    """
    dt = 0.25
    tracer = tracers["pui_tracer_32"]
    # delta_E = tracer.data["E"][:, 400] - tracer.data["E"][:, 0]
    delta_E = tracer.data["E"][:, 400]
    # delta_E = tracer.data["E"][:, 0]
    print(delta_E.max())
    index_beam = np.where(np.max(tracer.data["E"], axis=1) > 250)[0]
    shock_position_fit = tracer.x_shock_arr_fit()
    # plt.plot(shock_position_fit)
    # plt.plot(x_shock_34_inerp)
    # plt.show()
    #%%
    Tracer.plot_energy_variation_ShockFrame(tracers["pui_tracer_35"], iptl_list=[1527], x_shock_arr=x_shock_35_inerp)
    #%%
    nptl = tracer.data["nptl"]
    # delta_E_bin = [10, 20, 30, 50, 80, 128, 200]
    # delta_E_bin = np.logspace(np.log10(120), np.log10(160), 6)
    # delta_E_bin = np.concatenate([np.logspace(1, np.log10(70), 20), np.logspace(np.log10(70)+0.1, np.log10(200), 25)])
    delta_E_bin = np.logspace(np.log10(10), np.log10(160), 40)
    dx = 3
    Lx = 150
    Wy_map = np.zeros((len(delta_E_bin)-1, round(Lx/dx)))
    Wx_map = np.zeros((len(delta_E_bin) - 1, round(Lx / dx)))
    Wz_map = np.zeros((len(delta_E_bin) - 1, round(Lx / dx)))

    for i in range(len(delta_E_bin)-1):
        condition_1 = (delta_E >= delta_E_bin[i]) & (delta_E < delta_E_bin[i + 1]) # & (tracer.count_particle_crossings(x_shock_34_inerp)>=1)# & (np.max(tracer.data["uz"], axis=1)<15)
        x_tmp = tracer.data["x"][condition_1, :]
        uy_tmp = tracer.data["uy"][condition_1, :]
        ey_tmp = tracer.data["ey"][condition_1, :]
        ux_tmp = tracer.data["ux"][condition_1, :]
        ex_tmp = tracer.data["ex"][condition_1, :]
        uz_tmp = tracer.data["uz"][condition_1, :]
        ez_tmp = tracer.data["ez"][condition_1, :]
        for j in range(Wy_map.shape[1]-1):
            # print(1)
            condition_2 = (x_tmp - shock_position_fit >= -dx*Wy_map.shape[1]/2+j*dx) & (x_tmp - shock_position_fit < -dx*Wy_map.shape[1]/2+(j+1)*dx)
            Wy_tmp = uy_tmp * ey_tmp* condition_2
            Wx_tmp = ux_tmp * ex_tmp * condition_2
            Wz_tmp = uz_tmp * ez_tmp * condition_2
            # print((tracer.data["uy"][condition_2]).shape)
            Wy_map[i, j] += np.mean(np.sum(Wy_tmp, axis=1)) * dt
            Wx_map[i, j] += np.mean(np.sum(Wx_tmp, axis=1)) * dt
            Wz_map[i, j] += np.mean(np.sum(Wz_tmp, axis=1)) * dt
    #%%
    """
    PLOT ACCELERATION IN DIFFERENT REGIONS
    """
    Wx_max = 1.5
    Wy_max = 12
    Wz_max = 1.5
    i_col = 1
    set_cbar = True
    theta_bn = 90
    # fig, axes = plt.subplots(7, 2, figsize=(32, 30))
    ax = axes[0][i_col]
    pclr = ax.pcolormesh(np.linspace(-Lx/2, Lx/2, round(Lx/dx)), delta_E_bin[1:], Wx_map, cmap="RdBu_r", vmax=Wx_max, vmin=-Wx_max)
    # plt.colorbar(pclr)
    # ax.set_xlabel("distance from shock front", fontsize=15)
    ax.set_ylabel(r"$E_{\mathrm{fina}l}$", fontsize=40)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_title(fr"$\theta_{{Bn}}={theta_bn}\degree, \sigma^2=0.1$", fontsize=40)
    formatter = ScalarFormatter()
    # 禁用科学计数法
    formatter.set_scientific(False)
    # 将格式化器应用到 y 轴
    ax.yaxis.set_major_formatter(formatter)
    # ax.set_yticks([100, 110, 120, 130, 140, 150])
    # ax.set_title("Wx", fontsize=15)
    plt.subplots_adjust(hspace=0, right=0.85)
    ax_ylim = ax.get_ylim()
    face_color = (1, 1, 1, 0)
    edge_color = (0, 0, 0, 1)
    rect_sa = patches.Rectangle((-40, ax_ylim[0]), 25, ax_ylim[1] - ax_ylim[0], facecolor=face_color,
                                edgecolor=edge_color, linewidth=5, linestyle="--")
    # ax.add_patch(rect_sa)
    # ax.set_yscale("log")
    pos = ax.get_position()
    if set_cbar:
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0+0.01, 0.02, pos.y1 - pos.y0 - 0.01])
        # im = ax.imshow(a_smooth[:, 0, 8, :].T, cmap='jet')
        cbar = plt.colorbar(pclr, cax=cax)
        cbar.set_label(r"   $\mathrm{W}_x$", fontsize=35, rotation=0)
    ax = axes[1][i_col]
    pclr = ax.pcolormesh(np.linspace(-Lx / 2, Lx / 2, round(Lx / dx)), delta_E_bin[1:], Wy_map, cmap="RdBu_r", vmax=Wy_max,
                         vmin=-Wy_max)
    # plt.colorbar(pclr)
    # ax.set_xlabel("distance from shock front", fontsize=15)
    ax.set_ylabel(r"$E_{\mathrm{fina}l}$", fontsize=35)
    ax.tick_params(axis='y', labelsize=25)
    ax_ylim = ax.get_ylim()
    rect_sda = patches.Rectangle((-12, ax_ylim[0]), 24, ax_ylim[1] - ax_ylim[0], facecolor=face_color,
                                 edgecolor=edge_color, linewidth=5)
    # ax.add_patch(rect_sda)
    # ax.set_yscale("log")
    # ax.set_title("Wy", fontsize=15)
    # plt.subplots_adjust(hspace=0, right=0.85)
    pos = ax.get_position()
    if set_cbar:
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0+0.01, 0.02, pos.y1 - pos.y0 - 0.01])
        # im = ax.imshow(a_smooth[:, 0, 8, :].T, cmap='jet')
        cbar = plt.colorbar(pclr, cax=cax)
        cbar.set_label(r"   $\mathrm{W}_y$", fontsize=35, rotation=0)
    ax = axes[2][i_col]
    pclr = ax.pcolormesh(np.linspace(-Lx / 2, Lx / 2, round(Lx / dx)), delta_E_bin[1:], Wz_map, cmap="RdBu_r", vmax=Wz_max,
                         vmin=-Wz_max)
    # plt.colorbar(pclr)
    # ax.set_xlabel("distance from shock front", fontsize=13)
    ax.set_ylabel(r"$E_{\mathrm{fina}l}$", fontsize=35)
    ax.tick_params(axis='y', labelsize=25)
    rect_para = patches.Rectangle((-60, ax_ylim[0]), 20, ax_ylim[1] - ax_ylim[0], facecolor=face_color,
                                  edgecolor=edge_color, linewidth=5, linestyle="-.", label="SA region")
    # ax.add_patch(rect_para)
    # ax.set_title("Wz", fontsize=15)
    # plt.subplots_adjust(hspace=0, right=0.85)
    pos = ax.get_position()
    if set_cbar:
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0+0.01, 0.02, pos.y1 - pos.y0 - 0.01])
        # ax.set_yscale("log")
        # im = ax.imshow(a_smooth[:, 0, 8, :].T, cmap='jet')
        cbar = plt.colorbar(pclr, cax=cax)
        cbar.set_label(r"   $\mathrm{W}_z$", fontsize=35, rotation=0)
#%%
    uy_pui_mat=np.zeros((nx, nz))
    for i in range(nx):
        print(i)
        for j in range(nz):
            condition = (p_1.x >= i) & (p_1.x < i+1) & (p_1.z >= j-nz/2) & (p_1.z < j+1-nz/2)
            uy_pui_mat[i, j] = p_1.uy[condition].mean()
    #%%
    ax = axes[3][i_col]

    case = 32
    epoch = 50
    field_dir = f"data_ip_shock/field_data/field_data_{case}/"
    infoarr = loadinfo(field_dir)
    nx = int(infoarr[0])
    nz = int(infoarr[2])
    bz = load_data_at_certain_t(field_dir + "bz.gda", epoch, nx, nz)
    by = load_data_at_certain_t(field_dir + "by.gda", epoch, nx, nz)
    bx = load_data_at_certain_t(field_dir + "bx.gda", epoch, nx, nz)
    ez = load_data_at_certain_t(field_dir + "ez.gda", epoch, nx, nz)
    ey = load_data_at_certain_t(field_dir + "ey.gda", epoch, nx, nz)
    ex = load_data_at_certain_t(field_dir + "ex.gda", epoch, nx, nz)
    pclr = ax.pcolormesh(range(nx)-shock_position_fit[200], range(nz), bx.T, cmap="jet",vmin=-1,vmax=1.5)
    ax.set_xlim([-75, 75])
    ax.tick_params(axis='y', labelsize=25)
    # plt.colorbar(pclr, ax=ax)
    # plt.show()
    pos = ax.get_position()
    if set_cbar:
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.01, 0.02, pos.y1 - pos.y0 - 0.01])
        # ax.set_yscale("log")
        # im = ax.imshow(a_smooth[:, 0, 8, :].T, cmap='jet')
        cbar = plt.colorbar(pclr, cax=cax)
    ax.set_ylabel("Bx", fontsize=35)
    ax = axes[4][i_col]
    pclr = ax.pcolormesh(range(nx) - shock_position_fit[200], range(nz), by.T, cmap="jet",vmin=-1.5,vmax=1.5)
    ax.set_xlim([-75, 75])
    ax.tick_params(axis='y', labelsize=25)
    pos = ax.get_position()
    if set_cbar:
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.01, 0.02, pos.y1 - pos.y0 - 0.01])
        # ax.set_yscale("log")
        # im = ax.imshow(a_smooth[:, 0, 8, :].T, cmap='jet')
        cbar = plt.colorbar(pclr, cax=cax)
    ax.set_ylabel("By", fontsize=35)
    ax = axes[5][i_col]
    pclr = ax.pcolormesh(range(nx) - shock_position_fit[200], range(nz), bz.T, cmap="jet",vmin=0.5,vmax=4)
    ax.set_xlim([-75, 75])
    ax.set_ylabel("Bz", fontsize=35)
    ax.tick_params(axis='y', labelsize=25)
    pos = ax.get_position()
    if set_cbar:
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.01, 0.02, pos.y1 - pos.y0 - 0.01])
        # ax.set_yscale("log")
        # im = ax.imshow(a_smooth[:, 0, 8, :].T, cmap='jet')
        cbar = plt.colorbar(pclr, cax=cax)
    ax = axes[6][i_col]
    p_2.plot_phase_space_2D(sample_step=1, x_plot_name="x", y_plot_name="E",
                            color=np.log(p_2.E), size=1, ax=ax, vmin=np.log(5), vmax=np.log(180), fig=fig, set_cbar=set_cbar, x_offset=shock_position_fit[200])
    ax.set_ylim([0, 400])
    ax.set_xlim([-75, 75])
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=25)
    # ax = axes[4]
    # counts_mat, xedges, zedges = np.histogram2d(p_1.x, p_1.z, bins=[p_1.nx, p_1.nz],
    #                                             range=[[0, p_1.nx], [-p_1.nz / 2, p_1.nz / 2]])
    # # ax.pcolormesh(xedges[:-1]-x_shock_34[epoch], zedges[:-1], counts_mat.T, cmap="jet", vmin=0, vmax=40)
    # ax.pcolormesh(range(nx)-x_shock_34[epoch], range(nz), uy_pui_mat.T, cmap="bwr")
    # plt.suptitle(r"$\theta_{Bn}=75\degree, \sigma^2=0.1$", fontsize=25)
    #%%
    plt.show()
    #%%
    # Species.plot_velocity_distribution(p_1, p_3, p_2, x_ranges=[(50, 60), (50, 60), (50, 60)],
    #                                    sigmas=[0, 1, 2])

    # FIT THE ENERGY SPECTRUM OF PUIs
    # x_range = [50, 60]
    # index = np.where((p_2.x >= x_range[0]) & (p_2.x < x_range[1]))
    # bin_min = 0  # np.min(species.uy[index])
    # bin_max = 600  # np.max(species.uy[index])
    # counts, bins = np.histogram(p_2.E[index], bins=np.linspace(bin_min, bin_max, 30))
    # bin_center = 0.5*(bins[:-1]+bins[1:])
    # p0 = [100, 20, 1.5, 2]
    # p, _ = curve_fit(counts_pui_E, bin_center, counts, p0=p0)
    # A_fit, vc_fit, alpha_fit, eta_fit = p
    # plt.plot(bin_center, counts, label="simulated")
    # plt.plot(bin_center, counts_pui_E(bin_center, A_fit, vc_fit+1.5, alpha_fit, eta_fit-0.55), label="fitted")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    # plt.show()

    # EVALUATE THE POWER SPECTRUM OF TURBULENCE

    # print(get_shock_position(field_dir=field_dir, epoch=50, bz_threshold=1.5, nx=256, nz=64))
    # plot_power_spectra_at_different_positions(50, field_dir=field_dir, epoch=0, nx=256, nz=64, lambda_min=10)

    # it = 50# p_2.it
    # nx = p_2.nx
    # nz = p_2.nz
    # kz_min = 2 * np.pi / nz
    # kz_max = 2 * np.pi / 1
    # bx = load_data_at_certain_t("data_ip_shock/field_data_19/bx.gda", it, nx, nz)
    # by = load_data_at_certain_t("data_ip_shock/field_data_19/by.gda", it, nx, nz)
    # epoch = 45
    cases = [34]
    Pk_hi = np.zeros((len(cases), 100))
    k_min_factor = [10, 1, 5]
    mean_var = np.zeros(256)
    mean_bz = np.zeros(256)
    # fig, ax = plt.subplots(figsize=(15, 5))
    ax = axes[3]
    # ax2 = ax.twinx()
    for i in range(len(cases)):
        case = cases[i]
        field_dir = f"data_ip_shock/field_data/field_data_{case}/"
        infoarr = loadinfo(field_dir)
        nx = int(infoarr[0])
        nz = int(infoarr[2])
        Pk_hi_ez_arr = np.zeros(nx)
        Pk_hi_ex_arr = np.zeros(nx)
        for epoch in range(40, 60):
            bz = load_data_at_certain_t(field_dir + "bz.gda", epoch, nx, nz)
            by = load_data_at_certain_t(field_dir + "by.gda", epoch, nx, nz)
            bx = load_data_at_certain_t(field_dir + "bx.gda", epoch, nx, nz)
            ez = load_data_at_certain_t(field_dir + "ez.gda", epoch, nx, nz)
            ey = load_data_at_certain_t(field_dir + "ey.gda", epoch, nx, nz)
            ex = load_data_at_certain_t(field_dir + "ex.gda", epoch, nx, nz)
            var_arr = np.mean(bx * bx + by * by)
            var_2 = np.var(bx + by, axis=1)
            for j in range(nx):
                x_plot = 80
                kz = fftfreq(bx.shape[1], d=0.5) * 2 * np.pi
                # kz = np.logspace(np.log10(0.01), np.log10(kz_max), nz)
                bx_k = fft(ez[j, :])

                Pk = np.abs(bx_k) ** 2
                Pk_integral = np.sum(Pk) * (kz[1] - kz[0])
                # Pk_1d = np.mean(Pk, axis=0)
                kz_min = 2 * np.pi / 64
                kz_max = 2 * np.pi / 1
                k_bins = np.linspace(kz_min, kz_max, 10)
                Pk_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk)
                Pk_hi[i, epoch] = np.sum(Pk_avg[-2:])
                Pk_hi_ez_arr[j] = np.sum(Pk_avg)
                bx_k = fft(ex[j, :])

                Pk = np.abs(bx_k) ** 2
                Pk_integral = np.sum(Pk) * (kz[1] - kz[0])
                # Pk_1d = np.mean(Pk, axis=0)
                kz_min = 2 * np.pi / 64
                kz_max = 2 * np.pi / 1
                k_bins = np.linspace(kz_min, kz_max, 10)
                Pk_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk)
                Pk_hi_ex_arr[j] = np.sum(Pk_avg)
            # plt.figure(figsize=(10, 7))
            # plt.plot(var_2, label="variation of B")
            mean_bz += np.mean(bz, axis=1)
            # ax2.plot(np.linspace(0, 255, 256) - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.6, nx=nx, nz=nz), np.var(bz, axis=1), c="r")
            # ax2.plot(
            #     np.linspace(0, 255, 256) - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.6, nx=nx,
            #                                                   nz=nz), Pk_hi_ex_arr, c="yellow")
            # ax.plot(
            #     np.linspace(0, 255, 256) - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.6, nx=nx,
            #                                                   nz=nz), np.mean(by, axis=1), c="r")
            # ax2.plot(np.linspace(0, 255, 256) - x_shock_34[epoch], np.var(bz, axis=1), c="b")
            if epoch == 40:
                ax.plot(
                    np.linspace(0, 255, 256) - x_shock_34[epoch], np.var(bx, axis=1), c="r", label=r"$\sigma_{Bx}^2$")
                ax.plot(
                    np.linspace(0, 255, 256) - x_shock_34[epoch], np.var(by, axis=1), c="b", label=r"$\sigma_{By}^2$")
                ax.plot(
                    np.linspace(0, 255, 256) - x_shock_34[epoch], np.var(bz, axis=1), c="k", label=r"$\sigma_{Bz}^2$")
            else:
                ax.plot(
                    np.linspace(0, 255, 256) - x_shock_34[epoch], np.var(bx, axis=1), c="r")
                ax.plot(
                    np.linspace(0, 255, 256) - x_shock_34[epoch], np.var(by, axis=1), c="b")
                ax.plot(
                    np.linspace(0, 255, 256) - x_shock_34[epoch], np.var(bz, axis=1), c="k")
            # ax.text(-83, 0.1, r"$\sigma_{Bx}^2$", c="r")
            # ax.text(-83, 0.2, r"$\sigma_{By}^2$", c="b")
            # ax.text(-83, 0.3, r"$\sigma_{Bz}^2$", c="k")
            ax.set_yscale("log")
            ax.legend(fontsize=14)
            axes[4].plot(np.linspace(0, 255, 256) - x_shock_34[epoch], np.mean(bz, axis=1), c="g")
            # ax.plot(np.linspace(0, 255, 256) - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.6, nx=nx, nz=nz), np.mean(bz, axis=1), c="b")
            # ax2.set_yscale("log")
            # plt.plot(
            #     np.linspace(0, 255, 256) - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.6, nx=nx,
            #                                                   nz=nz), np.mean(ey, axis=1), c="r")
            # plt.plot(bx[200, :], label="Bx")
            # plt.plot(by[200, :], label="By")
            # # plt.plot(bz[200, :], label="Bz")
            # plt.plot(np.sqrt(bx[200, :]**2+by[200, :]**2), label="B")
            # plt.xlim([10,20])
            # a = bx[200, :]
            # b = by[200, :]
            # c = bz[200, :]
            # d = np.sqrt(bx[200, :] ** 2 + by[200, :] ** 2)
            # print(bx[30, :].mean())
            # plt.axvline(get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=256, nz=64),
            #             c="k", linestyle="--", label="shock front")
            # plt.ylabel("variation of Bx", fontsize=15)
            # plt.axvline(0, linestyle="--", c="k")
            # plt.xlabel("distance to shock front", fontsize=16)

            ax.set_xlim([-75, 75])
            ax.set_ylim([1e-3, 1])
            axes[4].set_xlim([-75, 75])
            # plt.ylim([-3, 3])

            # plt.title(
            #     f"t={epoch}, x={x_plot}, x-x_shock={x_plot - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=nx, nz=nz)}, {kz_min*k_min_factor[i]:.2f}<k<{kz_max:.2f}",
            #     fontsize=14)
            # plt.legend()
            # plt.text(75, 2.5, f"P({k_bins[-3]:.2f}<k<{k_bins[-1]:.2f})=1e{np.log10(np.sum(Pk_avg[-2:])):.3f}",
            #          fontsize=15)
            # save_fig_path = f"data_ip_shock/field_data_{cases[i]}/ex_fig"
            # if not os.path.exists(save_fig_path):
            #     os.makedirs(save_fig_path)
            # plt.savefig(save_fig_path + f"/ex_{epoch}.png")
            # plt.close()
            print(epoch)
        # ax2.set_ylim([0.5, 3])
        # ax.set_ylim([0, 6])
        # ax.set_ylabel(r"$\sigma_{\perp}^2$", fontsize=23, c="r")
        # ax2.set_ylabel(r"$\sigma_{\parallel}^2$", fontsize=23, c="b")
        # ax2.tick_params(axis='y', labelcolor='b')
        ax.tick_params(axis='y', labelcolor='r')
        ax_ylim = ax.get_ylim()
        ax4_ylim = axes[4].get_ylim()
        face_color = (1, 1, 1, 0)
        edge_color = (0, 0, 0, 1)
        rect_sda = patches.Rectangle((-12, ax4_ylim[0]), 24, ax4_ylim[1] - ax4_ylim[0], facecolor=face_color,
                                     edgecolor=edge_color, linewidth=5, label="SDA region")
        rect_sa = patches.Rectangle((-40, ax_ylim[0]), 25, ax_ylim[1] - ax_ylim[0], facecolor=face_color,
                                    edgecolor=edge_color, linewidth=5, linestyle="--", label=r"SA region($\perp$)")
        rect_para = patches.Rectangle((-60, ax_ylim[0]), 20, ax_ylim[1] - ax_ylim[0], facecolor=face_color,
                                      edgecolor=edge_color, linewidth=5, linestyle="-.", label=r"SA region($\parallel$)")

        # 将补丁添加到坐标轴
        # axes[4].add_patch(rect_sda)
        # ax.add_patch(rect_sa)
        # ax.add_patch(rect_para)

        # 设置坐标轴刻度标签大小和颜色
        ax.tick_params(axis='x', labelsize=15, labelcolor='k')

        # 设置坐标轴标签
        axes[4].set_xlabel(r"distance to shock front [$v_A/\omega_{ci}$]", fontsize=20)
        axes[4].set_ylabel(r"$B_z$", fontsize=20)
        ax4 = axes[4]

        # 获取图例句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        handles4, labels4 = ax4.get_legend_handles_labels()

        # 合并句柄和标签
        handles.extend(handles4)
        labels.extend(labels4)

        # 添加图例
        # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=26)

        # 设置标题
        turbulence = True
        turbulence_variance = 0.0
        theta_bn = 60
        if turbulence:
            plt.suptitle(fr"$\sigma_1^2$={turbulence_variance}, $\theta_{{Bn}}={theta_bn}\degree$, $E \geq {np.min(delta_E_bin)/np.max(delta_E_bin):.2f}E_{{max}}$", fontsize=25)
        else:
            plt.suptitle(fr"$\sigma_1^2$=0, $\theta_{{Bn}}={theta_bn}\degree$, $E \geq {np.min(delta_E_bin)/np.max(delta_E_bin):.2f}E_{{max}}$", fontsize=25)



        # plt.tight_layout()
        # ax2.spines["right"].set_color("b")
        # plt.plot(np.linspace(-75, 75, 256), mean_bz)
    plt.show()
    plt.close()
    #%%
    """
    PLOT PSD variation
    """
    cases = [7]
    Pk_hi = np.zeros((len(cases), 100))
    Pk_sum = np.zeros((len(cases), 100))
    k_min_factor = [10, 1, 5]

    for i in range(len(cases)):
        case = cases[i]
        field_dir = f"data_ip_shock/field_data_{case}/"
        infoarr = loadinfo(field_dir)
        nx = int(infoarr[0])
        nz = int(infoarr[2])
        para_to_perp = np.zeros((nx, 100))
        b_perp_var = np.zeros((nx, 100))
        b_para_var = np.zeros((nx, 100))
        Pk_hi_arr = np.zeros(nz)
        for epoch in range(1, 100):
            bz = load_data_at_certain_t(field_dir + "bz.gda", epoch, nx, nz)
            by = load_data_at_certain_t(field_dir + "by.gda", epoch, nx, nz)
            bx = load_data_at_certain_t(field_dir + "bx.gda", epoch, nx, nz)
            ez = load_data_at_certain_t(field_dir + "ez.gda", epoch, nx, nz)
            ey = load_data_at_certain_t(field_dir + "ey.gda", epoch, nx, nz)
            ex = load_data_at_certain_t(field_dir + "ex.gda", epoch, nx, nz)
            var_arr = np.mean(bx * bx + by * by)
            var_perp = np.var(bx + by, axis=1)
            var_para = np.var(bz, axis=1)
            b_para_var[:, epoch] = var_para
            b_perp_var[:, epoch] = var_perp
            para_to_perp[:, epoch] = var_para/var_perp
            x_plot = 80
            kz = fftfreq(bx.shape[1], d=0.5) * 2 * np.pi
            # kz = np.logspace(np.log10(0.01), np.log10(kz_max), nz)
            bz_k = fft(bz[x_plot, :])
            bx_k = fft(bx[x_plot, :])
            Pk_z = np.abs(bz_k) ** 2
            Pk_x = np.abs(bx_k) ** 2
            Pk_z_integral = np.sum(Pk_z) * (kz[1] - kz[0])
            # Pk_1d = np.mean(Pk, axis=0)
            kz_min = 2 * np.pi / 64
            kz_max = 2 * np.pi / 1
            k_bins = np.linspace(kz_min, kz_max, 10)
            Pk_z_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk_z)
            Pk_x_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk_x)
            Pk_hi[i, epoch] = np.sum(Pk_z_avg[-2:])
            Pk_sum[i, epoch] = np.sum(Pk_z_avg)
            plt.figure(figsize=(10, 7))
            plt.plot(edges[:-1], Pk_z_avg, label=r"$\delta B_z$")
            plt.plot(edges[:-1], Pk_x_avg, label=r"$\delta B_x$")
            plt.yscale("log")
            plt.xscale("log")
            plt.ylim([1e-1, 1e5])
            plt.title(f"PSD of Bx(t={epoch}, x={x_plot}, x-x_shock={x_plot - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=nx, nz=nz)})")
            # plt.plot(var_2, label="variation of B")
            # plt.plot(bx[x_plot, :])
            # plt.plot(bx[200, :], label="Bx")
            # plt.plot(by[200, :], label="By")
            # # plt.plot(bz[200, :], label="Bz")
            # plt.plot(np.sqrt(bx[200, :]**2+by[200, :]**2), label="B")
            # plt.xlim([10,20])
            # a = bx[200, :]
            # b = by[200, :]
            # c = bz[200, :]
            # d = np.sqrt(bx[200, :] ** 2 + by[200, :] ** 2)
            # print(bx[30, :].mean())
            # plt.axvline(get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=256, nz=64),
            #             c="k", linestyle="--", label="shock front")
            # plt.ylabel("variation of Bx", fontsize=15)
            # plt.xlabel("z", fontsize=16)
            # plt.ylabel("ex", fontsize=16)
            # plt.ylim([-3, 3])

            # plt.title(
            #     f"t={epoch}, x={x_plot}, x-x_shock={x_plot - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=nx, nz=nz)}, {kz_min * k_min_factor[i]:.2f}<k<{kz_max:.2f}",
            #     fontsize=14)
            # plt.legend()
            # plt.text(75, 2.5, f"P({k_bins[-3]:.2f}<k<{k_bins[-1]:.2f})=1e{np.log10(np.sum(Pk_avg[-2:])):.3f}",
            #          fontsize=15)
            plt.legend()
            save_fig_path = f"data_ip_shock/field_data_{cases[i]}/psd_z_fig"
            if not os.path.exists(save_fig_path):
                os.makedirs(save_fig_path)
            plt.savefig(save_fig_path + f"/psd_z_{epoch}.png")
            plt.close()
            print(epoch)
#%%
    """
    PLOT 2D magnetic field map
    """
    cases = [28]
    Pk_hi = np.zeros((len(cases), 100))
    Pk_sum = np.zeros((len(cases), 100))
    k_min_factor = [10, 1, 5]

    for i in range(len(cases)):
        case = cases[i]
        field_dir = f"data_ip_shock/field_data_{case}/"
        infoarr = loadinfo(field_dir)
        nx = int(infoarr[0])
        nz = int(infoarr[2])
        para_to_perp = np.zeros((nx, 100))
        b_perp_var = np.zeros((nx, 100))
        b_para_var = np.zeros((nx, 100))
        Pk_hi_arr = np.zeros(nz)
        for epoch in range(1, 100):
            bz = load_data_at_certain_t(field_dir + "bz.gda", epoch, nx, nz)
            by = load_data_at_certain_t(field_dir + "by.gda", epoch, nx, nz)
            bx = load_data_at_certain_t(field_dir + "bx.gda", epoch, nx, nz)
            ez = load_data_at_certain_t(field_dir + "ez.gda", epoch, nx, nz)
            ey = load_data_at_certain_t(field_dir + "ey.gda", epoch, nx, nz)
            ex = load_data_at_certain_t(field_dir + "ex.gda", epoch, nx, nz)

            var_arr = np.mean(bx * bx + by * by)
            var_perp = np.var(bx + by, axis=1)
            var_para = np.var(bz, axis=1)
            b_para_var[:, epoch] = var_para
            b_perp_var[:, epoch] = var_perp
            # para_to_perp[:, epoch] = var_para / var_perp
            # x_plot = 80
            # kz = fftfreq(bx.shape[1], d=0.5) * 2 * np.pi
            # # kz = np.logspace(np.log10(0.01), np.log10(kz_max), nz)
            # bz_k = fft(bz[x_plot, :])
            # bx_k = fft(bx[x_plot, :])
            # Pk_z = np.abs(bz_k) ** 2
            # Pk_x = np.abs(bx_k) ** 2
            # Pk_z_integral = np.sum(Pk_z) * (kz[1] - kz[0])
            # # Pk_1d = np.mean(Pk, axis=0)
            # kz_min = 2 * np.pi / 64
            # kz_max = 2 * np.pi / 1
            # k_bins = np.linspace(kz_min, kz_max, 10)
            # Pk_z_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk_z)
            # Pk_x_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk_x)
            # Pk_hi[i, epoch] = np.sum(Pk_z_avg[-2:])
            # Pk_sum[i, epoch] = np.sum(Pk_z_avg)
            plt.figure(figsize=(10, 7))
            # plt.plot(edges[:-1], Pk_z_avg, label=r"$\delta B_z$")
            # plt.plot(edges[:-1], Pk_x_avg, label=r"$\delta B_x$")
            plot_param = "ey"
            plt.pcolormesh(range(nx), range(nz), ey.T, cmap="jet", vmin=-2, vmax=2)
            cbar = plt.colorbar()
            cbar.set_label(f"{plot_param}", fontsize=16)
            plt.xlabel(r"x [$v_A/\omega_{ci}$]", fontsize=15)
            plt.ylabel(r"z [$v_A/\omega_{ci}$]", fontsize=15)
            # plt.yscale("log")
            # plt.xscale("log")
            # plt.ylim([1e-1, 1e5])
            plt.title(f"epoch={epoch}")
            # plt.plot(var_2, label="variation of B")
            # plt.plot(bx[x_plot, :])
            # plt.plot(bx[200, :], label="Bx")
            # plt.plot(by[200, :], label="By")
            # # plt.plot(bz[200, :], label="Bz")
            # plt.plot(np.sqrt(bx[200, :]**2+by[200, :]**2), label="B")
            # plt.xlim([10,20])
            # a = bx[200, :]
            # b = by[200, :]
            # c = bz[200, :]
            # d = np.sqrt(bx[200, :] ** 2 + by[200, :] ** 2)
            # print(bx[30, :].mean())
            # plt.axvline(get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=256, nz=64),
            #             c="k", linestyle="--", label="shock front")
            # plt.ylabel("variation of Bx", fontsize=15)
            # plt.xlabel("z", fontsize=16)
            # plt.ylabel("ex", fontsize=16)
            # plt.ylim([-3, 3])

            # plt.title(
            #     f"t={epoch}, x={x_plot}, x-x_shock={x_plot - get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=nx, nz=nz)}, {kz_min * k_min_factor[i]:.2f}<k<{kz_max:.2f}",
            #     fontsize=14)
            # plt.legend()
            # plt.text(75, 2.5, f"P({k_bins[-3]:.2f}<k<{k_bins[-1]:.2f})=1e{np.log10(np.sum(Pk_avg[-2:])):.3f}",
            #          fontsize=15)
            # plt.legend()
            save_fig_path = f"data_ip_shock/field_data_{cases[i]}/{plot_param}_map_fig"
            if not os.path.exists(save_fig_path):
                os.makedirs(save_fig_path)
            plt.savefig(save_fig_path + f"/{plot_param}_{epoch}.png")
            plt.close()
            print(epoch)
#%%
    ex = load_data_at_certain_t(field_dir + "ex.gda", 40, nx, nz)
    plt.plot(range(nx), ex[:, 30])
    plt.axvline(get_shock_position(field_dir,40,bz_threshold=1.5, nx=nx, nz=nz), c="k")
    plt.show()
    #%%
    epoch = 50
    field_dir = "data_ip_shock/field_data/field_data_46/"
    Lx, Ly, Lz = 256, 1, 64
    nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
    hx, hy, hz = Lx/nx, Ly/ny, Lz/nz
    bx = load_data_at_certain_t(field_dir+"bx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    bz = load_data_at_certain_t(field_dir+"bz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    by = load_data_at_certain_t(field_dir + "by.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    Jx, Jy, Jz = calculate_current_density(bx, by, bz, hx, hy, hz)

    ex = load_data_at_certain_t(field_dir+"ex.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    ey = load_data_at_certain_t(field_dir + "ey.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    uiy = load_data_at_certain_t(field_dir + "uiy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    uiz = load_data_at_certain_t(field_dir + "uiz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    pxx = load_data_at_certain_t(field_dir + "pi-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    pyy = load_data_at_certain_t(field_dir + "pi-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    pzz = load_data_at_certain_t(field_dir + "pi-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    ni = load_data_at_certain_t(field_dir + "ni.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    grad_p = np.diff(pxx+pyy+pzz, axis=0)
    Ay = load_data_at_certain_t(field_dir + "Ay.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    # plt.plot(1/para_to_perp[:, epoch]/10)
    # plt.plot(b_para_var[:, epoch], label="b_para")
    # plt.plot(b_perp_var[:, epoch]/10, label="b_perp")
    # plt.plot(bz)
    u_cross_b_x = uiy*bz - uiz*by
    J_cross_B = Jy*bz-Jz*by
    #plt.pcolormesh(range(nx), range(nz), ex.T, cmap="bwr", vmin=-1, vmax=1)
    # plt.pcolormesh(range(nx), range(nz), -u_cross_b_x.T, cmap="bwr", vmin=-1, vmax=1)
   # plt.contour(np.linspace(0, Lx, nx), np.linspace(-Lz/2, Lz/2, nz), Ay.T, colors="k", levels=300, linestyle="--")
    plt.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz/2, Lz/2, nz), np.log(pzz.T/pxx.T), cmap="bwr", vmin=-1, vmax=1)
    # plt.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), bz.T, cmap="jet")
    cbar = plt.colorbar()
    cbar.set_label("Ex", fontsize=15)
    # plt.plot(ex[:, 30])
    # plt.axvline(get_shock_position(field_dir=field_dir, epoch=epoch, bz_threshold=1.5, nx=nx, nz=nz),
    #             c="k", linestyle="--")
    # plt.yscale('log')
    # plt.xlim([70, 130])
    # plt.ylim([10, 32])
    # plt.legend()
    plt.xlabel("x")
    plt.ylabel(r"$z$", fontsize=15)
    plt.title(rf"$\Omega t={epoch}$", fontsize=15)
    plt.show()
#%%
    # # print(np.mean(bz, axis=1))
    # b_turb = np.sqrt(bx**2+by**2)
    # print(np.var(bx[10, :]))
    kz = fftfreq(bx.shape[1], d=0.5) * 2 * np.pi
    # kz = np.logspace(np.log10(0.01), np.log10(kz_max), nz)
    bx_k = fft(bx[50, :])

    Pk = np.abs(bx_k) ** 2
    Pk_integral = np.sum(Pk)*(kz[1]-kz[0])
    # Pk_1d = np.mean(Pk, axis=0)
    kz_min = 2*np.pi/64
    kz_max = 2*np.pi/1
    k_bins = np.linspace(kz_min, kz_max, 25)
    Pk_avg, edges = np.histogram(np.abs(kz), bins=k_bins, weights=Pk)
    plt.scatter(edges[:-1], Pk_avg/Pk_integral)
    print(np.sum(Pk_avg))
    plt.plot(edges, edges**(-5/3)/(kz_min**(-2/3)-kz_max**(-2/3))/1.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([0.05, 5000])
    # plt.show()
    #%%
    plt.plot(range(100), Pk_sum[0, :]
             )
    plt.plot(range(100), Pk_sum[1, :])
    plt.plot(range(100), Pk_sum[2, :])
    plt.yscale("log")
    plt.show()
    #%%
    field_dir = "data_ip_shock/field_data_27"
    # plot_power_spectra_at_different_positions(50, 100, 150, field_dir=field_dir, epoch=0,
    #                                           nx=256, nz=128, lambda_min=1)
    # fig, ax =plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(np.mean(bz, axis=1))
    # plt.show()

    # swi_2.plot_phase_space_2D(sample_step=10, x_plot_name="x", y_plot_name="uz", color="k", size=1)
    # counts_swi_up, counts_swi_dn, bins = swi.get_counts_in_SWAP_view(energy_bin=energy_bin, up_region_start=230, up_region_end=240
    #                            , dn_region_start=20, dn_region_end=30, v_spacesraft=11-14/va)
    # swi.plot_velocity_distribution(x_ranges=[(150, 160), (200, 210), (240, 250)])
    # swi_1.plot_temperature_variation(field_dir)
    # v_arr = np.linspace(-12, 10, 100)

    # plt.show()
    # v0, pheader, data = read_particle_data(base_fname_swi.format(0))
    # # print(round(v0.nt*v0.dt))
    # data = pd.read_csv("particle_injection_info_3.txt", sep=" ", header=None)
    # particle_data = np.array(data.T)
    # pui_index = np.where(particle_data[1, :] == "pui")[0]
    # pui_data = particle_data[2:, pui_index]
    # vx_pui = pui_data[3, :]
    # vy_pui = pui_data[4, :]
    # vz_pui = pui_data[5, :]
    # vr_pui = (vy_pui**2 + vz_pui**2)**0.5
    # #%%
    # bin_min = np.min(vx_pui)
    # bin_max = np.max(vx_pui)
    # vx_bins = np.linspace(bin_min, bin_max, 20)
    # vr_bins = np.linspace(0, np.max(vr_pui), 20)
    # empirical_dist, _, _ = np.histogram2d(vx_pui, vr_pui, bins=[vx_bins, vr_bins])
    # empirical_dist = empirical_dist/np.sum(empirical_dist)
    # # bin_center = 0.5 * (bins[1:] + bins[:-1])
    # vx_grid, vr_grid = np.meshgrid(vx_bins[:-1], vr_bins[:-1])
    # theoretical_dist = g_pui_cylin(vx_grid, vr_grid, v_drift=1.86)
    # theoretical_dist /= np.sum(theoretical_dist)
    # # 可视化理论分布和经验分布
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #
    # axes[0].imshow(theoretical_dist, origin='lower', extent=[vx_bins[0], vx_bins[-1], vr_bins[0], vr_bins[-1]],
    #                aspect='auto', cmap="jet")
    # axes[0].set_title('theoretical')
    # axes[0].set_xlabel('vx')
    # axes[0].set_ylabel('vr')
    #
    # axes[1].imshow(empirical_dist.T, origin='lower', extent=[vx_bins[0], vx_bins[-1], vr_bins[0], vr_bins[-1]],
    #                aspect='auto', cmap="jet")
    # axes[1].set_title('empirical')
    # axes[1].set_xlabel('vx')
    # axes[1].set_ylabel('vr')
    #
    # plt.tight_layout()
    # fig_2, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].plot(np.sum(empirical_dist, axis=0))
    # axes[0].plot(np.sum(theoretical_dist, axis=1))
    # axes[1].plot(np.sum(empirical_dist, axis=1))
    # axes[1].plot(np.sum(theoretical_dist, axis=0))
    # plt.show()




