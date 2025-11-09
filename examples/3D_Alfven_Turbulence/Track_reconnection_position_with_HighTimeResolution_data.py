import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import struct
import scipy.fft
from scipy.fft import fftshift
from scipy.fft import fftfreq
from scipy.io import loadmat,savemat
from scipy.interpolate import griddata
matplotlib.rcParams['font.size'] = 16
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

######### loadinfo function
def loadinfo(dir):
    fstr = dir + "info"
    fd = open(fstr,"rb")
    infocontent = fd.read()
    fd.close
    arr = struct.unpack("fIIIffffff", infocontent[:40])
    infoarr=np.zeros(6)
    infoarr[0] = arr[1]
    infoarr[1] = arr[2]
    infoarr[2] = arr[3]
    infoarr[3] = arr[6]
    infoarr[4] = arr[7]
    infoarr[5] = arr[8]
    print(infoarr)
    return infoarr
######### end loadSlice


######### loadSlice function
def loadSlice(dir,q,sl,nx,ny,nz,interval=11):
    fstr = dir + q + f"_{sl*interval}.gda"
    fd = open(fstr,"rb")
    fd.seek(0*4*sl*nx*ny*nz,1)
    arr = np.fromfile(fd,dtype=np.float32,count=nx*ny*nz)
    fd.close()
    arr = np.reshape(arr,(nz, ny, nx))
    arr = np.transpose(arr, axes=(2, 1, 0))
    return arr
######### end loadSlice

def get_psd(array,dx=0.5,dy=0.5,dt=2.56):
    dim = array.ndim
    if dim == 2:
        FS = (scipy.fft.fftn(array, axes=(0,1)))
        psd = np.abs(FS)**2/(FS.size * dx * dy)
    elif dim == 3:
        FS = (scipy.fft.fftn(array, axes=(0,1,2)))
        psd = np.abs(FS)**2/(FS.size * dx * dy * dt)
    return psd  


######### Make a gif
def makeGIF(imdir, basename, slicenums, imageext):
    images = [(imdir + basename + '_' + str(index) + imageext) for index in slicenums]
    filename = imdir+'../'+basename+'.gif'
    with open(imdir + basename+'_list.txt','w') as fil:
        for item in images:
            fil.write("%s\n" % item)
    os.chdir(imdir)
    os.system('convert @'+basename+'_list.txt '+filename)
########
def get_parameter_index_in_np_array(array, sub_array):
    sub_idx = np.array([], dtype=int)
    for i in sub_array:
        indices = np.where(array == i)[0]  # 获取所有匹配索引
        sub_idx = np.concatenate((sub_idx, indices))
    return sub_idx
cmap = plt.get_cmap("Spectral")

Q = {}
Q_3d = {}
qs = ["ni","bx","by","bz","uix","uiy","uiz","pi-xx","pi-yy","pi-zz","ex","ey","ez"]
qs = ["bx","by","bz","uix","uiy","uiz","ni"]
# qs = ["bx","by","bz","ex","ey","ez"]
dir = "data_imbalanced_highTimeResolution/"
# dir = "/Users/chuanpeng/research/alfven_turbulence/data_2kx_amp002/"
infoarr = loadinfo(dir)
nx = int(infoarr[0])
ny = int(infoarr[1])
nz = int(infoarr[2])
Lx = int(infoarr[3])
Ly = int(infoarr[4])
Lz = int(infoarr[5])
print(infoarr)

dx = Lx/nx
dy = Ly/ny
dz = Lz/nz
dt = 11/32/np.sqrt(3)#2.56


#mark
t_idx_start = 95
t_idx_end = 105
t_idx_step = 1

calculate_w_k = 1
use_cwt_2DT_CPU = 0
use_cwt_2DT_GPU = 1
re_calc = 0

W_thermal_arr = np.array([])

xv = np.linspace(0,Lx,nx)-Lx/2.0
yv = np.linspace(0,Ly,ny)-Ly/2.0
zv = np.linspace(0,Lz,nz)-Lz/2.0
slicenums = []

for q in qs:
    slices = []
    for slice_ in range(t_idx_start, t_idx_end, t_idx_step):
            print(slice_)
            tmp = loadSlice(dir,q,slice_,nx,ny,nz)
            slices.append(tmp[:,:,:])
    Q_3d[q] = np.stack(slices, axis=-1)

variable = Q_3d['bx'][:,:,:,0]
vari_rms = np.sqrt(np.mean(variable**2) - np.mean(variable)**2)
print("vari_rms: ", vari_rms)

import math
def find_nearest_lattice(r1, direction, l):
    """
    在三维格点中，找到沿指定方向、距离起始格点r1为l的最近格点。
    
    参数:
        r1 (tuple/list): 起始格点坐标，需为3个整数，如(1, 2, 3)
        direction (tuple/list): 方向向量，需为3个数字，如(1.0, 1.0, 0.0)
        l (float/int): 距离，需为正数
    
    返回:
        tuple: 最近的格点坐标（3个整数）
    
    异常:
        ValueError: 输入参数不符合要求时抛出
    """
    # 验证起始格点r1的有效性（3个整数）
    if not (isinstance(r1, (tuple, list)) and len(r1) == 3 and all(isinstance(x, int) for x in r1)):
        raise ValueError("r1必须是包含3个整数的元组或列表")
    
    # 验证方向向量的有效性（3个数字）
    if not (isinstance(direction, (tuple, list)) and len(direction) == 3):
        raise ValueError("direction必须是包含3个数字的元组或列表")
    
    # 验证距离l的有效性（正数）
    if not (isinstance(l, (int, float)) and l > 0):
        raise ValueError("l必须是正数")
    
    # 解析方向向量分量
    vx, vy, vz = direction
    
    # 计算方向向量的模长（避免零向量）
    mod = math.sqrt(vx**2 + vy**2 + vz**2)
    if mod < 1e-10:  # 考虑浮点数精度误差
        raise ValueError("方向向量不能是零向量")
    
    # 归一化方向向量（得到单位向量）
    nx = vx / mod
    ny = vy / mod
    nz = vz / mod
    
    # 计算沿方向移动距离l后的目标点坐标（非格点）
    px = r1[0] + l * nx
    py = r1[1] + l * ny
    pz = r1[2] + l * nz
    
    # 四舍五入得到最近的格点（各分量取最近整数）
    nearest_lattice = (round(px), round(py), round(pz))
    
    return nearest_lattice

def calculate_curl(Bx, By, Bz, x, y, z):
    # 计算偏导数
    dBz_dy = np.gradient(Bz, y, axis=1)
    dBy_dz = np.gradient(By, z, axis=2)
    dBx_dz = np.gradient(Bx, z, axis=2)
    dBz_dx = np.gradient(Bz, x, axis=0)
    dBy_dx = np.gradient(By, x, axis=0)
    dBx_dy = np.gradient(Bx, y, axis=1)

    # 计算旋度
    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy

    return curl_x, curl_y, curl_z
# import numpy as np
# from numba import njit, prange
# from tqdm import tqdm
# from numba import get_num_threads, set_num_threads
# # set_num_threads(256)
# print(f"Numba默认线程数：{get_num_threads()}")

# @njit(fastmath=True, parallel=True)
# def _numba_batch_eig(jacobian_batch, use_symmetric):
#     n_total = jacobian_batch.shape[0]
#     # 预定义复数类型数组（固定为complex128）
#     eigenvals = np.zeros((n_total, 3), dtype=np.complex128)
#     eigenvecs = np.zeros((n_total, 3, 3), dtype=np.complex128)
    
#     for idx in prange(n_total):
#         jac = jacobian_batch[idx]
#         # 显式转为复数矩阵，避免类型混乱
#         jac_complex = jac.astype(np.complex128)
        
#         if use_symmetric:
#             # 处理对称矩阵：eigh返回实数，需转为复数
#             jac_sym = 0.5 * (jac_complex + jac_complex.T.conj())
#             evals_real, evecs_real = np.linalg.eigh(jac_sym)
#             # 关键：强制转换为complex128，与预定义数组类型匹配
#             evals = evals_real.astype(np.complex128)
#             evecs = evecs_real.astype(np.complex128)
#         else:
#             # 处理非对称矩阵：eig直接返回复数，无需额外转换
#             evals, evecs = np.linalg.eig(jac_complex)
        
#         # 赋值（此时类型完全匹配）
#         eigenvals[idx] = evals
#         eigenvecs[idx] = evecs
#     return eigenvals, eigenvecs


# def calc_vector_field_jacobian_eigen_numba(vector_field, hx, hy, hz, use_symmetric=False):
#     # 输入校验
#     if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
#         raise ValueError(f"矢量场必须是 (nx, ny, nz, 3)，当前形状 {vector_field.shape}")
#     if hx <= 0 or hy <= 0 or hz <= 0:
#         raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
#     nx, ny, nz, _ = vector_field.shape
#     n_total = nx * ny * nz
    
#     # 一阶偏导数计算（确保输入为float64）
#     def first_deriv(arr, axis, h):
#         # 强制arr为float64，避免混合类型
#         arr_float = arr.astype(np.float64)
#         deriv = np.zeros_like(arr_float, dtype=np.float64)
#         n = arr_float.shape[axis]
        
#         # 内部点：中心差分
#         slices = [slice(None)] * 3
#         slices[axis] = slice(1, n-1)
#         slices_plus = [slice(None)] * 3
#         slices_plus[axis] = slice(2, n)
#         slices_minus = [slice(None)] * 3
#         slices_minus[axis] = slice(0, n-2)
#         deriv[tuple(slices)] = (arr_float[tuple(slices_plus)] - arr_float[tuple(slices_minus)]) / (2 * h)
        
#         # 首边界：向前差分
#         slices_first = [slice(None)] * 3
#         slices_first[axis] = 0
#         slices_first_plus = [slice(None)] * 3
#         slices_first_plus[axis] = 1
#         deriv[tuple(slices_first)] = (arr_float[tuple(slices_first_plus)] - arr_float[tuple(slices_first)]) / h
        
#         # 尾边界：向后差分
#         slices_last = [slice(None)] * 3
#         slices_last[axis] = -1
#         slices_last_minus = [slice(None)] * 3
#         slices_last_minus[axis] = -2
#         deriv[tuple(slices_last)] = (arr_float[tuple(slices_last)] - arr_float[tuple(slices_last_minus)]) / h
        
#         return deriv
    
#     # 计算偏导数（统一为float64）
#     d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
#     d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
#     d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
#     d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
#     d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
#     d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
#     d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
#     d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
#     d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
#     # 构建批量雅可比矩阵（统一为float64）
#     d_Bx_dx_flat = d_Bx_dx.reshape(-1, 1)
#     d_Bx_dy_flat = d_Bx_dy.reshape(-1, 1)
#     d_Bx_dz_flat = d_Bx_dz.reshape(-1, 1)
#     d_By_dx_flat = d_By_dx.reshape(-1, 1)
#     d_By_dy_flat = d_By_dy.reshape(-1, 1)
#     d_By_dz_flat = d_By_dz.reshape(-1, 1)
#     d_Bz_dx_flat = d_Bz_dx.reshape(-1, 1)
#     d_Bz_dy_flat = d_Bz_dy.reshape(-1, 1)
#     d_Bz_dz_flat = d_Bz_dz.reshape(-1, 1)
    
#     jacobian_batch = np.stack([
#         np.hstack([d_Bx_dx_flat, d_Bx_dy_flat, d_Bx_dz_flat]),
#         np.hstack([d_By_dx_flat, d_By_dy_flat, d_By_dz_flat]),
#         np.hstack([d_Bz_dx_flat, d_Bz_dy_flat, d_Bz_dz_flat])
#     ], axis=1).astype(np.float64)  # 强制float64，避免类型混合
    
#     # 计算特征值/向量（带进度条）
#     with tqdm(total=1, desc="Numba批量计算特征值（类型统一）") as pbar:
#         eigenvals_batch, eigenvecs_batch = _numba_batch_eig(jacobian_batch, use_symmetric)
#         pbar.update(1)
    
#     # 重塑回原三维形状
#     eigenvals = eigenvals_batch.reshape(nx, ny, nz, 3)
#     eigenvecs = eigenvecs_batch.reshape(nx, ny, nz, 3, 3)
    
#     return eigenvals, eigenvecs
import numpy as np
from tqdm import tqdm


def calc_vector_field_jacobian_eigen_vectorized(vector_field, hx, hy, hz, use_symmetric=False):
    """
    向量化版本：计算三维矢量场的雅可比矩阵特征值和特征向量（无Python循环）。
    
    参数:
        vector_field (np.ndarray): 三维矢量场，形状 (nx, ny, nz, 3)。
        hx, hy, hz (float): 各方向空间步长（正数）。
        use_symmetric (bool): 若为True，假设雅可比矩阵对称，用eigh加速（快2~3倍）。
    
    返回:
        eigenvals (np.ndarray): 特征值数组 (nx, ny, nz, 3)（复数）。
        eigenvecs (np.ndarray): 特征向量数组 (nx, ny, nz, 3, 3)（复数）。
    """
    # 输入校验
    if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
        raise ValueError(f"矢量场必须是 (nx, ny, nz, 3)，当前形状 {vector_field.shape}")
    if hx <= 0 or hy <= 0 or hz <= 0:
        raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
    nx, ny, nz, _ = vector_field.shape
    n_total = nx * ny * nz  # 总网格点数
    eigenvals = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    eigenvecs = np.zeros((nx, ny, nz, 3, 3), dtype=np.complex128)
    
    # --------------------------
    # 1. 向量化计算一阶偏导数（同原逻辑）
    # --------------------------
    def first_deriv(arr, axis, h):
        """向量化计算沿指定轴的一阶偏导数（中心/向前/向后差分）"""
        deriv = np.zeros_like(arr, dtype=np.float64)
        n = arr.shape[axis]
        # 内部点：中心差分 (f[i+1] - f[i-1])/(2h)
        slices = [slice(None)] * 3
        slices[axis] = slice(1, n-1)
        slices_plus = [slice(None)] * 3
        slices_plus[axis] = slice(2, n)
        slices_minus = [slice(None)] * 3
        slices_minus[axis] = slice(0, n-2)
        deriv[tuple(slices)] = (arr[tuple(slices_plus)] - arr[tuple(slices_minus)]) / (2 * h)
        # 首边界：向前差分 (f[1] - f[0])/h
        slices_first = [slice(None)] * 3
        slices_first[axis] = 0
        slices_first_plus = [slice(None)] * 3
        slices_first_plus[axis] = 1
        deriv[tuple(slices_first)] = (arr[tuple(slices_first_plus)] - arr[tuple(slices_first)]) / h
        # 尾边界：向后差分 (f[-1] - f[-2])/h
        slices_last = [slice(None)] * 3
        slices_last[axis] = -1
        slices_last_minus = [slice(None)] * 3
        slices_last_minus[axis] = -2
        deriv[tuple(slices_last)] = (arr[tuple(slices_last)] - arr[tuple(slices_last_minus)]) / h
        return deriv
    
    # 计算9个偏导数（向量化操作，无循环）
    d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
    d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
    d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
    d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
    d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
    d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
    d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
    d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
    d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
    # --------------------------
    # 2. 向量化构建批量雅可比矩阵
    # --------------------------
    # 将所有偏导数从 (nx, ny, nz) 重塑为 (n_total, 1)，便于堆叠
    d_Bx_dx_flat = d_Bx_dx.reshape(-1, 1)  # 形状 (n_total, 1)
    d_Bx_dy_flat = d_Bx_dy.reshape(-1, 1)
    d_Bx_dz_flat = d_Bx_dz.reshape(-1, 1)
    d_By_dx_flat = d_By_dx.reshape(-1, 1)
    d_By_dy_flat = d_By_dy.reshape(-1, 1)
    d_By_dz_flat = d_By_dz.reshape(-1, 1)
    d_Bz_dx_flat = d_Bz_dx.reshape(-1, 1)
    d_Bz_dy_flat = d_Bz_dy.reshape(-1, 1)
    d_Bz_dz_flat = d_Bz_dz.reshape(-1, 1)
    
    # 堆叠成 (n_total, 3, 3) 的批量雅可比矩阵：每个元素是一个网格点的3×3矩阵
    # 结构：[[∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z],
    #        [∂By/∂x, ∂By/∂y, ∂By/∂z],
    #        [∂Bz/∂x, ∂Bz/∂y, ∂Bz/∂z]]
    jacobian_batch = np.stack([
        np.hstack([d_Bx_dx_flat, d_Bx_dy_flat, d_Bx_dz_flat]),  # 第0行：Bx的三个偏导数
        np.hstack([d_By_dx_flat, d_By_dy_flat, d_By_dz_flat]),  # 第1行：By的三个偏导数
        np.hstack([d_Bz_dx_flat, d_Bz_dy_flat, d_Bz_dz_flat])   # 第2行：Bz的三个偏导数
    ], axis=1)  # 最终形状：(n_total, 3, 3)
    
    # --------------------------
    # 3. 向量化批量计算特征值/向量
    # --------------------------
    # 进度条（监控批量计算进度，可选）
    with tqdm(total=1, desc="批量计算特征值") as pbar:
        if use_symmetric:
            # 若矩阵对称，用eigh加速（速度快2~3倍，数值更稳定）
            # 对称化处理：J_sym = (J + J.T)/2
            jacobian_batch = 0.5 * (jacobian_batch + np.transpose(jacobian_batch, axes=(0, 2, 1)))
            eigenvals_batch, eigenvecs_batch = np.linalg.eigh(jacobian_batch)
        else:
            # 通用非对称矩阵，用eig计算
            eigenvals_batch, eigenvecs_batch = np.linalg.eig(jacobian_batch)
        pbar.update(1)
    
    # --------------------------
    # 4. 重塑回原三维网格形状
    # --------------------------
    eigenvals = eigenvals_batch.reshape(nx, ny, nz, 3)  # 从 (n_total, 3) → (nx, ny, nz, 3)
    eigenvecs = eigenvecs_batch.reshape(nx, ny, nz, 3, 3)  # 从 (n_total, 3, 3) → (nx, ny, nz, 3, 3)
    
    return eigenvals, eigenvecs
# import numpy as np
# import concurrent.futures
# from concurrent.futures import ProcessPoolExecutor
# import itertools
# import os
# from tqdm import tqdm  # 导入进度条库


# def calc_vector_field_jacobian_eigen(vector_field, hx, hy, hz, max_workers=None):
#     """
#     计算三维矢量场的雅可比矩阵的特征值和特征向量（多线程版本，带进度条）。
    
#     参数:
#         vector_field (np.ndarray): 三维矢量场数据，形状为 (nx, ny, nz, 3)。
#         hx, hy, hz (float): 各方向空间步长（正数）。
#         max_workers (int, optional): 线程池最大线程数，默认使用CPU核心数。
    
#     返回:
#         eigenvals (np.ndarray): 特征值数组 (nx, ny, nz, 3)（可能为复数）。
#         eigenvecs (np.ndarray): 特征向量数组 (nx, ny, nz, 3, 3)（可能为复数）。
#     """
#     # 输入校验
#     if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
#         raise ValueError(f"矢量场必须是形状为 (nx, ny, nz, 3) 的4维数组，当前形状为 {vector_field.shape}")
#     if hx <= 0 or hy <= 0 or hz <= 0:
#         raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
#     nx, ny, nz, _ = vector_field.shape
#     total_points = nx * ny * nz  # 总网格点数（用于进度条）
#     eigenvals = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
#     eigenvecs = np.zeros((nx, ny, nz, 3, 3), dtype=np.complex128)
    
#     # 一阶偏导数计算函数（同原逻辑）
#     def first_deriv(arr, axis, h):
#         deriv = np.zeros_like(arr, dtype=np.float64)
#         n = arr.shape[axis]
#         # 内部点：中心差分
#         slices = [slice(None)] * 3
#         slices[axis] = slice(1, n-1)
#         slices_plus = [slice(None)] * 3
#         slices_plus[axis] = slice(2, n)
#         slices_minus = [slice(None)] * 3
#         slices_minus[axis] = slice(0, n-2)
#         deriv[tuple(slices)] = (arr[tuple(slices_plus)] - arr[tuple(slices_minus)]) / (2 * h)
#         # 首边界：向前差分
#         slices_first = [slice(None)] * 3
#         slices_first[axis] = 0
#         slices_first_plus = [slice(None)] * 3
#         slices_first_plus[axis] = 1
#         deriv[tuple(slices_first)] = (arr[tuple(slices_first_plus)] - arr[tuple(slices_first)]) / h
#         # 尾边界：向后差分
#         slices_last = [slice(None)] * 3
#         slices_last[axis] = -1
#         slices_last_minus = [slice(None)] * 3
#         slices_last_minus[axis] = -2
#         deriv[tuple(slices_last)] = (arr[tuple(slices_last)] - arr[tuple(slices_last_minus)]) / h
#         return deriv
    
#     # 计算所有偏导数（同原逻辑）
#     d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
#     d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
#     d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
#     d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
#     d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
#     d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
#     d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
#     d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
#     d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
#     # 单个网格点的处理函数（供线程调用）
#     def process_point(i, j, k):
#         # 构建雅可比矩阵
#         jacobian = np.array([
#             [d_Bx_dx[i, j, k], d_Bx_dy[i, j, k], d_Bx_dz[i, j, k]],
#             [d_By_dx[i, j, k], d_By_dy[i, j, k], d_By_dz[i, j, k]],
#             [d_Bz_dx[i, j, k], d_Bz_dy[i, j, k], d_Bz_dz[i, j, k]]
#         ], dtype=np.float64)
#         # 计算特征值和特征向量
#         evals, evecs = np.linalg.eig(jacobian)  # evecs每一列是特征向量
#         # 写入结果（线程安全）
#         eigenvals[i, j, k] = evals
#         eigenvecs[i, j, k] = evecs
#         return  # 仅用于触发进度条更新
    
#     # 生成所有网格点的索引 (i,j,k)，并包装进度条
#     indices = itertools.product(range(nx), range(ny), range(nz))
#     # 用tqdm包装迭代器，显示进度（total指定总任务数）
#     pbar = tqdm(indices, total=total_points, desc="计算雅可比矩阵特征值", unit="点")
    
#     # 多线程处理所有点（结合进度条）
#     max_workers = max_workers or os.cpu_count()
#     print(max_workers)
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:  # 用ProcessPoolExecutor替代ThreadPoolExecutor
#         list(executor.map(lambda x: process_point(x[0], x[1], x[2]), pbar))
    
#     return eigenvals, eigenvecs
# def calc_vector_field_jacobian_eigen(vector_field, hx, hy, hz):
#     """
#     计算三维矢量场的雅可比矩阵（一阶梯度矩阵）的特征值和特征向量。
    
#     参数:
#         vector_field (np.ndarray): 三维矢量场数据，形状为 (nx, ny, nz, 3)，
#                                  最后一个维度对应3个分量（如 [Bx, By, Bz]）。
#         hx (float): x方向空间步长（相邻网格x坐标差），必须为正数。
#         hy (float): y方向空间步长，必须为正数。
#         hz (float): z方向空间步长，必须为正数。
    
#     返回:
#         eigenvals (np.ndarray): 每个网格点的雅可比矩阵特征值，形状为 (nx, ny, nz, 3)，
#                                特征值可能为复数（因雅可比矩阵不一定对称）。
#         eigenvecs (np.ndarray): 每个网格点的雅可比矩阵特征向量，形状为 (nx, ny, nz, 3, 3)，
#                                其中eigenvecs[i,j,k,:,m]对应第m个特征值的特征向量，
#                                特征向量可能为复数。
    
#     数值方法:
#         - 内部网格点：采用**中心差分**（二阶精度）计算一阶偏导数。
#         - 边界网格点：采用**向前/向后差分**（一阶精度）避免越界。
#     """
#     # 输入校验
#     if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
#         raise ValueError(f"矢量场必须是形状为 (nx, ny, nz, 3) 的4维数组，当前形状为 {vector_field.shape}")
#     if hx <= 0 or hy <= 0 or hz <= 0:
#         raise ValueError(f"空间步长必须为正数，当前 hx={hx}, hy={hy}, hz={hz}")
    
#     nx, ny, nz, _ = vector_field.shape
#     # 存储特征值（可能为复数）
#     eigenvals = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
#     # 存储特征向量（最后两个维度：3个特征向量，每个为3维向量）
#     eigenvecs = np.zeros((nx, ny, nz, 3, 3), dtype=np.complex128)
    
#     # 定义一阶偏导数计算函数（处理边界）
#     def first_deriv(arr, axis, h):
#         """
#         计算数组沿指定轴的一阶偏导数。
#         参数:
#             arr: 输入数组（形状为 (nx, ny, nz)）
#             axis: 求导轴（0=x, 1=y, 2=z）
#             h: 该轴的空间步长
#         返回:
#             deriv: 与arr同形状的一阶导数数组
#         """
#         deriv = np.zeros_like(arr, dtype=np.float64)
#         n = arr.shape[axis]  # 该轴的网格点数
        
#         # 内部点：中心差分 (f[i+1] - f[i-1])/(2h)
#         slices = [slice(None)] * 3
#         slices[axis] = slice(1, n-1)  # 内部索引
#         slices_plus = [slice(None)] * 3
#         slices_plus[axis] = slice(2, n)  # i+1
#         slices_minus = [slice(None)] * 3
#         slices_minus[axis] = slice(0, n-2)  # i-1
#         deriv[tuple(slices)] = (arr[tuple(slices_plus)] - arr[tuple(slices_minus)]) / (2 * h)
        
#         # 边界点：首边界用向前差分 (f[1] - f[0])/h
#         slices_first = [slice(None)] * 3
#         slices_first[axis] = 0
#         slices_first_plus = [slice(None)] * 3
#         slices_first_plus[axis] = 1
#         deriv[tuple(slices_first)] = (arr[tuple(slices_first_plus)] - arr[tuple(slices_first)]) / h
        
#         # 边界点：尾边界用向后差分 (f[-1] - f[-2])/h
#         slices_last = [slice(None)] * 3
#         slices_last[axis] = -1
#         slices_last_minus = [slice(None)] * 3
#         slices_last_minus[axis] = -2
#         deriv[tuple(slices_last)] = (arr[tuple(slices_last)] - arr[tuple(slices_last_minus)]) / h
        
#         return deriv
    
#     # 计算雅可比矩阵的9个元素（3个分量×3个方向的偏导数）
#     # Bx的偏导数：∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z
#     d_Bx_dx = first_deriv(vector_field[..., 0], axis=0, h=hx)
#     d_Bx_dy = first_deriv(vector_field[..., 0], axis=1, h=hy)
#     d_Bx_dz = first_deriv(vector_field[..., 0], axis=2, h=hz)
    
#     # By的偏导数：∂By/∂x, ∂By/∂y, ∂By/∂z
#     d_By_dx = first_deriv(vector_field[..., 1], axis=0, h=hx)
#     d_By_dy = first_deriv(vector_field[..., 1], axis=1, h=hy)
#     d_By_dz = first_deriv(vector_field[..., 1], axis=2, h=hz)
    
#     # Bz的偏导数：∂Bz/∂x, ∂Bz/∂y, ∂Bz/∂z
#     d_Bz_dx = first_deriv(vector_field[..., 2], axis=0, h=hx)
#     d_Bz_dy = first_deriv(vector_field[..., 2], axis=1, h=hy)
#     d_Bz_dz = first_deriv(vector_field[..., 2], axis=2, h=hz)
    
#     # 遍历每个网格点，构建雅可比矩阵并计算特征值和特征向量
#     for i in range(nx):
#         print(i)
#         for j in range(ny):
#             for k in range(nz):
                
#                 # 构建当前点的3×3雅可比矩阵
#                 jacobian = np.array([
#                     [d_Bx_dx[i, j, k], d_Bx_dy[i, j, k], d_Bx_dz[i, j, k]],
#                     [d_By_dx[i, j, k], d_By_dy[i, j, k], d_By_dz[i, j, k]],
#                     [d_Bz_dx[i, j, k], d_Bz_dy[i, j, k], d_Bz_dz[i, j, k]]
#                 ], dtype=np.float64)
#                 symmetric_jacobian = 0.5 * (jacobian + jacobian.T)  # 对称化雅可比矩阵，以防奇异
                
#                 # 计算特征值和特征向量（雅可比矩阵可能非对称，用eig求复特征值/向量）
#                 evals, evecs = np.linalg.eig(jacobian)  # evecs每一列是一个特征向量
#                 eigenvals[i, j, k] = evals
#                 # 存储特征向量（保持与特征值的对应关系）
#                 eigenvecs[i, j, k] = evecs  # 形状为(3,3)，每列对应一个特征值的特征向量
    
#     return eigenvals, eigenvecs
if __name__ == '__main__':
    epoch = 10
    for epoch in range(1):
        t_idx = 5
        Bx, By, Bz = Q_3d['bx'][:,:,:,t_idx],Q_3d['by'][:,:,:,t_idx],Q_3d['bz'][:,:,:,t_idx]
        uix, uiy, uiz = Q_3d['uix'][:,:,:,t_idx],Q_3d['uiy'][:,:,:,t_idx],Q_3d['uiz'][:,:,:,t_idx]
        B_vec = np.stack([Bx,By,Bz],axis=-1)
        u_vec = np.stack([uix,uiy,uiz],axis=-1)
        print(B_vec.shape)
        #算的时候得用梯度矩阵计算
        lambda_mat_B, eigen_vec_B = calc_vector_field_jacobian_eigen_vectorized(B_vec,hx=0.25,hy=0.25,hz=0.25)
        lambda_mat_u, eigen_vec_u = calc_vector_field_jacobian_eigen_vectorized(u_vec,hx=0.25,hy=0.25,hz=0.25)
        C = 5
        print(lambda_mat_B.shape)
        print(lambda_mat_B[0,0,0,:])
        sort_indices = np.argsort(np.abs(np.real(lambda_mat_B)),axis=3)
        sort_indices_u = np.argsort(np.abs(np.real(lambda_mat_u)),axis=3)
        print(sort_indices[0,0,0,:])
        lambda_mat_B_sorted = np.take_along_axis(lambda_mat_B,sort_indices,axis=3)
        lambda_mat_u_sorted = np.take_along_axis(lambda_mat_u,sort_indices_u,axis=3)
        eig_vec_B_sorted = np.take_along_axis(eigen_vec_B,sort_indices[...,np.newaxis],axis=3)
        eig_vec_u_sorted = np.take_along_axis(eigen_vec_u,sort_indices_u[...,np.newaxis],axis=3)
        n_vec_B = np.cross(eig_vec_B_sorted[:,:,:,1,:],eig_vec_B_sorted[:,:,:,2,:],axis=3)
        n_vec_u= np.cross(eig_vec_u_sorted[:,:,:,1,:],eig_vec_u_sorted[:,:,:,2,:],axis=3)
        print(eig_vec_B_sorted.shape)
        print(eig_vec_B_sorted[0,1,0,:,:],eigen_vec_B[0,1,0,:,:])
        condition_B = (np.abs(np.imag(lambda_mat_B[:,:,:,0]))<1e-10) & (np.abs(np.imag(lambda_mat_B[:,:,:,1]))<1e-10) & (np.abs(np.imag(lambda_mat_B[:,:,:,2]))<1e-10)  & (np.abs(lambda_mat_B_sorted[:,:,:,1])>C*np.abs(lambda_mat_B_sorted[:,:,:,0])) & (np.real(lambda_mat_B_sorted[:,:,:,1]*lambda_mat_B_sorted[:,:,:,2])<0)
        condition_u = (np.abs(np.imag(lambda_mat_u[:,:,:,0]))<1e-10) & (np.abs(np.imag(lambda_mat_u[:,:,:,1]))<1e-10) & (np.abs(np.imag(lambda_mat_u[:,:,:,2]))<1e-10)  & (np.abs(lambda_mat_u_sorted[:,:,:,1])>C*np.abs(lambda_mat_u_sorted[:,:,:,0])) & (np.real(lambda_mat_u_sorted[:,:,:,1]*lambda_mat_u_sorted[:,:,:,2])<0)
        condition_plane = np.abs(np.sum(n_vec_B*n_vec_u,axis=3)/np.linalg.norm(n_vec_B,axis=3)/np.linalg.norm(n_vec_u,axis=3))>0.9
        condition = np.where(condition_B & condition_u & condition_plane)
        i_lst, j_lst, k_lst = list(zip(condition))
        # lambda_mat_B_sorted[i_lst,j_lst,k_lst].shape
        # print(lambda_mat_B_sorted[condition[0]][0])
        # print(eigen_vec_B[condition[0]])
        # print(condition[0].shape)
        i_right = 0
        x_point_lst = []
        lambda_1_lst, lambda_2_lst = [], []
        eig_vec_b1_lst = []
        eig_vec_b2_lst = []
        delta_l = 3
        cos_lambda_1z_lst = []
        cos_lambda_2z_lst = []
        cos_lambda_max_lst = []
        cos_lambda_min_lst = []
        Bz_total = Bz
        # from scipy.interpolate import RegularGridInterpolator
        x = np.linspace(-32, 32, 256)
        y = np.linspace(-32, 32, 256)
        z = np.linspace(-32, 32, 256)
        # interp_Bx = RegularGridInterpolator((x, y, z), Bx)
        # interp_By = RegularGridInterpolator((x, y, z), By)
        # interp_Bz = RegularGridInterpolator((x, y, z), Bz_total)
        # interp_Ti = RegularGridInterpolator((x, y, z), Ti)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = np.linspace(-5, 5, 100)  # a0方向坐标
        v = np.linspace(-5, 5, 100)  # b0方向坐标
        U, V = np.meshgrid(u, v)
        for i in range(len(condition[0])):
            # print(i)
            cos_lambda_1z = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:],[0,0,1])/np.linalg.norm(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])
            cos_lambda_2z = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:],[0,0,1])/np.linalg.norm(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])
            cos_lambda_1z_lst.append(cos_lambda_1z)
            cos_lambda_2z_lst.append(cos_lambda_2z)
            cos_lambda_max_lst.append(max(np.abs(cos_lambda_1z),np.abs(cos_lambda_2z)))
            cos_lambda_min_lst.append(min(np.abs(cos_lambda_1z),np.abs(cos_lambda_2z)))

            # vec_1, vec_2 = np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:]), np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])
            # a0 = vec_1 / np.linalg.norm(vec_1)
            # b_perp = vec_2 - np.dot(vec_2, a0)*a0  # 施密特正交化
            # b0 = b_perp / np.linalg.norm(b_perp)
            # n = np.cross(vec_1, vec_2)
            # center = np.array([-32,-32,-32])+0.25*np.array([i_lst[0][i],j_lst[0][i],k_lst[0][i]])
            # R = (center + U[..., None]*a0 + V[..., None]*b0+32)%64-32 
            # Bx_proj = interp_Bx(R)
            # By_proj = interp_By(R)
            # Bz_proj = interp_Bz(R)
            # B = np.stack([Bx_proj, By_proj, Bz_proj], axis=-1)
            # B_plane = B - np.dot(B, n)[..., None]*n / np.linalg.norm(n)**2
            # B_u = np.dot(B_plane, a0)
            # B_v = np.dot(B_plane, b0)
            # u_near = np.linspace(-2, 2, 50)
            # v_naer = np.linspace(-2, 2, 50)
            # # print(interp_Bx(((center + u_near[0]*a0)+32)%64-32).shape)
            # for j in range(len(u_near)):
            #     bx_temp_1, by_temp_1, bz_temp_1 = interp_Bx(((center + u_near[0]*a0)+32)%64-32), interp_By(((center + u_near[0]*a0)+32)%64-32),interp_Bz(((center + u_near[0]*a0)+32)%64-32)
            #     b1_temp = np.dot([bx_temp_1, by_temp_1, bz_temp_1],vec_2)
            #     bx_temp_2, by_temp_2, bz_temp_2 = interp_Bx(((center + u_near[0]*a0)+32)%64-32), interp_By(((center + u_near[0]*a0)+32)%64-32),interp_Bz(((center + u_near[0]*a0)+32)%64-32)
            #     b2_temp = np.dot([bx_temp_2, by_temp_2, bz_temp_2],vec_1)
            #     if j==0:
            #         b1_temp_0 = b1_temp
            #         b2_temp_0 = b2_temp
            #     if np.sign(b1_temp)*np.sign(b1_temp_0)<0 and np.sign(b2_temp)*np.sign(b2_temp_0)<0:
            #         i_right += 1
            #         eig_vec_b1_lst.append(vec_1)
            #         eig_vec_b2_lst.append(vec_2)


            
            a_1,b_1,c_1 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])),delta_l)
            a_2,b_2,c_2 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(-np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])),delta_l)
            a_3,b_3,c_3 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])),delta_l)
            a_4,b_4,c_4 = find_nearest_lattice((int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])),list(-np.real(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])),delta_l)
            a_1,b_1,c_1 = a_1%256,b_1%256,c_1%256
            a_2,b_2,c_2 = a_2%256,b_2%256,c_2%256
            a_3,b_3,c_3 = a_3%256,b_3%256,c_3%256
            a_4,b_4,c_4 = a_4%256,b_4%256,c_4%256
            B1 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:],[Bx[a_1,b_1,c_1],By[a_1,b_1,c_1],Bz_total[a_1,b_1,c_1]])
            B2 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:],[Bx[a_2,b_2,c_2],By[a_2,b_2,c_2],Bz_total[a_2,b_2,c_2]])
            B3 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:],[Bx[a_3,b_3,c_3],By[a_3,b_3,c_3],Bz_total[a_3,b_3,c_3]])
            B4 = np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:],[Bx[a_4,b_4,c_4],By[a_4,b_4,c_4],Bz_total[a_4,b_4,c_4]])
            # print(B1,B2,B3,B4)
            if (B1*B2<0)&(B3*B4<0):
                i_right += 1
                eig_vec_b1_lst.append(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2,:])
                eig_vec_b2_lst.append(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1,:])
                x_point_lst.append([i_lst[0][i],j_lst[0][i],k_lst[0][i]])
                lambda_1_lst.append(np.real(lambda_mat_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],2]))
                lambda_2_lst.append(np.real(lambda_mat_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],1]))
                print(i_right)
            # if (i%100==0):
                print("origin",(int(i_lst[0][i]),int(j_lst[0][i]),int(k_lst[0][i])))
                print("new",(a_1,b_1,c_1),r"$\vec{B}\cdot\hat{l}$=",np.dot(eig_vec_B_sorted[i_lst[0][i],j_lst[0][i],k_lst[0][i],0,:],[Bx[a_1,b_1,c_1],By[a_1,b_1,c_1],Bz[a_1,b_1,c_1]]))
    target_point_lst = []
    for idx in range(t_idx_start, t_idx_end):
        t_idx = idx-t_idx_start
        from matplotlib.patches import Rectangle
        import matplotlib.gridspec as gridspec
        from matplotlib.image import imread
        i_plot = 239-1
        a = np.array([1, 0, 0])  # 示例向量a
        b = np.array([0, 1, 0])  # 示例向量b（此处为xy平面，可替换为任意不平行向量）
        a = np.real(eig_vec_b1_lst[i_plot])
        b = np.real(eig_vec_b2_lst[i_plot])
        # a_prime = eigenvecs_sorted_norm[0,2,:]
        # b_prime = eigenvecs_sorted_norm[0,1,:]
        # print(a,b)
        # print(-32+0.25*np.array(x_point_lst[:][2]))
        # print(np.where((-32+0.25*np.array(x_point_lst)[:,2]>-5)&(-32+0.25*np.array(x_point_lst)[:,2]<5)))
        center = np.array([-32,-32,-32])+0.25*np.array(x_point_lst[i_plot])
        
        
        #print(uix[x_point_lst[i_plot][0],x_point_lst[i_plot][1],x_point_lst[i_plot][2]],uiy[x_point_lst[i_plot][0],x_point_lst[i_plot][1],x_point_lst[i_plot][2]],uiz[x_point_lst[i_plot][0],x_point_lst[i_plot][1], x_point_lst[i_plot][2]])

        # print(center)
        # print(np.dot([Bx[i_lst[0][i],j_lst[0][i],k_lst[0][i]],By[i_lst[0][i],j_lst[0][i],k_lst[0][i]],Bz_total[i_lst[0][i],j_lst[0][i],k_lst[0][i]]],a))
        a0 = a / np.linalg.norm(a)
        b_perp = b - np.dot(b, a0)*a0  # 施密特正交化
        b0 = b_perp / np.linalg.norm(b_perp)
        n = np.cross(a, b)
        from scipy.interpolate import RegularGridInterpolator

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        # 3. 构建目标平面（xy平面）的二维网格
        u = np.linspace(-5, 5, 101)  # a0方向坐标

        v = np.linspace(-5, 5, 100)  # b0方向坐标
        u_near = np.linspace(-1.5, 1.5, 20)  # a0方向坐标
        v_near = np.linspace(-1.5, 1.5, 20)  # b0方向坐标
        U, V = np.meshgrid(u, v)
        U_near, v_near = np.meshgrid(u_near, v_near)
        # 将二维网格点映射到三维空间（xy平面，z=0）
        R = (center + U[..., None]*a0 + V[..., None]*b0+32)%64-32  # R.shape = (100,100,3)
        R_near = (center + U_near[..., None]*a0 + v_near[..., None]*b0+32)%64-32  # R.shape = (100,100,3)
        # print(center,R)
        # 4. 插值获取平面上的三维磁场并投影
        # 构建三维插值器
        Bx, By, Bz = Q_3d['bx'][:,:,:,t_idx],Q_3d['by'][:,:,:,t_idx],Q_3d['bz'][:,:,:,t_idx]
        ni = Q_3d['ni'][:,:,:,t_idx]
        uix, uiy, uiz = Q_3d['uix'][:,:,:,t_idx],Q_3d['uiy'][:,:,:,t_idx],Q_3d['uiz'][:,:,:,t_idx]
        Jx, Jy, Jz = calculate_curl(Bx, By, Bz, x, y, z)
        uex, uey, uez = uix-Jx/ni, uiy-Jy/ni, uiz-Jz/ni
        interp_uex = RegularGridInterpolator((x, y, z), uex)
        interp_uey = RegularGridInterpolator((x, y, z), uey)
        interp_uez = RegularGridInterpolator((x, y, z), uez)
        # print("test:",interp_uex(np.array([0,0,0])))
        if idx==t_idx_start:
            target_point_lst.append(center)
        else:
            uex_tmp = interp_uex(np.array(target_point_lst[-1]))
            uey_tmp = interp_uey(np.array(target_point_lst[-1]))
            uez_tmp = interp_uez(np.array(target_point_lst[-1]))
            print(f"idx={idx-t_idx_start}",interp_uex(np.array(target_point_lst[-1])))
            print(f"idx={idx-t_idx_start}",np.array(target_point_lst[-1]).shape,np.array([uex_tmp,uey_tmp,uez_tmp]).squeeze().shape)
            target_point_lst.append(np.array(target_point_lst[-1])+np.array([uex_tmp,uey_tmp,uez_tmp]).squeeze()*dt)

        interp_Bx = RegularGridInterpolator((x, y, z), Bx)
        interp_By = RegularGridInterpolator((x, y, z), By)
        interp_Bz = RegularGridInterpolator((x, y, z), Bz_total)
        interp_uix = RegularGridInterpolator((x, y, z), uix)
        interp_uiy = RegularGridInterpolator((x, y, z), uiy)
        interp_uiz = RegularGridInterpolator((x, y, z), uiz)
        # interp_Ti = RegularGridInterpolator((x, y, z), Ti)
        # interp_J_dot_e_prime = RegularGridInterpolator((x, y, z), J_dot_e_prime)
        # interp_J_dot_e_prime_parallel = RegularGridInterpolator((x, y, z), J_dot_e_prime_parallel)
        # interp_J_dot_e = RegularGridInterpolator((x, y, z), J_dot_e)
        # 插值得到平面上的B向量
        Bx_proj = interp_Bx(R)
        By_proj = interp_By(R)
        Bz_proj = interp_Bz(R)
        uix_proj = interp_uix(R)
        uiy_proj = interp_uiy(R)
        uiz_proj = interp_uiz(R)
        # Ti_proj = interp_Ti(R)
        # J_dot_e_prime_proj = interp_J_dot_e_prime(R)
        # J_dot_e_prime_parallel_proj = interp_J_dot_e_prime_parallel(R)
        # J_dot_e_proj = interp_J_dot_e(R)
        uix_proj_near = interp_uix(R_near)
        uiy_proj_near = interp_uiy(R_near)
        uiz_proj_near = interp_uiz(R_near)

        B = np.stack([Bx_proj, By_proj, Bz_proj], axis=-1)
        ui = np.stack([uix_proj, uiy_proj, uiz_proj], axis=-1)
        ui_near = np.stack([uix_proj_near, uiy_proj_near, uiz_proj_near], axis=-1)
        # 剔除法向分量（n=(0,0,1)，此处即剔除Bz）
        B_plane = B - np.dot(B, n)[..., None]*n / np.linalg.norm(n)**2
        ui_plane = ui - np.dot(ui, n)[..., None]*n / np.linalg.norm(n)**2
        ui_plane_near = ui_near - np.dot(ui_near, n)[..., None]*n / np.linalg.norm(n)**2
        # 转化为局部二维分量（B_u = B·a0，B_v = B·b0）
        B_u = np.dot(B_plane, a0)
        B_v = np.dot(B_plane, b0)
        u_u = np.dot(ui_plane, a0)
        u_v = np.dot(ui_plane, b0)
        u_u_near = np.dot(ui_plane_near, a0)
        u_v_near = np.dot(ui_plane_near, b0)
        u_u_prime = u_u-u_u_near.mean()
        u_v_prime = u_v-u_v_near.mean()
        rect = Rectangle(
            (-2, -1.5),  # 左下角坐标
            width=3,  # 沿x轴长度
            height=3,  # 沿y轴长度
            edgecolor='red',  # 边框颜色
            facecolor='none',  # 填充颜色（none为空心）
            linewidth=2,  # 边框线宽
            linestyle='-'  # 边框样式（虚线）
        )
        # 5. 绘制磁力线投影
        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(1, 2, figure=fig, height_ratios=[1], hspace=0.45)
        # ax = fig.add_subplot(gs[0, :])
        # ax.imshow(imread("screenshot_90.png"),aspect='auto')
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax = fig.add_subplot(gs[0])
        ax.streamplot(U, V, B_u, B_v, density=1, color='k', linewidth=0.8, broken_streamlines=False)
        # ax.scatter((np.array(target_point_lst[-1]).squeeze()-center)[0],)
        # print(u_u_prime.shape,u_v_prime.shape)
        # pclr = ax.pcolormesh(U,V,J_dot_e_prime_parallel_proj,cmap='bwr',shading='auto', vmin=-0.02, vmax=0.02)
        # cbar=plt.colorbar(pclr,ax=ax)
        # cbar.set_label(r'$(J\cdot E^\prime)_{\parallel}$', fontsize=20)
        ax.arrow(-1.2,0.8,10*(u_u_prime[58,38]), 10*u_v_prime[58,38], head_width=0.2,color='b')
        # ax.arrow(0.3,0,10*(u_u_prime[50,53]), 10*u_v_prime[50,53], head_width=0.2,color='b')
        ax.arrow(0.2,1,10*(u_u_prime[60,52]), 10*u_v_prime[60,52], head_width=0.2, color='r')
        ax.arrow(0.5,-0.8,10*(u_u_prime[42,55]), 10*u_v_prime[42,55], head_width=0.2, color='b')
        ax.arrow(-1,-0.8,10*(u_u_prime[42,40]), 10*u_v_prime[40,42], head_width=0.2, color='r')
        # ax.arrow(1.5,-0.3,10*(u_u_prime[50,65]), 10*u_v_prime[50,65], head_width=0.2, color='r')
        # # ax.streamplot(U, V, u_u-u_u_near.mean(), u_v-u_v_near.mean(), density=2, color='b', linewidth=0.8)
        # # plt.streamplot(U, V, u_u-u_u_near.mean(), u_v-u_v_near.mean(), density=2, color='b', linewidth=0.8)
        ax.add_patch(rect)
        ax.set_xlabel(f'$e_1$')
        ax.set_ylabel(f'$e_2$')
        ax.set_title(f'magnetic field line projection\n on the principal eigenvector plane', fontsize=25)
        ax.axis('equal')
        ax = fig.add_subplot(gs[1])
        rect = Rectangle(
            (-2, -1.5),  # 左下角坐标
            width=3,  # 沿x轴长度
            height=3,  # 沿y轴长度
            edgecolor='red',  # 边框颜色
            facecolor='none',  # 填充颜色（none为空心）
            linewidth=2,  # 边框线宽
            linestyle='-'  # 边框样式（虚线）
        )
        # ax.streamplot(U, V, B_u, B_v, density=2, color='k', linewidth=0.8)
        ax.streamplot(U, V, u_u-u_u_near.mean(), u_v-u_v_near.mean(), density=3, color='k', linewidth=0.8)
        
        # pclr = ax.pcolormesh(U,V,J_dot_e_prime_parallel_proj,cmap='bwr',shading='auto', vmin=-0.02, vmax=0.02)
        # cbar=plt.colorbar(pclr,ax=ax)
        # cbar.set_label(r'$(J\cdot E^\prime)_{\parallel}$', fontsize=20)
        ax.arrow(-1.2,0.8,10*(u_u_prime[58,38]), 10*u_v_prime[58,38], head_width=0.2,color='b')
        # ax.arrow(0.3,0,10*(u_u_prime[50,53]), 10*u_v_prime[50,53], head_width=0.2,color='b')
        ax.arrow(0.2,1,10*(u_u_prime[60,52]), 10*u_v_prime[60,52], head_width=0.2, color='r')
        ax.arrow(0.5,-0.8,10*(u_u_prime[42,55]), 10*u_v_prime[42,55], head_width=0.2, color='b')
        ax.arrow(-1,-0.8,10*(u_u_prime[42,40]), 10*u_v_prime[40,42], head_width=0.2, color='r')
        ax.add_patch(rect)
        ax.set_xlabel(f'$e_1$')
        ax.set_ylabel(f'$e_2$')
        ax.set_title(f'velocity field line projection\n on the principal eigenvector plane', fontsize=25)
        ax.axis('equal')
        plt.savefig(f'./figures/fig_{i_plot}_epoch_{idx}.png')
        plt.close(fig=fig)
    for i, item in enumerate(target_point_lst):
        print(f"第{i}个元素: {item}, 长度: {len(item)}")
    plt.figure(figsize=(10, 7))
    plt.plot(np.array(target_point_lst)[:,0],label='x')
    plt.plot(np.array(target_point_lst)[:,1],label='y')
    plt.plot(np.array(target_point_lst)[:,2],label='z')
    plt.savefig("./figures/trajectory.png")

                # print(a,b,c)
    # print(target_point_lst)
    
        # print(a)
        # print(sort_indices.shape)