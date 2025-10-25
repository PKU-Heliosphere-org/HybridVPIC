import sys
sys.path.append("D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection")
from plot_simulation_results_new import Species as Species_tag
from plot_simulation_results import Species
from read_field_data import load_data_at_certain_t, loadinfo, load_data
import matplotlib.pyplot as plt
import numpy as np
from double_maxwellian_fit import double_maxwellian
from scipy.optimize import curve_fit
import matplotlib as mpl
from tracking_data_read import Tracer
import os
from scipy import ndimage
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from loky import get_reusable_executor
# # 设置使用 LaTeX 渲染文本
# mpl.rcParams['text.usetex'] = True
# # 设置 LaTeX 预编译指令（可选，根据需要添加宏包等）
# # mpl.rcParams['text.latex.preamble'] = [
# #     r'$\usepackage{amsmath}$'  # 例如，如果使用了amsmath 宏包相关环境
# # ]
#%%
topo_x, topo_y, topo_z = 32, 1, 8


def process_rank(rank, p_c, icell_lst, icell_to_grid, rank_to_position, nx, nz, param_name='ux'):
    """处理单个rank的计算任务

    Args:
        param_name: 要计算方差的p_c参数名称，默认为'ux'
    """
    rank_result = np.zeros((nx, nz))
    rank_mask = p_c.rank == rank  # 预筛选当前rank的数据

    # 创建子进度条（显示当前rank的处理进度）
    with tqdm(icell_lst, desc=f"Rank {rank}", leave=False) as pbar:
        for i in pbar:
            ix, _, iz = icell_to_grid[i]
            condition = (p_c.icell == i) & rank_mask

            if np.any(condition):
                i_pos, j_pos = rank_to_position[rank]
                # 通过getattr动态获取属性并计算方差
                param_values = getattr(p_c, param_name, np.zeros(0))
                rank_result[i_pos + int(ix) - 1, j_pos + int(iz) - 1] = param_values[condition].var()
                print(f"{i_pos + int(ix) - 1}, {j_pos + int(iz) - 1}, {rank}\n")

    return rank_result


def optimized_ux_mean_calculation(p_c, topo_x, topo_y, topo_z, param_name='ux', show_progress=True):
    """优化的参数方差计算函数，支持进度条显示和参数选择

    Args:
        param_name: 要计算方差的p_c参数名称，默认为'ux'
    """
    # 预计算网格参数
    nx, ny, nz = p_c.nx, p_c.ny, p_c.nz
    _nx, _ny, _nz = nx // topo_x + 2, ny // topo_y + 2, nz // topo_z + 2

    # 预计算icell到网格索引的映射
    icell_lst = np.unique(p_c.icell)
    icell_to_grid = {i: (i % _nx, (i // _nx) % _ny, i // (_nx * _ny)) for i in icell_lst}

    # 预计算rank到输出位置的映射
    rank_to_position = {(rank): ((rank % 8) * 64, (rank // 8) * 128) for rank in range(16)}

    # 初始化结果数组
    result_array = np.zeros((nx, nz))

    # 准备并行处理
    worker_func = partial(
        process_rank,
        p_c=p_c,
        icell_lst=icell_lst,
        icell_to_grid=icell_to_grid,
        rank_to_position=rank_to_position,
        nx=nx,
        nz=nz,
        param_name=param_name  # 传递参数名称
    )

    # 显示总进度条
    if show_progress:
        total_work = 16  # 16个rank

        # 创建主进度条（显示总体进度）
        with tqdm(total=total_work, desc="Total Progress", unit="rank") as pbar:
            with ThreadPoolExecutor() as executor:
                # 使用tqdm包装结果迭代器以更新主进度条
                results = list(tqdm(
                    executor.map(worker_func, range(16)),
                    total=total_work,
                    desc="Processing ranks",
                    leave=False
                ))
                pbar.update(total_work)  # 更新总进度
    else:
        # 不显示进度条时的处理
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(worker_func, range(16)))

    # 合并结果
    for rank_result in results:
        result_array += rank_result

    return result_array

def calculate_gradient_drift_velocity(Bx, By, Bz, B_mag, W_perp, dx, dz):
    """
    计算梯度漂移速度

    参数:
    q -- 粒子电荷
    Bx -- 二维数组，x 方向的磁场分量，形状为 (nx, nz)
    By -- 二维数组，y 方向的磁场分量，形状为 (nx, nz)
    Bz -- 二维数组，z 方向的磁场分量，形状为 (nx, nz)
    B_mag -- 二维数组，磁场强度，形状为 (nx, nz)
    W_perp -- 二维数组，粒子垂直于磁场方向的动能，形状为 (nx, nz)
    dx -- x 方向的空间步长
    dz -- z 方向的空间步长

    返回:
    v_grad -- 梯度漂移速度，形状为 (nx, nz, 3) 的三维数组
    """
    # 计算磁场强度的梯度
    # 由于磁场在 y 方向均匀，所以 dB/dy = 0
    q = 1
    dB_dx = (np.roll(B_mag, -1, axis=0) - np.roll(B_mag, 1, axis=0)) / (2 * dx)
    dB_dz = (np.roll(B_mag, -1, axis=1) - np.roll(B_mag, 1, axis=1)) / (2 * dz)
    # 扩展维度以方便后续计算
    dB_dx = np.expand_dims(dB_dx, axis=-1)
    dB_dz = np.expand_dims(dB_dz, axis=-1)
    dB = np.zeros((Bx.shape[0], Bx.shape[1], 3))
    dB[:, :, 0] = dB_dx.squeeze()
    dB[:, :, 2] = dB_dz.squeeze()

    # 计算叉乘 B × ∇B
    cross_product = np.zeros((Bx.shape[0], Bx.shape[1], 3))
    cross_product[:, :, 0] = By * dB[:, :, 2] - Bz * dB[:, :, 1]
    cross_product[:, :, 1] = Bz * dB[:, :, 0] - Bx * dB[:, :, 2]
    cross_product[:, :, 2] = Bx * dB[:, :, 1] - By * dB[:, :, 0]

    # 计算分母 q * B_mag^3
    denominator = q * B_mag ** 3
    denominator = np.expand_dims(denominator, axis=-1)

    # 计算梯度漂移速度
    v_grad = (W_perp / denominator) * cross_product
    return v_grad


def calculate_vc(v_parallel, Bx, By, Bz, dx, dz):
    """
    计算公式中的 vC

    参数:
    v_parallel -- 平行速度
    Bx -- 二维数组，x 方向的磁场分量，形状为 (nx, nz)
    By -- 二维数组，y 方向的磁场分量，形状为 (nx, nz)
    Bz -- 二维数组，z 方向的磁场分量，形状为 (nx, nz)
    q -- 粒子电荷
    dx -- x 方向的空间步长
    dz -- z 方向的空间步长

    返回:
    vC -- 计算得到的 vC 值，形状为 (nx, nz, 3)
    """
    q = 1
    nx, nz = Bx.shape
    # 计算 B 在 x 方向的梯度 (B·∇)B_x
    dBx_dx = (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / (2 * dx)
    dBx_dz = (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / (2 * dz)
    B_dot_grad_Bx = Bx * dBx_dx + Bz * dBx_dz

    # 计算 B 在 z 方向的梯度 (B·∇)B_z
    dBz_dx = (np.roll(Bz, -1, axis=0) - np.roll(Bz, 1, axis=0)) / (2 * dx)
    dBz_dz = (np.roll(Bz, -1, axis=1) - np.roll(Bz, 1, axis=1)) / (2 * dz)
    B_dot_grad_Bz = Bx * dBz_dx + Bz * dBz_dz

    # 构建 (B·∇)B 向量
    B_dot_grad_B = np.zeros((nx, nz, 3))
    B_dot_grad_B[:, :, 0] = B_dot_grad_Bx
    B_dot_grad_B[:, :, 2] = B_dot_grad_Bz

    # 计算叉乘 [(B·∇)B]×B
    cross_product = np.zeros((nx, nz, 3))
    cross_product[:, :, 0] = -By * B_dot_grad_B[:, :, 2] + Bz * B_dot_grad_B[:, :, 1]
    cross_product[:, :, 1] = -Bz * B_dot_grad_B[:, :, 0] + Bx * B_dot_grad_B[:, :, 2]
    cross_product[:, :, 2] = -Bx * B_dot_grad_B[:, :, 1] + By * B_dot_grad_B[:, :, 0]

    # 计算分母 B 的模的四次方
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    B_magnitude_fourth = B_magnitude ** 4
    # 扩展 B_magnitude_fourth 的维度，使其能与 cross_product 广播
    B_magnitude_fourth = np.expand_dims(B_magnitude_fourth, axis=-1)
    B_magnitude_fourth = np.repeat(B_magnitude_fourth, 3, axis=-1)
    # 计算 vC
    vC = - (v_parallel * cross_product) / (q * B_magnitude_fourth)

    return vC
def calculate_ux_mean_parallel(x, z, ux_c, nx, nz):
    # nx = p.nx
    # nz = p.nz

    def calculate_single_element(i, j):
        condition_ij = region(x, z, i, i + 1, -nz / 2 + j, -nz / 2 + j + 1)
        return np.mean(ux_c[condition_ij])

    results = Parallel(n_jobs=-1)(delayed(calculate_single_element)(i, j) for i in range(nx) for j in range(nz))
    ux_mean = np.array(results).reshape((nx, nz))
    return ux_mean
def smooth_matrix(matrix, kernel_size=3):
    # 创建一个卷积核，其元素值都为1，用于计算周围元素的和
    kernel = np.ones((kernel_size, kernel_size))
    # 计算卷积核元素的总和，用于后续求平均值
    kernel_sum = np.sum(kernel)
    # 使用scipy的ndimage.convolve函数进行卷积操作
    smoothed = ndimage.convolve(matrix, kernel, mode='constant', cval=0.0)
    # 将卷积结果除以卷积核元素总和，得到平均值
    return smoothed / kernel_sum
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


def mat_shift(M):
    n1, n2 = M.shape[0], M.shape[1]
    M_shift = np.zeros_like(M)
    M_shift[0:n1//2, :] = M[n1//2:, :]
    M_shift[n1//2:, :] = M[0:n1//2, :]
    return M_shift


def process_array(arr, a):
    """
    对数组或矩阵中的元素进行处理：
    - 若元素 < a，则元素 += a
    - 若元素 > a，则元素 -= a
    - 若元素 == a，则保持不变

    参数:
        arr: 输入的数组或矩阵（可是list或numpy数组）
        a: 阈值（数值型）

    返回:
        numpy数组: 处理后的数组/矩阵
    """
    # 将输入转换为numpy数组，方便元素级操作
    np_arr = np.asarray(arr)

    # 使用numpy的where函数进行条件判断和操作
    # 先处理元素 < a 的情况，再处理元素 > a 的情况，其余（==a）保持不变
    result = np.where(
        np_arr < a,  # 条件1：元素小于a
        np_arr + a,  # 满足条件1时的操作
        np.where(
            np_arr > a,  # 条件2：元素大于a
            np_arr - a,  # 满足条件2时的操作
            np_arr - a  # 既不满足条件1也不满足条件2（即等于a）时保持不变
        )
    )

    return result
#%%


def region(x_data, z_data, x_left, x_right, z_bottom, z_top):
    if x_data is not None and z_data is not None:
        return (x_data >= x_left) & (x_data < x_right) & (z_data >= z_bottom) & (z_data < z_top)
    else:
        return np.zeros_like(x_data, dtype=bool)
#%%


def calculate_2d_skewness_kurtosis(data, alpha=0.001, normalize=True):
    # 去除全零的行
    data = data[~np.all(data == 0, axis=1)]
    # 去除全零的列
    data = data[:, ~np.all(data == 0, axis=0)]

    if normalize:
        # 归一化数据
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    n = data.shape[0]
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    S = np.cov(data, rowvar=False)

    # 添加正则化项
    S_reg = S + alpha * np.eye(S.shape[0])
    S_inv = np.linalg.inv(S_reg)

    b12 = 0
    b22 = 0
    for i in range(n):
        zi = centered_data[i].reshape(-1, 1)
        b22 += (zi.T @ S_inv @ zi) ** 2
        for j in range(n):
            zj = centered_data[j].reshape(-1, 1)
            b12 += (zi.T @ S_inv @ zj) ** 3

    b12 /= n ** 2
    b22 /= n

    return b12, b22

#%%
if __name__ == "__main__":
    #%%
    x = np.linspace(-1, 1, 8)
    y = np.linspace(-1, 1, 8)
    X, Y = np.meshgrid(x, y)


    # 自定义函数，根据坐标计算类似图中的数值分布，这里只是示例模拟，可根据实际需求调整
    def calculate_value(x_coord, y_coord):
        # 简单模拟色彩映射相关的计算，你可根据原图规律精细调整
        value = 2.5 * (1 - (np.abs(x_coord) + np.abs(y_coord)) / 2)
        return value


    rows, cols = 8, 8
    result_array = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # 计算从左上角到右下角的递减权重
            weight = 1.0 + 30*(-i + j) / (rows + cols - 2)
            result_array[i, j] = weight

    # 可视化数组（可选，方便查看分布是否符合预期）
    plt.pcolormesh(np.linspace(-1.5, 1, 8), np.linspace(-1.5, 1, 8), (result_array+np.random.uniform(0, 1, size=(8, 8))).T, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label(label=r'$y_{drift}[d_i]$', fontsize=15)
    plt.xlabel('Vx, 0/VA')
    plt.ylabel('Vy, 0/VA')
    plt.show()
    #%%
    counts_mat = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000)
    skewness, kurtosis = calculate_2d_skewness_kurtosis(counts_mat)
    print(f"二维偏度: {skewness}")
    print(f"二维峰度: {kurtosis}")
    # %%
    """
    READ TRACER DATA
    """
    num_particle_traj = 2000
    ratio_emax = 1
    species_name_lst = ["ion_c", "ion_b"]
    species_fullname_lst = ["core", "beam"]
    sample_lst = [1, 1]
    ntraj_lst = [30000, 30000]
    # %%
    fdir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/trace_data/"
    # 用于存储文件名的字典
    file_names = {}
    # 用于存储 Tracer 对象的字典
    tracers = {}
    # %%
    # 遍历 index 列表
    for index in [25]:
        # 遍历物种名称列表
        for j in range(len(species_name_lst)):
            # 生成文件名
            fname_key = f"fname{index}_{species_name_lst[j]}"
            file_names[
                fname_key] = f"{species_name_lst[j]}_tracer/{species_name_lst[j]}s_ntraj{ntraj_lst[j]}_{ratio_emax}emax_{index}.h5p"
            # 生成 Tracer 对象
            tracer_key = f"{species_name_lst[j]}_tracer_{index}"
            tracers[tracer_key] = Tracer(
                species_name_lst[j],
                species_fullname_lst[j],
                fdir,
                file_names[fname_key],
                sample_step=sample_lst[j]
            )
    # %%
    dt = 0.5
    resolution = 0.5
    x_left, x_right, z_bottom, z_top = 375, 425, -1, 1
    x_arr = np.arange(30, 80 + 0.5, 0.5)
    z_arr = np.arange(-6, 6 + resolution, resolution)
    ux_0_bin = np.linspace(-1, 1.5, 9)
    y_drift_bin = np.zeros(len(ux_0_bin) - 1)
    delta_ux_bin = np.zeros(len(ux_0_bin) - 1)
    tracer_b = tracers["ion_b_tracer_25"]
    nframe = tracer_b.data["nframe"]
    max_step = nframe - 1
    x_b, z_b, y_b = tracer_b.data["x"], tracer_b.data["z"], dt * np.cumsum(tracer_b.data["uy"], axis=1)
    ux_b, uz_b, uy_b = tracer_b.data["ux"], tracer_b.data["uz"], tracer_b.data["uy"]
    ex_b, ez_b, ey_b = tracer_b.data["ex"], tracer_b.data["ez"], tracer_b.data["ey"]
    bx_b, bz_b, by_b = tracer_b.data["bx"], tracer_b.data["bz"], tracer_b.data["by"]
    Fy_b = ey_b + uz_b * bx_b - ux_b * bz_b
    Fz_b = ez_b + ux_b * by_b - uy_b * bx_b
    E_b = tracer_b.data["E"]
    tracer_c = tracers["ion_c_tracer_25"]

    z_time_mat_c = np.zeros((len(z_arr) - 1, tracer_c.data["nframe"]))
    z_time_mat_b = np.zeros((len(z_arr) - 1, tracer_c.data["nframe"]))
    x_c, z_c, y_c = tracer_c.data["x"], tracer_c.data["z"], dt * np.cumsum(tracer_c.data["uy"], axis=1)

    ux_c, uz_c, uy_c = tracer_c.data["ux"], tracer_c.data["uz"], tracer_c.data["uy"]
    print(x_c[:, 0].max())
    ex_c, ez_c, ey_c = tracer_c.data["ex"], tracer_c.data["ez"], tracer_c.data["ey"]
    bx_c, bz_c, by_c = tracer_c.data["bx"], tracer_c.data["bz"], tracer_c.data["by"]
    Fy_c = ey_c + uz_c * bx_c - ux_c * bz_c
    Fz_c = ez_c + ux_c * by_c - uy_c * bx_c
    E_c = tracer_c.data["E"]
    x_total, y_total, z_total = np.concatenate((x_c, x_b), axis=0), np.concatenate((y_c, y_b), axis=0), np.concatenate(
        (z_c, z_b), axis=0)
    ux_total, uy_total, uz_total = np.concatenate((ux_c, ux_b), axis=0), np.concatenate((uy_c, uy_b),
                                                                                        axis=0), np.concatenate(
        (uz_c, uz_b), axis=0)
    ex_total, ey_total, ez_total = np.concatenate((ex_c, ex_b), axis=0), np.concatenate((ey_c, ey_b),
                                                                                        axis=0), np.concatenate(
        (ez_c, ez_b), axis=0)
    bx_total, by_total, bz_total = np.concatenate((bx_c, bx_b), axis=0), np.concatenate((by_c, by_b),
                                                                                        axis=0), np.concatenate(
        (bz_c, bz_b), axis=0)
    # plt.scatter(x, tracer.data["ux"][:, 0], s=1)
    # condition_b = region(x_b[:, 0], z_b[:, 0], 0, 250, -5, 5)
    # plt.scatter(ux_b[condition_b, 0], uy_b[condition_b, 0], s=5, c="b")
    # plt.scatter(ux_b[condition_b, 200], uy_b[condition_b, 200], s=5, c="g", marker="x")
    condition_c_position = region(x_c[:, 100], z_c[:, 100], x_left, x_right, z_bottom, z_top)
    condition_b_position = region(x_b[:, 100], z_b[:, 100], x_left, x_right, z_bottom, z_top)
    condition_t_position = region(x_total[:, 0], z_total[:, 0], x_left, x_right, z_bottom, z_top)
    condition_c_1 = ux_c[:, 0] < -0.4  # region(x_c[:, 0], z_c[:, 0], 0, 250, -5, 5)
    condition_c_2 = ux_c[:, 0] > 0.3
    delta_Ec = E_c[:, max_step] - E_c[:, 0]
    error_y = np.zeros(len(y_drift_bin))
    error_ux = np.zeros(len(y_drift_bin))
    for i in range(len(y_drift_bin)):
        condition_tmp = (ux_total[:, 0] >= ux_0_bin[i]) & (ux_total[:, 0] < ux_0_bin[i + 1])
        y_drift_bin[i] = dt * np.sum(uy_total[condition_tmp * condition_t_position, :], axis=1).mean()
        error_y[i] = np.sum(uy_total[condition_tmp * condition_t_position, :] * dt, axis=1).std()
        delta_ux_bin[i] = (ux_total[condition_tmp * condition_t_position, max_step] - ux_total[
            condition_tmp * condition_t_position, 0]).mean()
        error_ux[i] = (ux_total[condition_tmp * condition_t_position, max_step] - ux_total[
            condition_tmp * condition_t_position, 0]).std()
    for iframe in range(tracer_c.data["nframe"]):
        n, bins = np.histogram(tracer_c.data["z"][condition_c_position, iframe], bins=z_arr)
        z_time_mat_c[:, iframe] = n / n.sum()
        n, bins = np.histogram(tracer_b.data["z"][condition_b_position, iframe], bins=z_arr)
        z_time_mat_b[:, iframe] = n / n.sum()
        # print(iframe, n.sum())
    delta_vc_1 = dt * np.sum((ey_c * uy_c)[condition_c_1 * condition_c_position, :],
                             axis=1)  # E_c[condition_c_1*condition_c_position, 400]-1*E_c[condition_c_1*condition_c_position, 0]
    delta_vc_2 = dt * np.sum((ey_c * uy_c)[condition_c_2 * condition_c_position, :],
                             axis=1)  # E_c[condition_c_2*condition_c_position, 400]-1*E_c[condition_c_2*condition_c_position, 0]
    delta_vb = dt * np.sum((ey_b * uy_b)[condition_b_position, :],
                           axis=1)  # E_b[condition_b_position, 400]-1*E_b[condition_b_position, 0]
    index_c_1 = np.where(condition_c_1 & condition_c_position)[0]
    index_c_2 = np.where(condition_c_2 & region(x_c[:, 0], z_c[:, 0], x_left, x_right, z_bottom, z_top))[0]
    index_b = np.where(condition_b_position)[0]
    label_c_2 = f"mean={delta_vc_2.mean():.2f}\n var={delta_vc_2.var():.2f}"
    label_c_1 = f"mean={delta_vc_1.mean():.2f}\n var={delta_vc_1.var():.2f}"
    label_b = f"mean={delta_vb.mean():.2f}\n var={delta_vb.var():.2f}"
    # plt.hist(delta_vc_1, alpha=1, label=r"core($v_{x,0}<0,$"+label_c_1+")", edgecolor="b", facecolor="w", linewidth=3, density=True)
    # plt.hist(delta_vc_2, alpha=0.8, label=r"core($(v_{x,0}>0,$"+label_c_2+")", edgecolor="r", facecolor="w", linewidth=3, density=True)
    # plt.hist(delta_vb, alpha=0.8, label=r"beam,("+label_b+")", edgecolor="g", facecolor="w", linewidth=3, density=True)
    # plt.xlabel(r"$W_y$", fontsize=16)
    # plt.ylabel("PDF", fontsize=16)
    # plt.title(r"$n_b/n_c=0.8$", fontsize=15)
    # # plt.text(0.8, 60, f"mean={delta_vc_2.mean():.2f}\n var={delta_vc_2.var():.2f}", fontsize=14, color="r")
    # # plt.text(0, 60, f"mean={delta_vc_1.mean():.2f}\n var={delta_vc_1.var():.2f}", fontsize=14, color="b")
    # # plt.text(0.5, 90, f"mean={delta_vb.mean():.2f}\n var={delta_vb.var():.2f}", fontsize=14, color="g")
    # plt.legend()
    # plt.ylim([0, 100])
    iptl_1 = index_c_1[3]
    iptl_2 = index_c_2[0]
    iptl_b = index_b[0]
    c1_plot = np.array([3, 4, 2], dtype=int)
    b_plot = np.array([0], dtype=int)
    # print(x_c[iptl, :])
    # plt.scatter(x_c[condition_c_2*condition_c_position, 0], z_c[condition_c_2*condition_c_position, 0])#, c=np.linspace(0, 400, 401), cmap="jet")
    # plt.scatter(x_c[condition_c_1*condition_c_position, 0], z_c[condition_c_1*condition_c_position, 0])
    # plt.ylim([-10, 10])
    b_c = np.sqrt(bx_c ** 2 + by_c ** 2 + bz_c ** 2)
    b_b = np.sqrt(bx_b ** 2 + by_b ** 2 + bz_b ** 2)
    vy_drift_c = (bx_c * ez_c + by_c * (bx_c * ux_c + by_c * uy_c)) / (bx_c ** 2 + by_c ** 2)
    vy_drift_b = (bx_b * ez_b + by_b * (bx_b * ux_b + by_b * uy_b)) / (bx_b ** 2 + by_b ** 2)

    amp_c = np.sqrt(((ez_c[iptl_1, :] + ux_c[iptl_1, :] * by_c[iptl_1, :] - uy_c[iptl_1, :] * bx_c[iptl_1, :]) / b_c[
                                                                                                                 iptl_1,
                                                                                                                 :]) ** 2 + uz_c[
                                                                                                                            iptl_1,
                                                                                                                            :] ** 2)
    # amp_2 = np.abs(ez_c[iptl_2, :]+ux_c[iptl_2, :]*by_c[iptl_2, :]-uy_c[iptl_2, :]*bx_c[iptl_2, :])
    amp_b = np.sqrt(((ez_b[iptl_b, :] + ux_b[iptl_b, :] * by_b[iptl_b, :] - uy_b[iptl_b, :] * bx_b[iptl_b, :]) / b_b[
                                                                                                                 iptl_b,
                                                                                                                 :]) ** 2 + uz_b[
                                                                                                                            iptl_b,
                                                                                                                            :] ** 2)
    fig = plt.figure()
    # 创建一个三维坐标轴对象
    ax = fig.add_subplot(111)  # , projection='3d')
    # plt.scatter(x_c[iptl_2, :], z_c[iptl_2, :], c=ux_c[iptl_2, :]*ex_c[iptl_2, :], cmap="bwr", vmin=-0.05, vmax=0.05, edgecolors="k")
    # point = ax.scatter(x_c[iptl_1, :200], z_c[iptl_1, :200], c=Fz_c[iptl_1, :200], cmap="bwr", vmax=1, vmin=-1)
    # point = ax.scatter(x_b[iptl_b, :150], z_b[iptl_b, :150], c=Fz_b[iptl_b, :150], vmin=-1, vmax=1, cmap="bwr")
    # # plt.plot(vy_drift_b[iptl_b, :])
    for index in c1_plot:
        ax.scatter(x_c[index_c_1[index], :100], z_c[index_c_1[index], :100], c=E_c[index_c_1[index], :100]/E_c[index_c_1[index], 0], vmin=0.5, vmax=2)
    for index in b_plot:
        ax.scatter(x_b[index_b[index], :100], z_b[index_b[index], :100], c=E_b[index_b[index], :100]/E_b[index_b[index], 0], vmin=0.5, vmax=2)
    # point = ax.scatter(x_b[iptl_b, :], z_b[iptl_b, :], c=uy_b[iptl_b, :], cmap="jet", vmin=-0.5, vmax=1)
    # cbar = plt.colorbar(point, ax=ax)
    # cbar.set_label(r"$\Delta v_x$", fontsize=15)
    # cbar.set_label(r"epoch", fontsize=15)
    a = uy_c[index_c_1, 0]
    b = uy_b[index_b, 0]
    print(bx_c[iptl_1, 0] * ez_c[iptl_1, 0], (by_c * (bx_c * ux_c))[iptl_1, 0], (by_c ** 2 * uy_c)[iptl_1, 0])
    print(bx_b[iptl_b, 0] * ez_b[iptl_b, 0], (by_b * (bx_b * ux_b))[iptl_b, 0], (by_b ** 2 * uy_b)[iptl_b, 0])
    # plt.scatter(x_c[iptl_1, 0], z_c[iptl_1, 0], c="k")
    # # plt.xlim([0, 100])
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    # ax.set_zlabel("z", fontsize=15)
    # plt.arrow(80, 0, -10, 0.2, head_width=0.2, fc="k", linewidth=2, head_length=1)
    # plt.text(81, -0.1, "core", fontsize=15)
    # plt.arrow(80, -2.4, -12, -0.2, head_width=0.2, fc="k", linewidth=2, head_length=1)
    # plt.text(81, -2.38, "beam", fontsize=15)

    # print(ux_c[iptl_1, 400]-ux_c[iptl_1, 0], ux_b[iptl_b, 400]-ux_b[iptl_b, 0])
    # print(amp_1[0], amp_b[0])
    # print(np.sum(ux_c[iptl_1, :]*ex_c[iptl_1, :]), np.sum(uy_c[iptl_1, :]*ey_c[iptl_1, :]), np.sum(uz_c[iptl_1, :]*ez_c[iptl_1, :]))
    # print(np.sum(ux_b[iptl_b, :]*ex_b[iptl_b, :]), np.sum(uy_b[iptl_b, :]*ey_b[iptl_b, :]), np.sum(uz_b[iptl_b, :]*ez_b[iptl_b, :]))
    # print((ez_c[iptl_1, 0]+ux_c[iptl_1, 0]*by_c[iptl_1, 0]-uy_c[iptl_1, 0]*bx_c[iptl_1, 0]))
    # #print(ez_c[iptl_2, 0]+ux_c[iptl_2, 0]*by_c[iptl_2, 0]-uy_c[iptl_2, 0]*bx_c[iptl_2, 0])
    # print(ez_b[iptl_b, 0]+ux_b[iptl_b, 0]*by_b[iptl_b, 0]-uy_b[iptl_b, 0]*bx_b[iptl_b, 0])
    # plt.axis("equal")
    # plt.plot(z_c[iptl, :], ux_c[iptl, :], c="k")
    # plt.scatter(z_c[iptl, :], ex_c[iptl, :])
    # plt.yscale("log")
    # fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    # ax = axes[1]
    # ax.scatter(ux_c[condition_c_1*condition_c_position, 400], uy_c[condition_c_1*condition_c_position, 400], s=5, c="r", alpha=0.5)
    # ax.scatter(ux_c[condition_c_2*condition_c_position, 400], uy_c[condition_c_2*condition_c_position, 400], s=5, c="b", marker="x", alpha=0.5)
    # ax.set_xlim([-2, 1.5])
    # ax = axes[0]
    # ax.scatter(ux_c[condition_c_1*condition_c_position, 0], uy_c[condition_c_1*condition_c_position, 0], s=5, c="r", alpha=0.5)
    # ax.scatter(ux_c[condition_c_2*condition_c_position, 0], uy_c[condition_c_2*condition_c_position, 0], s=5, c="b", marker="x", alpha=0.5)
    # ax.set_xlim([-2, 1.5])
    # print(ux_c[condition_c_1*condition_c_position, 0].mean(), ux_c[condition_c_2*condition_c_position, 0].mean())
    # print(ux_c[condition_c_1*condition_c_position, 400].mean(), ux_c[condition_c_2*condition_c_position, 400].mean())
    plt.show()
    # %%
    case_index = 25
    figs_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_figs/particles_{case_index}/"
    for epoch in range(nframe):
        print(epoch)
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(x_c[::10, epoch], z_c[::10, epoch], c="r", s=3)
        plt.scatter(x_b[::10, epoch], z_b[::10, epoch], c="b", s=3)
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)
        plt.title(f"epoch={epoch}")
        plt.savefig(figs_dir + f"particles_{epoch}")
        plt.close(fig)
    # plt.show()
    # %%
    z_arr = np.linspace(-5, 5, 11)
    vx0_arr = np.linspace(-1, 2, 20)
    Wy = np.zeros((len(vx0_arr) - 1, len(z_arr) - 1))
    Wx = np.zeros((len(vx0_arr) - 1, len(z_arr) - 1))
    Wz = np.zeros((len(vx0_arr) - 1, len(z_arr) - 1))
    prob = np.zeros((len(vx0_arr) - 1, len(z_arr) - 1))
    for i in range(Wy.shape[0]):
        condition_vx0 = (ux_total[:, 0] >= vx0_arr[i]) & (ux_total[:, 0] < vx0_arr[i + 1])
        for j in range(Wy.shape[1]):
            condition_z = (z_total >= z_arr[j]) & (z_total < z_arr[j + 1]) & (x_total >= 350) & (x_total < 450)
            Wy_tmp = uy_total * ey_total * condition_z
            Wx_tmp = ux_total * ex_total * condition_z
            Wz_tmp = uz_total * ez_total * condition_z

            Wy[i, j] = np.sum(Wy_tmp[condition_vx0, :], axis=1).mean() * dt
            Wx[i, j] = np.sum(Wx_tmp[condition_vx0, :], axis=1).mean() * dt
            Wz[i, j] = np.sum(Wz_tmp[condition_vx0, :], axis=1).mean() * dt

    # %%
    print(ux_total[:, 0].min())
    # %%
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    ax = axes[0]
    pclr = ax.pcolormesh(vx0_arr[1:], z_arr[1:], Wx.T, cmap="bwr", vmin=-0.02, vmax=0.02)
    ax.set_xlabel("vx0", fontsize=15)
    ax.set_ylabel("z", fontsize=15)
    # ax.set_title("Wx", fontsize=15)
    cbar = plt.colorbar(pclr, ax=ax)
    cbar.set_label("Wx", fontsize=17)
    ax = axes[1]
    pclr = ax.pcolormesh(vx0_arr[1:], z_arr[1:], Wy.T, cmap="bwr", vmin=-0.02, vmax=0.02)
    ax.set_xlabel("vx0", fontsize=15)
    ax.set_ylabel("z", fontsize=15)
    # ax.set_title("Wy", fontsize=15)
    cbar = plt.colorbar(pclr, ax=ax)
    cbar.set_label("Wy", fontsize=17)
    ax = axes[2]
    pclr = ax.pcolormesh(vx0_arr[1:], z_arr[1:], Wz.T, cmap="bwr", vmin=-0.02, vmax=0.02)
    ax.set_xlabel("vx0", fontsize=15)
    ax.set_ylabel("z", fontsize=15)
    # ax.set_title("Wz", fontsize=15)
    cbar = plt.colorbar(pclr, ax=ax)
    cbar.set_label("Wz", fontsize=17)
    plt.subplots_adjust(hspace=0)
    plt.suptitle("350<x<450", fontsize=18)
    plt.show()
    # %%
    condition_1 = region(x_c[:, 100], z_c[:, 100], 400, 450, -2, 2)
    condition_2 = region(x_b[:, 100], z_b[:, 100], 400, 450, -4, 0)
    # print(len(x_b[condition, 0]))
    # plt.scatter(x_b[condition_1|condition_2, 0], z_b[condition_1|condition_2, 0], label="initial")
    # plt.scatter(x_b[condition_1|condition_2, 100], z_b[condition_1|condition_2, 100], label="final")
    plt.scatter(x_c[condition_1, 0], z_c[condition_1, 0])
    plt.ylim([-5, 5])
    plt.xlabel("x", fontsize=16)
    plt.ylabel("z", fontsize=16)
    plt.legend()
    plt.show()
    # %%
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax2 = ax.twinx()
    ax.errorbar((ux_0_bin[1:] + ux_0_bin[:-1]) / 2, y_drift_bin, error_y, c="r", capsize=5, fmt="-", marker=".",
                markersize=20)
    ax2.errorbar((ux_0_bin[1:] + ux_0_bin[:-1]) / 2, delta_ux_bin, error_ux, c="b", capsize=5, fmt="-", marker=".",
                 markersize=20)

    ax.set_ylabel(r"$y_{\mathrm{drift}}$", fontsize=25, c="r")
    ax.set_xlabel(r"$v_{\mathrm{x,0}}$", fontsize=25)
    ax2.set_ylabel(r"$\Delta v_{x}$", fontsize=25, c="b")
    ax.tick_params(axis='y', labelcolor="r", color="r")
    ax2.tick_params(axis='y', labelcolor="b", color="b")
    plt.show()
    # %%
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(z_arr[1:], range(tracer_c.data["nframe"]), z_time_mat_c.T, cmap="jet", vmin=0, vmax=0.3)
    plt.xlabel("z", fontsize=15)
    plt.ylabel("epoch", fontsize=15)
    plt.title("core", fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label("probability", fontsize=14)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(z_arr[1:], range(tracer_c.data["nframe"]), z_time_mat_b.T, cmap="jet", vmin=0, vmax=0.3)
    plt.xlabel("z", fontsize=15)
    plt.ylabel("epoch", fontsize=15)
    cbar = plt.colorbar()
    cbar.set_label("probability", fontsize=14)
    plt.title("beam", fontsize=16)
    plt.show()
    # %%
    dt = 0.125
    Wy_c = np.sum(uy_c[condition_c_position * condition_c_1, :] * ey_c[condition_c_position * condition_c_1, :], axis=1)
    Wy_b = np.sum(uy_b[condition_b_position, :] * ey_b[condition_b_position, :], axis=1)
    Wx_c = np.sum(ux_c[condition_c_position * condition_c_1, :] * ex_c[condition_c_position * condition_c_1, :], axis=1)
    Wx_b = np.sum(ux_b[condition_b_position, :] * ex_b[condition_b_position, :], axis=1)
    Wz_c = np.sum(uz_c[condition_c_position * condition_c_1, :] * ez_c[condition_c_position * condition_c_1, :], axis=1)
    Wz_b = np.sum(uz_b[condition_b_position, :] * ez_b[condition_b_position, :], axis=1)
    plt.hist(Wy_c, alpha=0.5)
    plt.hist(Wy_b, alpha=0.5)
    print(Wy_c.mean(), Wy_b.mean())
    plt.show()
    # %%
    run_case_index = 70
    num_files = 256
    x_left, x_right, z_bottom, z_top = 50, 60, -0.5, 0.5
    step = 30000
    field_dir = "D:\Research\Codes\Hybrid-vpic\HybridVPIC-main/reconnection/field_data/field_data_68/"
    base_fname_swi_c = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_c.{step}.{{}}"
    base_fname_swi_b = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_b.{step}.{{}}"
    p_c = Species_tag(name="ion_c", fullname="Ion_core", filename=base_fname_swi_c, num_files=num_files,
                  region=[x_left, x_right, z_bottom, z_top])

    # p_b = Species_tag(name="ion_b", fullname="Ion_beam", filename=base_fname_swi_b, num_files=num_files,
    #               region=[x_left, x_right, z_bottom, z_top], field_dir=field_dir)
    #%%
    vx_bin = np.linspace(-1.5, 1, 8)
    vy_bin = np.linspace(-1.5, 1, 8)
    delta_vx_c = np.zeros((len(vx_bin)-1, len(vy_bin)-1))
    delta_vx_b = np.zeros((len(vx_bin) - 1, len(vy_bin) - 1))
    step = 10000
    for i in range(len(vx_bin)-1):
        for j in range(len(vy_bin)-1):
            print(i, j)
            condition_tmp_c = (p_c.ux >= vx_bin[i]) & (p_c.ux < vx_bin[i+1]) & (p_c.uy>=vy_bin[j]) & (p_c.uy<vy_bin[j+1])
            idx_c = np.where(condition_tmp_c)
            condition_tmp_b = (p_b.ux >= vx_bin[i]) & (p_b.ux < vx_bin[i + 1]) & (p_b.uy >= vy_bin[j]) & (
                        p_b.uy < vy_bin[j + 1])
            idx_b = np.where(condition_tmp_b)
            tag_c_tmp = p_c.tag[condition_tmp_c]
            tag_b_tmp = p_b.tag[condition_tmp_b]
            vx0_c = p_c.ux[condition_tmp_c][np.argsort(tag_c_tmp)]
            vx0_b = p_b.ux[condition_tmp_b][np.argsort(tag_b_tmp)]
            base_fname_swi_c = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_c.{step}.{{}}"
            base_fname_swi_b = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_b.{step}.{{}}"
            p_cf = Species_tag(name="ion_c", fullname="Ion_core", filename=base_fname_swi_c, num_files=num_files,
                              field_dir=field_dir, tag_chosen=tag_c_tmp)

            # p_bf = Species_tag(name="ion_b", fullname="Ion_beam", filename=base_fname_swi_b, num_files=num_files,
            #                   field_dir=field_dir, tag_chosen=tag_b_tmp)
            vxf_c = p_cf.ux[np.argsort(p_cf.tag)]
            # vxf_b = p_bf.ux[np.argsort(p_bf.tag)]
            delta_vx_c[i, j] = np.mean(vxf_c-vx0_c)
            #index_f = np.argsort(p_cf.)



    #%%
    plt.pcolormesh(vx_bin[1:], vy_bin[1:], delta_vx_c.T, cmap='jet')
    plt.xlabel(r"$v_{x,0}/v_A$", fontsize=15)
    plt.ylabel(r"$v_{y,0}/v_A$", fontsize=15)
    cbar = plt.colorbar()
    cbar.set_label(r'$\Delta v_x/v_A$', fontsize=15)
    plt.savefig("D:\Research\Figures\VPIC磁重联模拟/delta_vx_for_different_vdfs.png")
    plt.show()
    #%%
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ux_total = np.concatenate((p_c.ux, p_b.ux))
    ax = axes[0]
    counts_ux, bins = np.histogram(p_c.ux, bins=np.linspace(-2, 2, 41))
    ax.plot(bins[1:], counts_ux)


    ax.set_xlabel(r"$v_x$", fontsize=18)
    ax.set_ylabel("Count", fontsize=18)
    ax.set_ylim(top=500000)
    y0, y1 = ax.get_ylim()
    ax.axvline(1, ymin=0, ymax=(counts_ux[29] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.axvline(0.5, ymin=0, ymax=(counts_ux[24] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.axvline(0, ymin=0, ymax=(counts_ux[19] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.axvline(-0.5, ymin=0, ymax=(counts_ux[14] - y0) / (y1 - y0), c="k", linestyle="--")
    # ax.axvline(0.5, ymin=0, ymax=(counts_ux[24] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.axvline(-1, ymin=0, ymax=(counts_ux[9] - y0) / (y1 - y0), c="k", linestyle="--")
    # ax.axvline(-0.2, ymin=0, ymax=(counts_ux[17] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.set_title(fr"Core", fontsize=20)
    ax.tick_params(labelsize=16)
    ax = axes[1]
    counts_ux, bins = np.histogram(p_b.ux, bins=np.linspace(-2, 2, 41))
    ax.plot(bins[1:], counts_ux)

    plt.xlabel(r"$v_x$", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    # ax.set_ylim(top=150000)
    ax.set_ylim(top=50000)
    y0, y1 = ax.get_ylim()
    # ax.axvline(1, ymin=0, ymax=(counts_ux[29] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.axvline(0.5, ymin=0, ymax=(counts_ux[24] - y0) / (y1 - y0), c="k", linestyle="--")
    # ax.axvline(0.5, ymin=0, ymax=(counts_ux[24] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.axvline(0.2, ymin=0, ymax=(counts_ux[21] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.axvline(-0.1, ymin=0, ymax=(counts_ux[18] - y0) / (y1 - y0), c="k", linestyle="--")
    ax.set_title("Beam",
                 fontsize=20)
    ax.tick_params(labelsize=16)
    plt.suptitle(fr"$v_x$ distribution of ions of {x_left}<x<{x_right}, {z_bottom}<z<{z_top}, $\Omega$t=50",
                 fontsize=20)
    plt.savefig(f"D:\Research\Figures\VPIC磁重联模拟/vx distribution of ions x{x_left}~{x_right},z{z_bottom}~{z_top}_3.png")

    plt.show()
    #%%
    tag_c = p_c.tag
    tag_c_hi_vx = tag_c[p_c.ux > 2]
    tag_c_lo_vx = tag_c[p_c.ux <= 0]
    # tag_b = p_b.tag
    run_case_index = 70
    num_files = 256
    # x_left, x_right, z_bottom, z_top = 246, 266, -1, 1
    step = 0
    base_fname_swi_c = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_c.{step}.{{}}"
    base_fname_swi_b = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_b.{step}.{{}}"
    p_c0_hi_vx = Species_tag(name="ion_c", fullname="Ion_core", filename=base_fname_swi_c, num_files=num_files,
                      tag_chosen=tag_c_hi_vx)
    p_c0_lo_vx = Species_tag(name="ion_c", fullname="Ion_core", filename=base_fname_swi_c, num_files=num_files,
                         tag_chosen=tag_c_lo_vx)
    # p_b0 = Species_tag(name="ion_b", fullname="Ion_beam", filename=base_fname_swi_b, num_files=num_files,
    #                   tag_chosen=tag_b)
    #%%
    x_shift_hi = process_array(p_c0_hi_vx.x, 128)
    x_shift_lo = process_array(p_c0_lo_vx.x, 128)
    counts_max_hi_vx_pos, x_edge_hi, z_edge_hi = np.histogram2d(x_shift_hi, p_c0_hi_vx.z, bins=[40, 20],
                                                          range=[[x_shift_hi.min(), x_shift_hi.max()], [p_c0_hi_vx.z.min(), p_c0_hi_vx.z.max()]])
    counts_max_lo_vx_pos, x_edge_lo, z_edge_lo = np.histogram2d(x_shift_lo, p_c0_lo_vx.z, bins=[40, 20],
                                                          range=[[x_shift_lo.min(), x_shift_lo.max()],
                                                                 [p_c0_lo_vx.z.min(), p_c0_lo_vx.z.max()]])
    # plt.pcolormesh(x_edge_lo[1:], z_edge_lo[1:], counts_max_lo_vx_pos.T)

    plt.pcolormesh(x_edge_hi[1:], z_edge_hi[1:], counts_max_hi_vx_pos.T)
    plt.colorbar()
    plt.show()
    #%%
    counts_max_hi_vx_vel, vx_edge_hi, vz_edge_hi = np.histogram2d(p_c0_hi_vx.ux, p_c0_hi_vx.uz, bins=[40, 20],
                                                                range=[[p_c0_hi_vx.ux.min(), p_c0_hi_vx.ux.max()],
                                                                       [p_c0_hi_vx.uz.min(), p_c0_hi_vx.uz.max()]])
    counts_max_lo_vx_vel, vx_edge_lo, vz_edge_lo = np.histogram2d(p_c0_lo_vx.ux, p_c0_lo_vx.uz, bins=[40, 20],
                                                                range=[[p_c0_lo_vx.ux.min(), p_c0_lo_vx.ux.max()],
                                                                       [p_c0_lo_vx.uz.min(), p_c0_lo_vx.uz.max()]])
    plt.pcolormesh(vx_edge_hi[1:], vz_edge_hi[1:], counts_max_hi_vx_vel.T)
    plt.colorbar()
    plt.show()
    #%%
    tag_i = np.sort(p_c0.tag)
    tag_f = np.sort(p_c.tag)
    idx_i = np.argsort(p_c0.tag)
    idx_f = np.argsort(p_c.tag)
    #%%
    counts_mat_c, xedges_c, zedges_c = np.histogram2d(p_c0.x, p_c0.z,
                                                      bins=[80, 80],
                                                      range=[[350, 450],
                                                             [-10, 10]])
    plt.pcolormesh(xedges_c[1:], zedges_c[1:], counts_mat_c.T)

    plt.show()
    #%%
    print(p_c.ux.min(), p_c.ux.max())
    print(p_b.ux.min(), p_b.ux.max())
    #%%
    condition_tmp_c_1 = (p_c.ux > -2) & (p_c.ux < -1)#(p_c.E < 0.125) & (p_c.E > 0)
    condition_tmp_b_1 = (p_b.ux > -2) & (p_b.ux < -0.1)
    step = 0
    base_fname_swi_c = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_c.{step}.{{}}"
    base_fname_swi_b = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_b.{step}.{{}}"
    p_c0_tmp_1 = Species_tag(name="ion_c", fullname="Ion_core", filename=base_fname_swi_c, num_files=num_files,
                      tag_chosen=tag_c[condition_tmp_c_1])
    # p_b0_tmp_1 = Species_tag(name="ion_b", fullname="Ion_beam", filename=base_fname_swi_b, num_files=num_files,
    #                          tag_chosen=tag_b[condition_tmp_b_1])
    #%%
    condition = (np.abs(p_c0_tmp_1.z) <= 1)
    print(len(p_c0_tmp_1.ux[condition])/len(p_c0_tmp_1.ux))
    #%%
    counts_mat_c, xedges_c, zedges_c = np.histogram2d(p_c0_tmp_1.ux, p_c0_tmp_1.uy,
                                                      bins=[80, 80],
                                                      range=[[-2, 2],
                                                             [-2, 2]])
    # counts_mat_b, xedges_b, zedges_b = np.histogram2d(p_b0_tmp_1.x, p_b0_tmp_1.z,
    #                                                   bins=[80, 80],
    #                                                   range=[[350, 450],
    #                                                          [-3, 3]])
    plt.pcolormesh(xedges_c[1:], zedges_c[1:], counts_mat_c.T, cmap='jet')
    # plt.contourf(xedges_c[1:], zedges_c[1:], counts_mat_c.T, cmap='jet')
    x_plot = np.linspace(-1.25, -0.0, 100)
    cbar = plt.colorbar()
    cbar.set_label(r"$\log_{10}(count)$", fontsize=15)
    # cbar.set_label("count", fontsize=15)
    # plt.axvline(-0.6, ymin=0.1, ymax=0.8, c="white", linestyle="--")
    plt.plot(x_plot, x_plot+1.3, c="white", linestyle="--", label=r"$v_y>v_x+1.2v_A$")
    # condition_position_a = p_c0_tmp_1.ux < -0.6
    # ratio_1 = len(p_c0_tmp_1.ux[condition_position_a & (np.abs(p_c0_tmp_1.z) <= 1)]) / len(
    #     p_c0_tmp_1.ux[condition_position_a])
    # ratio_2 = len(p_c0_tmp_1.ux[~condition_position_a & (np.abs(p_c0_tmp_1.z) <= 1)]) / len(
    #     p_c0_tmp_1.ux[~condition_position_a])
    # plt.scatter(-1/3, 0, c="white", label=r"$(v_{x,0}^c,v_{y,0}^c)$")
    # plt.text(-1.4, 0.7, fr"{100*ratio_1:.2f}% in HCS", fontsize=15, c="white")
    # plt.text(-0.45, -0.7, fr"{100 * ratio_2:.2f}% in HCS", fontsize=15, c="white")
    # plt.text(-0.27, 0, r"$(v_{x,0}^c,v_{y,0}^c)$", c="white", fontsize=15)
    plt.title("core ion(1<vx<1.3)", fontsize=15)
    # plt.xlabel(r"$v_x/v_A$", fontsize=15)
    # plt.ylabel(r"$v_y/v_A$", fontsize=15)
    plt.xlabel("x[di]", fontsize=15)
    plt.ylabel("z[di]", fontsize=15)
    # plt.savefig("D:\Research\Figures\VPIC磁重联模拟\initial beam ions vdf(bottom=-2, top=-0.1).png")
    # plt.legend()
    plt.show()
    #%%
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    ax = axes[0]
    counts_mat_c, xedges_c, zedges_c = np.histogram2d(p_c0_tmp_1.x, p_c0_tmp_1.z,
                                                      bins=[80, 80],
                                                      range=[[350, 500],
                                                             [-10, 10]])
    pclr = ax.pcolormesh(xedges_c[1:], zedges_c[1:], counts_mat_c.T, cmap='jet')
    cbar = plt.colorbar(pclr, ax=ax)
    ax.set_title("Initial position distribution", fontsize=15)
    # ax.set_title(r"$-v_A<vx_f<-0.5v_A$", fontsize=15)
    ax.set_xlabel("x[di]", fontsize=15)
    ax.set_ylabel("z[di]", fontsize=15)

    ax = axes[1]
    counts_mat_c, xedges_c, zedges_c = np.histogram2d(p_c0_tmp_1.ux,
                                                      p_c0_tmp_1.uy,
                                                      bins=[80, 80],
                                                      range=[[-2, 2],
                                                             [-2, 2]])
    pclr = ax.pcolormesh(xedges_c[1:], zedges_c[1:], counts_mat_c.T, cmap='jet')
    cbar = plt.colorbar(pclr, ax=ax)
    ax.set_title("Initial velocity distribution", fontsize=15)
    ax.set_xlabel(r"$v_x[v_A]$", fontsize=15)
    ax.set_ylabel(r"$v_y[v_A]$", fontsize=15)
    plt.suptitle(r"$-2v_A<vx_f<-v_A$", fontsize=17)
    # plt.savefig("D:\Research\Figures\VPIC磁重联模拟\initial core ions pdf&vdf(bottom=-0.5, top=0).png")
    plt.savefig("D:\Research\Figures\VPIC磁重联模拟\initial core ions pdf&vdf(bottom=-2, top=-1)_verticle.png")
    plt.show()
    #%%
    """
    PLOT 2D-VDF
    """
    run_case_indecies = [66]
    b2c_ratios = [0.1]
    sample_step = [1, 1, 1]
    steps = [10000]
    fig, axes = plt.subplots(len(steps), len(run_case_indecies), figsize=(10, 10))
    idx = 0
    x_left, x_right, z_bottom, z_top = 30, 40, 3, 6
    x_center = (x_left + x_right) // 2
    z_center = (z_bottom + z_top) // 2
    num_files = 256
    for i_run_case_index in range(len(run_case_indecies)):
        for i_step in range(len(steps)):
            step = steps[i_step]
            run_case_index = run_case_indecies[i_run_case_index]

            # field_dir = f"data_ip_shock/field_data_{run_case_index}/"
            base_fname_swi_c = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_c.{step}.{{}}"
            base_fname_swi_b = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_b.{step}.{{}}"
            p_c = Species(name="ion_c", fullname="Ion_core", filename=base_fname_swi_c, num_files=num_files, region=[x_left, x_right, z_bottom, z_top])
            p_b = Species(name="ion_b", fullname="Ion_beam", filename=base_fname_swi_b, num_files=num_files, region=[x_left, x_right, z_bottom, z_top])
            field_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data/field_data_{run_case_index}/"
            # %%
            epoch = step // 200
            nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
            # bx = load_data_at_certain_t(field_dir + "bx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
            # by = load_data_at_certain_t(field_dir + "by.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
            # bz = load_data_at_certain_t(field_dir + "bz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
            # condition_total = region(x_total, z_total, x_left, x_right, z_bottom, z_top)
            condition_b = region(p_b.x, p_b.z, x_left, x_right, z_bottom, z_top)
            condition_c = region(p_c.x, p_c.z, x_left, x_right, z_bottom, z_top)
            x_total, y_total, z_total = np.concatenate((p_c.x[condition_c], p_b.x[condition_b])), np.concatenate((p_c.y[condition_c], p_b.y[condition_b])), np.concatenate(
                (p_c.z[condition_c], p_b.z[condition_b]))
            ux_total, uy_total, uz_total = np.concatenate((p_c.ux[condition_c], p_b.ux[condition_b])), np.concatenate(
                (p_c.uy[condition_c], p_b.uy[condition_b])), np.concatenate((p_c.uz[condition_c], p_b.uz[condition_b]))
            # %%
            # p_c.plot_phase_space_2D(sample_step=40, x_plot_name="x", y_plot_name="ux", color="k", size=1)
            # p_b.plot_phase_space_2D(sample_step=40, x_plot_name="x", y_plot_name="ux", color="r", size=1)
            # plt.show()
            # %%
            nx, nz = p_c.nx, p_c.nz

            ux_mean_c = np.zeros(nx)
            ux_mean_b = np.zeros(nx)
            ux_var_c = np.zeros(nx)
            ux_var_b = np.zeros(nx)
            # for i in range(nx):
            #     condition_c_tmp = region(p_c.x, p_c.z, i, i + 1, -1, 1)
            #     ux_mean_c[i] = p_c.ux[condition_c_tmp].mean()
            #     ux_var_c[i] = p_c.ux[condition_c_tmp].var() + p_c.uy[condition_c_tmp].var() + p_c.uz[
            #         condition_c_tmp].var()
            #     condition_b_tmp = region(p_b.x, p_b.z, i, i + 1, -1, 1)
            #     ux_mean_b[i] = p_b.ux[condition_b_tmp].mean()
            #     ux_var_b[i] = p_b.ux[condition_b_tmp].var() + p_b.uy[condition_b_tmp].var() + p_b.uz[
            #         condition_b_tmp].var()
            # ax = axes  # [i_step]# [i_run_case_index]
            # ax.plot(ux_var_c, c="b", label="core")
            #
            # ax.plot(ux_var_b, c="r", label="beam")
            # # ax.set_xlim([0, 100])
            # ax.set_xlabel("x", fontsize=15)
            # ax.set_ylabel("T", fontsize=15)
            # ax.legend()
            # ax.set_ylim([-1.2, 0.4])
            # plt.plot(ux_var_b/1.25)

            # plt.ylim([-1, 0.4])
            # ax.set_ylim([-0.5, 1.5])
            # plt.show()
            # print(len(ux_total[condition_total]))
            # counts_ux, bins = np.histogram(p_b.ux[condition_b], bins=np.linspace(np.min(p_b.ux), np.max(p_b.ux), 20))

            counts_mat_b, xedges_b, zedges_b = np.histogram2d(p_b.ux[condition_b][::sample_step[i_run_case_index]],
                                                        p_b.uz[condition_b][::sample_step[i_run_case_index]],
                                                        bins=[80, 80],
                                                        range=[[np.min(p_b.ux), np.max(p_b.ux)],
                                                               [np.min(p_b.uz), np.max(p_b.uz)]])
            counts_mat_c, xedges_c, zedges_c = np.histogram2d(p_c.ux[condition_c][::sample_step[i_run_case_index]],
                                                        p_c.uz[condition_c][::sample_step[i_run_case_index]], bins=[80, 80],
                                                        range=[[np.min(p_c.ux), np.max(p_c.ux)],
                                                               [np.min(p_c.uz), np.max(p_c.uz)]])
            counts_mat_t, xedges_t, zedges_t = np.histogram2d(ux_total[::sample_step[i_run_case_index]],
                                                        uz_total[::sample_step[i_run_case_index]], bins=[80, 80],
                                                        range=[[np.min(ux_total), np.max(ux_total)],
                                                               [np.min(uz_total), np.max(uz_total)]])

            n_core = len(p_c.ux[condition_c][::sample_step[i_run_case_index]])
            # n_beam = len(p_b.ux[condition_b][::sample_step[i_run_case_index]])

            ax = axes
            ctf = ax.contourf(xedges_c[1:], zedges_c[1:], np.log10(counts_mat_c.T), cmap="jet")
            ax.contour(xedges_c[1:], zedges_c[1:], np.log10(counts_mat_c.T), colors="k")
            cbar = plt.colorbar(ctf, ax=ax)
            cbar.set_label(r"$\log_{10}{(Counts)}$", fontsize=15)
            ax.scatter(-1/3, 0, c="k", s=45, label=r"$(v_{x,0}^c, v_{y,0}^c)$")
            ax.scatter(2/3, 0, c="k", marker="x", s=45, label=r"$(v_{x,0}^b, v_{y,0}^b)$")
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)
            # ax.scatter(0, 0, color="k", marker="x")
            # ax.axhline(0, linestyle="--", color="k")

            # ax.arrow(0.5, -1.5, bx[x_center, nz//2+8]*7,
            #           by[x_center, nz//2+8]*7, head_width=0.1, fc="k")
            # ax.text(1, -1.5, r"$\mathbf{B}$", fontsize=13)

            # if i_step==1:
            #     plt.plot(bins[1:], counts_ux/counts_ux.sum(), label=f"nb/nc={b2c_ratios[i_run_case_index]}")
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])

            # ax.axis("equal")
            ax.set_xlabel("vx", fontsize=25)
            ax.set_ylabel("vy", fontsize=25)
            ax.grid(True)
            title_total = "total\n"\
            # rf"({100*n_core/(n_core+n_beam):.2f}%core+{100*n_beam/(n_core+n_beam):.2f}%beam)"
            ax.set_title(title_total, fontsize=25)
            # ax.legend(fontsize=25)
            # ax = axes[1]
            # ctf = ax.contourf(xedges_c[1:], zedges_c[1:], np.log10(counts_mat_c.T), cmap="jet")
            # ax.contour(xedges_c[1:], zedges_c[1:], np.log10(counts_mat_c.T), colors="k")
            # cbar = plt.colorbar(ctf, ax=ax)
            # cbar.set_label(r"$\log_{10}{(Counts)}$", fontsize=15)
            # # ax.scatter(0, 0, color="k", marker="x")
            # # ax.axhline(0, linestyle="--", color="k")
            #
            # # ax.arrow(0.5, -1.5, bx[x_center, nz//2+8]*7,
            # #           by[x_center, nz//2+8]*7, head_width=0.1, fc="k")
            # # ax.text(1, -1.5, r"$\mathbf{B}$", fontsize=13)
            #
            # # if i_step==1:
            # #     plt.plot(bins[1:], counts_ux/counts_ux.sum(), label=f"nb/nc={b2c_ratios[i_run_case_index]}")
            # ax.set_xlim([-2, 3])
            # ax.set_ylim([-2, 2])
            # # ax.axis("equal")
            # ax.set_xlabel("vx", fontsize=20)
            # ax.set_ylabel("vy", fontsize=20)
            # ax.grid(True)
            # ax.set_title(rf"core", fontsize=20)
            # ax.scatter(-1 / 3, 0, c="k", s=45)# label=r"$(v_{x,0}^c, v_{y,0}^c)$")
            # ax.scatter(2 / 3, 0, c="k", marker="x", s=45)# label=r"$(v_{x,0}^b, v_{y,0}^b)$")
            # ax.tick_params(axis='x', labelsize=25)
            # ax.tick_params(axis='y', labelsize=25)
            # ax = axes[2]
            # ctf = ax.contourf(xedges_b[1:], zedges_b[1:], np.log10(counts_mat_b.T), cmap="jet")
            # ax.contour(xedges_b[1:], zedges_b[1:], np.log10(counts_mat_b.T), colors="k")
            # cbar = plt.colorbar(ctf, ax=ax)
            # cbar.set_label(r"$\log_{10}{(Counts)}$", fontsize=15)
            # # ax.scatter(0, 0, color="k", marker="x")
            # # ax.axhline(0, linestyle="--", color="k")
            #
            # # ax.arrow(0.5, -1.5, bx[x_center, nz//2+8]*7,
            # #           by[x_center, nz//2+8]*7, head_width=0.1, fc="k")
            # # ax.text(1, -1.5, r"$\mathbf{B}$", fontsize=13)
            #
            # # if i_step==1:
            # #     plt.plot(bins[1:], counts_ux/counts_ux.sum(), label=f"nb/nc={b2c_ratios[i_run_case_index]}")
            # ax.set_xlim([-2, 3])
            # ax.set_ylim([-2, 2])
            # # ax.axis("equal")
            # ax.set_xlabel("vx", fontsize=20)
            # ax.set_ylabel("vy", fontsize=20)
            # ax.grid(True)
            # ax.set_title(rf"beam", fontsize=20)
            # ax.scatter(-1 / 3, 0, c="k", s=45)# label=r"$(v_{x,0}^c, v_{y,0}^c)$")
            # ax.scatter(2 / 3, 0, c="k", marker="x", s=45)# label=r"$(v_{x,0}^b, v_{y,0}^b)$")
            # ax.tick_params(axis='x', labelsize=25)
            # ax.tick_params(axis='y', labelsize=25)
            # if i_step == 1:
            #     ax.text(-1.8, 1.8, fr"$ux_{{core}}$={p_c.ux[condition_c].mean():.3f}"
            #             , fontsize=13)
            idx += 1
    fig.legend(loc='lower center',
               bbox_to_anchor=(0.5, 0),  # 0.5 表示水平居中，0 表示底部
               ncol=1,  # 图例分为3列，对应3个子图
               fontsize=20)
    # 调整子图布局，为图例腾出空间
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle(rf"$\Omega t={step // 400}$, {x_left:.2f}<x<{x_right:.2f}, {z_bottom}<z<{z_top}", fontsize=28)
    # plt.legend()
    plt.show()

    # %%
    step = 1000
    run_case_index = 34
    num_files = 16
    base_fname_swi_c = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_c.{step}.{{}}"
    base_fname_swi_b = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data/particle_data_{run_case_index}/T.{step}/Hparticle_b.{step}.{{}}"
    p_c = Species(name="ion_c", fullname="Ion_core", filename=base_fname_swi_c, num_files=num_files)
    p_b = Species(name="ion_b", fullname="Ion_beam", filename=base_fname_swi_b, num_files=num_files)
    # p_c.plot_phase_space_2D(sample_step=100, x_plot_name="x", y_plot_name="ux", color="k", size=1)
    # plt.show()
    x_total, y_total, z_total = np.concatenate((p_c.x, p_b.x)), np.concatenate((p_c.y, p_b.y)), np.concatenate(
        (p_c.z, p_b.z))
    ux_total, uy_total, uz_total = np.concatenate((p_c.ux, p_b.ux)), np.concatenate((p_c.uy, p_b.uy)), np.concatenate(
        (p_c.uz, p_b.uz))
    # tag_b = np.sort(p_b.tag)
    # %%
    ni_c, xedges, zedges = np.histogram2d(p_c.x, p_c.z, bins=[p_c.nx, p_c.nz],
                                                range=[[0, p_c.x.max()], [p_c.z.min(), p_c.z.max()]])
    ni_b, xedges, zedges = np.histogram2d(p_b.x, p_b.z, bins=[p_b.nx, p_b.nz],
                                          range=[[0, p_b.x.max()], [p_b.z.min(), p_b.z.max()]])
    #%%
    ux_var_b = optimized_ux_mean_calculation(p_b, topo_x, topo_y, topo_z, param_name="ux")
    ux_var_c = optimized_ux_mean_calculation(p_c, topo_x, topo_y, topo_z, param_name="ux")
    print(p_c.iz.max())
    #%%
    ux_var_t = (ux_var_c*ni_c+ux_var_b*ni_b)/(ni_b+ni_c)
    # print(p_c.z[p_c.rank == 7].min())
    # %%
    from tqdm import tqdm

    nx, ny, nz = p_c.nx, p_c.ny, p_c.nz
    _nx, _ny, _nz = nx // topo_x + 2, ny // topo_y + 2, nz // topo_z + 2
    icell_lst = list(set(p_c.icell))
    ux_mean = np.zeros((p_c.nx, p_c.nz))

    # 计算总迭代次数
    total_iterations = 16 * len(icell_lst)

    # 创建进度条
    with tqdm(total=total_iterations, desc="Processing", unit="iter") as pbar:
        for rank in range(16):
            for i in icell_lst:
                condition = (p_c.icell == i) & (p_c.rank == rank)
                ix = i % _nx
                iy = (i // _nx) % _ny
                iz = i // (_nx * _ny)
                # print(i, ix, iz, rank)
                ux_mean[(rank // 2) * 64 + int(ix), (rank // 8) * 128 + int(iz)] = p_c.ux[condition].mean()

                # 更新进度条
                pbar.update(1)
    # %%
    print(p_c.rank.min(), p_c.rank.max())
    # %%
    # ux_mean = np.zeros((p_c.nx, p_c.nz))
    Lx = 512
    Lz = 64
    hx, hz = Lx / p_c.nx, Lz / p_c.nz
    #%%
    for i in range(p_c.nx):
        print(i)
        for j in range(p_c.nz):
            condition_ij = region(x_total[::100], z_total[::100], i * hx, (i + 1) * hx, -Lz / 2 + j * hz,
                                  -Lz / 2 + (j + 1) * hz)
            if len(ux_total[::100][condition_ij]) > 0:
                ux_mean[i, j] = ux_total[::100][condition_ij].var()
    # ux_mean_c = calculate_ux_mean_parallel(p_c.x, p_c.z, p_c.ux, p_c.nx, p_c.nz)
    # ux_mean_b = calculate_ux_mean_parallel(p_b.x, p_b.z, p_b.ux, p_c.nx, p_c.nz)
    # ux_mean_t = calculate_ux_mean_parallel(x_total, z_total, ux_total, p_c.nx, p_c.nz)

    # %%
    # p_b.plot_counts_dis_map(vmin=50, vmax=300)
    p_c.plot_phase_space_2D(x_plot_name="z", y_plot_name="ux", color="k", sample_step=10, size=1)
    plt.show()
    # %%
    plt.pcolormesh(range(p_c.nx), np.linspace(-Lz / 2, Lz / 2, p_c.nz), smooth_matrix(ux_var_c).T, cmap="jet")
    # plt.ylim([-5, 5])
    plt.colorbar()
    plt.show()
    # %%
    p = p_c
    v_para = p.ux
    v_perp = np.sqrt(p.uy ** 2 + p.uz ** 2)
    x_left, x_right = 150, 200
    z_bottom, z_top = -1, 1
    x_center_idx = round((x_left + x_right) / 2)
    z_center_idx = round((z_bottom + z_top) / 2 + 32)
    condition = (p.x > x_left) & (p.x < x_right) & (p.z > z_bottom) & (p.z < z_top)
    counts_arr_x, bins_x = np.histogram(p.ux[condition],
                                        bins=np.linspace(p.ux[condition].min(), p.ux[condition].max(), 25)
                                        , density=True)
    counts_arr_y, bins_y = np.histogram(p.uy[condition],
                                        bins=np.linspace(p.uy[condition].min(), p.uy[condition].max(), 25)
                                        , density=True)
    # %%
    p0 = [0.5, 1, 0, 1, 0]
    bounds = [[0, 0, -10, 0, -10], [1, 10, 10, 10, 10]]
    # 进行曲线拟合
    popt_x, pcov_x = curve_fit(double_maxwellian, bins_x[1:], counts_arr_x, p0=p0, bounds=bounds)
    bounds_y = [0.99 * popt_x, 1.01 * popt_x]
    bounds_y[0][2], bounds_y[0][4], bounds_y[1][2], bounds_y[1][4] = -1, -1, 1, 1
    bounds_y[0][1], bounds_y[0][3], bounds_y[1][1], bounds_y[1][3] = 0, 0, 1, 1
    # %%
    popt_y, pcov_y = curve_fit(double_maxwellian, bins_y[1:], counts_arr_y, p0=popt_x
                               , bounds=bounds_y)
    if popt_x[0] > 0.5:
        xi_c, Tx_c, vx_c, Tx_b, vx_b = popt_x
        Ty_c, vy_c, Ty_b, vy_b = popt_y[1:]
    else:
        xi_c, Tx_b, vx_b, Tx_c, vx_c = popt_x
        Ty_b, vy_b, Ty_c, vy_c = popt_y[1:]
        xi_c = 1 - xi_c
    print(f"xi_c={xi_c:.3f}, vx_c={vx_c:.3f}, vy_c={vy_c:.3f}, vx_b={vx_b:.3f}, vy_b={vy_b:.3f}")
    print(f"Tx_c={Tx_c:.3f}, Ty_c={Ty_c:.3f}, Tx_b={Tx_b:.3f}, Ty_b={Ty_b:.3f}")
    plt.plot(bins_x[1:], double_maxwellian(bins_x[1:], *popt_x), c="b")
    plt.plot(bins_x[1:], counts_arr_x, c="r")
    plt.plot(bins_y[1:], double_maxwellian(bins_y[1:], *popt_y), c="b", linestyle="--")
    plt.plot(bins_y[1:], counts_arr_y, c="r", linestyle="--")
    plt.show()
    # plt.plot(bins[1:], counts_arr_1/np.sum(counts_arr_1))
    # plt.yscale("log")
    # plt.xscale("log")
    #%%
    n_sheet = 0
    for i in range(0, 32):
        for j in range(-32, 0):
            xx, zz = 0.5*i+0.25, 0.25*j+0.25/2
            n_sheet += (0.05+1/np.cosh(zz)**2)*0.25*0.5
    print(n_sheet)




    #%%
    """
    PLOT THE FIELD
    """
    epoch = 50
    run_case_index = "47"
    plot_particle_tracking = False
    nb = 0.2
    field_plot = "field_total"
    field_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data/field_data_{run_case_index}/"
    # field_dir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/field_data/field_data_{run_case_index}/"
    nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
    Lx, Lz = int(loadinfo(field_dir)[3]), int(loadinfo(field_dir)[5])
    X, Z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz))
    hx, hz = Lx / nx, Lz / nz
    ni_c0 = load_data_at_certain_t(field_dir + "ni.gda", i_t=0, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    Ay_total = load_data(field_dir+'Ay.gda', num_dim1=nx, num_dim2=nz, num_dim3=2)
    Ay_0 = Ay_total[:, :, 0]
    # plt.plot(Ay_total[511, 127, :]-Ay_total[850, 127, :])
    # # plt.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), -Ay_total[:,:,99].T, cmap='jet')
    # # plt.colorbar()
    # plt.show()
    # #%%
    for epoch in range(1):
        print(epoch)
        bx = load_data_at_certain_t(field_dir + "bx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        by = load_data_at_certain_t(field_dir + "by.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        bz = load_data_at_certain_t(field_dir + "bz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # bx_shift=np.zeros_like(bx)
        # bx_shift[0:nx//2, :], bx_shift = bx[nx//2:, :]

        b_mag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        ex = load_data_at_certain_t(field_dir + "ex.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ey = load_data_at_certain_t(field_dir + "ey.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ez = load_data_at_certain_t(field_dir + "ez.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ni_c = load_data_at_certain_t(field_dir + "ni.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # ni_b = load_data_at_certain_t(field_dir + "ni2.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uix_c = load_data_at_certain_t(field_dir + "uix.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_c = load_data_at_certain_t(field_dir + "uiy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_c = load_data_at_certain_t(field_dir + "uiz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uix_b = load_data_at_certain_t(field_dir + "ui2x.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_b = load_data_at_certain_t(field_dir + "ui2y.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_b = load_data_at_certain_t(field_dir + "ui2z.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_xx = load_data_at_certain_t(field_dir + "pi-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_yy = load_data_at_certain_t(field_dir + "pi-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_zz = load_data_at_certain_t(field_dir + "pi-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        Jx, Jy, Jz = calculate_current_density(bx, by, bz, 0.5, 1, 0.25)
        ez_hall = -Jy*bx/ni_c
        ez_pressure = -np.gradient((pic_xx+pic_yy+pic_zz)/3, axis=1)*4/ni_c
        ez_ion = uiy_c*bx
        ez_calc = ez_hall + ez_pressure + ez_ion
        ez_hall_theory = -np.sinh(Z.T)/np.cosh(Z.T)**3/(nb+1/np.cosh(Z.T)**2)
        ez_pressure_theory = 0.5*np.sinh(Z.T)/np.cosh(Z.T)**3/(nb+1/np.cosh(Z.T)**2)
        uiy_theory = 0.5/np.cosh(Z.T)**2/(nb+1/np.cosh(Z.T)**2)
        ez_ion_theory = uiy_theory*np.tanh(Z.T)
        ez_theoty = ez_ion_theory+ez_hall_theory+ez_pressure_theory
        # pib_xx = load_data_at_certain_t(field_dir + "pi2-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # pib_yy = load_data_at_certain_t(field_dir + "pi2-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # pib_zz = load_data_at_certain_t(field_dir + "pi2-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # tx_c, ty_c, tz_c = pic_xx/ni_c, pic_yy/ni_c, pic_zz/ni_c
        # tx_b, ty_b, tz_b = pib_xx / ni_b, pib_yy / ni_b, pib_zz / ni_b
        # uix_total = (ni_c * uix_c + ni_b * uix_b) / (ni_c + ni_b)
        # uiy_total = (ni_c * uiy_c + ni_b * uiy_b) / (ni_c + ni_b)
        # uiz_total = (ni_c * uiz_c + ni_b * uiz_b) / (ni_c + ni_b)
        Ay = load_data_at_certain_t(field_dir + "Ay.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # p_total = (pic_xx+pic_zz+pic_yy+pib_xx+pib_yy+pib_zz)*1/3+0.5*b_mag**2
        # n_total = ni_c+ni_b
        f_bz = (bz**2-bx**2-by**2)/2

        fig, axes = plt.subplots(3, 2, figsize=(20, 10))
        # ax = axes[0]
        # ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), Ay.T, colors="k", levels=40)
        #
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(ni_c.T),
        #                      cmap="jet")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        # ax.set_ylabel("z", fontsize=20)
        # ax.set_title(r"$n_c$", fontsize=20)
        # ax = axes[1]
        # ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), Ay.T, colors="k", levels=40)
        #
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(ni_b.T),
        #                      cmap="jet")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        # ax.set_ylabel("z", fontsize=20)
        # ax.set_title(r"$n_b$", fontsize=20)
        # plt.show()

        ax = axes[0][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr=ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(ni_c-ni_c0).T),
                                 cmap="bwr", vmin=-0.1, vmax=0.1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$\Delta n$", fontsize=20)
        ax = axes[0][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), Ay.T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix((ex).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$E_x$", fontsize=20)
        ax = axes[1][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(by).T),
                             cmap="bwr", vmin=-0.2, vmax=0.2)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$B_y$", fontsize=20)
        ax = axes[1][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=80)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(ey).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$E_y$", fontsize=20)
        ax = axes[2][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(bz.T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$B_z$", fontsize=20)
        ax = axes[2][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(ez).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$E_z$", fontsize=20)
        plt.suptitle(rf"$t={epoch}\Omega_{{ci}}^{{-1}}$", fontsize=25)
        # cold_x = ex+uiy_total*bz-uiz_total*by
        # cold_y = ey + uiz_total * bx - uix_total * bz
        # cold_z = ez + uix_total * by - uiy_total * bx
        # pclr = axes.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(cold_z.T),
        #                        cmap="bwr", vmin=-0.05, vmax=0.05)
        # plt.show()
        # cbar = plt.colorbar(pclr, ax=axes)
        # axes.set_ylim([-10, 10])
        # axes.set_xlabel("x", fontsize=20)
        # axes.set_ylabel("z", fontsize=20)
        # axes.set_title(r"$(\vec{E}+\vec{U}\times\vec{B})_z$", fontsize=20)
        figs_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_figs/{field_plot}s_{run_case_index}/"
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)
        plt.savefig(figs_dir + f"{field_plot}_{epoch}.png")
        plt.close(fig)
    # plt.close()
    # plt.scatter(400, 0, c="k")
    # plt.scatter(400, 2, c="k")
    # plt.scatter(400, 4, c="k")
    # plt.scatter(256, 0, c="k")
    # plt.scatter(256, 2, c="k")
    # plt.scatter(256, 4, c="k")

    #%%
    """
    PLOT THE MOMENT
    """
    epoch = 50
    run_case_index = "59"
    plot_particle_tracking = False
    nb = 0.2
    field_plot = "moment_total"
    field_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data/field_data_{run_case_index}/"
    # field_dir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/field_data/field_data_{run_case_index}/"
    nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
    Lx, Lz = int(loadinfo(field_dir)[3]), int(loadinfo(field_dir)[5])
    X, Z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz))
    hx, hz = Lx / nx, Lz / nz
    ni_c0 = load_data_at_certain_t(field_dir + "ni.gda", i_t=0, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    Ay_total = load_data(field_dir + 'Ay.gda', num_dim1=nx, num_dim2=nz, num_dim3=2)
    Ay_0 = Ay_total[:, :, 0]
    # plt.plot(Ay_total[511, 127, :]-Ay_total[850, 127, :])
    # # plt.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), -Ay_total[:,:,99].T, cmap='jet')
    # # plt.colorbar()
    # plt.show()
    # #%%
    for epoch in range(50):
        print(epoch)
        bx = load_data_at_certain_t(field_dir + "bx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        by = load_data_at_certain_t(field_dir + "by.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        bz = load_data_at_certain_t(field_dir + "bz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        b_mag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        ex = load_data_at_certain_t(field_dir + "ex.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ey = load_data_at_certain_t(field_dir + "ey.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ez = load_data_at_certain_t(field_dir + "ez.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ni_c = load_data_at_certain_t(field_dir + "ni.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ni_b = load_data_at_certain_t(field_dir + "ni2.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uix_c = load_data_at_certain_t(field_dir + "uix.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_c = load_data_at_certain_t(field_dir + "uiy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_c = load_data_at_certain_t(field_dir + "uiz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uix_b = load_data_at_certain_t(field_dir + "ui2x.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_b = load_data_at_certain_t(field_dir + "ui2y.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_b = load_data_at_certain_t(field_dir + "ui2z.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_xx = load_data_at_certain_t(field_dir + "pi-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_yy = load_data_at_certain_t(field_dir + "pi-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_zz = load_data_at_certain_t(field_dir + "pi-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        Jx, Jy, Jz = calculate_current_density(bx, by, bz, hx, 1, hz)
        ez_hall = -Jy * bx / ni_c
        ez_pressure = -np.gradient((pic_xx + pic_yy + pic_zz) / 3, axis=1) * 4 / ni_c
        ez_ion = uiy_c * bx
        ez_calc = ez_hall + ez_pressure + ez_ion
        ez_hall_theory = -np.sinh(Z.T) / np.cosh(Z.T) ** 3 / (nb + 1 / np.cosh(Z.T) ** 2)
        ez_pressure_theory = 0.5 * np.sinh(Z.T) / np.cosh(Z.T) ** 3 / (nb + 1 / np.cosh(Z.T) ** 2)
        uiy_theory = 0.5 / np.cosh(Z.T) ** 2 / (nb + 1 / np.cosh(Z.T) ** 2)
        ez_ion_theory = uiy_theory * np.tanh(Z.T)
        ez_theoty = ez_ion_theory + ez_hall_theory + ez_pressure_theory
        # pib_xx = load_data_at_certain_t(field_dir + "pi2-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # pib_yy = load_data_at_certain_t(field_dir + "pi2-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # pib_zz = load_data_at_certain_t(field_dir + "pi2-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # tx_c, ty_c, tz_c = pic_xx/ni_c, pic_yy/ni_c, pic_zz/ni_c
        # tx_b, ty_b, tz_b = pib_xx / ni_b, pib_yy / ni_b, pib_zz / ni_b
        # uix_total = (ni_c * uix_c + ni_b * uix_b) / (ni_c + ni_b)
        # uiy_total = (ni_c * uiy_c + ni_b * uiy_b) / (ni_c + ni_b)
        # uiz_total = (ni_c * uiz_c + ni_b * uiz_b) / (ni_c + ni_b)
        Ay = load_data_at_certain_t(field_dir + "Ay.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # p_total = (pic_xx+pic_zz+pic_yy+pib_xx+pib_yy+pib_zz)*1/3+0.5*b_mag**2
        # n_total = ni_c+ni_b
        f_bz = (bz ** 2 - bx ** 2 - by ** 2) / 2

        fig, axes = plt.subplots(4, 2, figsize=(20, 15))
        # ax = axes[0]
        # ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), Ay.T, colors="k", levels=40)
        #
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(ni_c.T),
        #                      cmap="jet")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        # ax.set_ylabel("z", fontsize=20)
        # ax.set_title(r"$n_c$", fontsize=20)
        # ax = axes[1]
        # ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), Ay.T, colors="k", levels=40)
        #
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(ni_b.T),
        #                      cmap="jet")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        # ax.set_ylabel("z", fontsize=20)
        # ax.set_title(r"$n_b$", fontsize=20)
        # plt.show()

        ax = axes[0][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz),
                             smooth_matrix(mat_shift(ni_b).T),
                             cmap="jet", vmin=0.2, vmax=1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$n_b$", fontsize=20)
        ax = axes[0][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(ni_c).T),
                             cmap="jet", vmin=0.2, vmax=1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$n_c$", fontsize=20)
        ax = axes[1][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz),
                             smooth_matrix(mat_shift(uix_b).T),
                             cmap="jet", vmin=0, vmax=1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$u_{x,b}$", fontsize=20)
        ax = axes[1][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=80)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz),
                             smooth_matrix(mat_shift(uix_c).T),
                             cmap="jet", vmin=-0.5, vmax=0.5)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$u_{x,c}$", fontsize=20)
        ax = axes[2][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(uiy_b).T),
                             cmap="jet", vmin=0, vmax=1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$u_{y,b}$", fontsize=20)
        ax = axes[2][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=80)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(uiy_c).T),
                             cmap="jet", vmin=0.0, vmax=1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$u_{y,c}$", fontsize=20)
        ax = axes[3][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(uiz_b).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$u_{b,z}$", fontsize=20)
        ax = axes[3][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(uiz_c).T),
                             cmap="bwr", vmin=-0.1, vmax=0.1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$u_{c,z}$", fontsize=20)
        plt.suptitle(rf"$t={epoch}\Omega_{{ci}}^{{-1}}$", fontsize=25)
        # cold_x = ex+uiy_total*bz-uiz_total*by
        # cold_y = ey + uiz_total * bx - uix_total * bz
        # cold_z = ez + uix_total * by - uiy_total * bx
        # pclr = axes.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(cold_z.T),
        #                        cmap="bwr", vmin=-0.05, vmax=0.05)
        # plt.show()
        # cbar = plt.colorbar(pclr, ax=axes)
        # axes.set_ylim([-10, 10])
        # axes.set_xlabel("x", fontsize=20)
        # axes.set_ylabel("z", fontsize=20)
        # axes.set_title(r"$(\vec{E}+\vec{U}\times\vec{B})_z$", fontsize=20)
        figs_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_figs/{field_plot}s_{run_case_index}/"
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)
        plt.savefig(figs_dir + f"{field_plot}_{epoch}.png")
        plt.close(fig)
    # plt.close()
    # plt.scatter(400, 0, c="k")
    # plt.scatter(400, 2, c="k")
    # plt.scatter(400, 4, c="k")
    # plt.scatter(256, 0, c="k")
    # plt.scatter(256, 2, c="k")
    # plt.scatter(256, 4, c="k")
    #%%
    """
        PLOT THE FROZEN CONDITION
        """
    epoch = 50
    run_case_index = "57"
    plot_particle_tracking = False
    nb = 0.2
    field_plot = "frozen_total"
    field_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data/field_data_{run_case_index}/"
    # field_dir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/field_data/field_data_{run_case_index}/"
    nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
    Lx, Lz = int(loadinfo(field_dir)[3]), int(loadinfo(field_dir)[5])
    X, Z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz))
    hx, hz = Lx / nx, Lz / nz
    ni_c0 = load_data_at_certain_t(field_dir + "ni.gda", i_t=0, num_dim1=nx, num_dim2=ny, num_dim3=nz)
    Ay_total = load_data(field_dir + 'Ay.gda', num_dim1=nx, num_dim2=nz, num_dim3=2)
    Ay_0 = Ay_total[:, :, 0]
    # plt.plot(Ay_total[511, 127, :]-Ay_total[850, 127, :])
    # # plt.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), -Ay_total[:,:,99].T, cmap='jet')
    # # plt.colorbar()
    # plt.show()
    # #%%
    for epoch in range(50):
        print(epoch)
        bx = load_data_at_certain_t(field_dir + "bx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        by = load_data_at_certain_t(field_dir + "by.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        bz = load_data_at_certain_t(field_dir + "bz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # bx_shift=np.zeros_like(bx)
        # bx_shift[0:nx//2, :], bx_shift = bx[nx//2:, :]

        b_mag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        ex = load_data_at_certain_t(field_dir + "ex.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ey = load_data_at_certain_t(field_dir + "ey.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ez = load_data_at_certain_t(field_dir + "ez.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ni_c = load_data_at_certain_t(field_dir + "ni.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # ni_b = load_data_at_certain_t(field_dir + "ni2.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uix_c = load_data_at_certain_t(field_dir + "uix.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_c = load_data_at_certain_t(field_dir + "uiy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_c = load_data_at_certain_t(field_dir + "uiz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uix_b = load_data_at_certain_t(field_dir + "ui2x.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_b = load_data_at_certain_t(field_dir + "ui2y.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_b = load_data_at_certain_t(field_dir + "ui2z.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_xx = load_data_at_certain_t(field_dir + "pi-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_yy = load_data_at_certain_t(field_dir + "pi-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_zz = load_data_at_certain_t(field_dir + "pi-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        Jx, Jy, Jz = calculate_current_density(bx, by, bz, 0.5, 1, 0.25)
        ez_hall = -Jy * bx / ni_c
        ez_pressure = -np.gradient((pic_xx + pic_yy + pic_zz) / 3, axis=1) * 4 / ni_c
        ez_ion = uiy_c * bx
        ez_calc = ez_hall + ez_pressure + ez_ion
        ez_hall_theory = -np.sinh(Z.T) / np.cosh(Z.T) ** 3 / (nb + 1 / np.cosh(Z.T) ** 2)
        ez_pressure_theory = 0.5 * np.sinh(Z.T) / np.cosh(Z.T) ** 3 / (nb + 1 / np.cosh(Z.T) ** 2)
        uiy_theory = 0.5 / np.cosh(Z.T) ** 2 / (nb + 1 / np.cosh(Z.T) ** 2)
        ez_ion_theory = uiy_theory * np.tanh(Z.T)
        ez_theoty = ez_ion_theory + ez_hall_theory + ez_pressure_theory
        cold_x = ex+uiy_c*bz-uiz_c*by
        cold_y = ey + uiz_c * bx - uix_c * bz
        cold_z = ez + uix_c * by - uiy_c * bx
        # pib_xx = load_data_at_certain_t(field_dir + "pi2-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # pib_yy = load_data_at_certain_t(field_dir + "pi2-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # pib_zz = load_data_at_certain_t(field_dir + "pi2-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # tx_c, ty_c, tz_c = pic_xx/ni_c, pic_yy/ni_c, pic_zz/ni_c
        # tx_b, ty_b, tz_b = pib_xx / ni_b, pib_yy / ni_b, pib_zz / ni_b
        # uix_total = (ni_c * uix_c + ni_b * uix_b) / (ni_c + ni_b)
        # uiy_total = (ni_c * uiy_c + ni_b * uiy_b) / (ni_c + ni_b)
        # uiz_total = (ni_c * uiz_c + ni_b * uiz_b) / (ni_c + ni_b)
        Ay = load_data_at_certain_t(field_dir + "Ay.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # p_total = (pic_xx+pic_zz+pic_yy+pib_xx+pib_yy+pib_zz)*1/3+0.5*b_mag**2
        # n_total = ni_c+ni_b
        f_bz = (bz ** 2 - bx ** 2 - by ** 2) / 2

        fig, axes = plt.subplots(3, 2, figsize=(20, 10))
        # ax = axes[0]
        # ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), Ay.T, colors="k", levels=40)
        #
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(ni_c.T),
        #                      cmap="jet")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        # ax.set_ylabel("z", fontsize=20)
        # ax.set_title(r"$n_c$", fontsize=20)
        # ax = axes[1]
        # ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), Ay.T, colors="k", levels=40)
        #
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(ni_b.T),
        #                      cmap="jet")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        # ax.set_xlabel("x", fontsize=20)
        # ax.set_ylabel("z", fontsize=20)
        # ax.set_title(r"$n_b$", fontsize=20)
        # plt.show()

        ax = axes[0][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz),
                             smooth_matrix(mat_shift(cold_x).T),
                             cmap="bwr", vmin=-0.1, vmax=0.1)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$\Delta n$", fontsize=20)
        ax = axes[0][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix((mat_shift(ex)).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$E_x$", fontsize=20)
        ax = axes[1][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(cold_y).T),
                             cmap="bwr", vmin=-0.2, vmax=0.2)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$B_y$", fontsize=20)
        ax = axes[1][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=80)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(ey).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$E_y$", fontsize=20)
        ax = axes[2][0]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(cold_z).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$B_z$", fontsize=20)
        ax = axes[2][1]
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), mat_shift(Ay).T, colors="k", levels=40)

        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(mat_shift(ez).T),
                             cmap="bwr", vmin=-0.05, vmax=0.05)
        cbar = plt.colorbar(pclr, ax=ax)
        # cbar.set_label("Bx", fontsize=15)
        # ax.set_ylim([-10, 10])
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        ax.set_title(r"$E_z$", fontsize=20)
        plt.suptitle(rf"$t={epoch}\Omega_{{ci}}^{{-1}}$", fontsize=25)

        # pclr = axes.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(cold_z.T),
        #                        cmap="bwr", vmin=-0.05, vmax=0.05)
        # plt.show()
        # cbar = plt.colorbar(pclr, ax=axes)
        # axes.set_ylim([-10, 10])
        # axes.set_xlabel("x", fontsize=20)
        # axes.set_ylabel("z", fontsize=20)
        # axes.set_title(r"$(\vec{E}+\vec{U}\times\vec{B})_z$", fontsize=20)
        figs_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_figs/{field_plot}s_{run_case_index}/"
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)
        plt.savefig(figs_dir + f"{field_plot}_{epoch}.png")
        plt.close(fig)
    # plt.close()
    # plt.scatter(400, 0, c="k")
    # plt.scatter(400, 2, c="k")
    # plt.scatter(400, 4, c="k")
    # plt.scatter(256, 0, c="k")
    # plt.scatter(256, 2, c="k")
    # plt.scatter(256, 4, c="k")

    #%%

    field_dir_1 = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data/field_data_56/"
    field_dir_2 = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data/field_data_57/"
    ey_total_1 = load_data(field_dir_1 + "uix.gda", num_dim1=nx, num_dim2=nz, num_dim3=50)
    ey_total_2 = load_data(field_dir_1 + "ui2x.gda", num_dim1=nx, num_dim2=nz, num_dim3=50)
    # plt.plot(mat_shift(ey_total_1)[:, nz//2, 25])
    plt.plot(mat_shift(ey_total_2)[:, nz // 2, 25]-mat_shift(ey_total_1)[:, nz//2, 25])
    plt.plot(mat_shift(ey_total_2)[:, nz // 2, 0] - mat_shift(ey_total_1)[:, nz // 2, 0])
    plt.show()
    #%%
    ey_total = load_data(field_dir+"ey.gda", num_dim1=nx, num_dim2=nz, num_dim3=100)
    Ay_total = load_data(field_dir + "ey.gda", num_dim1=nx, num_dim2=nz, num_dim3=100)
    # plt.plot(ey_total[0, nz//2, :])
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(-Ay_total[nx//2, nz//2, :]+Ay_total[0, nz//2, :])
    plt.xlabel(r"$t[\Omega_{ci}^{-1}]$", fontsize=15)
    plt.ylabel(r"$\Delta \psi$", fontsize=15)
    plt.subplot(122)
    plt.plot(ey_total[0, nz // 2, :])
    plt.xlabel(r"$t[\Omega_{ci}^{-1}]$", fontsize=15)
    plt.ylabel(r"$E_y$", fontsize=15)
    plt.show()
    #%%
    epoch = 20
    run_case_index = 36
    plot_particle_tracking = False
    field_plot = "T"
    field_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data/field_data_{run_case_index}/"
    # field_dir = f"D:/Research/Codes/Hybrid-vpic/data_ip_shock/field_data/field_data_{run_case_index}/"
    nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
    Lx, Lz = 768, 64
    hx, hz = Lx / nx, Lz / nz

    for epoch in range(100):
        print(epoch)
        bx = load_data_at_certain_t(field_dir + "bx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        by = load_data_at_certain_t(field_dir + "by.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        bz = load_data_at_certain_t(field_dir + "bz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        b_mag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        ex = load_data_at_certain_t(field_dir + "ex.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ey = load_data_at_certain_t(field_dir + "ey.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ez = load_data_at_certain_t(field_dir + "ez.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)

        uix_c = load_data_at_certain_t(field_dir + "uix.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_c = load_data_at_certain_t(field_dir + "uiy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_c = load_data_at_certain_t(field_dir + "uiz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uix_b = load_data_at_certain_t(field_dir + "ui2x.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiy_b = load_data_at_certain_t(field_dir + "ui2y.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        uiz_b = load_data_at_certain_t(field_dir + "ui2z.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_xx = load_data_at_certain_t(field_dir + "pi-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_yy = load_data_at_certain_t(field_dir + "pi-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pic_zz = load_data_at_certain_t(field_dir + "pi-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)

        pib_xx = load_data_at_certain_t(field_dir + "pi2-xx.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pib_yy = load_data_at_certain_t(field_dir + "pi2-yy.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        pib_zz = load_data_at_certain_t(field_dir + "pi2-zz.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        p_total = (pib_xx+pib_yy+pib_zz+pic_zz+pic_yy+pic_xx)/8+b_mag**2/2
        Ay = load_data_at_certain_t(field_dir + "Ay.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        Jx, Jy, Jz = calculate_current_density(bx, by, bz, 1, 1, 1)
        ni_c = load_data_at_certain_t(field_dir + "ni.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        ni_b = load_data_at_certain_t(field_dir + "ni2.gda", i_t=epoch, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        tx_c, ty_c, tz_c = pic_xx / ni_c, pic_yy / ni_c, pic_zz / ni_c
        tx_b, ty_b, tz_b = np.nan_to_num(pib_xx / ni_b, nan=0), np.nan_to_num(pib_yy / ni_b, nan=0), np.nan_to_num(pib_zz / ni_b, nan=0)
        uix_total = (ni_c * uix_c + ni_b * uix_b) / (ni_c + ni_b)
        uiy_total = (ni_c * uiy_c + ni_b * uiy_b) / (ni_c + ni_b)
        uiz_total = (ni_c * uiz_c + ni_b * uiz_b) / (ni_c + ni_b)
        tx_total = (ni_c * tx_c + ni_b * tx_b) / (ni_c + ni_b)
        ty_total = (ni_c * ty_c + ni_b * ty_b) / (ni_c + ni_b)
        tz_total = (ni_c * tz_c + ni_b * tz_b) / (ni_c + ni_b)
        t_total = tx_total+ty_total+tz_total
        ex_prime = ex + uiy_total * bz - uiz_total * by
        ey_prime = ey + uiz_total * bx - uix_total * bz
        ez_prime = ez + uix_total * by - uiy_total * bx
        bx_prime = bx - uiy_total * ez + uiz_total * ey
        by_prime = by - uiz_total * ex + uix_total * ez
        bz_prime = bz - uix_total * ey + uiy_total * ex
        vc = calculate_vc(1, bx, by, bz, hx, hz)
        vg = calculate_gradient_drift_velocity(bx, by, bz, b_mag, 0.5, hx, hz)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        (grad_x, grad_z) = np.gradient(ez)
        ax.contour(np.linspace(0, Lx, nx), np.linspace(-Lz/2, Lz/2, nz), Ay.T, colors="k", levels=40)
        # # ax = axes[0]
        # ax.scatter(400, 0, c="k")
        # ax.scatter(400, 2, c="k")
        # ax.scatter(400, 4, c="k")
        pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz / 2, Lz / 2, nz), smooth_matrix(t_total.T),
                             cmap="jet", vmin=0, vmax=0.6)
        c1_plot = np.array([3, 40, 2], dtype=int)
        b_plot = np.array([1, 2, 3, 0], dtype=int)
        if plot_particle_tracking:
            for index in c1_plot:
                if index == c1_plot[0]:
                    ax.plot(x_c[index_c_1[index], :200], z_c[index_c_1[index], :200], c="k", label="core", alpha=0.5)
                else:
                    ax.plot(x_c[index_c_1[index], :200], z_c[index_c_1[index], :200], c="k", alpha=0.5)
                ax.scatter(x_c[index_c_1[index], 0], z_c[index_c_1[index], 0], c="k", alpha=0.5, marker="*", s=160)
            for index in b_plot:
                if index == b_plot[0]:
                    ax.plot(x_b[index_b[index], :100], z_b[index_b[index], :100], c="b", label="beam", alpha=0.5)
                else:
                    ax.plot(x_b[index_b[index], :100], z_b[index_b[index], :100], c="b", alpha=0.5)
                ax.scatter(x_b[index_b[index], 0], z_b[index_b[index], 0], c="b", alpha=0.5, marker="*", s=160)

            ax.legend(fontsize=15)
        # point = ax.scatter(x_c[iptl_1, :], z_c[iptl_1, :], c=uy_c[iptl_1, :], vmin=-1, vmax=1, cmap="bwr")
        # point = ax.scatter(x_b[iptl_b, :], z_b[iptl_b, :], c=uy_b[iptl_b, :], vmin=-1, vmax=1, cmap="bwr")
        # ax.scatter(x_c[iptl_1, 0], z_c[iptl_1, 0], marker="x", s=80, c="k")
        # ax.scatter(x_b[iptl_b, 0], z_b[iptl_b, 0], marker="x", s=80, c="k")
        cbar = plt.colorbar(pclr, ax=ax)
        cbar.set_label(f"{field_plot}", fontsize=15)
        # cbar.set_label(r"$\frac{Ti_{b,x}}{Ti_{b,y}}-1$", fontsize=18)
        # cbar.set_label(r"$\log{\frac{T_{c, x}}{T_{b,x}}}$", fontsize=20)
        # ax.set_ylim([-5, 5])
        # ax.set_xlim([x_c[iptl_1, :].min()-35, x_c[iptl_1, :].max()+5])
        # ax.arrow(360, 2, 5, -2, head_width=0.1, fc="k", linewidth=2)
        # ax.text(360, 2.1, "beam", fontsize=17)
        # ax.arrow(400, -2, -5, 2, head_width=0.1, fc="k", linewidth=2)
        # ax.text(400, -2.3, "core", fontsize=17)
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("z", fontsize=20)
        # cbar.set_label(r"$v_y$", fontsize=19)
        # ax.set_xlabel("x", fontsize=19)
        # ax.set_ylabel("z", fontsize=19)
        # ax = axes[1]
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz/2, Lz/2, nz), smooth_matrix(by).T, cmap="bwr", vmin=0, vmax=1)
        # # plt.plot(x_b[iptl_b, :], z_b[iptl_b, :], c="k")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label(f"{field_plot}", fontsize=15)
        # cbar.set_label(r"$B_y$", fontsize=19)
        # ax.set_xlabel("x", fontsize=19)
        # ax.set_ylabel("z", fontsize=19)
        # ax = axes[2]
        # pclr = ax.pcolormesh(np.linspace(0, Lx, nx), np.linspace(-Lz/2, Lz/2, nz), smooth_matrix(bz).T, cmap="bwr", vmin=-0.2, vmax=0.2)
        # # plt.plot(x_b[iptl_b, :], z_b[iptl_b, :], c="k")
        # cbar = plt.colorbar(pclr, ax=ax)
        # # cbar.set_label(f"{field_plot}", fontsize=15)
        # cbar.set_label(r"$B_z$", fontsize=19)
        # ax.set_xlabel("x", fontsize=19)
        # ax.set_ylabel("z", fontsize=19)
        plt.ylim([-10, 10])
        # plt.xlim([300, 512])
        ax.set_title(f"epoch={epoch}, nb/nc=0.0", fontsize=16)

        # plt.show()

        figs_dir = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_figs/{field_plot}s_{run_case_index}/"
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)
        plt.savefig(figs_dir + f"{field_plot}_{epoch}.png")
        plt.close(fig)
    # plt.show()
    # %%
    v_2d = np.zeros((len(v_para[condition]), 2))
    v_2d[:, 0] = v_para[condition]
    v_2d[:, 1] = p.uy[condition]
    # %%
    counts_mat, xedges, zedges = np.histogram2d(v_para[condition], p.uy[condition], bins=[80, 80],
                                                range=[[np.min(v_para), np.max(v_para)], [np.min(p.uy), np.max(p.uy)]])
    # %%
    skewness, kurtosis = calculate_2d_skewness_kurtosis(v_2d)
    print([skewness, kurtosis])
    # %%
    plt.pcolormesh(np.linspace(np.min(v_para), np.max(v_para), 80), np.linspace(np.min(p.uy), np.max(p.uy), 80),
                   np.log10(counts_mat).T
                   , vmin=0, vmax=1.5, cmap="jet")
    plt.xlim([-5, 5])
    plt.ylim([-4, 5])
    cbar = plt.colorbar()
    cbar.set_label(r"$\log_{10}$(counts)", fontsize=15)
    v_para_max_index, vy_max_index = np.unravel_index(np.argmax(counts_mat), counts_mat.shape)
    print(xedges[v_para_max_index], zedges[vy_max_index])
    plt.arrow(xedges[v_para_max_index], zedges[vy_max_index], bx[x_center_idx, z_center_idx],
              by[x_center_idx, z_center_idx], head_width=0.1, fc="k")
    if by[x_center_idx, z_center_idx] > 0:
        plt.text(xedges[v_para_max_index] + bx[x_center_idx, z_center_idx],
                 zedges[vy_max_index] + by[x_center_idx, z_center_idx] + 0.25, r"$\mathbf{B_0}$", fontsize=13)
    else:
        plt.text(xedges[v_para_max_index] + bx[x_center_idx, z_center_idx],
                 zedges[vy_max_index] + by[x_center_idx, z_center_idx] - 0.25, r"$\mathbf{B_0}$", fontsize=13)
    # latex_txt_1 = rf"$\xi_{{c}}={xi_c:.3f}$"
    # latex_txt_2 = rf"$v_{{x,c}}={vx_c:.3f}, v_{{y,c}}={vy_c:.3f}$"
    # latex_txt_3 = rf"$v_{{x,b}}={vx_b:.3f}, v_{{y,b}}={vy_b:.3f}$"
    # latex_txt_4 = rf"$T_{{x,c}}={Tx_c:.3f}, T_{{y,c}}={Ty_c:.3f}$"
    # latex_txt_5 = rf"$T_{{x,b}}={Tx_b:.3f}, T_{{y,b}}={Ty_b:.3f}$"
    # plt.text(-4.5, 4, latex_txt_1)
    # plt.text(-4.5, 3.7, latex_txt_2)
    # plt.text(-4.5, 3.4, latex_txt_3)
    # plt.text(-4.5, 3.1, latex_txt_4)
    # plt.text(-4.5, 2.8, latex_txt_5)
    plt.title(rf"{x_left}<x<{x_right}, {z_bottom}<z<{z_top}, $\Omega_i t$={step // 200}", fontsize=15)
    plt.xlabel(r"$v_x$", fontsize=15)
    plt.ylabel(r"$v_y$", fontsize=15)
    plt.show()
    # %%

    # %%
    E_mat = np.zeros((256, 64))
    for i in range(256):
        print(i)
        for j in range(-32, 32, 1):
            condition = (p.x >= i) & (p.x < i + 1) & (p.z >= j) & (p.z < j + 1)
            E_mat[i, j] = p.E[condition].mean()
    # %%
    plt.pcolormesh(range(256), range(64), E_mat.T)
    plt.colorbar()
    plt.show()
    # %%
    condition = p.x
    # %%

    # plt.pcolormesh(range(256), range(64), bx.T)
    plt.plot(b_mag[:, 31])
    plt.ylim([1, 1.5])
    plt.ylabel("|B|", fontsize=15)
    plt.xlabel("x", fontsize=15)
    # plt.colorbar()
    plt.show()
