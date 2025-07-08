import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import shutil
import matplotlib.colors as colors
from mpi4py import MPI

# 设置Matplotlib后端为非交互式，避免多进程冲突
plt.switch_backend('Agg')

# 初始化MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # 当前进程ID
size = comm.Get_size()  # 总进程数

# 数据读取函数
def loadinfo(dir):
    fstr = dir + "info"
    with open(fstr, "rb") as fd:
        infocontent = fd.read()
    arr = struct.unpack("fIIIffffff", infocontent[:40])
    infoarr = np.zeros(6)
    infoarr[0] = arr[1]  # nx
    infoarr[1] = arr[2]  # ny (实际应为nz，根据二维半模拟调整)
    infoarr[2] = arr[3]  # nz
    infoarr[3] = arr[6]  # Lx
    infoarr[4] = arr[7]  # Ly (未使用)
    infoarr[5] = arr[8]  # Lz
    return infoarr

def loadSlice(dir, q, sl, nx, ny):
    fstr = dir + q + ".gda"
    with open(fstr, "rb") as fd:
        fd.seek(4 * sl * nx * ny)  # 跳过前sl个切片
        arr = np.fromfile(fd, dtype=np.float32, count=nx*ny)
    arr = arr.reshape((ny, nx)).T  # 转置为(nx, ny)
    return arr

def compute_current_density(Bx, By, Bz, dx=0.25, dy=1.0, dz=0.25, mu0=1):
    """
    计算电流密度 J = (Jx, Jy, Jz) 通过安培定律 ∇×B = μ₀J
    输入：
        Bx, By, Bz : 形状为 (nx, ny, nz) 的磁场分量
        dx, dy, dz  : 网格步长（默认假设为1.0）
        mu0         : 真空磁导率（默认值 4π×1e-7 H/m）
    输出：
        Jx, Jy, Jz : 电流密度分量，形状与输入相同
    """
    # 检查各维度长度
    axes = []
    deltas = []
    for i, (axis_size, delta) in enumerate(zip(Bx.shape, [dx, dy, dz])):
        if axis_size > 1:
            axes.append(i)
            deltas.append(delta)
    
    # 计算磁场分量的梯度（仅处理维度>1的轴）
    gradients_Bx = np.gradient(Bx, *deltas, axis=axes)
    gradients_By = np.gradient(By, *deltas, axis=axes)
    gradients_Bz = np.gradient(Bz, *deltas, axis=axes)
    
    # 初始化导数为零（匹配原始维度）
    dBx_dx = np.zeros_like(Bx)
    dBx_dy = np.zeros_like(Bx)
    dBx_dz = np.zeros_like(Bx)
    
    dBy_dx = np.zeros_like(By)
    dBy_dy = np.zeros_like(By)
    dBy_dz = np.zeros_like(By)
    
    dBz_dx = np.zeros_like(Bz)
    dBz_dy = np.zeros_like(Bz)
    dBz_dz = np.zeros_like(Bz)
    
    # 填充有效梯度值
    for i, axis in enumerate(axes):
        if axis == 0:  # x方向
            dBx_dx = gradients_Bx[i]
            dBy_dx = gradients_By[i]
            dBz_dx = gradients_Bz[i]
        elif axis == 1:  # y方向
            dBx_dy = gradients_Bx[i]
            dBy_dy = gradients_By[i]
            dBz_dy = gradients_Bz[i]
        elif axis == 2:  # z方向
            dBx_dz = gradients_Bx[i]
            dBy_dz = gradients_By[i]
            dBz_dz = gradients_Bz[i]
    
    # 计算旋度 ∇×B
    Jx = (dBz_dy - dBy_dz) / mu0
    Jy = (dBx_dz - dBz_dx) / mu0
    Jz = (dBy_dx - dBx_dy) / mu0
    
    return Jx, Jy, Jz

# 新增函数：扩展数据并应用周期性边界条件
def extend_with_periodic_boundaries(data, n_extra=4):
    """
    在数据四周各扩展n_extra个网格，并用对面边界的值填充
    
    参数:
        data: 原始数据，形状为(nx, ny)
        n_extra: 每个边界扩展的网格数
    
    返回:
        扩展后的数据，形状为(nx+2*n_extra, ny+2*n_extra)
    """
    nx, ny = data.shape
    
    # 创建扩展后的数组
    extended_data = np.zeros((nx + 2*n_extra, ny + 2*n_extra))
    
    # 填充原始数据到中心区域
    extended_data[n_extra:n_extra+nx, n_extra:n_extra+ny] = data
    
    # 填充左右边界
    extended_data[:n_extra, n_extra:n_extra+ny] = data[-n_extra:, :]  # 左边界 = 右边界
    extended_data[n_extra+nx:, n_extra:n_extra+ny] = data[:n_extra, :]  # 右边界 = 左边界
    
    # 填充上下边界
    extended_data[n_extra:n_extra+nx, :n_extra] = data[:, -n_extra:]  # 上边界 = 下边界
    extended_data[n_extra:n_extra+nx, n_extra+ny:] = data[:, :n_extra]  # 下边界 = 上边界
    
    # 填充四个角
    extended_data[:n_extra, :n_extra] = data[-n_extra:, -n_extra:]  # 左上角 = 右下角
    extended_data[:n_extra, n_extra+ny:] = data[-n_extra:, :n_extra]  # 右上角 = 左下角
    extended_data[n_extra+nx:, :n_extra] = data[:n_extra, -n_extra:]  # 左下角 = 右上角
    extended_data[n_extra+nx:, n_extra+ny:] = data[:n_extra, :n_extra]  # 右下角 = 左上角
    
    return extended_data

# 主逻辑
if __name__ == "__main__":
    # 参数设置
    dir = "./data/"
    infoarr = loadinfo(dir)
    nx = int(infoarr[0])
    nz = int(infoarr[2])  # 注意：这里使用infoarr[2]作为nz
    Lx = infoarr[3]
    Lz = infoarr[5]
    dx = Lx / nx
    dz = Lz / nz
    dt = 1  # 时间间隔
    nt = 800  # 时间切片数
    n_extra = 4  # 扩展边界的网格数
    angle = 85  # 磁场方向角度

    # 创建输出目录（仅主进程执行，避免多进程重复创建）
    if rank == 0:
        for d in ["B_fluct_plots", "E_fluct_plots", "Ui_plots", "energy_conversion_plots", "figgs", "figss",'tem','dens','energy']:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        print("输出目录创建完成")

    # 等待主进程创建目录完成
    comm.Barrier()

    # 主进程计算全局统计量（vmin/vmax），并广播给所有进程
    if rank == 0:
        print("计算全局统计量（抽样10%的切片）...")
        # 定义需要计算统计量的字段
        fields = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'ni', 'uix', 'uiy', 'uiz']
        stats = {}
        
        # 抽样10%的切片计算统计量（平衡精度和速度）
        sample_slices = np.linspace(0, nt-1, int(nt*0.1), dtype=int)
        
        # 计算各字段的vmin和vmax
        for field in fields:
            data_sample = np.array([loadSlice(dir, field, sl, nx, nz) for sl in sample_slices])
            stats[f'{field}_vmin'] = np.percentile(data_sample, 5)
            stats[f'{field}_vmax'] = np.percentile(data_sample, 95)
            print(f"{field} vmin: {stats[f'{field}_vmin']:.4f}, vmax: {stats[f'{field}_vmax']:.4f}")
        
        # 计算温度相关统计量
        # 先计算单位磁场向量
        Bx_sample = np.array([loadSlice(dir, 'Bx', sl, nx, nz) for sl in sample_slices])
        By_sample = np.array([loadSlice(dir, 'By', sl, nx, nz) for sl in sample_slices])
        Bz_sample = np.array([loadSlice(dir, 'Bz', sl, nx, nz) for sl in sample_slices])
        
        # 计算磁场幅值
        b_mag = np.sqrt(Bx_sample**2 + By_sample**2 + Bz_sample**2)
        # 避免除以零，添加极小值
        b_mag = np.where(b_mag == 0, 1e-12, b_mag)
        
        # 计算单位磁场向量
        bx_unit = Bx_sample / b_mag
        by_unit = By_sample / b_mag
        bz_unit = Bz_sample / b_mag
        
        # 读取压力张量分量
        pi_xx_sample = np.array([loadSlice(dir, 'pi-xx', sl, nx, nz) for sl in sample_slices])
        pi_yy_sample = np.array([loadSlice(dir, 'pi-yy', sl, nx, nz) for sl in sample_slices])
        pi_zz_sample = np.array([loadSlice(dir, 'pi-zz', sl, nx, nz) for sl in sample_slices])
        pi_xy_sample = np.array([loadSlice(dir, 'pi-xy', sl, nx, nz) for sl in sample_slices])
        pi_xz_sample = np.array([loadSlice(dir, 'pi-xz', sl, nx, nz) for sl in sample_slices])
        pi_yz_sample = np.array([loadSlice(dir, 'pi-yz', sl, nx, nz) for sl in sample_slices])
        
        # 计算平行和垂直温度
        T_parallel = (bx_unit**2 * pi_xx_sample +
                      by_unit**2 * pi_yy_sample +
                      bz_unit**2 * pi_zz_sample +
                      2 * bx_unit * by_unit * pi_xy_sample +
                      2 * bx_unit * bz_unit * pi_xz_sample +
                      2 * by_unit * bz_unit * pi_yz_sample)
        Tr_pi = pi_xx_sample + pi_yy_sample + pi_zz_sample
        T_perp = (Tr_pi - T_parallel) / 2
        
        # 计算温度统计量
        stats['T_perp_vmin'] = np.percentile(T_perp, 5)
        stats['T_perp_vmax'] = np.percentile(T_perp, 95)
        stats['T_para_vmin'] = np.percentile(T_parallel, 5)
        stats['T_para_vmax'] = np.percentile(T_parallel, 95)
        print(f"T_perp vmin: {stats['T_perp_vmin']:.4f}, vmax: {stats['T_perp_vmax']:.4f}")
        print(f"T_parallel vmin: {stats['T_para_vmin']:.4f}, vmax: {stats['T_para_vmax']:.4f}")
    else:
        stats = None

    # 广播统计量到所有进程
    stats = comm.bcast(stats, root=0)
    print(f"进程 {rank} 收到统计量数据")
    
    # 解析统计量
    bx_vmin, bx_vmax = stats['Bx_vmin'], stats['Bx_vmax']
    by_vmin, by_vmax = stats['By_vmin'], stats['By_vmax']
    bz_vmin, bz_vmax = stats['Bz_vmin'], stats['Bz_vmax']
    ex_vmin, ex_vmax = stats['Ex_vmin'], stats['Ex_vmax']
    ey_vmin, ey_vmax = stats['Ey_vmin'], stats['Ey_vmax']
    ez_vmin, ez_vmax = stats['Ez_vmin'], stats['Ez_vmax']
    uix_vmin, uix_vmax = stats['uix_vmin'], stats['uix_vmax']
    uiy_vmin, uiy_vmax = stats['uiy_vmin'], stats['uiy_vmax']
    uiz_vmin, uiz_vmax = stats['uiz_vmin'], stats['uiz_vmax']
    ni_vmin, ni_vmax = stats['ni_vmin'], stats['ni_vmax']
    T_perp_vmin, T_perp_vmax = stats['T_perp_vmin'], stats['T_perp_vmax']
    T_para_vmin, T_para_vmax = stats['T_para_vmin'], stats['T_para_vmax']

    # 任务分配：每个进程处理一部分时间切片
    chunk_size = nt // size
    start = rank * chunk_size
    end = start + chunk_size if rank != size - 1 else nt  # 最后一个进程处理剩余切片
    print(f"进程 {rank} 负责处理切片 {start} 到 {end-1}")

    # 并行处理分配的切片
    for sl in range(start, end):
        print(f"进程 {rank} 正在处理切片 {sl}/{nt-1}")

        # 读取当前切片的所有场数据
        Bx = loadSlice(dir, 'Bx', sl, nx, nz)
        By = loadSlice(dir, 'By', sl, nx, nz)
        Bz = loadSlice(dir, 'Bz', sl, nx, nz)
        Ex = loadSlice(dir, 'Ex', sl, nx, nz)
        Ey = loadSlice(dir, 'Ey', sl, nx, nz)
        Ez = loadSlice(dir, 'Ez', sl, nx, nz)
        ni = loadSlice(dir, 'ni', sl, nx, nz)
        uix = loadSlice(dir, 'uix', sl, nx, nz)
        uiy = loadSlice(dir, 'uiy', sl, nx, nz)
        uiz = loadSlice(dir, 'uiz', sl, nx, nz)
        pi_xx = loadSlice(dir, 'pi-xx', sl, nx, nz)
        pi_yy = loadSlice(dir, 'pi-yy', sl, nx, nz)
        pi_zz = loadSlice(dir, 'pi-zz', sl, nx, nz)
        pi_xy = loadSlice(dir, 'pi-xy', sl, nx, nz)
        pi_xz = loadSlice(dir, 'pi-xz', sl, nx, nz)
        pi_yz = loadSlice(dir, 'pi-yz', sl, nx, nz)

        # 计算单位磁场向量
        b_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        # 避免除以零，添加极小值
        b_mag = np.where(b_mag == 0, 1e-12, b_mag)
        bx_unit = Bx / b_mag
        by_unit = By / b_mag
        bz_unit = Bz / b_mag

        # 计算温度
        T_parallel = (bx_unit**2 * pi_xx +
                      by_unit**2 * pi_yy +
                      bz_unit**2 * pi_zz +
                      2 * bx_unit * by_unit * pi_xy +
                      2 * bx_unit * bz_unit * pi_xz +
                      2 * by_unit * bz_unit * pi_yz)
        Tr_pi = pi_xx + pi_yy + pi_zz
        T_perp = (Tr_pi - T_parallel) / 2

        # 扩展数据（周期性边界）
        Bx_extended = extend_with_periodic_boundaries(Bx, n_extra)
        By_extended = extend_with_periodic_boundaries(By, n_extra)
        Bz_extended = extend_with_periodic_boundaries(Bz, n_extra)
        Ex_extended = extend_with_periodic_boundaries(Ex, n_extra)
        Ey_extended = extend_with_periodic_boundaries(Ey, n_extra)
        Ez_extended = extend_with_periodic_boundaries(Ez, n_extra)
        uix_extended = extend_with_periodic_boundaries(uix, n_extra)
        uiy_extended = extend_with_periodic_boundaries(uiy, n_extra)
        uiz_extended = extend_with_periodic_boundaries(uiz, n_extra)
        ni_extended = extend_with_periodic_boundaries(ni, n_extra)
        T_para_extended = extend_with_periodic_boundaries(T_parallel, n_extra)
        T_perp_extended = extend_with_periodic_boundaries(T_perp, n_extra)

        # 生成当前切片的所有图像
        # 1. 密度图
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        im0 = axes.imshow(ni.T, origin='lower', extent=[0, Lx, 0, Lz], cmap='RdBu_r',
                         vmin=ni_vmin, vmax=ni_vmax)
        axes.set_title(f'ni Distribution at time = {sl*dt:.1f} wci-1')
        fig.colorbar(im0, ax=axes)
        ylims = axes.get_xlim()
        y_vals = np.array(ylims)
        x_vals = 1 / np.tan(np.deg2rad(angle)) * y_vals
        line, = axes.plot(x_vals, y_vals, 'k-', linewidth=2)
        axes.legend([line], ['B0'])
        plt.tight_layout()
        plt.savefig(f'dens/dens_slice_{sl:04d}.png')
        plt.close()

        # 2. Ui分布图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        im0 = axes[0].imshow(uix_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=uix_vmin, vmax=uix_vmax)
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(f'Uix Distribution at time = {sl*dt:.1f} wci-1')
        
        im1 = axes[1].imshow(uiy_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=uiy_vmin, vmax=uiy_vmax)
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(f'Uiy Distribution at time = {sl*dt:.1f} wci-1')
        
        im2 = axes[2].imshow(uiz_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=uiz_vmin, vmax=uiz_vmax)
        fig.colorbar(im2, ax=axes[2])
        axes[2].set_title(f'Uiz Distribution at time = {sl*dt:.1f} wci-1')

        for ax in axes:
            ax.set_xlabel('X [di]')
            ax.set_ylabel('Z [di]')
            
            # 添加磁场方向线
            y_vals = np.linspace(0, Lz, 100)
            x_vals = y_vals / np.tan(np.deg2rad(angle))
            line, = ax.plot(x_vals, y_vals, 'k-', linewidth=2)
            ax.legend([line], ['B0'])
        
        plt.tight_layout()
        plt.savefig(f'Ui_plots/Ui_slice_{sl:04d}.png')
        plt.close()

        # 3. B场分布图
        fig, axes = plt.subplots(3, 2, figsize=(18, 10))
        im0 = axes[0,0].imshow(Bx_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r',vmin=bx_vmin, vmax=bx_vmax)
        axes[0,0].set_title(f'Bx Distribution at time = {sl*dt:.1f} wci-1')
        axes[0,0].set_xlabel('X [di]')
        axes[0,0].set_ylabel('Z [di]')
        fig.colorbar(im0, ax=axes[0,0])

        im1 = axes[1,0].imshow(By_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=by_vmin, vmax=by_vmax)
        axes[1,0].set_title(f'By Distribution at time = {sl*dt:.1f} wci-1')
        fig.colorbar(im1, ax=axes[1,0])
        axes[1,0].set_xlabel('X [di]')
        axes[1,0].set_ylabel('Z [di]')

        im2 = axes[2,0].imshow(Bz_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=bz_vmin, vmax=bz_vmax)
        axes[2,0].set_title(f'Bz Distribution at time = {sl*dt:.1f} wci-1')
        fig.colorbar(im2, ax=axes[2,0])
        axes[2,0].set_xlabel('X [di]')
        axes[2,0].set_ylabel('Z [di]')

        delta_bx_sq = (Bx_extended[:,:])**2
        delta_by_sq = (By_extended[:,:])**2
        delta_bz_sq = (Bz_extended[:,:])**2

        im3 = axes[0,1].imshow(delta_bx_sq.T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='jet')
        axes[0,1].set_title(f'Bx sq Distribution at time = {sl*dt:.1f} wci-1')
        fig.colorbar(im3, ax=axes[0,1])
        axes[0,1].set_xlabel('X [di]')
        axes[0,1].set_ylabel('Z [di]')

        im4 = axes[1,1].imshow(delta_by_sq.T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='jet')
        axes[1,1].set_title(f'By sq Distribution at time = {sl*dt:.1f} wci-1')
        fig.colorbar(im4, ax=axes[1,1])
        axes[1,1].set_xlabel('X [di]')
        axes[1,1].set_ylabel('Z [di]')

        im5 = axes[2,1].imshow(delta_bz_sq.T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='jet')
        axes[2,1].set_title(f'Bz sq Distribution at time = {sl*dt:.1f} wci-1')
        fig.colorbar(im5, ax=axes[2,1])
        axes[2,1].set_xlabel('X [di]')
        axes[2,1].set_ylabel('Z [di]')

        for ax in axes.flat:
            # 添加磁场方向线
            y_vals = np.linspace(0, Lz, 100)
            x_vals = y_vals / np.tan(np.deg2rad(angle))
            line, = ax.plot(x_vals, y_vals, 'k-', linewidth=2)
            ax.legend([line], ['B0'])

        plt.tight_layout()
        plt.savefig(f'B_fluct_plots/B_slice_{sl:04d}.png')
        plt.close()

        # 4. E场分布图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        im0 = axes[0].imshow(Ex_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=ex_vmin, vmax=ex_vmax)
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(f'Ex Distribution at time = {sl*dt:.1f} wci-1')
        
        im1 = axes[1].imshow(Ey_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=ey_vmin, vmax=ey_vmax)
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(f'Ey Distribution at time = {sl*dt:.1f} wci-1')
        
        im2 = axes[2].imshow(Ez_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r', vmin=ez_vmin, vmax=ez_vmax)
        fig.colorbar(im2, ax=axes[2])
        axes[2].set_title(f'Ez Distribution at time = {sl*dt:.1f} wci-1')
        
        # 绘制原始边界
        axes[2].axvline(x=0, color='k', linestyle='--', linewidth=1)
        axes[2].axvline(x=Lx, color='k', linestyle='--', linewidth=1)
        axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[2].axhline(y=Lz, color='k', linestyle='--', linewidth=1)

        for ax in axes:
            ax.set_xlabel('X [di]')
            ax.set_ylabel('Z [di]')
            
            # 添加磁场方向线
            y_vals = np.linspace(0, Lz, 100)
            x_vals = y_vals / np.tan(np.deg2rad(angle))
            line, = ax.plot(x_vals, y_vals, 'k-', linewidth=2)
            ax.legend([line], ['B0'])

        plt.tight_layout()
        plt.savefig(f'E_fluct_plots/E_slice_{sl:04d}.png')
        plt.close()

        # 5. 温度分布图
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        im0 = axes[0].imshow(T_para_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], vmin=T_para_vmin, vmax=T_para_vmax, cmap='RdBu_r')
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(f'T_parallel Distribution at time = {sl*dt} wci-1')

        im1 = axes[1].imshow(T_perp_extended[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], vmin=T_perp_vmin, vmax=T_perp_vmax, cmap='RdBu_r')
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(f'T_perp Distribution at time = {sl*dt} wci-1')

        for ax in axes:
            ax.set_xlabel('X [di]')
            ax.set_ylabel('Z [di]')
            
            # 添加磁场方向线
            y_vals = np.linspace(0, Lz, 100)
            x_vals = y_vals / np.tan(np.deg2rad(angle))
            line, = ax.plot(x_vals, y_vals, 'k-', linewidth=2)
            ax.legend([line], ['B0'])

        plt.tight_layout()
        plt.savefig(f'tem/temp_slice_{sl:04d}.png')
        plt.close()

        # 6. 能量分布图
        Eb_tot = (Bx_extended**2 + By_extended**2 + Bz_extended**2) / 2
        Ek_tot = (uix_extended**2 + uiy_extended**2 + uiz_extended**2) / 2
        E_th = 1.5 * ni_extended * (T_perp_extended * 2 + T_para_extended)
        E_tot = Eb_tot + Ek_tot + E_th

        fig, axes = plt.subplots(4, 1, figsize=(12, 18))
        im0 = axes[0].imshow(Eb_tot[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r')
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(f'mag energy Distribution at time = {sl*dt} wci-1')

        im1 = axes[1].imshow(Ek_tot[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r')
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(f'kinetic energy Distribution at time = {sl*dt} wci-1')

        im2 = axes[2].imshow(E_th[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r')
        fig.colorbar(im2, ax=axes[2])
        axes[2].set_title(f'thermal energy Distribution at time = {sl*dt} wci-1')

        im3 = axes[3].imshow(E_tot[:, :].T, origin='lower', extent=[-n_extra*dx, Lx+n_extra*dx, -n_extra*dz, Lz+n_extra*dz], cmap='RdBu_r')
        fig.colorbar(im3, ax=axes[3])
        axes[3].set_title(f'total energy Distribution at time = {sl*dt} wci-1')

        for ax in axes:
            ax.set_xlabel('X [di]')
            ax.set_ylabel('Z [di]')
            
            # 添加磁场方向线
            y_vals = np.linspace(0, Lz, 100)
            x_vals = y_vals / np.tan(np.deg2rad(angle))
            line, = ax.plot(x_vals, y_vals, 'k-', linewidth=2)
            ax.legend([line], ['B0'])

        plt.tight_layout()
        plt.savefig(f'energy/energy_slice_{sl:04d}.png')
        plt.close()

        # 保存每个切片的温度平均值（仅主进程收集所有数据后绘制）
        if rank == 0:
            T_perp_arr = np.zeros(nt)
            T_para_arr = np.zeros(nt)
            E_b = np.zeros(nt)
            E_k = np.zeros(nt)
            E_t = np.zeros(nt)
            E_total = np.zeros(nt)

            # 主进程需要读取所有切片的数据来计算时间演化
            if sl < nt:  # 防止越界
                T_perp_arr[sl] = np.mean(T_perp)
                T_para_arr[sl] = np.mean(T_parallel)
                E_b[sl] = np.mean(Eb_tot[:,:])
                E_k[sl] = np.mean(Ek_tot[:,:])
                E_t[sl] = np.mean(E_th[:,:])
                E_total[sl] = np.mean(E_tot[:,:])

    # 等待所有进程完成
    comm.Barrier()
    print(f"进程 {rank} 已完成所有任务")

    # 仅主进程绘制时间演化图
    if rank == 0:
        print("正在生成时间演化图...")
        
        # 收集所有进程的温度和能量数据
        for proc in range(1, size):
            if proc < (nt // chunk_size):  # 防止进程数超过需要的数量
                proc_start = proc * chunk_size
                proc_end = proc_start + chunk_size if proc != size - 1 else nt
                
                # 接收子进程的数据
                T_perp_data = comm.recv(source=proc, tag=100)
                T_para_data = comm.recv(source=proc, tag=101)
                E_b_data = comm.recv(source=proc, tag=102)
                E_k_data = comm.recv(source=proc, tag=103)
                E_t_data = comm.recv(source=proc, tag=104)
                E_total_data = comm.recv(source=proc, tag=105)
                
                # 更新主进程的数组
                T_perp_arr[proc_start:proc_end] = T_perp_data
                T_para_arr[proc_start:proc_end] = T_para_data
                E_b[proc_start:proc_end] = E_b_data
                E_k[proc_start:proc_end] = E_k_data
                E_t[proc_start:proc_end] = E_t_data
                E_total[proc_start:proc_end] = E_total_data
        
        # 绘制温度演化图
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(nt) * dt, T_perp_arr, label='T_perp')
        plt.plot(np.arange(nt) * dt, T_para_arr, label='T_parallel')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.title('Temperature Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig('temperature_evolution.png')
        plt.close()
        
        # 绘制能量演化图
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(nt) * dt, E_b, label='E_mag')
        plt.plot(np.arange(nt) * dt, E_k, label='E_kinetic')
        plt.plot(np.arange(nt) * dt, E_t, label='E_themal')
        plt.plot(np.arange(nt) * dt, E_total, label='E_total')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Evolution')
        plt.legend()
        plt.savefig('energy_evolution.png')
        plt.close()
        
        print("所有图像生成完成！")
    # else:
        # 子进程发送数据给主进程
        # if start < nt:  # 防止越界
        #     comm.send(T_perp_arr[start:end], dest=0, tag=100)
        #     comm.send(T_para_arr[start:end], dest=0, tag=101)
        #     comm.send(E_b[start:end], dest=0, tag=102)
        #     comm.send(E_k[start:end], dest=0, tag=103)
        #     comm.send(E_t[start:end], dest=0, tag=104)
        #     comm.send(E_total[start:end], dest=0, tag=105)    