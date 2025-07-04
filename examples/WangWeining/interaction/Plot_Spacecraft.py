# from datetime import datetime
# import numpy as np
# import pyvista as pv
# import spiceypy as spice
#
# from read_field_data import loadinfo, load_data_at_certain_t
#
# def load_spacecraft_model(stl_path, scale=10):
#     """
#     加载航天器STL模型并进行缩放处理
#     :param stl_path: STL文件路径
#     :param scale: 缩放因子，默认10
#     :return: pyvista网格对象
#     """
#     mesh = pv.read(stl_path)
#     # 按模型最大尺寸缩放
#     mesh.points = scale * mesh.points / np.max(mesh.points)
#     return mesh
#
#
# def rotate_model(mesh, theta_x=0, theta_y=0, theta_z=0, origin=(0, 0, 0)):
#     """
#     旋转模型
#     :param mesh: pyvista网格对象
#     :param theta_x: X轴旋转角度(度)
#     :param theta_y: Y轴旋转角度(度)
#     :param theta_z: Z轴旋转角度(度)
#     :param origin: 旋转中心点
#     :return: 旋转后的网格
#     """
#     if theta_x != 0:
#         mesh = mesh.rotate_x(theta_x, point=origin, inplace=False)
#     if theta_y != 0:
#         mesh = mesh.rotate_y(theta_y, point=origin, inplace=False)
#     if theta_z != 0:
#         mesh = mesh.rotate_z(theta_z, point=origin, inplace=False)
#     return mesh
#
#
# def get_span_a_position(species='ion', scale=10):
#     """
#     获取SPAN-A仪器在航天器坐标系中的位置
#     :param species: 'ion'或'electron'，指定仪器类型
#     :param scale: 缩放因子
#     :return: 三维坐标数组
#     """
#     if species == 'ion':
#         # SPAN-A离子探测器相对位置(原始单位:米)
#         rel_pos = np.array([0.128, -0.0298, -0.293])
#     else:  # electron
#         # SPAN-A电子探测器相对位置(原始单位:米)
#         rel_pos = np.array([0.1040395, -0.0940903, -0.29254955])
#     return rel_pos * scale
#
#
# def plot_span_a_ion(plotter, pos, rot_theta, dt, scale=10, color='silver'):
#     """
#     在Plotter中添加SPAN-A离子探测器及其FOV
#     :param plotter: pyvista.Plotter对象
#     :param pos: 航天器中心位置
#     :param rot_theta: 绕Z轴旋转角度(度)
#     :param dt: 时间，用于SPICE坐标转换
#     :param scale: 缩放因子
#     :param color: 模型颜色
#     """
#     # 1. 加载航天器模型
#     stl_path = '/Users/ephe/Desktop/SolHelio-Viewer/ParkerSolarProbe.stl'
#     mesh = load_spacecraft_model(stl_path, scale)
#
#     # 2. 定义旋转参数
#     theta_x = 180  # X轴旋转角度
#     theta_z = 90 + rot_theta  # Z轴旋转角度（包含用户指定的旋转）
#     origin = (0, 0, 0)  # 旋转中心
#
#     # 3. 执行旋转
#     mesh = rotate_model(mesh, theta_x, theta_z=theta_z, origin=origin)
#
#     # 4. 添加到绘图器（带位置偏移）
#     plotter.add_mesh(mesh.translate(pos), color=color, opacity=0.8,
#                      name="Parker Solar Probe", label="Spacecraft")
#
#     # 5. 获取SPAN-A离子探测器位置
#     spana_center = get_span_a_position('ion', scale) + np.array(pos)
#
#     # 6. 添加SPAN-A仪器标记
#     plotter.add_mesh(pv.Sphere(radius=0.05*scale), center=spana_center,
#                      color='blue', name="SPAN-A Ion", label="SPAN-A Ion")
#
#     # 7. 绘制FOV（简化版，实际需结合SPICE数据）
#     try:
#         # 加载SPICE内核（需提前准备）
#         spice.furnsh('naif0012.tls')  # 闰秒内核
#         spice.furnsh('spp_ik.tf')     # 仪器内核
#         et = spice.datetime2et(dt)
#
#         # 获取FOV参数
#         sweap_param = spice.getfov(-96201, 26)
#         edges = np.array(sweap_param[4][:])
#
#         # 坐标转换矩阵
#         M_arr = spice.sxform('SPP_SWEAP_SPAN_A_ION', 'SPP_SPACECRAFT', et)[0:3, 0:3]
#
#         # 绘制FOV边界（简化为射线）
#         length_ray = 2 * scale
#         for edge in edges:
#             tmp_ray = np.dot(M_arr, edge) * length_ray
#             ray = pv.Line(spana_center, spana_center + tmp_ray)
#             plotter.add_mesh(ray, color='blue', line_width=3,
#                              name="FOV Edge", label="FOV Edge")
#
#     except spice.utils.exceptions.SpiceError as e:
#         print(f"SPICE数据加载错误: {e}")
#         print("请确保已正确加载SPICE内核文件")
#
#
# def plot_span_a_electron(plotter, pos, rot_theta, dt, scale=10, color='silver'):
#     """
#     在Plotter中添加SPAN-A电子探测器及其FOV
#     :param plotter: pyvista.Plotter对象
#     :param pos: 航天器中心位置
#     :param rot_theta: 绕Z轴旋转角度(度)
#     :param dt: 时间，用于SPICE坐标转换
#     :param scale: 缩放因子
#     :param color: 模型颜色
#     """
#     # 1. 加载航天器模型（此处使用SOHO模型示例，需替换为实际SPP模型）
#     stl_path = 'D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/interaction/SOHO/SOHO_all_parts.stl'
#     mesh = load_spacecraft_model(stl_path, scale)
#
#     # 2. 定义旋转参数
#     theta_x = 90   # X轴旋转角度
#     theta_z = 90 + rot_theta  # Z轴旋转角度（包含用户指定的旋转）
#     origin = (0, 0, 0)  # 旋转中心
#
#     # 3. 执行旋转
#     mesh = rotate_model(mesh, theta_x, theta_z=theta_z, origin=origin)
#
#     # 4. 添加到绘图器（带位置偏移）
#     plotter.add_mesh(mesh.translate(pos), color=color, opacity=0.9,
#                      name="Parker Solar Probe", label="Spacecraft")
#
#     # 5. 获取SPAN-A电子探测器位置
#     spana_center = get_span_a_position('electron', scale) + np.array(pos)
#
#     # 6. 添加SPAN-A仪器标记
#     plotter.add_mesh(pv.Sphere(radius=0.05*scale, center=spana_center),
#                      color='green', name="SPAN-A Electron", label="SPAN-A Electron")
#
#     # 7. （可选）绘制电子探测器FOV，逻辑与离子探测器类似
#
#
# def plot_arc(center, R, pos_A, pos_B, longer_arc=False, n=50):
#     """
#     绘制三维圆弧（PyVista版本）
#     :param center: 圆弧中心
#     :param R: 圆弧半径
#     :param pos_A: 圆弧起点
#     :param pos_B: 圆弧终点
#     :param longer_arc: 是否绘制优弧
#     :param n: 插值点数
#     :return: pyvista PolyData对象
#     """
#     OA = np.array(pos_A - center)
#     OB = np.array(pos_B - center)
#
#     # 计算插值点
#     p = np.linspace(0, 1, n)
#     OP = np.array([(1-p[i])*OA + p[i]*OB for i in range(n)])
#
#     # 归一化并缩放至半径R
#     OP = np.array([R * OP[i] / np.linalg.norm(OP[i], 2) for i in range(n)])
#     if longer_arc:
#         OP = -OP  # 优弧方向
#
#     # 生成圆弧点
#     pos_P = OP + center
#     points = np.column_stack((pos_P[:, 0], pos_P[:, 1], pos_P[:, 2]))
#
#     # 创建线对象
#     lines = np.zeros((n-1, 3), dtype=np.int_)
#     lines[:, 0] = 2  # 每条线有2个点
#     for i in range(n-1):
#         lines[i, 1] = i
#         lines[i, 2] = i+1
#
#     arc = pv.PolyData(points)
#     arc.lines = lines
#     return arc
#
#
# if __name__ == '__main__':
#     # 创建Plotter对象（可与现有磁场绘图的Plotter集成）
#     plotter = pv.Plotter()
#     field_dir = 'D:\Research\Codes\Hybrid-vpic\HybridVPIC-main\interaction/data/'
#     nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
#     x = np.linspace(0, 256, nx)
#     y = np.linspace(0, 64, ny)
#     z = np.linspace(0, 64, nz)
#     X, Y, Z = np.meshgrid(x, y, z)
#     Bx = load_data_at_certain_t(field_dir + "bx.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
#     By = load_data_at_certain_t(field_dir + "by.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
#     Bz = load_data_at_certain_t(field_dir + "bz.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
#     uix = load_data_at_certain_t(field_dir + "uix.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
#     ni = load_data_at_certain_t(field_dir + "ni.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
#     B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
#
#     # 创建 PyVista 网格
#     grid = pv.StructuredGrid(X, Y, Z)
#
#     # 添加向量场和标量场到网格
#     grid["magnetic_field"] = np.column_stack((Bx.flatten(), By.flatten(), Bz.flatten()))
#     grid["field_magnitude"] = B_magnitude.flatten()
#
#     # 创建绘图器
#     # plotter = pv.Plotter()
#
#     # # 添加向量场 - 使用箭头表示磁场方向
#     # arrows = grid.glyph(orient="magnetic_field", scale="field_magnitude", factor=0.8,
#     #                     tolerance=0.08)
#     # plotter.add_mesh(arrows, cmap="plasma", scalars="field_magnitude",
#     #                  clim=[0, np.percentile(B_magnitude, 95)],
#     #                  lighting=True, show_scalar_bar=True,
#     #                  scalar_bar_args={"title": "磁场强度"})
#
#     # 添加等值面 - 表示磁场强度相同的区域
#     # contours = grid.contour(isosurfaces=8, scalars="field_magnitude")
#     # plotter.add_mesh(contours, cmap="viridis", opacity=0.3,
#     #                  show_scalar_bar=False)
#     #
#     # # 添加一个平面表示 XY 平面
#     # plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
#     #                  i_size=10, j_size=10, i_resolution=50, j_resolution=50)
#     # plane_contour = plane.sample(grid).contour(isosurfaces=10, scalars="field_magnitude")
#     # plotter.add_mesh(plane_contour, cmap="jet", line_width=2,
#     #                  show_scalar_bar=False)
#     #
#     # 设置场景
#     plotter.set_background("white")
#     plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
#     plotter.add_title("三维磁场分布图 (磁偶极子场)", font_size=14)
#     # 创建渲染窗口
#     plotter = pv.Plotter(off_screen=True)
#     plotter.set_background("white")  # 设置背景色
#     plotter.add_axes()  # 添加坐标轴
#
#
#     # 添加坐标轴
#     # plotter.add_axes(actor_scale=5.0, line_width=10)
#     plotter.add_axes()
#
#     # 绘制SPAN-A离子探测器模型
#     plot_span_a_electron(plotter, pos=[100, 20, 20], rot_theta=0,
#                     dt=datetime(2021, 3, 14), scale=50)
#
#     # （示例）添加磁场强度分布（需替换为实际磁场数据）
#     # 假设已有磁场数据grid，可以这样添加：
#     # plotter.add_volume(grid, cmap='viridis', opacity=[0, 1], label="Magnetic Field")
#     plotter.add_volume(
#         ni,
#         cmap="jet",
#         # scalar_range=(bx.min(), bx.max()),
#         opacity=[0, 0.3, 0.6],  # 透明度传递函数
#         shade=True,
#     )
#     plotter.export_html("D:\Research\Codes\Hybrid-vpic\HybridVPIC-main\interaction/n_field_interactive.html")
#     # 显示图形
#     # plotter.show()
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import spiceypy as spice
import os
from read_field_data import loadinfo, load_data_at_certain_t


def load_spacecraft_model(stl_path, scale=10):
    """加载并缩放航天器模型"""
    mesh = pv.read(stl_path)
    mesh.points = scale * mesh.points / np.max(mesh.points)
    return mesh


def create_volume_seeds(volume, num_points=10):
    """在volume内部生成三维均匀分布的种子点"""
    bounds = volume.bounds  # 获取volume的边界

    # 在volume内部生成随机点（也可以使用均匀网格）
    x = np.linspace(bounds[0], bounds[1], num_points // 10)
    y = np.linspace(bounds[2], bounds[3], num_points // 10)
    z = np.linspace(bounds[4], bounds[5], num_points // 10)

    # 生成网格点
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # 创建PolyData对象
    seeds = pv.PolyData(points)
    return seeds

def create_satellite_seeds(satellite_pos, num_points=20, radius=5):
    """
    在卫星位置周围生成球形分布的种子点（用于生成穿过卫星的磁力线）
    :param satellite_pos: 卫星三维坐标
    :param num_points: 种子点数量
    :param radius: 种子点分布的球体半径
    :return: pyvista.PolyData 种子点对象
    """
    # 生成球坐标系随机点
    theta = np.random.uniform(0, np.pi, num_points)  # 极角
    phi = np.random.uniform(0, 2*np.pi, num_points)  # 方位角
    r = np.random.uniform(0, radius, num_points)     # 半径

    # 转换为笛卡尔坐标
    x = satellite_pos[0] + r * np.sin(theta) * np.cos(phi)
    y = satellite_pos[1] + r * np.sin(theta) * np.sin(phi)
    z = satellite_pos[2] + r * np.cos(theta)

    # 合并为点集
    points = np.column_stack((x, y, z))
    return pv.PolyData(points)
def rotate_model(mesh, theta_x=0, theta_y=0, theta_z=0, origin=(0, 0, 0)):
    """旋转模型"""
    if theta_x != 0:
        mesh = mesh.rotate_x(theta_x, point=origin, inplace=False)
    if theta_y != 0:
        mesh = mesh.rotate_y(theta_y, point=origin, inplace=False)
    if theta_z != 0:
        mesh = mesh.rotate_z(theta_z, point=origin, inplace=False)
    return mesh


def get_span_a_position(species='ion', scale=10):
    """获取SPAN-A仪器位置"""
    if species == 'ion':
        rel_pos = np.array([0.128, -0.0298, -0.293])
    else:
        rel_pos = np.array([0.1040395, -0.0940903, -0.29254955])
    return rel_pos * scale


def plot_span_a_ion(plotter, pos, rot_theta, dt, scale=10, color=(1, 1, 1)):
    """绘制SPAN-A离子探测器"""
    stl_path = '/Users/ephe/Desktop/SolHelio-Viewer/ParkerSolarProbe.stl'
    mesh = load_spacecraft_model(stl_path, scale)
    theta_x = 180
    theta_z = 90 + rot_theta
    mesh = rotate_model(mesh, theta_x, theta_z=theta_z)
    plotter.add_mesh(mesh.translate(pos), color=color, opacity=0.0)

    spana_center = get_span_a_position('ion', scale) + np.array(pos)
    plotter.add_mesh(pv.Sphere(radius=0.05 * scale, center=spana_center),
                     color='r', name="SPAN-A Ion", opacity=0.0)

    try:
        spice.furnsh('naif0012.tls')
        spice.furnsh('spp_ik.tf')
        et = spice.datetime2et(dt)
        sweap_param = spice.getfov(-96201, 26)
        edges = np.array(sweap_param[4][:])
        M_arr = spice.sxform('SPP_SWEAP_SPAN_A_ION', 'SPP_SPACECRAFT', et)[:3, :3]

        length_ray = 2 * scale
        for edge in edges:
            tmp_ray = np.dot(M_arr, edge) * length_ray
            ray = pv.Line(spana_center, spana_center + tmp_ray)
            plotter.add_mesh(ray, color='blue', line_width=3)

    except spice.utils.exceptions.SpiceError as e:
        print(f"SPICE错误: {e}")


def plot_span_a_electron(plotter, pos, rot_theta, dt, scale=10, color='#FFFF00'):
    """绘制SPAN-A电子探测器"""
    stl_path = 'D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/interaction/soho.stl'
    mesh = load_spacecraft_model(stl_path, scale)
    theta_x = 90
    theta_z = 90 + rot_theta
    mesh = rotate_model(mesh, theta_x, theta_z=theta_z)
    plotter.add_mesh(mesh.translate(pos), color=color, opacity=1)

    spana_center = get_span_a_position('electron', scale) + np.array(pos)
    plotter.add_mesh(pv.Sphere(radius=0.05 * scale, center=spana_center),
                     color='w', name="SPAN-A Electron")


def get_grid_point_coordinate(i, j, k, x, y, z):
    """根据网格索引计算坐标"""
    return [x[i], y[j], z[k]]


def get_para_cut_at_time_t(field_dir, para_name,  target_grid_idx, start_t=10, end_t=30, step=1, resolution_factor=2):
    time_steps = list(range(start_t, end_t + 1, step))
    nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(
        loadinfo(field_dir)[2])
    # para_arr = np.zeros_like(time_steps)
    para_lst = []
    for t in time_steps:
        para = load_data_at_certain_t(field_dir+para_name+".gda", i_t=t, num_dim1=nx, num_dim2=ny, num_dim3=nz)
        # para_2 = downsample_3d(para, resolution_factor=resolution_factor)
        i, j, k = target_grid_idx[0], target_grid_idx[1], target_grid_idx[2]
        para_lst.append(para[i*resolution_factor, j*resolution_factor, k*resolution_factor])
    para_arr = np.array(para_lst)
    return para_arr


def create_magnetic_field_animation(field_dir, output_dir='animation_frames',
                                    start_t=10, end_t=30, step=1, scale=50,
                                    resolution_factor=2, target_grid_idx=None,
                                    rotate_x=0, rotate_y=0, rotate_z=90):
    """创建磁场演化动画并导出为PNG图像序列（无需FFmpeg）"""
    import logging
    import os
    import numpy as np
    import pyvista as pv
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, 'frames')
    os.makedirs(image_dir, exist_ok=True)
    logger.info(f"开始创建磁场演化动画，输出目录: {image_dir}")

    # 加载网格信息
    try:
        from read_field_data import loadinfo, load_data_at_certain_t
        nx_original, ny_original, nz_original = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(
            loadinfo(field_dir)[2])
        logger.info(f"原始网格尺寸: {nx_original}x{ny_original}x{nz_original}")
    except Exception as e:
        logger.error(f"加载网格信息失败: {e}")
        return

    nx = nx_original // resolution_factor
    ny = ny_original // resolution_factor
    nz = nz_original // resolution_factor
    x = np.linspace(0, 256, nx)
    y = np.linspace(0, 64, ny)
    z = np.linspace(0, 64, nz)
    X, Y, Z = np.meshgrid(x, y, z)
    logger.info(f"降采样后网格尺寸: {nx}x{ny}x{nz}")

    # 创建Plotter对象
    plotter = pv.Plotter(off_screen=True, window_size=[1000, 500])
    plotter.set_background("white")
    plotter.add_axes()

    # 设置目标位置（示例：定位到网格中心）
    if target_grid_idx is None:
        target_grid_idx = (nx//2, ny//2, nz//2)  # 网格中心点
    target_pos = get_grid_point_coordinate(*target_grid_idx, x, y, z)
    print(f"航天器将定位到坐标: {target_pos} (网格索引: {target_grid_idx})")
    # 绘制航天器模型


    # 预创建网格对象
    # grid = pv.StructuredGrid(X, Y, Z)
    magnetic_field_actor = None
    particle_density_actor = None

    time_steps = list(range(start_t, end_t + 1, step))
    logger.info(f"待处理时间步: {time_steps}")
    rotation_center = (x[nx // 2], y[ny // 2], z[nz // 2])
    for t in time_steps:
        logger.info(f"开始处理时间步: {t}")

        # 清空之前的磁场数据
        if magnetic_field_actor:
            plotter.remove_actor(magnetic_field_actor)
        if particle_density_actor:
            plotter.remove_actor(particle_density_actor)
        velocity_arrow_actors = []

        # 加载并降采样磁场数据
        try:
            Bx_original = load_data_at_certain_t(field_dir + "bx.gda", i_t=t,
                                                 num_dim1=nx_original, num_dim2=ny_original, num_dim3=nz_original)
            By_original = load_data_at_certain_t(field_dir + "by.gda", i_t=t,
                                                 num_dim1=nx_original, num_dim2=ny_original, num_dim3=nz_original)
            Bz_original = load_data_at_certain_t(field_dir + "bz.gda", i_t=t,
                                                 num_dim1=nx_original, num_dim2=ny_original, num_dim3=nz_original)
            ni_original = load_data_at_certain_t(field_dir + "ni.gda", i_t=t,
                                                 num_dim1=nx_original, num_dim2=ny_original, num_dim3=nz_original)
            uix_original = load_data_at_certain_t(field_dir + "uix.gda", i_t=t,
                                                  num_dim1=nx_original, num_dim2=ny_original, num_dim3=nz_original)
            uiy_original = load_data_at_certain_t(field_dir + "uiy.gda", i_t=t,
                                                  num_dim1=nx_original, num_dim2=ny_original, num_dim3=nz_original)
            uiz_original = load_data_at_certain_t(field_dir + "uiz.gda", i_t=t,
                                                  num_dim1=nx_original, num_dim2=ny_original, num_dim3=nz_original)
            logger.info(f"时间步 {t} 数据加载完成")
        except Exception as e:
            logger.error(f"加载时间步 {t} 数据失败: {e}")
            continue

        # 降采样数据
        try:
            Bx = downsample_3d(Bx_original, resolution_factor)
            By = downsample_3d(By_original, resolution_factor)
            Bz = downsample_3d(Bz_original, resolution_factor)
            ni = downsample_3d(ni_original, resolution_factor)
            # 降采样速度场数据
            uix = downsample_3d(uix_original, resolution_factor)
            uiy = downsample_3d(uiy_original, resolution_factor)
            uiz = downsample_3d(uiz_original, resolution_factor)
            velocity = np.sqrt(uix**2+uiy**2+uiz**2)
            logger.info(f"时间步 {t} 数据降采样完成")
        except Exception as e:
            logger.error(f"时间步 {t} 数据降采样失败: {e}")
            continue

        # 更新网格数据
        # grid["magnetic_field"] = np.column_stack((Bx.flatten(), By.flatten(), Bz.flatten()))
        # grid["field_magnitude"] = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2).flatten()

        # # 添加磁场可视化
        # try:
        #     magnetic_field_actor = plotter.add_volume(
        #         grid,
        #         cmap="viridis",
        #         scalars="field_magnitude",
        #         opacity=[0, 0.3, 0.6],
        #         shade=True
        #     )
        #     logger.info(f"时间步 {t} 磁场可视化添加完成")
        # except Exception as e:
        #     logger.error(f"时间步 {t} 添加磁场可视化失败: {e}")

        # 添加粒子密度可视化
        try:
            ni = np.transpose(ni, axes=(1, 0, 2))
            uix, uiy, uiz = np.transpose(uix, axes=(1, 0, 2)), np.transpose(uiy, axes=(1, 0, 2)), np.transpose(uiz, axes=(1, 0, 2))
            Bx, By, Bz = np.transpose(Bx, axes=(1, 0, 2)), np.transpose(By, axes=(1, 0, 2)), np.transpose(Bz, axes=(1, 0, 2))#GSE坐标系
            grid = pv.UniformGrid(dimensions=ni.shape)
            grid['density'] = ni.flatten()
            grid["magnetic_field"] = np.column_stack((Bx.flatten(), By.flatten(), Bz.flatten()))
            grid["field_magnitude"] = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2).flatten()
            grid["velocity"] = np.column_stack((uix.flatten(), uiy.flatten(), uiz.flatten()))
            grid["speed"] = 0.01*np.sqrt(uix ** 2 + uiy ** 2 + uiz ** 2).flatten()
            scaled_grid = grid.scale([4, 1, 4], inplace=False)
            if rotate_x != 0:
                scaled_grid = scaled_grid.rotate_x(rotate_x, point=rotation_center, inplace=False)
            if rotate_y != 0:
                scaled_grid = scaled_grid.rotate_y(rotate_y, point=rotation_center, inplace=False)
            if rotate_z != 0:
                scaled_grid = scaled_grid.rotate_z(rotate_z, point=rotation_center, inplace=False)
            volume_center = [x[nx // 2], y[ny // 2], z[nz // 2]]
            # plotter.camera_position = [
            #     (volume_center[0]+20, volume_center[1]+20, volume_center[2]),  # 相机位置（Z轴正方向，增大此值使体积图变小）
            #     (volume_center[0], volume_center[1], volume_center[2]),  # 焦点
            #     (0, 0, 1)  # 上方向
            # ]
            plotter.camera.position = (volume_center[0]+150, volume_center[1]+400, volume_center[2]+225)
            plotter.camera.focal_point = (volume_center[0]-150, volume_center[1]-100, volume_center[2])
            plotter.camera.up = (0, 0, 1)
            plotter.camera.zoom(1.0)
            particle_density_actor = plotter.add_volume(
                scaled_grid,
                scalars='density',
                cmap="jet",
                opacity=[0.005, 0.005],
                shade=True,
                clim=[1, 5]
            )
            points = scaled_grid.points  # 形状为 (n_points, 3) 的 numpy 数组
            try:
                i_s, j_s, k_s = 16, 192, 16
                from datetime import datetime
                satellite_pos = points[i_s + ny * j_s + nx * ny * k_s]  # 记录卫星位置
                plot_span_a_electron(plotter, pos=satellite_pos, rot_theta=180,
                                     dt=datetime(2021, 3, 14), scale=scale)
                logger.info("航天器模型绘制完成")
            except Exception as e:
                logger.error(f"绘制航天器模型失败: {e}")
                plotter.close()
                return
            # # 获取特定网格点的坐标（例如第 i 个点）
            # i = 1000  # 示例索引
            # x, y, z = points[i]
            # print(len(points))
            # # 遍历所有点
            # for i in range(len(points)):
            #     x, y, z = points[i]
                # print(x,y,z)
            # 或者使用outline_corners()创建带拐角的边框
            outline = scaled_grid.outline()
            plotter.add_mesh(outline, color="black", line_width=2)
            # 创建速度流线场
            # 定义种子点（在感兴趣区域均匀分布）
            # 1. 生成全局种子点（保证整体磁场分布）
            global_seeds = create_volume_seeds(scaled_grid, num_points=50)
            # 2. 生成卫星周围种子点（保证穿过卫星的磁力线）
            satellite_seeds = create_satellite_seeds(satellite_pos, num_points=15, radius=10)
            # 3. 合并种子点
            combined_seeds = global_seeds + satellite_seeds
            # seed_points = pv.Plane(
            #     center=(x[nx // 2], y[ny // 2], z[nz // 2]),  # 种子平面中心
            #     direction=(0, 0, 1),  # 平面法线方向
            #     i_size=50, j_size=50,  # 平面大小
            #     i_resolution=5, j_resolution=5  # 种子点密度
            # )
            logger.info(f"时间步 {t} 定义种子点完成")
            # 计算流线
            # 计算流线（使用三维种子点）
            streamlines = scaled_grid.streamlines_from_source(
                satellite_seeds,  # 使用包含卫星种子点的集合
                vectors="magnetic_field",
                max_time=700,  # 延长流线长度（确保穿过卫星）
                integration_direction="both",  # 双向积分（从卫星向前后延伸）
                max_steps=1000  # 增加步数，避免流线过早终止
            )
            logger.info(f"时间步 {t} 计算流线完成")
            # 检查是否生成了有效流线
            if streamlines.n_points == 0:
                print(f"警告：时间步 {t} 生成的流线为空，改用箭头显示速度场")
                # 使用箭头代替流线
                arrows = scaled_grid.glyph(
                    orient="velocity",
                    scale="speed",
                    factor=0.1,  # 调整箭头大小
                    tolerance=0.9,  # 控制采样密度
                    geom=pv.Arrow()
                )
                plotter.add_mesh(
                    arrows,
                    cmap="jet",
                    scalars="speed",
                    show_scalar_bar=True
                )
            else:

                # 添加流线到渲染器
                streamlines_actor = None
                streamlines_actor = plotter.add_mesh(
                    streamlines,
                    #scalars="field_magnitude",  # 使用速度大小着色
                    #cmap="jet",  # 颜色映射
                    color="white",
                    line_width=2,  # 流线宽度

                    render_lines_as_tubes=True,  # 渲染为管道
                    show_scalar_bar=True,  # 显示颜色条
                    scalar_bar_args={"title": "速度大小"}  # 颜色条标题
                )
                logger.info(f"时间步 {t} 添加流线到渲染器完成")
            logger.info(f"时间步 {t} 粒子密度可视化添加完成")
        except Exception as e:
            logger.error(f"时间步 {t} 添加粒子密度可视化失败: {e}")
        # 在volume右侧添加固定箭头
        try:
            # 定义箭头参数
            arrow_start = np.array([volume_center[0]-400, volume_center[1]-20, volume_center[2]])  # 箭头起点
            arrow_direction = np.array([1, 0, 0])  # 箭头方向（指向X轴正方向）
            arrow_length = 150  # 箭头长度
            arrow_end = arrow_start + arrow_direction * arrow_length
            # 文字位置在箭头终点右侧，Y坐标与箭头中心对齐
            text_position = (
                arrow_start[0] + 5,  # X方向偏移（箭头右侧5个单位）
                arrow_start[1],  # Y方向与箭头对齐
                arrow_start[2]  # Z方向与箭头对齐
            )
            # 创建箭头
            arrow = pv.Arrow(start=arrow_start, direction=arrow_direction, scale=arrow_length)

            # 添加箭头到渲染器
            plotter.add_mesh(
                arrow,
                color='red',  # 箭头颜色
                opacity=0.8,  # 透明度
                show_scalar_bar=False
            )

            # 添加箭头标签
            plotter.add_text(r"V_flow", position=[800, 200, 0.5], font_size=15)

            logger.info(f"时间步 {t} 固定箭头添加完成")
        except Exception as e:
            logger.error(f"时间步 {t} 添加固定箭头失败: {e}")
        try:
            # ... 现有粒子密度和流线代码 ...

            # 新增：生成流速场箭头
            # 控制箭头采样密度（tolerance越小，箭头越密集，性能消耗越大）
            velocity_arrows_actor = None
            tolerance = 1  # 根据网格大小调整，值越大箭头越少
            # 生成箭头（基于速度向量场）

            for i_tmp in range(0, 256, 40):
                print(i_tmp)
                for j_tmp in range(0, 32, 10):
                    for k_tmp in range(0, 32, 10):
                        # i_tmp, j_tmp, k_tmp = 10, 10, 10
                        # print(velocity[i_tmp, j_tmp, k_tmp])
                        arrow = pv.Arrow(start=points[j_tmp + ny * i_tmp + nx * ny * k_tmp, :], direction=[-uix[j_tmp, i_tmp, k_tmp],-uiy[j_tmp, i_tmp, k_tmp], -uiz[j_tmp, i_tmp, k_tmp]], scale=float(2*velocity[i_tmp, j_tmp, k_tmp]),
                                         tip_radius=0.2)

                        # 添加箭头到绘图器，并保存actor
                        velocity_arrow_actor = plotter.add_mesh(
                            arrow,
                            color="black",
                            opacity=1,
                            show_scalar_bar=False,
                            line_width=3

                        )
                        velocity_arrow_actors.append(velocity_arrow_actor)
            logger.info(f"时间步 {t} 流速箭头添加完成")

        except Exception as e:
            logger.error(f"时间步 {t} 生成流速箭头失败: {e}")
        # 设置标题
        plotter.add_title(f"epoch: {t}", font_size=12)
        # volume_center = [x[nx // 2], y[ny // 2], z[nz // 2]]
        # plotter.camera_position = [
        #     (volume_center[0], volume_center[1], volume_center[2] + 300),  # 相机位置（Z轴正方向）
        #     (volume_center[0], volume_center[1], volume_center[2]),  # 焦点
        #     (0, 0, 1)  # 上方向（Z轴正方向）
        # ]
        # plotter.reset_camera_clipping()
        # 渲染并保存图像
        image_file = os.path.join(image_dir, f"frame_{t:04d}.png")
        try:
            plotter.show(auto_close=False)  # 渲染场景但不关闭窗口
            image_3d = plotter.screenshot(image_file, transparent_background=False)
            logger.info(f"时间步 {t} 图像已保存: {image_file}")
            if streamlines_actor:
                plotter.remove_actor(streamlines_actor)
            if velocity_arrow_actor:
                plotter.remove_actor(velocity_arrow_actor)
                velocity_arrows_actor = None  # 重置为None
            tag = 0
            # for actor in velocity_arrow_actors:
            #     print(tag)
            #     tag += 1
            #     plotter.remove_actor(actor)
            logger.info("plotter去除完成")
            velocity_arrow_actors.clear()
        except Exception as e:
            logger.error(f"时间步 {t} 保存图像失败: {e}")
            continue
        fig, axes = plt.subplots(6, 1, figsize=(8, 24))
        ax = axes[0]
        ax.imshow(image_3d, aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1]

        vx_arr = get_para_cut_at_time_t(field_dir, para_name="uix", target_grid_idx=[j_s, i_s, k_s], step=step,
                                        start_t=0, end_t=50, resolution_factor=resolution_factor)
        # print(ni_arr.shape)
        ax.plot(np.linspace(0, 50, 51), vx_arr*50)
        ax.axvline(t, c="k", linestyle="--")
        combined_img_path = os.path.join(output_dir, f'combined_t{t}.png')
        # ax.set_xlabel("time", fontsize=15)
        ax.set_ylabel("vx[km/s]", fontsize=15)
        ax = axes[2]

        ni_arr = get_para_cut_at_time_t(field_dir, para_name="ni", target_grid_idx=[j_s, i_s, k_s], step=step, start_t=0, end_t=50, resolution_factor=resolution_factor)
        # print(ni_arr.shape)
        ax.plot(np.linspace(0, 50, 51), ni_arr*5)
        ax.axvline(t, c="k", linestyle="--")
        combined_img_path = os.path.join(output_dir, f'combined_t{t}.png')
        # ax.set_xlabel("time", fontsize=15)
        ax.set_ylabel(r"density[$cm^{-3}$]", fontsize=15)

        ax = axes[3]

        bx_arr = get_para_cut_at_time_t(field_dir, para_name="bx", target_grid_idx=[j_s, i_s, k_s], step=step,
                                        start_t=0, end_t=50, resolution_factor=resolution_factor)
        ax.plot(np.linspace(0, 50, 51), bx_arr*5)
        ax.axvline(t, c="k", linestyle="--")
        combined_img_path = os.path.join(output_dir, f'combined_t{t}.png')
        # ax.set_xlabel("time", fontsize=15)
        ax.set_ylabel("Bx[nT]", fontsize=15)

        ax = axes[4]

        by_arr = get_para_cut_at_time_t(field_dir, para_name="by", target_grid_idx=[j_s, i_s, k_s], step=step,
                                        start_t=0, end_t=50, resolution_factor=resolution_factor)
        ax.plot(np.linspace(0, 50, 51), -by_arr*5)
        ax.axvline(t, c="k", linestyle="--")
        combined_img_path = os.path.join(output_dir, f'combined_t{t}.png')
        # ax.set_xlabel("time", fontsize=15)
        ax.set_ylabel("By[nT]", fontsize=15)

        ax = axes[5]

        bz_arr = get_para_cut_at_time_t(field_dir, para_name="bz", target_grid_idx=[j_s, i_s, k_s], step=step,
                                        start_t=0, end_t=50, resolution_factor=resolution_factor)
        ax.plot(np.linspace(0, 50, 51), -bz_arr*5)
        ax.axvline(t, c="k", linestyle="--")
        combined_img_path = os.path.join(output_dir, f'combined_t{t}.png')
        ax.set_xlabel("time", fontsize=15)
        ax.set_ylabel("Bz[nT]", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

        plt.savefig(combined_img_path)
        plt.close()
        print(f"已处理时间步 {t}")

    # 关闭Plotter
    plotter.close()
    logger.info(f"动画帧生成完成，共生成 {len(time_steps)} 帧")
    print(f"动画帧已导出至: {image_dir}")
    print("请使用图像查看器或视频编辑软件打开该文件夹查看序列动画")
    print("提示: 可以使用Windows照片查看器、ImageJ或其他工具将PNG序列合并为视频")




def downsample_3d(data, resolution_factor=2):
    """对3D数据进行降采样（分块平均）"""
    shape = data.shape
    new_shape = (shape[0] // resolution_factor, shape[1] // resolution_factor, shape[2] // resolution_factor)
    downsampled = np.zeros(new_shape, dtype=data.dtype)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                # 计算块内平均值
                i_start = i * resolution_factor
                i_end = min(i_start + resolution_factor, shape[0])
                j_start = j * resolution_factor
                j_end = min(j_start + resolution_factor, shape[1])
                k_start = k * resolution_factor
                k_end = min(k_start + resolution_factor, shape[2])

                block = data[i_start:i_end, j_start:j_end, k_start:k_end]
                downsampled[i, j, k] = np.mean(block)

    return downsampled


def convert_frames_to_video(frame_dir, output_video, fps=10):
    """将图像帧转换为视频"""
    try:
        import imageio
        import glob

        print("正在将帧转换为视频...")
        frames = sorted(glob.glob(f"{frame_dir}/frame_*.png"))
        if not frames:
            print(f"警告: 未找到帧文件 in {frame_dir}")
            return

        with imageio.get_writer(output_video, fps=fps) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(frame))

        print(f"视频已保存至: {output_video}")
    except ImportError:
        print("错误: 无法导入imageio，无法转换帧为视频。请安装imageio: pip install imageio")
    except Exception as e:
        print(f"转换视频时出错: {e}")


if __name__ == '__main__':
    field_dir = 'D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/interaction/data/'

    # 生成HTML动画
    for i in range(9, 51):
        create_magnetic_field_animation(
            field_dir,
            start_t=i,
            end_t=i,
            step=1,
            scale=40,
            target_grid_idx=(64, 30, 30)
        )
