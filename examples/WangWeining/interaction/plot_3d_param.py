import numpy as np
import pyvista as pv
from read_field_data import loadinfo, load_data_at_certain_t
from pyvista import examples
# 生成三维网格数据
#%%

#%%
# # 定义磁场函数 - 这里使用磁偶极子场作为示例
# def magnetic_dipole_field(X, Y, Z, mx=0, my=0, mz=1):
#     """
#     计算磁偶极子产生的磁场
#
#     参数:
#     X, Y, Z: 网格点坐标
#     mx, my, mz: 磁偶极矩的三个分量
#
#     返回:
#     Bx, By, Bz: 磁场的三个分量
#     """
#     # 计算到偶极子的距离
#     r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
#     r[r == 0] = 1e-10  # 避免除零错误
#
#     # 计算磁场分量
#     Bx = 3 * X * Z / r ** 5 - mx / r ** 3
#     By = 3 * Y * Z / r ** 5 - my / r ** 3
#     Bz = (3 * Z ** 2 / r ** 5 - 1 / r ** 3) - mz / r ** 3
#
#     return Bx, By, Bz
#
#
# # 计算磁场分量
# Bx, By, Bz = magnetic_dipole_field(X, Y, Z)

# 计算磁场强度
field_dir = 'D:\Research\Codes\Hybrid-vpic\HybridVPIC-main\interaction/data/'
nx, ny, nz = int(loadinfo(field_dir)[0]), int(loadinfo(field_dir)[1]), int(loadinfo(field_dir)[2])
x = np.linspace(0, 256, nx)
y = np.linspace(0, 64, ny)
z = np.linspace(0, 64, nz)
X, Y, Z = np.meshgrid(x, y, z)
Bx = load_data_at_certain_t(field_dir+"bx.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
By= load_data_at_certain_t(field_dir+"by.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
Bz = load_data_at_certain_t(field_dir+"bz.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
uix = load_data_at_certain_t(field_dir+"uix.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
ni = load_data_at_certain_t(field_dir+"ni.gda", i_t=20, num_dim1=nx, num_dim2=ny, num_dim3=nz)
B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

# 创建 PyVista 网格
grid = pv.StructuredGrid(X, Y, Z)

# 添加向量场和标量场到网格
grid["magnetic_field"] = np.column_stack((Bx.flatten(), By.flatten(), Bz.flatten()))
grid["field_magnitude"] = B_magnitude.flatten()

# 创建绘图器
plotter = pv.Plotter()

# # 添加向量场 - 使用箭头表示磁场方向
# arrows = grid.glyph(orient="magnetic_field", scale="field_magnitude", factor=0.8,
#                     tolerance=0.08)
# plotter.add_mesh(arrows, cmap="plasma", scalars="field_magnitude",
#                  clim=[0, np.percentile(B_magnitude, 95)],
#                  lighting=True, show_scalar_bar=True,
#                  scalar_bar_args={"title": "磁场强度"})

# 添加等值面 - 表示磁场强度相同的区域
# contours = grid.contour(isosurfaces=8, scalars="field_magnitude")
# plotter.add_mesh(contours, cmap="viridis", opacity=0.3,
#                  show_scalar_bar=False)
#
# # 添加一个平面表示 XY 平面
# plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
#                  i_size=10, j_size=10, i_resolution=50, j_resolution=50)
# plane_contour = plane.sample(grid).contour(isosurfaces=10, scalars="field_magnitude")
# plotter.add_mesh(plane_contour, cmap="jet", line_width=2,
#                  show_scalar_bar=False)
#
# 设置场景
plotter.set_background("white")
plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
plotter.add_title("三维磁场分布图 (磁偶极子场)", font_size=14)
# 创建渲染窗口
plotter = pv.Plotter(off_screen=True)
plotter.set_background("white")  # 设置背景色
plotter.add_axes()  # 添加坐标轴
plotter.add_volume(
    ni,
    cmap="jet",
    #scalar_range=(bx.min(), bx.max()),
    opacity=[0, 0.3, 0.6],  # 透明度传递函数
    shade=True,
)

# 显示交互式窗口
plotter.export_html("n_field_interactive.html")