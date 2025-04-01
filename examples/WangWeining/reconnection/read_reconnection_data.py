import sys
# sys.path.append("D:\Research\Codes\Hybrid-vpic")
from plot_simulation_results import Species
from read_field_data import load_data_at_certain_t
import matplotlib.pyplot as plt
import numpy as np
#%%
step = 6000
run_case_index = 5
num_files = 16
# field_dir = f"data_ip_shock/field_data_{run_case_index}/"
base_fname_swi_1 = f"D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/particle_data_{run_case_index}/T.{step}/Hparticle.{step}.{{}}"
p = Species(name="ion", fullname="Ion", filename=base_fname_swi_1, num_files=num_files)
field_dir = "D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/field_data_5/"
#%%
p.plot_phase_space_2D(sample_step=1, x_plot_name="z", y_plot_name="ux", color="k", size=1)
plt.show()
#%%
v_para = p.ux
v_perp = np.sqrt(p.uy**2+p.uz**2)
x_left, x_right = 200, 250
z_bottom, z_top = 25, 30
x_center_idx = round((x_left+x_right)/2)
z_center_idx = round((z_bottom+z_top)/2+32)
condition = (p.x > x_left) & (p.x < x_right) & (p.z > z_bottom) & (p.z < z_top)
counts_arr_1, bins = np.histogram(p.E[condition], bins=np.logspace(-1, np.log10(25), 20))

# plt.plot(bins[1:], counts_arr_1/np.sum(counts_arr_1))
# plt.yscale("log")
# plt.xscale("log")
#%%
epoch =30
bx = load_data_at_certain_t(field_dir+"bx.gda", i_t=epoch, num_dim1=256, num_dim2=64)
by = load_data_at_certain_t(field_dir+"by.gda", i_t=epoch, num_dim1=256, num_dim2=64)
bz = load_data_at_certain_t(field_dir+"bz.gda", i_t=epoch, num_dim1=256, num_dim2=64)
b_mag = np.sqrt(bx**2+by*2+bz**2)
plt.plot(by)
plt.show()
#%%
counts_mat, xedges, zedges = np.histogram2d(v_para[condition], p.uy[condition], bins=[80, 80],
                                                    range=[[np.min(v_para), np.max(v_para)], [np.min(p.uy), np.max(p.uy)]])
plt.pcolormesh(np.linspace(np.min(v_para), np.max(v_para), 80), np.linspace(np.min(p.uy), np.max(p.uy), 80), np.log10(counts_mat).T
               , vmin=0, vmax=2, cmap="jet")
plt.xlim([-5, 5])
plt.ylim([-4, 4])
cbar = plt.colorbar()
cbar.set_label(r"$\log_{10}$(counts)", fontsize=15)
v_para_max_index, vy_max_index = np.unravel_index(np.argmax(counts_mat), counts_mat.shape)
print(xedges[v_para_max_index], zedges[vy_max_index])
plt.arrow(xedges[v_para_max_index], zedges[vy_max_index], bx[x_center_idx, z_center_idx], by[x_center_idx, z_center_idx], head_width=0.1, fc="k")
if by[x_center_idx, z_center_idx] > 0:
    plt.text(xedges[v_para_max_index]+bx[x_center_idx, z_center_idx], zedges[vy_max_index]+by[x_center_idx, z_center_idx]+0.25, r"$\mathbf{B_0}$", fontsize=13)
else:
    plt.text(xedges[v_para_max_index] + bx[x_center_idx, z_center_idx],
             zedges[vy_max_index] + by[x_center_idx, z_center_idx]-0.25, r"$\mathbf{B_0}$", fontsize=13)
plt.title(rf"{x_left}<x<{x_right}, {z_bottom}<z<{z_top}, $\Omega_i t$={epoch}", fontsize=15)
plt.xlabel(r"$v_x$", fontsize=15)
plt.ylabel(r"$v_y$", fontsize=15)
plt.show()
#%%

#%%
E_mat = np.zeros((256, 64))
for i in range(256):
    print(i)
    for j in range(-32, 32, 1):

        condition = (p.x >= i) & (p.x < i+1) & (p.z >= j) & (p.z < j+1)
        E_mat[i, j] = p.E[condition].mean()
#%%
plt.pcolormesh(range(256), range(64), E_mat.T)
plt.colorbar()
plt.show()
#%%
condition = p.x
#%%

# plt.pcolormesh(range(256), range(64), bx.T)
plt.plot(b_mag[:, 31])
plt.ylim([1, 1.5])
plt.ylabel("|B|", fontsize=15)
plt.xlabel("x", fontsize=15)
# plt.colorbar()
plt.show()