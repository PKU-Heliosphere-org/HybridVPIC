import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import struct
import matplotlib.cm as cm

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
    print('info array:',infoarr)
    return infoarr

# parameters
mu_0 = 1.0

cmap = plt.get_cmap("Spectral")
directory = "./data/"
save_dir = "./image/"
os.makedirs(save_dir, exist_ok=True)
infoarr = loadinfo(directory)
nx = int(infoarr[0])
ny = int(infoarr[1])
Lx = int(infoarr[3])
Ly = int(infoarr[4])
print("nx, ny, Lx, Ly", nx, ny, Lx, Ly)

xv = np.linspace(0,Lx,nx)
yv = np.linspace(0,Ly,ny)
dx = xv[1] - xv[0]
dy = yv[1] - yv[0]

def load_data_at_certain_t(fname, i_t, num_dim1=nx, num_dim2=ny):
    with open(fname, 'rb') as f:
        f.seek(4 * i_t * nx * ny, 1)
        arr = np.fromfile(f,dtype=np.float32,count=nx*ny)
    arr = np.reshape(arr,(ny, nx))
    arr = np.transpose(arr)
    return arr

def determine_time_steps(file_path, num_dim1, num_dim2):
    """
    Calculate the total number of time steps (nt) in the file.

    Parameters:
    file_path : str
        Path to the data file.
    num_dim1 : int
        Size of the first dimension of the data (e.g., nx).
    num_dim2 : int
        Size of the second dimension of the data (e.g., ny).

    Returns:
    int
        Total number of time steps (nt) in the file.
    """
    # Get the file size in bytes
    file_size = os.path.getsize(file_path)

    # Each float32 data element occupies 4 bytes
    bytes_per_element = 4

    # Determine the size of data for a single time step
    elements_per_time_step = num_dim1 * num_dim2
    bytes_per_time_step = elements_per_time_step * bytes_per_element

    # Calculate the number of time steps
    if file_size % bytes_per_time_step != 0:
        raise ValueError("File size does not match the specified dimensions, unable to calculate time steps.")

    nt = file_size // bytes_per_time_step

    return nt

if __name__ == "__main__":
    file_path = directory + 'bx.gda'  # Use any one of the physical quantity files
    nt = determine_time_steps(file_path, nx, ny)
    print(f"Total number of time steps in the file: {nt}")

    colorbar_limits = {
        "|B|": {"min": None, "max": None},
        "Bx": {"min": None, "max": None},
        "By": {"min": None, "max": None},
        "Bz": {"min": None, "max": None},
        "Ex": {"min": None, "max": None},
        "Ey": {"min": None, "max": None},
        "Ez": {"min": None, "max": None},
        "uix": {"min": None, "max": None},
        "uiy": {"min": None, "max": None},
        "uiz": {"min": None, "max": None},
        "ni": {"min": None, "max": None},
        "T_para": {"min": None, "max": None},
        "T_perp": {"min": None, "max": None},
        "T_iso": {"min": None, "max": None},
    }
    T_center = {"T_para": None, "T_perp": None, "T_iso": None}

    print("Getting colorbar limits ...")
    for it in range(nt):
        print(f"{it}/{nt}",end = ' ')

        time_norm = 10 # omega_ci^-1
        time = it*time_norm

        Bx = load_data_at_certain_t(directory+'bx.gda',it,nx,ny)
        By = load_data_at_certain_t(directory+'by.gda',it,nx,ny)
        Bz = load_data_at_certain_t(directory+'bz.gda',it,nx,ny)
        Btot = np.sqrt(Bx**2+By**2+Bz**2)
        Bx_norm = Bx / Btot
        By_norm = By / Btot
        Bz_norm = Bz / Btot

        Ex = load_data_at_certain_t(directory+'ex.gda',it,nx,ny)
        Ey = load_data_at_certain_t(directory+'ey.gda',it,nx,ny)
        Ez = load_data_at_certain_t(directory+'ez.gda',it,nx,ny)
        ni = load_data_at_certain_t(directory+'ni.gda',it,nx,ny)
        Pixx = load_data_at_certain_t(directory+'pi-xx.gda',it,nx,ny)
        Piyy = load_data_at_certain_t(directory+'pi-yy.gda',it,nx,ny)
        Pizz = load_data_at_certain_t(directory+'pi-zz.gda',it,nx,ny)
        Pixy = load_data_at_certain_t(directory+'pi-xy.gda',it,nx,ny)
        Pixz = load_data_at_certain_t(directory+'pi-xz.gda',it,nx,ny)
        Piyz = load_data_at_certain_t(directory+'pi-yz.gda',it,nx,ny)
        uix = load_data_at_certain_t(directory+'uix.gda',it,nx,ny)
        uiy = load_data_at_certain_t(directory+'uiy.gda',it,nx,ny)
        uiz = load_data_at_certain_t(directory+'uiz.gda',it,nx,ny)

        P_th = np.zeros((120, 960, 3, 3))
        P_th[:, :, 0, 0] = Pixx
        P_th[:, :, 1, 1] = Piyy
        P_th[:, :, 2, 2] = Pizz
        P_th[:, :, 0, 1] = P_th[:, :, 1, 0] = Pixy
        P_th[:, :, 0, 2] = P_th[:, :, 2, 0] = Pixz
        P_th[:, :, 1, 2] = P_th[:, :, 2, 1] = Piyz
        P_th_eigenvalues = np.linalg.eigvalsh(P_th)
        P_th_total = np.sum(P_th_eigenvalues, axis=-1)
        P_th_iso = P_th_total / 3

        P_th_rot = np.zeros((120, 960, 2))
        for i in range(120):
            for j in range(960):
                B_norm = np.array([Bx_norm[i, j], By_norm[i, j], Bz_norm[i, j]])
                B_norm_outer = np.outer(B_norm, B_norm)
        #         P_parallel_mat = np.dot(B_norm_outer, np.dot(P_th[i, j], B_norm_outer))
                P_th_para = np.dot(B_norm, np.dot(P_th[i, j], B_norm))
                P_th_perp = (P_th_total[i,j] - P_th_para) / 2 

                P_th_rot[i, j, 0] = P_th_para
                P_th_rot[i, j, 1] = P_th_perp    
                
        T_para = P_th_rot[:, :, 0] / ni
        T_perp = P_th_rot[:, :, 1] / ni
        T_iso = P_th_iso / ni
        
        variables = {
            "|B|": Btot, "Bx": Bx, "By": By, "Bz": Bz,
            "Ex": Ex, "Ey": Ey, "Ez": Ez,
            "uix": uix, "uiy": uiy, "uiz": uiz,
            "ni": ni, "T_para": T_para,
            "T_perp": T_perp, "T_iso": T_iso,
        }
        for key, var in variables.items():
            if key in ["|B|", "ni" , "By", "uiy"]:
                x_max = np.max(var)
                x_min = np.min(var)
                center = {"|B|":1, "ni":1 , "By":0.4, "uiy":-0.4}[key]
                new_max = center + (center - x_min)
                new_min = center - (x_max - center)
            elif key in ["Bx", "Bz", "Ex", "Ey", "Ez", "uix", "uiz"]:
                abs_max = np.max(np.abs(var))
                new_max = abs_max
                new_min = -abs_max
            elif key in ["T_para","T_perp","T_iso"]:
                if it == 0:
                    T_center[key] = np.mean(var[0:10,0:10])
                    print(T_center)
                center = T_center[key]
                x_max = np.max(var)
                x_min = np.min(var)
                max_dev = max(abs(x_max - center), abs(x_min - center))
                new_max = center + max_dev
                new_min = center - max_dev
            
            colorbar_limits[key]["max"] = max(colorbar_limits[key]["max"] or new_max, new_max)
            colorbar_limits[key]["min"] = min(colorbar_limits[key]["min"] or new_min, new_min)
    
    print("Final colorbar limits:", colorbar_limits)
    CL = colorbar_limits

    print("Printing ...")
    for it in range(nt):
        print(f"{it}/{nt}",end = ' ')

        time_norm = 10 # omega_ci^-1
        time = it*time_norm

        Bx = load_data_at_certain_t(directory+'bx.gda',it,nx,ny)
        By = load_data_at_certain_t(directory+'by.gda',it,nx,ny)
        Bz = load_data_at_certain_t(directory+'bz.gda',it,nx,ny)
        Btot = np.sqrt(Bx**2+By**2+Bz**2)
        Bx_norm = Bx / Btot
        By_norm = By / Btot
        Bz_norm = Bz / Btot

        Ex = load_data_at_certain_t(directory+'ex.gda',it,nx,ny)
        Ey = load_data_at_certain_t(directory+'ey.gda',it,nx,ny)
        Ez = load_data_at_certain_t(directory+'ez.gda',it,nx,ny)
        ni = load_data_at_certain_t(directory+'ni.gda',it,nx,ny)
        Pixx = load_data_at_certain_t(directory+'pi-xx.gda',it,nx,ny)
        Piyy = load_data_at_certain_t(directory+'pi-yy.gda',it,nx,ny)
        Pizz = load_data_at_certain_t(directory+'pi-zz.gda',it,nx,ny)
        Pixy = load_data_at_certain_t(directory+'pi-xy.gda',it,nx,ny)
        Pixz = load_data_at_certain_t(directory+'pi-xz.gda',it,nx,ny)
        Piyz = load_data_at_certain_t(directory+'pi-yz.gda',it,nx,ny)
        uix = load_data_at_certain_t(directory+'uix.gda',it,nx,ny)
        uiy = load_data_at_certain_t(directory+'uiy.gda',it,nx,ny)
        uiz = load_data_at_certain_t(directory+'uiz.gda',it,nx,ny)

        P_th = np.zeros((120, 960, 3, 3))
        P_th[:, :, 0, 0] = Pixx
        P_th[:, :, 1, 1] = Piyy
        P_th[:, :, 2, 2] = Pizz
        P_th[:, :, 0, 1] = P_th[:, :, 1, 0] = Pixy
        P_th[:, :, 0, 2] = P_th[:, :, 2, 0] = Pixz
        P_th[:, :, 1, 2] = P_th[:, :, 2, 1] = Piyz
        P_th_eigenvalues = np.linalg.eigvalsh(P_th)
        P_th_total = np.sum(P_th_eigenvalues, axis=-1)
        P_th_iso = P_th_total / 3

        P_th_rot = np.zeros((120, 960, 3))
        for i in range(120):
            for j in range(960):
                B_norm = np.array([Bx_norm[i, j], By_norm[i, j], Bz_norm[i, j]])
                B_norm_outer = np.outer(B_norm, B_norm)
                P_th_para = np.dot(B_norm, np.dot(P_th[i, j], B_norm))
                P_th_perp = (P_th_total[i,j] - P_th_para) / 2 

                P_th_rot[i, j, 0] = P_th_para
                P_th_rot[i, j, 1] = P_th_perp
                P_th_rot[i, j, 2] = P_th_perp
                
        T_para = P_th_rot[:, :, 0] / ni
        T_perp = P_th_rot[:, :, 1] / ni
        T_iso = P_th_iso / ni

        
        fig, axes = plt.subplots(2,7,figsize=(15,8),facecolor='white')
        plt.suptitle(f"$t = {time}"+"\Omega_{ci}^{-1}$",fontsize=15)

        for ax in axes.flatten():
            ax.set_aspect('equal')
            ax.set_xlabel('x[d_i]',fontsize=15)
            ax.set_ylabel('y[d_i]',fontsize=15)

        ax = axes[0,0]
        pclr = ax.pcolor(xv,yv,Btot.T,shading="nearest",cmap='RdBu_r',vmin=CL['|B|']['min'],vmax=CL['|B|']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$|B|$',fontsize=15)

        ax = axes[0,1]
        pclr = ax.pcolor(xv,yv,Bx.T,shading="nearest",cmap='RdBu_r',vmin=CL['Bx']['min'],vmax=CL['Bx']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$B_x$',fontsize=15)

        ax = axes[0,2]
        pclr = ax.pcolor(xv,yv,By.T,shading="nearest",cmap='RdBu_r',vmin=CL['By']['min'],vmax=CL['By']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$B_y$',fontsize=15)

        ax = axes[0,3]
        pclr = ax.pcolor(xv,yv,Bz.T,shading="nearest",cmap='RdBu_r',vmin=CL['Bz']['min'],vmax=CL['Bz']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$B_z$',fontsize=15)

        ax = axes[0,4]
        pclr = ax.pcolor(xv,yv,uix.T,shading="nearest",cmap='RdBu_r',vmin=CL['uix']['min'],vmax=CL['uix']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$u_{ix}$',fontsize=15)

        ax = axes[0,5]
        pclr = ax.pcolor(xv,yv,uiy.T,shading="nearest",cmap='RdBu_r',vmin=CL['uiy']['min'],vmax=CL['uiy']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$u_{iy}$',fontsize=15)

        ax = axes[0,6]
        pclr = ax.pcolor(xv,yv,uiz.T,shading="nearest",cmap='RdBu_r',vmin=CL['uiz']['min'],vmax=CL['uiz']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$u_{iz}$',fontsize=15)

        ax = axes[1,0]
        pclr = ax.pcolor(xv,yv,ni.T,shading="nearest",cmap='RdBu_r',vmin=CL['ni']['min'],vmax=CL['ni']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$n_i$',fontsize=15)

        ax = axes[1,1]
        pclr = ax.pcolor(xv,yv,Ex.T,shading="nearest",cmap='RdBu_r',vmin=CL['Ex']['min'],vmax=CL['Ex']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$E_x$',fontsize=15)

        ax = axes[1,2]
        pclr = ax.pcolor(xv,yv,Ey.T,shading="nearest",cmap='RdBu_r',vmin=CL['Ey']['min'],vmax=CL['Ey']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$E_y$',fontsize=15)

        ax = axes[1,3]
        pclr = ax.pcolor(xv,yv,Ez.T,shading="nearest",cmap='RdBu_r',vmin=CL['Ez']['min'],vmax=CL['Ez']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$E_z$',fontsize=15)

        ax = axes[1,4]
        pclr = ax.pcolor(xv,yv,T_para.T,shading="nearest",cmap='RdBu_r',vmin=CL['T_para']['min'],vmax=CL['T_para']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$T_{para}$',fontsize=15)

        ax = axes[1,5]
        pclr = ax.pcolor(xv,yv,T_perp.T,shading="nearest",cmap='RdBu_r',vmin=CL['T_perp']['min'],vmax=CL['T_perp']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$T_{perp}$',fontsize=15)

        ax = axes[1,6]
        pclr = ax.pcolor(xv,yv,T_iso.T,shading="nearest",cmap='RdBu_r',vmin=CL['T_iso']['min'],vmax=CL['T_iso']['max'])
        cbar=plt.colorbar(pclr,ax=ax)
        ax.set_title('$T_{iso}$',fontsize=15)

        plt.tight_layout()
        plt.savefig(save_dir+f't={time}Omega^-1.png',dpi=200)
        plt.close(fig)
        



