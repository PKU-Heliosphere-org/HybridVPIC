import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import struct


######### loadinfo function
def loadinfo(dir):
    fstr = dir + "info"
    fd = open(fstr, "rb")
    infocontent = fd.read()
    fd.close()
    arr = struct.unpack("fIIIffffff", infocontent[:40])
    infoarr=np.zeros(6)
    infoarr[0] = arr[1]
    infoarr[1] = arr[2]
    infoarr[2] = arr[3]
    infoarr[3] = arr[6]
    infoarr[4] = arr[7]
    infoarr[5] = arr[8]
    print(infoarr)
#     print(arr)
    return infoarr


def load_data(fname, num_dim1, num_dim2, num_dim3):
    # nx, nz, nt
    with open(fname, 'rb') as f:
        res = np.fromfile(f, dtype=np.float32, count=num_dim1 * num_dim2 * num_dim3)
        res = res.reshape(num_dim3, num_dim2, num_dim1).T
    return res


def load_data_at_certain_t(fname, i_t, num_dim1, num_dim2):
    with open(fname, 'rb') as f:
        f.seek(4 * i_t * num_dim1 * num_dim2, 1)
        arr = np.fromfile(f, dtype=np.float32, count=num_dim1 * num_dim2)
    arr = np.reshape(arr, (num_dim2, num_dim1))
    arr = np.transpose(arr)
    return arr


def load_data_dynamic(fname, num_dim1, num_dim2):
    # 获取文件大小（字节数）
    file_size = os.path.getsize(fname)

    # 计算文件中 float32 数量
    total_elements = file_size // 4  # 每个 float32 占用 4 字节

    # 计算 num_dim3（nt）
    if total_elements % (num_dim1 * num_dim2) != 0:
        raise ValueError("文件大小与指定的 num_dim1 和 num_dim2 不匹配，无法确定 nt 值。")

    num_dim3 = total_elements // (num_dim1 * num_dim2)

    # 加载数据
    with open(fname, 'rb') as f:
        res = np.fromfile(f, dtype=np.float32, count=num_dim1 * num_dim2 * num_dim3)
        res = res.reshape(num_dim3, num_dim2, num_dim1).T

    return res, num_dim3


if __name__ == "__main__":
    it = 42
    fdir = "data_ip_shock/field_data_7/data/"
    infoarr = loadinfo(fdir)
    nx = int(infoarr[0])
    nz = int(infoarr[2])
    Lx = int(infoarr[3])
    Lz = int(infoarr[5])
    Bz1 = load_data_at_certain_t(fdir + 'bz.gda', it, nx, nz)


