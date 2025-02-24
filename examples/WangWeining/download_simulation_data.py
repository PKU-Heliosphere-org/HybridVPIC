import paramiko
import os

# 服务器信息
server_ip = '162.105.246.210'
server_port = 22
username = 'wwn'
password = 'my_password'

# 服务器上的文件夹路径
remote_folder_path = '/home/wwn/HybridVPIC-main/patch/particle/T.10000/'
# 本地目标文件夹路径
local_folder_path = 'D://Research/Codes/Hybrid-vpic/data_ip_shock/particle_data_8/T.10000/'


def download_folder(sftp, remote_folder, local_folder):
    # 确保本地目标文件夹存在
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    # 获取远程文件夹中的所有文件和子文件夹
    for item in sftp.listdir_attr(remote_folder):
        remote_item_path = os.path.join(remote_folder, item.filename)
        local_item_path = os.path.join(local_folder, item.filename)
        if item.st_mode & 0o40000:  # 如果是文件夹
            download_folder(sftp, remote_item_path, local_item_path)
        else:  # 如果是文件
            sftp.get(remote_item_path, local_item_path)
            print(f'文件 {remote_item_path} 已成功下载到 {local_item_path}')


# 创建 SSH 客户端
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # 连接到服务器
    ssh.connect(server_ip, port=server_port, username=username, password=password)

    # 创建 SFTP 客户端
    sftp = ssh.open_sftp()

    # 下载文件夹
    download_folder(sftp, remote_folder_path, local_folder_path)

    print(f'文件夹 {remote_folder_path} 已成功下载到 {local_folder_path}')

    # 关闭 SFTP 连接
    sftp.close()
except Exception as e:
    print(f'下载文件夹时出错: {e}')
finally:
    # 关闭 SSH 连接
    ssh.close()
