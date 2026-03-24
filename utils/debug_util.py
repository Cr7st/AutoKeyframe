import os
# import imageio
# import matlib.pyplot as plt
import numpy as np
# import torch
# from termcolor import colored
import subprocess
import inspect
import time
import datetime
import shutil

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H %M %S')

def toc():
    return time.time() * 1000

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(cmd, color)

def mylog(text):
    myprint(text, 'info')

def log_time(text):
    strf = get_time()
    print(strf, text)

def mywarn(text):
    myprint(text, 'warn')


def myerror(text):
    myprint(text, 'error')

def get_path_pre(name, use_time = False):
    if use_time:
        output_dir = f"debug/{get_time()}_{name}"
    else: output_dir = f"debug/{name}" 
    return output_dir


def run_cmd(cmd, verbo=True, bg=False):
    if verbo: myprint('[run] ' + cmd, 'run')
    if bg:
        args = cmd.split()
        # print(args)
        p = subprocess.Popen(args)
        return [p]
    else:
        exit_status = os.system(cmd)
        if exit_status != 0:
            raise RuntimeError
        return []

def mkdir(path):
    if os.path.exists(path):
        return 0
    mylog('mkdir {}'.format(path))
    os.makedirs(path, exist_ok=True)

def cp(srcname, dstname):
    mkdir(os.join(os.path.dirname(dstname)))
    shutil.copyfile(srcname, dstname)

def check_exists(path):
    flag1 = os.path.isfile(path) and os.path.exists(path)
    flag2 = os.path.isdir(path) and len(os.listdir(path)) >= 10
    return flag1 or flag2


def to_numpy(a)->np.ndarray:
    if type(a) == torch.Tensor:
        if a.is_cuda:
            a = a.cpu()
        return a.detach().numpy()
    elif type(a) == np.ndarray:
        return a
    else:
        try:
            return np.array(a)
        except:
            raise TypeError('Unsupported data type')

def copy_file_with_increment(file_path, target_file_path):
    # 分离文件名和扩展名
    base, extension = os.path.splitext(target_file_path)
    counter = 1

    # 如果文件已存在，增加计数器直到找到一个不存在的文件名
    while os.path.exists(target_file_path):
        target_file_path = f"{base}_{counter}{extension}"
        counter += 1

    # 复制文件
    shutil.copy(file_path, target_file_path)
    return target_file_path

def copy_folder_with_increment(source_folder, target_folder):
    # 获取目标文件夹的基本名称和父目录
    base_name = os.path.basename(target_folder)
    parent_dir = os.path.dirname(target_folder)
    counter = 1

    # 如果文件夹已存在，增加计数器直到找到一个不存在的文件夹名
    while os.path.exists(target_folder):
        target_folder = os.path.join(parent_dir, f"{base_name}_{counter}")
        counter += 1

    # 复制文件夹
    shutil.copytree(source_folder, target_folder)
    return target_folder

'''
def save_debug(name, a, time = False):
    if not cfg.debug: return
    np.save( f'{get_pre(name, time)}.npy', to_numpy(a))


def save_point_cloud(point_cloud, filename):
    if not cfg.debug: return
    return
    point_cloud = to_numpy(point_cloud)
    
    # 将numpy数组转换为open3d的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 保存PointCloud到文件
    o3d.io.write_point_cloud(get_pre(filename), pcd)

def output_debug_log(output, name):
    if not cfg.debug: return
    if type(output) != str:
        output = str(output)

    with open(f"{get_pre(name)}.log", 'w') as f:
        f.write(output)
        f.write('\n')

def save_debug(a, name, time = False):
    if not cfg.debug: return
    np.save( f'{get_pre(name, time)}.npy', to_numpy(a))

def save_img(img, name, time = False):
    if not cfg.debug: return
    img = to_numpy(img)
    img = img * 255
    img = img.astype(np.uint8)
    imageio.imwrite(f'{get_pre(name, time)}.png', img)

def save_imgs(msks, name, time = False):
    if not cfg.debug: return
    """Save imgs in a grid"""
    n = len(msks)
    fig, axs = plt.subplots(-(-n//3), 3, figsize=(15, 5*-(-n//3)))
    for i, ax in enumerate(axs.flat):
        ax.axis('off')
        if i<n:
            ax.imshow(msks[i])
    plt.tight_layout()
    plt.savefig(f'{get_pre(name, time)}.png')
    plt.close()
'''
