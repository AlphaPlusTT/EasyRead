import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
# from src.tools import gen_dx_bx, cumsum_trick, QuickCumsum

import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob
import torch.utils.data
# from src.tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx

from src.data import SegmentationData
from src.models import Up, BevEncode
from src.tools import get_rot
from src.data import worker_rnd_init
import pandas as pd
from nuscenes.map_expansion.map_api import locations
from nuscenes.map_expansion.map_api import NuScenesMap


# UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
matplotlib.use('TkAgg')


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_3d(data, grid=True, axis=True, invert_x=False, invert_y=False, y_range=None, x_range=None):
    data = data.view(-1, 3)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if not grid:
        ax.grid(None)
    if not axis:
        ax.axis('off')
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if y_range:
        assert isinstance(y_range, list) and len(y_range) == 2 and y_range[0] <= y_range[1]
        ax.set_ylim(ymin=y_range[0], ymax=y_range[1])
        keep = (data[:, 1] < y_range[1]) & (data[:, 1] > y_range[0])
        data = data[keep]
    if x_range:
        assert isinstance(x_range, list) and len(x_range) == 2 and x_range[0] <= x_range[1]
        ax.set_xlim(xmin=x_range[0], xmax=x_range[1])
        keep = (data[:, 0] < x_range[1]) & (data[:, 0] > x_range[0])
        data = data[keep]
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


same_seeds(0)
# -----------------------------------------config start-----------------------------------------
dataroot = '/home/zed/data/nuscenes'  # 数据集路径
nepochs = 10000  # 训练最大的epoch数
gpuid = 0  # gpu的序号

H = 900
W = 1600  # 图片大小
resize_lim = (0.193, 0.225)  # resize的范围
final_dim = (128, 352)  # 数据预处理之后最终的图片大小
bot_pct_lim = (0.0, 0.22)  # 裁剪图片时，图像底部裁剪掉部分所占比例范围
rot_lim = (-5.4, 5.4)  # 训练时旋转图片的角度范围
rand_flip = True  # # 是否随机翻转
ncams = 5  # 训练时选择的相机通道数
max_grad_norm = 5.0
pos_weight = 2.13  # 损失函数中给正样本项损失乘的权重系数
logdir = './runs'  # 日志的输出文件

xbound = [-50.0, 50.0, 0.5]  # 限制x方向的范围并划分网格
ybound = [-50.0, 50.0, 0.5]  # 限制y方向的范围并划分网格
zbound = [-10.0, 10.0, 20.0]  # 限制z方向的范围并划分网格
dbound = [4.0, 45.0, 1.0]  # 限制深度方向的范围并划分网格

bsz = 2  # batchsize
nworkers = 0  # 线程数
lr = 1e-3  # 学习率
weight_decay = 1e-7  # 权重衰减系数

grid_conf = {  # 网格配置
    'xbound': xbound,
    'ybound': ybound,
    'zbound': zbound,
    'dbound': dbound,
}
data_aug_conf = {  # 数据增强配置
    'resize_lim': resize_lim,
    'final_dim': final_dim,
    'rot_lim': rot_lim,
    'H': H, 'W': W,
    'rand_flip': rand_flip,
    'bot_pct_lim': bot_pct_lim,
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': ncams,
}
device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
# -----------------------------------------config end-----------------------------------------


# -----------------------------------------------SegmentationData start-----------------------------------------------
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)  # 加载nuScenes数据集

traindata = SegmentationData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                             grid_conf=grid_conf)
'''
# ---------------------------------------------prepro start---------------------------------------------
is_train = True
# filter by scene split
split = {
    'v1.0-trainval': {True: 'train', False: 'val'},
    'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
}[nusc.version][is_train]

scenes = create_splits_scenes()[split]
# print('scenes:', scenes)  # scenes: ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796',
# 'scene-1077', 'scene-1094', 'scene-1100']

samples = [samp for samp in nusc.sample]

# remove samples that aren't in this split
samples = [samp for samp in samples if
           nusc.get('scene', samp['scene_token'])['name'] in scenes]

# sort by scene, timestamp (only to make chronological viz easier)
samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

index = 100
rec = samples[index]  # 按索引取出sample
# ---------------------------------------------prepro end---------------------------------------------
# ------------------------------------------choose_cams start-----------------------------------------
if is_train and data_aug_conf['Ncams'] < len(data_aug_conf['cams']):
    cams = np.random.choice(data_aug_conf['cams'], data_aug_conf['Ncams'], replace=False)
else:
    cams = data_aug_conf['cams']
# print('cmas:', cams) # cmas: ['CAM_FRONT_RIGHT' 'CAM_BACK' 'CAM_FRONT_LEFT' 'CAM_BACK_LEFT' 'CAM_BACK_RIGHT']
# ------------------------------------------choose_cams end-------------------------------------------
# ----------------------------------------get_image_data start----------------------------------------
# cam = next(iter(cams))
cam = 'CAM_FRONT'
samp = nusc.get('sample_data', rec['data'][cam])  # 根据相机通道选择对应的sample_data
imgname = os.path.join(nusc.dataroot, samp['filename'])  # 图片的路径
# print(samp['filename'])  # samples/CAM_BACK_RIGHT/n008-2018-08-30-15-16-55-0400__CAM_BACK_RIGHT__1535657108278113.jpg
img = Image.open(imgname)
# print('img.size:', img.size)  # img.size: (1600, 900)
post_rot = torch.eye(2)  # 增强前后像素点坐标的旋转对应关系
post_tran = torch.zeros(2)  # 增强前后像素点坐标的平移关系

sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])  # 相机record
intrin = torch.Tensor(sens['camera_intrinsic'])  # 相机内参
rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  # 相机坐标系相对于ego坐标系的旋转矩阵
tran = torch.Tensor(sens['translation'])  # 相机坐标系相对于ego坐标系的平移矩阵

# map expansion
egopose = nusc.get('ego_pose', samp['ego_pose_token'])
ego_rot = torch.tensor(Quaternion(egopose['rotation']).rotation_matrix)
ego_tran = torch.tensor(egopose['translation'])

# ---------------------------------sample_augmentation start---------------------------------
H, W = data_aug_conf['H'], data_aug_conf['W']  # H:900, W:1600
fH, fW = data_aug_conf['final_dim']  # (128, 352)，表示变换之后最终的图像大小
if is_train:  # 训练集数据增强
    resize = np.random.uniform(*data_aug_conf['resize_lim'])  # resize_lim = (0.193, 0.225)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim'])) * newH) - fH
    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    if data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
        flip = True
    rotate = np.random.uniform(*data_aug_conf['rot_lim'])
# else:  # 测试集数据增强
#     resize = max(fH / H, fW / W)  # 缩小的倍数取二者较大值
#     resize_dims = (int(W * resize), int(H * resize))  # 保证H和W以相同的倍数缩放
#     newW, newH = resize_dims  # (352,198)
#     crop_h = int((1 - np.mean(data_aug_conf['bot_pct_lim'])) * newH) - fH  # 48
#     crop_w = int(max(0, newW - fW) / 2)  # 0
#     crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
#     flip = False  # 不翻转
#     rotate = 0  # 不旋转
# print('resize, resize_dims, crop, flip, rotate:', resize, resize_dims, crop, flip, rotate)
# resize, resize_dims, crop, flip, rotate: 0.22410423924148012 (358, 201) (2, 54, 354, 182) True 2.887731852897275
# ----------------------------------sample_augmentation end----------------------------------
# ------------------------------------img_transform start------------------------------------
# adjust image
img = img.resize(resize_dims)  # 图像缩放
img = img.crop(crop)  # 图像裁剪
if flip:
    img = img.transpose(method=Image.FLIP_LEFT_RIGHT)  # 左右翻转
img = img.rotate(rotate)  # 旋转
# img.show()
# img.save('image_aug.png')
# post-homography transformation
# 数据增强后的图像上的某一点的坐标需要对应回增强前的坐标
post_rot *= resize
post_tran -= torch.Tensor(crop[:2])
if flip:
    A = torch.Tensor([[-1, 0], [0, 1]])
    b = torch.Tensor([crop[2] - crop[0], 0])
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
A = get_rot(rotate / 180 * np.pi)  # 得到数据增强时旋转操作的旋转矩阵
b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2  # 裁剪保留部分图像的中心坐标 (176, 64)
b = A.matmul(-b) + b  # 0
post_rot = A.matmul(post_rot)
post_tran = A.matmul(post_tran) + b
# -------------------------------------img_transform end-------------------------------------
# for convenience, make augmentation matrices 3x3
post_tran_2 = torch.zeros(3)
post_rot_2 = torch.eye(3)
post_tran_2[:2] = post_tran
post_rot_2[:2, :2] = post_rot

# imgs.append(normalize_img(img))
# intrins.append(intrin)
# rots.append(rot)
# trans.append(tran)
# post_rots.append(post_rot)
# post_trans.append(post_tran)
# -----------------------------------------get_image_data end-----------------------------------------
# ------------------------------------------get_binimg start------------------------------------------
# xbound = [-50.0, 50.0, 0.5]  # 限制x方向的范围并划分网格
# ybound = [-50.0, 50.0, 0.5]  # 限制y方向的范围并划分网格
# zbound = [-10.0, 10.0, 20.0]  # 限制z方向的范围并划分网格
# dbound = [4.0, 45.0, 1.0]  # 限制深度方向的范围并划分网格
dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
# print('dx:', dx)  # dx: tensor([ 0.5000, 0.5000, 20.0000]) 分别为x, y, z三个方向上的网格尺寸
bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
# print('bx:', bx)  # bx: tensor([-49.7500, -49.7500, 0.0000])  分别为x, y, z三个方向上第一个格子的坐标
nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
# print('nx:', nx)  # nx: tensor([200, 200, 1])  分别为x, y, z三个方向上格子的数量
dx_np = dx.numpy()
bx_np = bx.numpy()
nx_np = nx.numpy()

# 得到自车坐标系相对于地图全局坐标系的位姿
egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
trans = -np.array(egopose['translation'])
rot = Quaternion(egopose['rotation']).inverse
binimg = np.zeros((nx[0], nx[1]))  # [200, 200]

patch_h = ybound[1] - ybound[0]
patch_w = xbound[1] - xbound[0]
map_pose = np.array(egopose['translation'])[:2]
patch_box = (map_pose[0], map_pose[1], patch_h, patch_w)
canvas_h = int(patch_h / ybound[2])
canvas_w = int(patch_w / xbound[2])
canvas_size = (canvas_h, canvas_w)

rotation = Quaternion(egopose['rotation']).rotation_matrix
v = np.dot(rotation, np.array([1, 0, 0]))
yaw = np.arctan2(v[1], v[0])
patch_angle = yaw / np.pi * 180

layer_names = ['drivable_area']
maps = {}
for location in locations:
    maps[location] = NuScenesMap(dataroot, location)

location = nusc.get('log', nusc.get('scene', rec['scene_token'])['log_token'])['location']
masks = maps[location].get_map_mask(
    patch_box=patch_box,
    patch_angle=patch_angle,
    layer_names=layer_names,
    canvas_size=canvas_size,
)
masks = masks.transpose(0, 2, 1)
masks = masks.astype(np.bool)
binimg[masks[0]] = 0.5
cv2.imshow('binimg', binimg)
key = cv2.waitKey(0)

for tok in rec['anns']:  # 遍历该sample的每个annotation token
    inst = nusc.get('sample_annotation', tok)  # 找到该annotation
    # add category for lyft
    if not inst['category_name'].split('.')[0] == 'vehicle':  # 只关注车辆
        continue
    box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))  # 参数分别为center, size, orientation
    box.translate(trans)  # 将box的center坐标从全局坐标系转换到自车坐标系下
    box.rotate(rot)  # 将box的center坐标从全局坐标系转换到自车坐标系下

    pts = box.bottom_corners()[:2].T  # 三维边界框取底面的四个角的(x,y)值后转置, 4x2
    pts = np.round(
        (pts - bx_np[:2] + dx_np[:2] / 2.) / dx_np[:2]
    ).astype(np.int32)  # 将box的实际坐标对应到网格坐标，同时将坐标范围[-50,50]平移到[0,100]
    pts[:, [1, 0]] = pts[:, [0, 1]]  # 把(x,y)的形式换成(y,x)的形式
    cv2.fillPoly(binimg, [pts], 1.0)  # 在网格中画出box
cv2.imshow('binimg', binimg)
key = cv2.waitKey(0)
# cv2.imshow('seg', masks.squeeze(0)*255)
# key = cv2.waitKey(0)
import matplotlib.image as mpimg
image_paths = []
for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT' , 'CAM_BACK', 'CAM_BACK_RIGHT']:
    samp = nusc.get('sample_data', rec['data'][cam])  # 根据相机通道选择对应的sample_data
    image_path = os.path.join(nusc.dataroot, samp['filename'])  # 图片的路径
    image_paths.append(image_path)
# 创建一个2行3列的子图布局
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
# 循环读取图片并在子图中显示
for i, ax in enumerate(axes.flat):
    if i < len(image_paths):
        # 读取图片
        img = mpimg.imread(image_paths[i])
        # 显示图片
        ax.imshow(img)
        # 设置子图标题
        ax.set_title(f"Image {i + 1}")
        # 隐藏坐标轴
        ax.axis("off")
# 调整布局
plt.tight_layout()
# 显示图形
plt.show()
# -------------------------------------------get_binimg end-------------------------------------------
# -----------------------------------------------SegmentationData end-----------------------------------------------
'''

# -----------------------------------------------load data start-----------------------------------------------
trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                          shuffle=False,
                                          num_workers=nworkers,
                                          drop_last=True,
                                          worker_init_fn=worker_rnd_init)  # 给每个线程设置随机种子

imgs, rots, trans, intrins, post_rots, post_trans, binimgs = next(iter(trainloader))
imgs = imgs.to(device)
rots = rots.to(device)
trans = trans.to(device)
intrins = intrins.to(device)
post_rots = post_rots.to(device)
post_trans = post_trans.to(device)
binimgs = binimgs.to(device)
# print('batchi:', batchi)  # batchi: 0
# print('imgs.shape:', imgs.shape)  # imgs.shape: torch.Size([2, 5, 3, 128, 352]), [B, N, C, H, W]
# print('rots.shape:', rots.shape)  # rots.shape: torch.Size([2, 5, 3, 3])
# print('trans.shape:', trans.shape)  # trans.shape: torch.Size([2, 5, 3])
# print('intrins.shape:', intrins.shape)  # intrins.shape: torch.Size([2, 5, 3, 3])
# print('post_rots.shape:', post_rots.shape)  # post_rots.shape: torch.Size([2, 5, 3, 3])
# print('post_trans.shape:', post_trans.shape)  # post_trans.shape: torch.Size([2, 5, 3])
# print('binimgs.shape:', binimgs.shape)  # binimgs.shape: torch.Size([2, 1, 200, 200])
# -----------------------------------------------load data end-----------------------------------------------


# -----------------------------------------------------model start-----------------------------------------------------
# --------------------------------------------model init start--------------------------------------------
outC = 1
downsample = 16  # 下采样倍数
camC = 64  # 图像特征维度
# xbound = [-50.0, 50.0, 0.5]  # 限制x方向的范围并划分网格
# ybound = [-50.0, 50.0, 0.5]  # 限制y方向的范围并划分网格
# zbound = [-10.0, 10.0, 20.0]  # 限制z方向的范围并划分网格
# dbound = [4.0, 45.0, 1.0]  # 限制深度方向的范围并划分网格
dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]], device=device)
# print('dx:', dx)  # dx: tensor([ 0.5000, 0.5000, 20.0000]) 分别为x, y, z三个方向上的网格尺寸
bx = torch.tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]], device=device)
# print('bx:', bx)  # bx: tensor([-49.7500, -49.7500, 0.0000])  分别为x, y, z三个方向上第一个点的坐标
nx = torch.tensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]], dtype=torch.long, device=device)
# print('nx:', nx)  # nx: tensor([200, 200, 1])  分别为x, y, z三个方向上格子的数量
dx = nn.Parameter(dx, requires_grad=False)  # [0.5, 0.5, 20]
bx = nn.Parameter(bx, requires_grad=False)  # [-49.5, -49.5, 0]
nx = nn.Parameter(nx, requires_grad=False)  # [200, 200, 1]

# make grid in image plane
ogfH, ogfW = data_aug_conf['final_dim']  # 数据预处理之后的图片大小  ogfH:128  ogfW:352
fH, fW = ogfH // downsample, ogfW // downsample  # 下采样16倍后图像大小  fH: 8  fW: 22
# dbound = [4.0, 45.0, 1.0]  # 限制深度方向的范围并划分网格
ds = torch.arange(*grid_conf['dbound'], dtype=torch.float, device=device).view(-1, 1, 1).expand(-1, fH, fW)
# ds.shape: torch.Size([41, 8, 22])
# print(ds[1, :, :])
D, _, _ = ds.shape  # D: 41 表示深度方向上网格的数量
xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float, device=device).view(1, 1, fW).expand(D, fH, fW)
# print('xs.shape:', xs.shape) #  xs.shape: torch.Size([41, 8, 22])
ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float, device=device).view(1, fH, 1).expand(D, fH, fW)
# print('ys.shape:', ys.shape)  # ys.shape: torch.Size([41, 8, 22])

frustum = torch.stack((xs, ys, ds), -1)
# print('frustum.shape:', frustum.shape)  # frustum.shape: torch.Size([41, 8, 22, 3]) # D x H x W x 3
frustum = nn.Parameter(frustum, requires_grad=False)
# -----------------------------------CamEncode init start-----------------------------------
CamEncode_trunk = EfficientNet.from_pretrained("efficientnet-b0").to(device)
CamEncode_up1 = Up(320 + 112, 512).to(device)
CamEncode_depthnet = nn.Conv2d(512, D + camC, kernel_size=1, padding=0).to(device)
# ------------------------------------CamEncode init end------------------------------------
bevencode = BevEncode(inC=camC, outC=outC).to(device)
use_quickcumsum = True
# ---------------------------------------------model init end---------------------------------------------
# ------------------------------------------model forward start-------------------------------------------
# def forward(self, x, rots, trans, intrins, post_rots, post_trans):
#     x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
#     x = self.bevencode(x)
#     return x
# x: imgs.shape: torch.Size([2, 5, 3, 128, 352]), [B, N, C, H, W]
# rots.shape: torch.Size([2, 5, 3, 3])
# trans.shape: torch.Size([2, 5, 3])
# intrins.shape: torch.Size([2, 5, 3, 3])
# post_rots.shape: torch.Size([2, 5, 3, 3])
# post_trans.shape: torch.Size([2, 5, 3])
# ------------------------------------get_geometry start------------------------------------
B, N, _ = trans.shape  # B:2(Batch size), N:5(Number of cameras)
# undo post-transformation
# frustum.shape: torch.Size([41, 8, 22, 3]) # D x H x W x 3(u, v, d)
# post_trans.shape: torch.Size([2, 5, 3])  # post_trans[:, :, 2] has no effect
points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
# print('points.shape:', points.shape) # points.shape: torch.Size([2, 5, 41, 8, 22, 3])
points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
# print('points.shape:', points.shape)  # points.shape: torch.Size([2, 5, 41, 8, 22, 3, 1])
# 上述代码将数据增强后的图像坐标映射回原图坐标

# cam_to_ego
points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
# (u,v,d)-->(ud,vd,d) 齐次坐标
# rots: 2D --> 3D
combine = rots.matmul(torch.inverse(intrins))
points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
points += trans.view(B, N, 1, 1, 1, 3)
# 像素坐标[u,v,1]-->车体坐标[x,y,z]
# print('points.shape:', points.shape)  # torch.Size([2, 5, 41, 8, 22, 3]) (3:x, y, z)
# plot_3d(points[0, 0, ...].cpu(), y_range=[-33, -30], x_range=[-13, -10])
# plot_3d(points[0, 2, ...].cpu())
# patial = points[0, 0, 5:41:15, 1:8:3, 2:22:3, :]
# print(patial.shape)  # torch.Size([3, 3, 7, 3])
# for d in [0, 1, 2]:
#     data = patial[d]
#     print(data.shape)
#     data = data.transpose(1, 2).contiguous().view(9, 7)
#     df = pd.DataFrame(data)
#     df.to_excel('./output{}.xlsx'.format(d), index=False, header=False)

# -------------------------------------get_geometry end-------------------------------------
# -----------------------------------get_cam_feats start -----------------------------------
x = imgs
# x: torch.Size([2, 5, 3, 128, 352]), [B, N, C, H, W]
B, N, C, imH, imW = x.shape
x = x.view(B * N, C, imH, imW)
# x: torch.Size([10, 3, 128, 352]), [BxN, C, H, W]
# -----------------------------CamEncode forward start-----------------------------
# ------------------------get_eff_depth start------------------------
# adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
endpoints = dict()
# Stem
x = CamEncode_trunk._swish(CamEncode_trunk._bn0(CamEncode_trunk._conv_stem(x)))  # x: 10 x 32 x 64 x 176
prev_x = x
# Blocks
for idx, block in enumerate(CamEncode_trunk._blocks):
    drop_connect_rate = CamEncode_trunk._global_params.drop_connect_rate
    if drop_connect_rate:
        drop_connect_rate *= float(idx) / len(CamEncode_trunk._blocks)  # scale drop connect_rate
    x = block(x, drop_connect_rate=drop_connect_rate)
    if prev_x.size(2) > x.size(2):
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
    prev_x = x
# 以下采样为分界线划分为若干个block，保存每个block的最后一层特征
# Head
endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
# reduction_1 torch.Size([10, 16, 64, 176])
# reduction_2 torch.Size([10, 24, 32, 88])
# reduction_3 torch.Size([10, 40, 16, 44])
# reduction_4 torch.Size([10, 112, 8, 22])
# reduction_5 torch.Size([10, 320, 4, 11])
x = CamEncode_up1(endpoints['reduction_5'], endpoints['reduction_4'])
# print(endpoints.keys())  # dict_keys(['reduction_1', 'reduction_2', 'reduction_3', 'reduction_4', 'reduction_5'])
# print('x.shape:', x.shape)  # torch.Size([10, 512, 8, 22])
# -------------------------get_eff_depth end-------------------------
# camC = 64  # 图像特征维度
# D = 41  # 深度方向上的栅格数量
# CamEncode_depthnet = nn.Conv2d(512, D + camC, kernel_size=1, padding=0)
x = CamEncode_depthnet(x)
# print(x.shape)  # torch.Size([10, 105, 8, 22])
depth = x[:, :D].softmax(dim=1)
# print(depth.shape)  # torch.Size([10, 41, 8, 22])
# print(depth.unsqueeze(1).shape)  # torch.Size([10, 1, 41, 8, 22])
# print(x[:, D:(D + camC)].unsqueeze(2).shape)  # torch.Size([10, 64, 1, 8, 22])
x = depth.unsqueeze(1) * x[:, D:(D + camC)].unsqueeze(2)
# print(x.shape)  # torch.Size([10, 64, 41, 8, 22])
# ------------------------------CamEncode forward end------------------------------
x = x.view(B, N, camC, D, imH // downsample, imW // downsample)  # torch.Size([2, 5, 64, 41, 8, 22])
x = x.permute(0, 1, 3, 4, 5, 2)
# print(x.shape)  # torch.Size([2, 5, 41, 8, 22, 64]) [B, N, D, H, W, C]
# ------------------------------------get_cam_feats end-------------------------------------
# -----------------------------------voxel_pooling start------------------------------------
geom_feats = points  # torch.Size([2, 5, 41, 8, 22, 3]) (3:x, y, z)
B, N, D, H, W, C = x.shape
Nprime = B * N * D * H * W
# flatten x
x = x.reshape(Nprime, C)  # torch.Size([72160, 64])
# flatten indices
# xbound = [-50.0, 50.0, 0.5]  # 限制x方向的范围并划分网格
# ybound = [-50.0, 50.0, 0.5]  # 限制y方向的范围并划分网格
# zbound = [-10.0, 10.0, 20.0]  # 限制z方向的范围并划分网格
# dbound = [4.0, 45.0, 1.0]  # 限制深度方向的范围并划分网格
# 将[-50,50] [-10 10]的范围平移到[0,100] [0,20]，计算栅格坐标并取整
# print('dx:', dx)  # dx: tensor([ 0.5000, 0.5000, 20.0000])  分别为x, y, z三个方向上的网格尺寸
# print('bx:', bx)  # bx: tensor([-49.7500, -49.7500, 0.0000])  分别为x, y, z三个方向上第一个格子的坐标
# print('points.shape:', points.shape)  # torch.Size([2, 5, 41, 8, 22, 3]) (3:x, y, z)
# 目前在车体坐标系定义了两个栅格组，一个栅格组是geom_feats（points），每个栅格包含xyz坐标；另一个是由bx、dx和nx定义的空栅格组
geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()  # 将xyz坐标转换为voxel索引
# plot_3d(geom_feats[0, 0, ...])

geom_feats = geom_feats.view(Nprime, 3)
batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
# print(batch_ix.shape)  # torch.Size([72160, 1])
geom_feats = torch.cat((geom_feats, batch_ix), 1)
# print(geom_feats.shape)  # torch.Size([72160, 4]) (4: x, y, z, b)

# filter out points that are outside box
# print('nx:', nx)  # nx: tensor([200, 200, 1])  分别为x, y, z三个方向上格子的数量
# 过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
       & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
       & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
x = x[kept]
geom_feats = geom_feats[kept]
# print(x.shape)  # torch.Size([69219, 64])
# print(geom_feats.shape)  # torch.Size([69219, 4]) (4: x, y, z, b)

# get tensors from the same voxel next to each other
# 每个batch里面的每个栅格有唯一确定rank值
ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
        + geom_feats[:, 1] * (nx[2] * B) \
        + geom_feats[:, 2] * B \
        + geom_feats[:, 3]
# rank就是voxel的索引且一一对应
sorts = ranks.argsort()
# print(ranks.shape)  # torch.Size([69219])
# print(sorts.shape)  # torch.Size([69219])
x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
# 通过排序将voxel索引相近的排在一起,更重要的是把在同一个voxel里面的排到一起
# print(x.shape)  # torch.Size([69219, 64])
# print(geom_feats.shape)  # torch.Size([69219, 4])
# print(ranks.shape)  # torch.Size([69219])

# cumsum trick
x = x.cumsum(0)
kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
kept[:-1] = (ranks[1:] != ranks[:-1])

x, geom_feats = x[kept], geom_feats[kept]
x = torch.cat((x[:1], x[1:] - x[:-1]))
# print(x.shape) # torch.Size([17375, 64])

# griddify (B x C x Z x X x Y)
final = torch.zeros((B, C, nx[2], nx[0], nx[1]), device=x.device)  # final: 2 x 64 x 1 x 200 x 200
final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

# collapse Z
final = torch.cat(final.unbind(dim=2), 1)  # implicit!  # splat
# final_test = final.squeeze(2)
# print(torch.sum((final-final_test)**2))  # tensor(0., grad_fn=<SumBackward0>)
# print(final.shape)  # torch.Size([2, 64, 200, 200])
# ------------------------------------voxel_pooling end-------------------------------------
x = bevencode(final)
# print(x.shape)  # torch.Size([2, 1, 200, 200])
# return x
# ------------------------------------------model forward end-------------------------------------------
# ------------------------------------------------------model end------------------------------------------------------


# ---------------------------------------------------SimpleLoss start---------------------------------------------------
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight])).to(device)
loss = loss_fn(x, binimgs)
# binimgs.shape: torch.Size([2, 1, 200, 200])
# ----------------------------------------------------SimpleLoss end----------------------------------------------------
