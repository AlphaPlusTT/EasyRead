from projects.mmdet3d_plugin.datasets.custom_script import load_data_instance
# from mmcv.parallel import DataContainer
import copy
import torch
from mmdet.models.backbones import ResNet
from mmdet.models.necks.fpn import FPN
import numpy as np
from PIL import Image, ImageDraw
import torch.nn as nn
from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding
from torchvision.transforms.functional import rotate
from mmcv.utils import TORCH_VERSION, digit_version
import warnings
from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import torch.nn.functional as F
# from mmcv.cnn.bricks.transformer import FFN
from torch.nn import Sequential
# from mmdet.models.utils.transformer import DetrTransformerDecoderLayer
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from projects.mmdet3d_plugin.bevformer.modules.decoder import inverse_sigmoid
# from mmdet.core import multi_apply
from mmdet.core.bbox.assigners import AssignResult
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
from mmdet.core.bbox.samplers import SamplingResult
from mmdet.models.losses.focal_loss import FocalLoss
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
import math


# ---------custom start-----------
def show_tensor_img(tensor: torch.Tensor, denorm=True):
    if denorm:
        assert torch.min(tensor) < 0
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    assert tensor.shape[2] == 3
    numpy_array = tensor.permute(0, 1, 3, 4, 2).cpu().numpy()
    batch_size, camera_num, height, width, channel = numpy_array.shape
    mean = mean[None, None, None, None, :]
    # std = np.expand_dims(std, axis=[0, 1, 2, 3])
    std = std[None, None, None, None, :]
    numpy_array = numpy_array * std + mean
    numpy_array = np.clip(numpy_array, 0, 255).astype('uint8')

    # 创建一个画布来合并图像
    canvas_width = width * camera_num
    canvas_height = height * batch_size
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    # 在画布上绘制图像
    draw = ImageDraw.Draw(canvas)
    for i in range(batch_size):
        for j in range(camera_num):
            # 获取当前图像的PIL Image
            current_image = Image.fromarray(numpy_array[i, j])  # 将数据缩放到0-255范围
            # 将当前图像粘贴到画布上
            canvas.paste(current_image, (j * width, i * height))

    # 显示画布
    canvas.show()
# ----------custom end------------


# ---------------------------------------------------load data start---------------------------------------------------
once_data = load_data_instance('once_data_idx_103.pkl')
# print(once_data)

img_metas = [once_data['img_metas'].data]
gt_bboxes_3d = [once_data['gt_bboxes_3d'].data]
gt_labels_3d = [once_data['gt_labels_3d'].data.cuda()]
img = once_data['img'].data[None, ...].cuda()  # torch.Size([1, 3, 6, 3, 480, 800])
# show_tensor_img(img[:, 0, ...])
# show_tensor_img(img[:, 1, ...])
# show_tensor_img(img[:, 2, ...])
# ----------------------------------------------------load data end----------------------------------------------------
# -------------------------------------------------forward train start-------------------------------------------------
len_queue = img.size(1)
prev_img = img[:, :-1, ...]  # torch.Size([1, 2, 6, 3, 480, 800])
img = img[:, -1, ...]  # torch.Size([1, 6, 3, 480, 800])
# print(torch.max(img), torch.min(img))
# show_tensor_img(img)


prev_img_metas = copy.deepcopy(img_metas)
# prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
# ------------------------------------------obtain_history_bev start------------------------------------------
imgs_queue = prev_img  # torch.Size([1, 2, 6, 3, 480, 800])
img_metas_list = prev_img_metas
with torch.no_grad():
    prev_bev = None
    bs, len_queue_1, num_cams, C, H, W = imgs_queue.shape  # torch.Size([1, 2, 6, 3, 480, 800])
    imgs_queue = imgs_queue.reshape(bs * len_queue_1, num_cams, C, H, W)  # torch.Size([2, 6, 3, 480, 800])
    # img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
    # ------------------------------------extract_feat start------------------------------------
    img_backbone_dict = dict(
        # type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch')
    img_backbone = ResNet(**img_backbone_dict).cuda()
    use_grid_mask = True

    """Extract features of images."""
    B = imgs_queue.size(0)  # torch.Size([2, 6, 3, 480, 800])
    if imgs_queue is not None:

        # input_shape = imgs_queue.shape[-2:]
        # # update real input shape of each single imgs_queue
        # for img_meta in img_metas:
        #     img_meta.update(input_shape=input_shape)

        if imgs_queue.dim() == 5 and imgs_queue.size(0) == 1:
            imgs_queue.squeeze_()
        elif imgs_queue.dim() == 5 and imgs_queue.size(0) > 1:
            B, N, C, H, W = imgs_queue.size()  # torch.Size([2, 6, 3, 480, 800])
            imgs_queue = imgs_queue.reshape(B * N, C, H, W)  # torch.Size([12, 3, 480, 800])
        if use_grid_mask:
            # imgs_queue = self.grid_mask(imgs_queue)
            # -----------------------------grid_mask start-----------------------------
            # x = imgs_queue
            gm_prob = 0.7
            gm_training = True
            gm_use_h = True
            gm_use_w = True
            gm_offset = False
            gm_mode = 1
            gm_rotate = 1
            gm_ratio = 0.5
            # self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            if np.random.rand() > gm_prob or not gm_training:
                # return x
                pass
            n, c, h, w = imgs_queue.size()
            imgs_queue = imgs_queue.view(-1, h, w)
            hh = int(1.5 * h)
            ww = int(1.5 * w)
            d = np.random.randint(2, h)
            l = min(max(int(d * gm_ratio + 0.5), 1), d - 1)
            mask = np.ones((hh, ww), np.float32)
            st_h = np.random.randint(d)
            st_w = np.random.randint(d)
            if gm_use_h:
                for i in range(hh // d):
                    s = d * i + st_h
                    t = min(s + l, hh)
                    mask[s:t, :] *= 0
            if gm_use_w:
                for i in range(ww // d):
                    s = d * i + st_w
                    t = min(s + l, ww)
                    mask[:, s:t] *= 0

            r = np.random.randint(gm_rotate)
            mask = Image.fromarray(np.uint8(mask))
            mask = mask.rotate(r)
            mask = np.array(mask)
            mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

            mask = torch.from_numpy(mask).to(imgs_queue.dtype).cuda()
            if gm_mode == 1:
                mask = 1 - mask
            mask = mask.expand_as(imgs_queue)
            if gm_offset:  # False
                offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(imgs_queue.dtype).cuda()
                imgs_queue = imgs_queue * mask + offset * (1 - mask)
            else:
                imgs_queue = imgs_queue * mask
            imgs_queue = imgs_queue.view(n, c, h, w)  # torch.Size([12, 3, 480, 800])
            # return x.view(n, c, h, w)
            # ------------------------------grid_mask end------------------------------
            # show_tensor_img(imgs_queue.view(bs * len_queue_1, num_cams, C, H, W))
        img_feats = img_backbone(imgs_queue)  # tuple(torch.Size([12, 2048, 15, 25]))
        # if isinstance(img_feats, dict):
        #     img_feats = list(img_feats.values())
    else:
        # return None
        pass
    with_img_neck = True
    if with_img_neck:
        img_neck_dict = dict(
            # type='FPN',
            in_channels=[2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=1,
            relu_before_extra_convs=True)
        img_neck = FPN(**img_neck_dict).cuda()
        img_feats = img_neck(img_feats)  # tuple(torch.Size([12, 256, 15, 25]))

    img_feats_reshaped = []
    for img_feat in img_feats:
        BN, C, H, W = img_feat.size()  # torch.Size([12, 256, 15, 25]
        if len_queue_1 is not None:  # len_queue_1 = 2
            img_feats_reshaped.append(img_feat.view(int(B / len_queue_1), len_queue_1, int(BN / B), C, H, W))
            # [torch.Size([1, 2, 6, 256, 15, 25]] # B = bs * len_queue_1 # BN = bs * len_queue_1 * num_cams
        else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
    # return img_feats_reshaped
    # -------------------------------------extract_feat end-------------------------------------
    img_feats_list = img_feats_reshaped  # [torch.Size([1, 2, 6, 256, 15, 25]]
    bev_h, bev_w = 50, 50
    embed_dims = 256
    num_query = 900  # TODO: Why 900?
    BH_bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims).cuda()
    BH_query_embedding = nn.Embedding(num_query, embed_dims * 2).cuda()
    pos_dim = 128
    BH_positional_encoding = LearnedPositionalEncoding(pos_dim, bev_h, bev_w).cuda()
    rotate_prev_bev = True
    rotate_center = [100, 100]
    can_bus_mlp = nn.Sequential(
        nn.Linear(18, embed_dims // 2),
        nn.ReLU(inplace=True),
        nn.Linear(embed_dims // 2, embed_dims),
        nn.ReLU(inplace=True),).cuda()
    use_can_bus = True
    use_cams_embeds = True
    cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims)).cuda()
    # TODO: cams_embeds和bev_pos是类似的作用吗？

    for idx in range(len_queue_1):  # len_queue_1 = 2
        ohb_img_metas = [each[idx] for each in img_metas_list]
        if not ohb_img_metas[0]['prev_bev_exists']:  # False #2 True prev_bev: torch.Size([1, 50*50, 256])
            prev_bev = None
        # img_feats = self.extract_feat(img=img, img_metas=ohb_img_metas)
        # 获得队列中当下索引（idx）的环视图像特征
        img_feats = [each_scale[:, idx] for each_scale in img_feats_list]  # [torch.Size([1, 6, 256, 15, 25])]
        # prev_bev = self.pts_bbox_head(img_feats, ohb_img_metas, prev_bev, only_bev=True)
        # -------------------------------------pts_bbox_head start-------------------------------------
        # BEVFormerHead
        mlvl_feats = img_feats  # [torch.Size([1, 6, 256, 15, 25])]

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = BH_query_embedding.weight.to(dtype)  # torch.Size([900, 512])
        bev_queries = BH_bev_embedding.weight.to(dtype)  # torch.Size([50 * 50, 256])
        # TODO: Is bev_queries necessary? What role can it play? Also other *_embed.

        bev_mask = torch.zeros((bs, bev_h, bev_w), device=bev_queries.device).to(dtype)
        # bev_pos = self.positional_encoding(bev_mask).to(dtype)
        # 这里只用到了bev_mask的shape和device信息，与bev_mask的值无关
        # bev_pos = BH_positional_encoding(bev_mask).to(dtype)  # torch.Size([1, 256, 50, 50])
        # ------------------------------positional_encoding start------------------------------
        # BH_positional_encoding
        pe_num_feats, pe_row_num_embed, pe_col_num_embed = pos_dim, bev_h, bev_w
        pe_row_embed = nn.Embedding(pe_row_num_embed, pe_num_feats).cuda()
        pe_col_embed = nn.Embedding(pe_col_num_embed, pe_num_feats).cuda()

        pe_h, pe_w = bev_mask.shape[-2:]  # torch.Size([1, 50, 50])
        pe_x = torch.arange(pe_w, device=bev_mask.device)
        pe_y = torch.arange(pe_h, device=bev_mask.device)
        pe_x_embed = pe_col_embed(pe_x)  # torch.Size([50, 128])
        pe_y_embed = pe_row_embed(pe_y)  # torch.Size([50, 128])
        pe_pos = torch.cat((pe_x_embed.unsqueeze(0).repeat(pe_h, 1, 1), pe_y_embed.unsqueeze(1).repeat(1, pe_w, 1)),
                           dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(bev_mask.shape[0], 1, 1, 1)
        # return pos
        bev_pos = pe_pos  # torch.Size([1, 256, 50, 50])  TODO: nn.Parameter([1, 256, 50, 50])?
        # -------------------------------positional_encoding end-------------------------------

        only_bev = True
        # if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
        #     return self.transformer.get_bev_features(
        #         mlvl_feats,
        #         bev_queries,
        #         self.bev_h,
        #         self.bev_w,
        #         grid_length=(self.real_h / self.bev_h,
        #                      self.real_w / self.bev_w),
        #         bev_pos=bev_pos,
        #         img_metas=img_metas,
        #         prev_bev=prev_bev,
        #     )
        # -------------------------------get_bev_features start--------------------------------
        """
        obtain bev features.
        """
        bs = mlvl_feats[0].size(0)  # [torch.Size([1, 6, 256, 15, 25])]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)  # torch.Size([50 * 50, 1, 256])
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # torch.Size([50 * 50, 1, 256])

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0] for each in ohb_img_metas])
        delta_y = np.array([each['can_bus'][1] for each in ohb_img_metas])
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in ohb_img_metas])

        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        real_w = pc_range[3] - pc_range[0]  # 102.4
        real_h = pc_range[4] - pc_range[1]  # 102.4
        grid_length = (real_h / bev_h, real_w / bev_w)
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        use_shift = True
        shift_y = shift_y * use_shift
        shift_x = shift_x * use_shift
        # TODO: 第一帧图像得到的shift是从哪个时刻到哪个时刻的shift？
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy # torch.Size([1, 2]) # normalize 0~1

        num_feature_levels = 4
        level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims)).cuda()

        if prev_bev is not None:  # None
            if prev_bev.shape[1] == bev_h * bev_w:  #2 # torch.Size([1, 50*50, 256])
                prev_bev = prev_bev.permute(1, 0, 2)  #2 # torch.Size([50*50, 1, 256])
            if rotate_prev_bev:  #2 True
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = ohb_img_metas[i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    # TODO: rotate_center为什么要设置成100而不是中心？rotate center没有根据bev的尺寸进行调整？
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor([each['can_bus'] for each in ohb_img_metas])  # torch.Size([1, 18])
        can_bus = can_bus_mlp(can_bus)[None, :, :]  # torch.Size([1, 1, 256])
        bev_queries = bev_queries + can_bus * use_can_bus

        feat_flatten = []
        spatial_shapes = []
        # mlvl_feats: # [torch.Size([1, 6, 256, 15, 25])]
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape  # torch.Size([1, 6, 256, 15, 25])
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # torch.Size([1, 6, 256, 15*25]) -> torch.Size([6, 1, 15*25, 256])
            if use_cams_embeds:  # True
                feat = feat + cams_embeds[:, None, None, :].to(feat.dtype)  # TODO: cams_embeds跟环视相机的径向对称性有没有关系？
            feat = feat + level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)  # torch.Size([6, 1, 15*25, 256])
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)  # torch.Size([1, 2])
        # 将不同（H, W）的平面flatten为（H*W）的向量后cat在一起每个平面的起始下标
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims) # torch.Size([6, 15*25, 1, 256])

        # bev_embed = self.encoder(
        #     bev_queries,
        #     feat_flatten,
        #     feat_flatten,
        #     bev_h=bev_h,
        #     bev_w=bev_w,
        #     bev_pos=bev_pos,
        #     spatial_shapes=spatial_shapes,
        #     level_start_index=level_start_index,
        #     prev_bev=prev_bev,
        #     shift=shift,
        #     **kwargs
        # )
        # ------------------------------encoder start------------------------------
        bev_query = bev_queries  # torch.Size([50 * 50, 1, 256])
        output = bev_query
        intermediate = []

        # ref_3d = self.get_reference_points(
        #     bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d',
        #     bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        # ---------------------get_reference_points start---------------------
        num_points_in_pillar = 4
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        device = 'cuda'
        ref_3d_z = pc_range[5] - pc_range[2]  # 8
        bs = bev_queries.size(1)  # 1

        # TODO:zs和xs、ys的尺度是不一致的，zs是真实高度，xs和ys是预定义的bev的Voxel尺度
        zs = torch.linspace(0.5, ref_3d_z - 0.5, num_points_in_pillar, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(num_points_in_pillar, bev_h, bev_w) / ref_3d_z
        xs = torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype,
                            device=device).view(1, 1, bev_w).expand(num_points_in_pillar, bev_h, bev_w) / bev_w
        ys = torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype,
                            device=device).view(1, bev_h, 1).expand(num_points_in_pillar, bev_h, bev_w) / bev_h
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # bs, num_points_in_pillar, bev_h * bev_w, xyz # torch.Size([1, 4, 50*50, 3])
        # ref_3d:
        #   zs: (0.5 ~ 8-0.5) / 8
        #   xs: (0.5 ~ 50-0.5) / 50
        #   ys: (0.5 ~ 50-0.5) / 50
        # return ref_3d
        # ----------------------get_reference_points end----------------------
        # ref_2d = self.get_reference_points(
        #     bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        # ---------------------get_reference_points start---------------------
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device),
            torch.linspace(
                0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / bev_h
        ref_x = ref_x.reshape(-1)[None] / bev_w
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        # bs, bev_h * bev_w, None, xy # torch.Size([1, 50*50, 1, 2])
        # return ref_2d
        # ----------------------get_reference_points end----------------------
        # reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, kwargs['img_metas'])
        # ------------------------point_sampling start------------------------
        reference_points = ref_3d
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in ohb_img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # torch.Size([1, 6, 4, 4])
        reference_points = reference_points.clone()

        # reference_points = ref_3d # normalize 0~1 # torch.Size([1, 4, 50*50, 3])
        # pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        # 映射到自车坐标系下, 尺度被缩放为真实尺度
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        # 变成齐次坐标
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        # bs, num_points_in_pillar, bev_h * bev_w, xyz1

        reference_points = reference_points.permute(1, 0, 2, 3)
        # num_points_in_pillar, bs, bev_h * bev_w, xyz1
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, xyz1, None

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, 4, 4

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, xys1
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)  # 只保留位于相机前方的点
        # 齐次坐标下除以比例系数得到图像平面的坐标真值
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, uv

        # 坐标归一化
        reference_points_cam[..., 0] /= ohb_img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= ohb_img_metas[0]['img_shape'][0][0]

        # 去掉图像以外的点
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        # num_cam, bs, bev_h * bev_w, num_points_in_pillar, uv
        # torch.Size([6, 1, 50*50, 4, 2])
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        # TODO: bev_mask: (num_cam, bs, bev_h * bev_w, num_points_in_pillar)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        # return reference_points_cam, bev_mask
        # -------------------------point_sampling end-------------------------

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        # bs, bev_h * bev_w, None, xy
        shift_ref_2d += shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        # torch.Size([50 * 50, 1, 256]) -> torch.Size([1, 50 * 50, 256])

        bev_pos = bev_pos.permute(1, 0, 2)  # torch.Size([50 * 50, 1, 256])
        bs, len_bev, num_bev_level, _ = ref_2d.shape  # bs, bev_h * bev_w, None, xy # torch.Size([1, 50*50, 1, 2])
        if prev_bev is not None:  # None #2 # torch.Size([50*50, 1, 256])
            prev_bev = prev_bev.permute(1, 0, 2)  #2 # torch.Size([1, 50*50, 256])
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)  #2 torch.Size([2, 50*50, 256])
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)  #2 torch.Size([2, 50*50, 1, 2])
            # TODO: Why stack and reshape instead of cat directly.
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)
            # 2*bs, bev_h * bev_w, None, xy # torch.Size([2, 50 * 50, 1, 2])

        # for lid, layer in enumerate(self.layers):
        #     output = layer(
        #         bev_query,
        #         key,
        #         value,
        #         *args,
        #         bev_pos=bev_pos,
        #         ref_2d=hybird_ref_2d,
        #         ref_3d=ref_3d,
        #         bev_h=bev_h,
        #         bev_w=bev_w,
        #         spatial_shapes=spatial_shapes,  # torch.Size([1, 2])
        #         level_start_index=level_start_index,
        #         reference_points_cam=reference_points_cam,
        #         bev_mask=bev_mask,
        #         prev_bev=prev_bev,
        #         **kwargs)
        #    bev_query = output
        # self.layers: [BEVFormerLayer, BEVFormerLayer, BEVFormerLayer] # 3个BEVFormerLayer的参数完全一致的
        # --------------------BEVFormerLayer forward start--------------------
        query = bev_query
        bfl_norm_index = 0
        bfl_attn_index = 0
        bfl_ffn_index = 0
        bfl_identity = query
        bfl_attn_masks = None
        bfl_num_attn = 2
        bfl_operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        bfl_pre_norm = False  #2 False

        if bfl_attn_masks is None:
            bfl_attn_masks = [None for _ in range(bfl_num_attn)]
        # elif isinstance(bfl_attn_masks, torch.Tensor):
        #     bfl_attn_masks = [
        #         copy.deepcopy(bfl_attn_masks) for _ in range(bfl_num_attn)
        #     ]
        #     warnings.warn(f'Use same attn_mask in all attentions in '
        #                   f'BEVFormerLayer one')
        # else:
        #     assert len(bfl_attn_masks) == bfl_num_attn, f'The length of ' \
        #                                         f'attn_masks {len(bfl_attn_masks)} must be equal ' \
        #                                         f'to the number of attention in ' \
        #                                         f'operation_order {bfl_num_attn}'

        for layer in bfl_operation_order:
            # temporal self attention
            if layer == 'self_attn':
                # query = self.attentions[attn_index](
                #     query,
                #     prev_bev,
                #     prev_bev,
                #     identity if self.pre_norm else None,
                #     query_pos=bev_pos,
                #     key_pos=bev_pos,
                #     attn_mask=attn_masks[attn_index],
                #     key_padding_mask=query_key_padding_mask,
                #     reference_points=ref_2d,
                #     spatial_shapes=torch.tensor(
                #         [[bev_h, bev_w]], device=query.device),
                #     level_start_index=torch.tensor([0], device=query.device),
                #     **kwargs)
                # --------------------self_attn start--------------------
                # TemporalSelfAttention forward
                tsa_value = prev_bev  # None  #2 [prev_bev, bev_query]
                tsa_batch_first = True
                tsa_identity = None
                tsa_query_pos = bev_pos
                tsa_value_proj = nn.Linear(embed_dims, embed_dims).cuda()
                tsa_output_proj = nn.Linear(embed_dims, embed_dims).cuda()
                tsa_key_padding_mask = None
                tsa_num_bev_queue = 2
                tsa_num_heads = 8
                tsa_num_levels = 1
                tsa_num_points = 4
                tsa_sampling_offsets = nn.Linear(embed_dims * tsa_num_bev_queue,
                                                 tsa_num_bev_queue * tsa_num_heads * tsa_num_levels * tsa_num_points * 2).cuda()
                tsa_attention_weights = nn.Linear(embed_dims * tsa_num_bev_queue,
                                                   tsa_num_bev_queue * tsa_num_heads * tsa_num_levels * tsa_num_points).cuda()
                # tsa_reference_points = ref_2d
                tsa_reference_points = hybird_ref_2d
                tsa_spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
                tsa_level_start_index = torch.tensor([0], device=query.device)
                tsa_im2col_step = 64
                tsa_dropout = nn.Dropout(0.1)

                if tsa_value is None:
                    assert tsa_batch_first
                    bs, len_bev, c = query.shape  # torch.Size([1, 50 * 50, 256])
                    tsa_value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)  # torch.Size([2, 50 * 50, 256])
                    # value = torch.cat([query, query], 0)
                if tsa_identity is None:
                    tsa_identity = query
                if tsa_query_pos is not None:
                    query = query + tsa_query_pos  # bev_query + bev_pos
                # if not tsa_batch_first:
                #     # change to (bs, num_query ,embed_dims)
                #     query = query.permute(1, 0, 2)
                #     tsa_value = tsa_value.permute(1, 0, 2)
                bs, num_query, embed_dims = query.shape  # torch.Size([1, 50 * 50, 256])
                _, num_value, _ = tsa_value.shape  # torch.Size([2, 50 * 50, 256])
                assert (tsa_spatial_shapes[:, 0] * tsa_spatial_shapes[:, 1]).sum() == num_value
                assert tsa_num_bev_queue == 2

                # tsa_value[:bs] == query # torch.Size([1, 50 * 50, 512])
                #2 tsa_value[:bs] == prev_bev #2 torch.Size([1, 50 * 50, 512])
                query = torch.cat([tsa_value[:bs], query], -1)  # torch.Size([1, 50 * 50, 512])
                # TODO:这里在特征维度cat未对齐的prev_bev（tsa_value[:bs]）和当前的bev_query并以此推理出oofsets和weights？
                tsa_value = tsa_value_proj(tsa_value)  # torch.Size([2, 50 * 50, 256])

                # if tsa_key_padding_mask is not None:
                #     tsa_value = tsa_value.masked_fill(tsa_key_padding_mask[..., None], 0.0)

                tsa_value = tsa_value.reshape(bs * tsa_num_bev_queue, num_value, tsa_num_heads, -1)  # torch.Size([2, 50 * 50, 8, 32])

                sampling_offsets = tsa_sampling_offsets(query)  # torch.Size([1, 50 * 50, 128])
                sampling_offsets = sampling_offsets.view(
                    bs, num_query, tsa_num_heads, tsa_num_bev_queue, tsa_num_levels, tsa_num_points, 2)
                attention_weights = tsa_attention_weights(query).view(
                    bs, num_query, tsa_num_heads, tsa_num_bev_queue, tsa_num_levels * tsa_num_points)
                attention_weights = attention_weights.softmax(-1)

                attention_weights = attention_weights.view(bs, num_query,
                                                           tsa_num_heads,
                                                           tsa_num_bev_queue,
                                                           tsa_num_levels,
                                                           tsa_num_points)

                attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5) \
                    .reshape(bs * tsa_num_bev_queue, num_query, tsa_num_heads, tsa_num_levels, tsa_num_points).contiguous()
                # torch.Size([1*2, 50*50, 8, 1, 4])
                sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6) \
                    .reshape(bs * tsa_num_bev_queue, num_query, tsa_num_heads, tsa_num_levels, tsa_num_points, 2)
                # torch.Size([1*2, 50*50, 8, 1, 4, 2])

                # TODO: sampling_offsets归一化的方式是否合理？
                if tsa_reference_points.shape[-1] == 2:  # torch.Size([2, 50*50, 1, 2]) # bs, bev_h * bev_w, None, xy
                    offset_normalizer = torch.stack([tsa_spatial_shapes[..., 1], tsa_spatial_shapes[..., 0]], -1)  # torch.Size([1, 2])
                    sampling_locations = tsa_reference_points[:, :, None, :, None, :] \
                                         + sampling_offsets \
                                         / offset_normalizer[None, None, None, :, None, :]
                    # torch.Size([1*2, 50*50, 8, 1, 4, 2])

                # elif tsa_reference_points.shape[-1] == 4:
                #     sampling_locations = tsa_reference_points[:, :, None, :, None, :2] \
                #                          + sampling_offsets / tsa_num_points \
                #                          * tsa_reference_points[:, :, None, :, None, 2:] \
                #                          * 0.5
                # else:
                #     raise ValueError(
                #         f'Last dim of tsa_reference_points must be'
                #         f' 2 or 4, but get {tsa_reference_points.shape[-1]} instead.')
                use_msda_cuda = False
                if torch.cuda.is_available() and tsa_value.is_cuda and use_msda_cuda:
                    if not sampling_locations.is_contiguous():
                        sampling_locations = sampling_locations.contiguous()
                    # using fp16 deformable attention is unstable because it performs many sum operations
                    if tsa_value.dtype == torch.float16:
                        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                    else:
                        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                    output = MultiScaleDeformableAttnFunction.apply(
                        tsa_value, tsa_spatial_shapes, tsa_level_start_index, sampling_locations,
                        attention_weights, tsa_im2col_step)
                else:
                    # output = multi_scale_deformable_attn_pytorch(
                    #     tsa_value, tsa_spatial_shapes, sampling_locations, attention_weights)

                    # query: nn.Embedding(bev_h * bev_w, embed_dims).weight + can_bus
                    # tsa_value: Linear([query, query])
                    # sampling_locations: meshgrid(val(0-1)len(50), val(0-1)len(50)) + norm(Linear(query))
                    # attention_weights: Softmax(Linear(query))
                    # -------------multi_scale_deformable_attn_pytorch start-------------
                    msda_bs, _, num_heads, msda_embed_dims = tsa_value.shape  # torch.Size([2, 50 * 50, 8, 32])
                    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape  # torch.Size([2, 50*50, 8, 1, 4, 2])
                    value_list = tsa_value.split([H_ * W_ for H_, W_ in tsa_spatial_shapes], dim=1)  # (torch.Size([2, 50*50, 8, 32]))
                    sampling_grids = 2 * sampling_locations - 1  # 为了对齐F.grid_sample函数中的grid坐标系范围[-1~1]
                    sampling_value_list = []
                    for level, (H_, W_) in enumerate(tsa_spatial_shapes):
                        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(msda_bs * num_heads, msda_embed_dims, H_, W_)
                        # msda_bs, H_*W_, num_heads, msda_embed_dims ->
                        # msda_bs, H_*W_, num_heads*msda_embed_dims ->
                        # msda_bs, num_heads*msda_embed_dims, H_*W_ ->
                        # msda_bs*num_heads, msda_embed_dims, H_, W_
                        # torch.Size([2*8, 32, 50, 50])

                        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
                        # msda_bs, num_queries, num_heads, num_points, 2 ->
                        # msda_bs, num_heads, num_queries, num_points, 2 ->
                        # msda_bs*num_heads, num_queries, num_points, 2
                        # torch.Size([2*8, 50*50, 4, 2])

                        sampling_value_l_ = F.grid_sample(
                            value_l_,
                            sampling_grid_l_,
                            mode='bilinear',
                            padding_mode='zeros',
                            align_corners=False)
                        # msda_bs*num_heads, msda_embed_dims, num_queries, num_points
                        # torch.Size([2*8, 32, 50*50, 4])
                        sampling_value_list.append(sampling_value_l_)
                    attention_weights = attention_weights.transpose(1, 2).reshape(msda_bs * num_heads, 1, num_queries, num_levels * num_points)
                    # (msda_bs, num_queries, num_heads, num_levels, num_points) ->
                    # (msda_bs, num_heads, num_queries, num_levels, num_points) ->
                    # (msda_bs*num_heads, 1, num_queries, num_levels*num_points)
                    # torch.Size([2*8, 1, 50*50， 1*4])

                    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
                              attention_weights).sum(-1).view(msda_bs, num_heads * msda_embed_dims, num_queries)
                    # torch.Size([2, 256, 50*50])
                    # return output.transpose(1, 2).contiguous()
                    output = output.transpose(1, 2).contiguous()
                    # torch.Size([2, 50*50, 256]) # (bs*num_bev_queue, num_query, embed_dims)
                    # --------------multi_scale_deformable_attn_pytorch end--------------
                # import copy
                # temp = copy.deepcopy(output)
                # temp = temp.mean(0)[None, ...]

                output = output.permute(1, 2, 0)
                # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)

                output = output.view(num_query, embed_dims, bs, tsa_num_bev_queue)
                # fuse history value and current value
                # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
                output = output.mean(-1)

                output = output.permute(2, 0, 1)  # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)

                output = tsa_output_proj(output)
                # (bs, num_query, embed_dims)
                # torch.Size([1, 50*50, 256])

                # if not tsa_batch_first:
                #     output = output.permute(1, 0, 2)

                # return self.dropout(output) + identity
                query = tsa_dropout(output) + tsa_identity
                # ---------------------self_attn end---------------------
                bfl_attn_index += 1
                identity = query

            elif layer == 'norm':
                # query = self.norms[norm_index](query)
                # -----------------------norm start-----------------------
                m = nn.LayerNorm(query.shape[-1]).cuda()
                query = m(query)
                # ------------------------norm end------------------------
                bfl_norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                # query = self.attentions[attn_index](
                #     query,
                #     key,
                #     value,
                #     identity if self.pre_norm else None,
                #     query_pos=query_pos,
                #     key_pos=key_pos,
                #     reference_points=ref_3d,
                #     reference_points_cam=reference_points_cam,
                #     mask=mask,
                #     attn_mask=attn_masks[attn_index],
                #     key_padding_mask=key_padding_mask,
                #     spatial_shapes=spatial_shapes,
                #     level_start_index=level_start_index,
                #     **kwargs)
                # --------------------cross_attn start--------------------
                ca_residual = None  # 默认参数
                ca_query_pos = None
                ca_key = feat_flatten  # only use ca_key.shape
                ca_value = feat_flatten  # (num_cam, H*W, bs, embed_dims) # torch.Size([6, 15*25, 1, 256])
                ca_output_proj = nn.Linear(embed_dims, embed_dims).cuda()
                ca_dropout = nn.Dropout(0.1)

                # if ca_key is None:
                #     ca_key = query
                # if ca_value is None:
                #     ca_value = ca_key

                if ca_residual is None:
                    inp_residual = query  # torch.Size([1, 50*50, 256])
                    slots = torch.zeros_like(query)
                # if ca_query_pos is not None:
                #     query = query + ca_query_pos

                bs, num_query, _ = query.size()  # torch.Size([1, 50*50, 256])

                D = reference_points_cam.size(3)  # torch.Size([6, 1, 50*50, 4, 2])
                indexes = []
                for i, mask_per_img in enumerate(bev_mask):  # torch.Size([6, 1, 50*50, 4]) # Reserved points
                    index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                    # torch.Size([num_nonzero_pillar]) # 包含的4个点至少有一个可以投影到图像上的pillar在50*50的bev中的索引
                    indexes.append(index_query_per_img)
                max_len = max([len(each) for each in indexes])

                # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
                queries_rebatch = query.new_zeros([bs, num_cams, max_len, embed_dims])
                reference_points_rebatch = reference_points_cam.new_zeros([bs, num_cams, max_len, D, 2])

                for j in range(bs):
                    for i, reference_points_per_img in enumerate(reference_points_cam):
                        index_query_per_img = indexes[i]
                        queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                        reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

                num_cams, l, bs, embed_dims = ca_key.shape  # (num_cam, H*W, bs, embed_dims) # torch.Size([6, 15*25, 1, 256])

                ca_key = ca_key.permute(2, 0, 1, 3).reshape(bs * num_cams, l, embed_dims)
                ca_value = ca_value.permute(2, 0, 1, 3).reshape(bs * num_cams, l, embed_dims)  # torch.Size([6, 15*25, 256])

                # queries = self.deformable_attention(
                #     query=queries_rebatch.view(bs * num_cams, max_len, embed_dims), key=ca_key, value=ca_value,
                #     reference_points=reference_points_rebatch.view(bs * num_cams, max_len, D, 2),
                #     spatial_shapes=spatial_shapes,
                #     level_start_index=level_start_index).view(bs, num_cams, max_len, embed_dims)
                queries_rebatch = queries_rebatch.view(bs * num_cams, max_len, embed_dims)
                reference_points_rebatch = reference_points_rebatch.view(bs * num_cams, max_len, D, 2)
                # -------------multi_scale_deformable_attn_pytorch start-------------
                msda_identity = None
                msda_query_pos = None
                msda_batch_first = True
                msda_value_proj = nn.Linear(embed_dims, embed_dims).cuda()
                msda_key_padding_mask = None
                msda_num_heads = 8
                msda_num_levels = 1
                msda_num_points = 8
                msda_sampling_offsets = nn.Linear(embed_dims, msda_num_heads * msda_num_levels * msda_num_points * 2).cuda()
                msda_attention_weights = nn.Linear(embed_dims, msda_num_heads * msda_num_levels * msda_num_points).cuda()
                msda_im2col_step = 64

                # if ca_value is None:
                #     ca_value = queries_rebatch
                if msda_identity is None:
                    msda_identity = queries_rebatch
                # if msda_query_pos is not None:
                #     queries_rebatch = queries_rebatch + msda_query_pos

                # if not msda_batch_first:
                #     # change to (msda_bs, num_query ,embed_dims)
                #     queries_rebatch = queries_rebatch.permute(1, 0, 2)
                #     ca_value = ca_value.permute(1, 0, 2)

                # msda_bs, num_query, _ = queries_rebatch.shape  # torch.Size([msda_bs * num_cams, max_len, embed_dims])
                msda_bs, max_len, _ = queries_rebatch.shape  # torch.Size([bs * num_cams, max_len, embed_dims])
                msda_bs, num_value, _ = ca_value.shape  # torch.Size([6, 15*25, 256])
                assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

                ca_value = msda_value_proj(ca_value)
                # if msda_key_padding_mask is not None:
                #     ca_value = ca_value.masked_fill(key_padding_mask[..., None], 0.0)
                ca_value = ca_value.view(msda_bs, num_value, msda_num_heads, -1)
                sampling_offsets = msda_sampling_offsets(queries_rebatch).view(
                    msda_bs, max_len, msda_num_heads, msda_num_levels, msda_num_points, 2)
                attention_weights = msda_attention_weights(queries_rebatch).view(
                    msda_bs, max_len, msda_num_heads, msda_num_levels * msda_num_points)

                attention_weights = attention_weights.softmax(-1)

                attention_weights = attention_weights.view(msda_bs, max_len,
                                                           msda_num_heads,
                                                           msda_num_levels,
                                                           msda_num_points)

                if reference_points_rebatch.shape[-1] == 2:  # torch.Size([6, 606, 4, 2])
                    """
                    For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
                    After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
                    For each referent point, we sample `num_points` sampling points.
                    For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
                    """
                    offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

                    msda_bs, max_len, num_Z_anchors, xy = reference_points_rebatch.shape  # torch.Size([6, 606, 4, 2])
                    reference_points_rebatch = reference_points_rebatch[:, :, None, None, None, :, :]
                    # [msda_bs, max_len, 1, 1, 1, num_Z_anchors, uv]
                    sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                    # [msda_bs, max_len, msda_num_heads, msda_num_levels, msda_num_points, 2]
                    # TODO: msda_num_levels和fpn的层数是否有关？
                    msda_bs, max_len, msda_num_heads, msda_num_levels, msda_num_all_points, xy = sampling_offsets.shape
                    # 6        606           8               1                  8            2
                    sampling_offsets = sampling_offsets.view(
                        msda_bs, max_len, msda_num_heads, msda_num_levels, msda_num_all_points // num_Z_anchors, num_Z_anchors, xy)
                    #      6       606          8                 1                            2                        4        2
                    sampling_locations = reference_points_rebatch + sampling_offsets
                    msda_bs, max_len, msda_num_heads, msda_num_levels, Z_anchor_points, num_Z_anchors, xy = sampling_locations.shape
                    assert msda_num_all_points == Z_anchor_points * num_Z_anchors

                    sampling_locations = sampling_locations.view(msda_bs, max_len, msda_num_heads, msda_num_levels, msda_num_all_points, xy)

                # elif reference_points_rebatch.shape[-1] == 4:
                #     assert False
                # else:
                #     raise ValueError(
                #         f'Last dim of reference_points_rebatch must be'
                #         f' 2 or 4, but get {reference_points_rebatch.shape[-1]} instead.')

                #  sampling_locations.shape: msda_bs, max_len, num_heads, num_levels, num_all_points, 2
                #  attention_weights.shape: msda_bs, max_len, num_heads, num_levels, num_all_points

                if torch.cuda.is_available() and ca_value.is_cuda and use_msda_cuda:
                    if ca_value.dtype == torch.float16:
                        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                    else:
                        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                    output = MultiScaleDeformableAttnFunction.apply(
                        ca_value, spatial_shapes, level_start_index, sampling_locations,
                        attention_weights, msda_im2col_step)
                else:
                    # ca_value: torch.Size([6, 15*25, 8, 32])
                    # spatial_shapes: torch.Size([1, 2])
                    # sampling_locations: torch.Size([6, 606, 8, 1, 8, 2])
                    # attention_weights: torch.Size([6, 606, 8, 1, 8])
                    output = multi_scale_deformable_attn_pytorch(ca_value, spatial_shapes, sampling_locations, attention_weights)
                    # torch.Size([6, 606, 256])
                # if not msda_batch_first:
                #     output = output.permute(1, 0, 2)
                # return output
                # --------------multi_scale_deformable_attn_pytorch end--------------
                ca_queries = output.view(bs, num_cams, max_len, embed_dims)  # torch.Size([1, 6, 606, 256])
                # slots: # torch.Size([1, 50*50, 256])
                for j in range(bs):
                    for i, index_query_per_img in enumerate(indexes):
                        # TODO: 不同的相机之间特征向量是直接相加的
                        slots[j, index_query_per_img] += ca_queries[j, i, :len(index_query_per_img)]

                # bev_mask: # torch.Size([6, 1, 50*50, 4])
                # slots: # torch.Size([1, 50*50, 256])
                count = bev_mask.sum(-1) > 0
                count = count.permute(1, 2, 0).sum(-1)
                count = torch.clamp(count, min=1.0)
                slots = slots / count[..., None]
                slots = ca_output_proj(slots)

                # return self.dropout(slots) + inp_residual
                query = ca_dropout(slots) + inp_residual  # TODO: WHAT THE FUCK?
                # TODO: 完全存在于图像可视范围之外的pillar也保留query的特征？应该按照bev_mask使能够投影到图像上的pillar加上query，其它的pillar应该置0
                # ---------------------cross_attn end---------------------
                bfl_attn_index += 1
                identity = query

            elif layer == 'ffn':
                # query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                # -----------------------ffn start-----------------------
                # ffn_cfgs = dict(
                #     # type='FFN',
                #     embed_dims=256,
                #     feedforward_channels=512,
                #     num_fcs=2,
                #     ffn_drop=0.1,
                #     act_cfg=dict(type='ReLU', inplace=True),
                # )
                # ffns = FFN(**ffn_cfgs).cuda()
                ffn_feedforward_channels = 512
                ffn_num_fcs = 2
                ffn_drop = 0.1
                ffn_add_identity = True
                ffn_identity = None
                ffn_dropout = nn.Dropout(ffn_drop)

                ffn_layers = []
                ffn_in_channels = embed_dims
                for _ in range(ffn_num_fcs - 1):
                    ffn_layers.append(Sequential(nn.Linear(ffn_in_channels, ffn_feedforward_channels),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(ffn_drop)))
                    ffn_in_channels = ffn_feedforward_channels
                ffn_layers.append(nn.Linear(ffn_feedforward_channels, embed_dims))
                ffn_layers.append(nn.Dropout(ffn_drop))
                # bevformer中ffn模块使用的是mmcv的Linear，但在pytorch1.6以上版本与nn.Linear没有区别
                ffn_layers = Sequential(*ffn_layers).cuda()

                out = ffn_layers(query)
                # if not ffn_add_identity:
                #     # return self.dropout_layer(out)
                if ffn_identity is None:
                    ffn_identity = query
                # return identity + self.dropout_layer(out)
                query = ffn_identity + ffn_dropout(out)
                # ------------------------ffn end------------------------
                bfl_ffn_index += 1
        # return query
        output = query
        # ---------------------BEVFormerLayer forward end---------------------
        bev_query = output
        # if self.return_intermediate:
        #     intermediate.append(output)
        #
        # if self.return_intermediate:
        #     return torch.stack(intermediate)

        # return output
        # -------------------------------encoder end-------------------------------
        bev_embed = output
        # return bev_embed
        # --------------------------------get_bev_features end---------------------------------
        # --------------------------------------pts_bbox_head end--------------------------------------
        prev_bev = bev_embed  # torch.Size([1, 50*50, 256])

    # self.train()
    # return prev_bev

# -------------------------------------------obtain_history_bev end-------------------------------------------
img_metas = [each[len_queue - 1] for each in img_metas]
# if not img_metas[0]['prev_bev_exists']:  # prev_bev_exists: True
#     prev_bev = None
# img_feats = self.extract_feat(img=img, img_metas=img_metas)
# ---------------------------------------------extract_feat start---------------------------------------------
img_feats = [torch.randn([1, 6, 256, 15, 25])]
# ----------------------------------------------extract_feat end----------------------------------------------
losses = dict()
# losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
#                                     gt_labels_3d, img_metas,
#                                     gt_bboxes_ignore, prev_bev)
# ---------------------------------------------forward_pts_train start---------------------------------------------
# outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
# ------------------------------------------pts_bbox_head start------------------------------------------
# BEVFormerHead
num_reg_fcs = 2
cls_out_channels = 10
code_size = 10
transformer_decoder_num_layers = 6

cls_branch = []
for _ in range(num_reg_fcs):
    cls_branch.append(nn.Linear(embed_dims, embed_dims))
    cls_branch.append(nn.LayerNorm(embed_dims))
    cls_branch.append(nn.ReLU(inplace=True))
cls_branch.append(nn.Linear(embed_dims, cls_out_channels))
cls_branch = nn.Sequential(*cls_branch)

reg_branch = []
for _ in range(num_reg_fcs):
    reg_branch.append(nn.Linear(embed_dims, embed_dims))
    reg_branch.append(nn.ReLU())
reg_branch.append(nn.Linear(embed_dims, code_size))
reg_branch = nn.Sequential(*reg_branch)

cls_branches = nn.ModuleList([copy.deepcopy(cls_branch) for i in range(transformer_decoder_num_layers)]).cuda()
reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for i in range(transformer_decoder_num_layers)]).cuda()

mlvl_feats = img_feats  # [torch.Size([1, 6, 256, 15, 25])]
only_bev = False

bs, num_cam, _, _, _ = mlvl_feats[0].shape
dtype = mlvl_feats[0].dtype
object_query_embeds = BH_query_embedding.weight.to(dtype)  # 之前声明过且未用
bev_queries = BH_bev_embedding.weight.to(dtype)

bev_mask = torch.zeros((bs, bev_h, bev_w), device=bev_queries.device).to(dtype)
bev_pos = BH_positional_encoding(bev_mask).to(dtype)

if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
    # return self.transformer.get_bev_features(
    #     mlvl_feats,
    #     bev_queries,
    #     self.bev_h,
    #     self.bev_w,
    #     grid_length=(self.real_h / self.bev_h,
    #                  self.real_w / self.bev_w),
    #     bev_pos=bev_pos,
    #     img_metas=img_metas,
    #     prev_bev=prev_bev,
    # )
    pass
else:
    # outputs = self.transformer(
    #     mlvl_feats,
    #     bev_queries,
    #     object_query_embeds,
    #     self.bev_h,
    #     self.bev_w,
    #     grid_length=(self.real_h / self.bev_h,
    #                  self.real_w / self.bev_w),
    #     bev_pos=bev_pos,
    #     reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
    #     cls_branches=self.cls_branches if self.as_two_stage else None,
    #     img_metas=img_metas,
    #     prev_bev=prev_bev
    # )
    # -----------------------------------self.transformer start-----------------------------------
    # PerceptionTransformer
    # bev_embed = self.get_bev_features(
    #     mlvl_feats,
    #     bev_queries,
    #     bev_h,
    #     bev_w,
    #     grid_length=grid_length,
    #     bev_pos=bev_pos,
    #     prev_bev=prev_bev,
    #     **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
    pt_reference_points_layer = nn.Linear(embed_dims, 3).cuda()
    # ------------------------------self.get_bev_features start------------------------------
    bev_embed = torch.randn([1, 2500, 256], device='cuda')  # [prev_bev, query, image]
    # -------------------------------self.get_bev_features end-------------------------------

    bs = mlvl_feats[0].size(0)
    # object_query_embeds = BH_query_embedding.weight.to(dtype)  # torch.Size([900, 512])
    pt_query_pos, query = torch.split(object_query_embeds, embed_dims, dim=1)
    pt_query_pos = pt_query_pos.unsqueeze(0).expand(bs, -1, -1)
    query = query.unsqueeze(0).expand(bs, -1, -1)
    pt_reference_points = pt_reference_points_layer(pt_query_pos)  # torch.Size([1, 900, 3])
    pt_reference_points = pt_reference_points.sigmoid()
    init_reference_out = pt_reference_points

    query = query.permute(1, 0, 2)  # torch.Size([900, 1, 256])
    pt_query_pos = pt_query_pos.permute(1, 0, 2)  # torch.Size([900, 1, 256])
    bev_embed = bev_embed.permute(1, 0, 2)  # torch.Size([50*50, 1, 256])

    # inter_states, inter_references = self.decoder(
    #     query=query,
    #     key=None,
    #     value=bev_embed,
    #     query_pos=query_pos,
    #     reference_points=pt_reference_points,
    #     reg_branches=reg_branches,
    #     cls_branches=cls_branches,
    #     spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
    #     level_start_index=torch.tensor([0], device=query.device),
    #     **kwargs)
    # -----------------------------------self.decoder start-----------------------------------
    dc_key = None
    dc_value = bev_embed
    dc_query_pos = pt_query_pos
    dc_reference_points = pt_reference_points
    dc_reg_branches = reg_branches
    dc_cls_branches = None
    dc_spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
    dc_level_start_index = torch.tensor([0], device=query.device)
    dc_key_padding_mask = None
    dc_return_intermediate = True

    output = query  # torch.Size([900, 1, 256])
    intermediate = []
    intermediate_reference_points = []
    # for lid, layer in enumerate(self.layers):
    #     reference_points_input = dc_reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
    #     output = layer(
    #         output,
    #         *args,
    #         reference_points=reference_points_input,
    #         key_padding_mask=key_padding_mask,
    #         **kwargs)
    #
    # nn.ModuleList{DetrTransformerDecoderLayer * 6}
    # --------------------------------layer start--------------------------------
    # DetrTransformerDecoderLayer
    dtdl_attn_masks = None
    dtdl_num_attn = 2
    dtdl_operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    dtdl_key = dc_key
    dtdl_value = dc_value
    dtdl_query_pos = dc_query_pos
    dtdl_reference_points = dc_reference_points[..., :2].unsqueeze(2)
    dtdl_reg_branches = dc_reg_branches
    dtdl_cls_branches = dc_cls_branches
    dtdl_spatial_shapes = dc_spatial_shapes
    dtdl_level_start_index = dc_level_start_index
    dtdl_key_padding_mask = dc_key_padding_mask

    dtdl_norm_index = 0
    dtdl_attn_index = 0
    dtdl_ffn_index = 0
    dtdl_identity = query
    if dtdl_attn_masks is None:
        dtdl_attn_masks = [None for _ in range(dtdl_num_attn)]
    # elif isinstance(dtdl_attn_masks, torch.Tensor):
    #     dtdl_attn_masks = [
    #         copy.deepcopy(dtdl_attn_masks) for _ in range(dtdl_num_attn)
    #     ]
    #     warnings.warn(f'Use same attn_mask in all attentions in '
    #                   f'DetrTransformerDecoderLayer')
    # else:
    #     assert len(dtdl_attn_masks) == dtdl_num_attn, f'The length of ' \
    #                                              f'attn_masks {len(dtdl_attn_masks)} must be equal ' \
    #                                              f'to the number of attention in ' \
    #                                              f'operation_order {dtdl_num_attn}'

    for layer in dtdl_operation_order:
        if layer == 'self_attn':
            # temp_key = temp_value = query
            # query = self.attentions[attn_index](
            #     query,
            #     temp_key,
            #     temp_value,
            #     identity if self.pre_norm else None,
            #     query_pos=dc_query_pos,
            #     key_pos=dc_query_pos,
            #     attn_mask=attn_masks[attn_index],
            #     key_padding_mask=query_key_padding_mask,
            #     **kwargs)
            # -----------------------------self.attentions start-----------------------------
            # MultiheadAttention
            mha_key = mha_value = query  # torch.Size([900, 1, 256])
            mha_identity = None
            mha_key_pos = dc_query_pos  # torch.Size([900, 1, 256]) # pt_query_pos
            mha_query_pos = dc_query_pos
            mha_batch_first = False
            mha_proj_drop = nn.Dropout(0.0)
            mha_dropout_layer = nn.Dropout(0.1)
            mha_attn_mask = dtdl_attn_masks[dtdl_attn_index]

            # if mha_key is None:
            #     mha_key = query
            # if mha_value is None:
            #     mha_value = mha_key
            if mha_identity is None:
                mha_identity = query  # torch.Size([900, 1, 256])
            # if mha_key_pos is None:
            #     if mha_query_pos is not None:
            #         # use mha_query_pos if mha_key_pos is not available
            #         if mha_query_pos.shape == mha_key.shape:
            #             mha_key_pos = mha_query_pos
            #         else:
            #             warnings.warn(f'position encoding of mha_key is'
            #                           f'missing in MultiheadAttention.')
            if mha_query_pos is not None:
                query = query + mha_query_pos
            if mha_key_pos is not None:
                mha_key = mha_key + mha_key_pos  # mha_key = query mha_key_pos = pt_query_pos

            # Because the dataflow('mha_key', 'query', 'mha_value') of
            # ``torch.nn.MultiheadAttention`` is (num_query, batch,
            # embed_dims), We should adjust the shape of dataflow from
            # batch_first (batch, num_query, embed_dims) to num_query_first
            # (num_query ,batch, embed_dims), and recover ``attn_output``
            # from num_query_first to batch_first.
            # if mha_batch_first:
            #     query = query.transpose(0, 1)
            #     mha_key = mha_key.transpose(0, 1)
            #     mha_value = mha_value.transpose(0, 1)

            # out = self.attn(
            #     query=query,
            #     key=mha_key,
            #     value=mha_value,
            #     attn_mask=attn_mask,
            #     key_padding_mask=key_padding_mask)[0]
            # -----------------------------self.attn start-----------------------------
            mhaf_embed_dim = 256
            mhaf_kdim = mhaf_embed_dim
            mhaf_vdim = mhaf_embed_dim
            mhaf_qkv_same_embed_dim = mhaf_kdim == mhaf_embed_dim and mhaf_vdim == mhaf_embed_dim  # True
            mhaf_num_heads = 8
            mhaf_dropout = 0.1
            mhaf_batch_first = False
            mhaf_head_dim = mhaf_embed_dim // mhaf_num_heads
            assert mhaf_head_dim * mhaf_num_heads == mhaf_embed_dim, "embed_dim must be divisible by num_heads"
            factory_kwargs = {'device': 'cuda', 'dtype': None}
            mhaf_in_proj_weight = nn.Parameter(torch.empty((3 * mhaf_embed_dim, mhaf_embed_dim), **factory_kwargs))
            mhaf_in_proj_bias = nn.Parameter(torch.empty(3 * mhaf_embed_dim, **factory_kwargs))
            mhaf_bias_k = mhaf_bias_v = None
            mhaf_add_zero_attn = False
            mhaf_out_proj = NonDynamicallyQuantizableLinear(mhaf_embed_dim, mhaf_embed_dim, bias=True, **factory_kwargs)
            mhaf_attn_mask = mha_attn_mask  # None

            # if mhaf_batch_first:
            #     query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

            if not mhaf_qkv_same_embed_dim:
                # attn_output, attn_output_weights = F.multi_head_attention_forward(
                #     query, key, value, self.embed_dim, self.num_heads,
                #     self.in_proj_weight, self.in_proj_bias,
                #     self.bias_k, self.bias_v, self.add_zero_attn,
                #     self.dropout, self.out_proj.weight, self.out_proj.bias,
                #     training=self.training,
                #     key_padding_mask=key_padding_mask, need_weights=need_weights,
                #     attn_mask=attn_mask, use_separate_proj_weight=True,
                #     q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                #     v_proj_weight=self.v_proj_weight)
                pass
            else:
                # attn_output, attn_output_weights = F.multi_head_attention_forward(
                #     query, mha_key, mha_value, mhaf_embed_dim, mhaf_num_heads, mhaf_in_proj_weight, mhaf_in_proj_bias,
                #     mhaf_bias_k, mhaf_bias_v, mhaf_add_zero_attn, mhaf_dropout, mhaf_out_proj.weight, mhaf_out_proj.bias,
                #     training=True, key_padding_mask=None, need_weights=True, attn_mask=mhaf_attn_mask)
                # -------------F.multi_head_attention_forward start-------------
                mhaf_out_proj_weight = mhaf_out_proj.weight
                mhaf_out_proj_bias = mhaf_out_proj.bias
                mhaf_key = mha_key
                mhaf_value = mha_value
                embed_dim_to_check = mhaf_embed_dim
                use_separate_proj_weight = False
                mhaf_training = True
                mhaf_key_padding_mask = None
                mhaf_need_weights = True
                mhaf_q_proj_weight, mhaf_k_proj_weight, mhaf_v_proj_weight = None, None, None
                mhaf_static_k, mhaf_static_v = None, None


                # tens_ops = (query, mhaf_key, mhaf_value, mhaf_in_proj_weight, mhaf_in_proj_bias, mhaf_bias_k, mhaf_bias_v,
                #             mhaf_out_proj_weight, mhaf_out_proj_bias)
                # if has_torch_function(tens_ops):
                #     return handle_torch_function(
                #         multi_head_attention_forward,
                #         tens_ops,
                #         query,
                #         key,
                #         value,
                #         ... ...
                #     )

                # set up shape vars
                tgt_len, bsz, embed_dim = query.shape  # torch.Size([900, 1, 256])
                src_len, _, _ = mhaf_key.shape
                assert embed_dim == embed_dim_to_check, \
                    f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
                if isinstance(embed_dim, torch.Tensor):
                #     # embed_dim can be a tensor when JIT tracing
                #     head_dim = embed_dim.div(mhaf_num_heads, rounding_mode='trunc')
                    pass
                else:
                    head_dim = embed_dim // mhaf_num_heads
                assert head_dim * mhaf_num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {mhaf_num_heads}"
                # if use_separate_proj_weight:  # False
                #     # # allow MHA to have different embedding dimensions when separate projection weights are used
                #     # assert key.shape[:2] == value.shape[:2], \
                #     #     f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
                #     pass
                # else:
                #     assert mhaf_key.shape == mhaf_value.shape, f"key shape {mhaf_key.shape} does not match value shape {mhaf_value.shape}"

                #
                # compute in-projection
                #
                if not use_separate_proj_weight:
                    # q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
                    # -----------_in_projection_packed start-----------
                    # q, k, v, w, b = query, mhaf_key, mhaf_value, mhaf_in_proj_weight, mhaf_in_proj_bias
                    # E = query.size(-1)
                    if mhaf_key is mhaf_value:
                        # if query is mhaf_key:
                        #     # self-attention
                        #     return linear(query, mhaf_in_proj_weight, mhaf_in_proj_bias).chunk(3, dim=-1)
                        # else:
                        #     # encoder-decoder attention
                        #     w_q, w_kv = mhaf_in_proj_weight.split([E, E * 2])
                        #     if mhaf_in_proj_bias is None:
                        #         b_q = b_kv = None
                        #     else:
                        #         b_q, b_kv = mhaf_in_proj_bias.split([E, E * 2])
                        #     return (linear(query, w_q, b_q),) + linear(mhaf_key, w_kv, b_kv).chunk(2, dim=-1)
                        pass
                    else:
                        w_q, w_k, w_v = mhaf_in_proj_weight.chunk(3)
                        if mhaf_in_proj_bias is None:
                            # b_q = b_k = b_v = None
                            pass
                        else:
                            b_q, b_k, b_v = mhaf_in_proj_bias.chunk(3)
                        # return linear(query, w_q, b_q), linear(mhaf_key, w_k, b_k), linear(mhaf_value, w_v, b_v)
                        # F.linear(x, A, b): return x @ A.T + b
                        query, mhaf_key, mhaf_value = F.linear(query, w_q, b_q), F.linear(mhaf_key, w_k, b_k), F.linear(mhaf_value, w_v, b_v)
                        #                                   query + pt_query_pos      query + pt_query_pos                 query
                    # ------------_in_projection_packed end------------
                # else:
                #     assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
                #     assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
                #     assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
                #     if in_proj_bias is None:
                #         b_q = b_k = b_v = None
                #     else:
                #         b_q, b_k, b_v = in_proj_bias.chunk(3)
                #     q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

                # prep attention mask
                # if mhaf_attn_mask is not None:
                #     if mhaf_attn_mask.dtype == torch.uint8:
                #         warnings.warn(
                #             "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                #         mhaf_attn_mask = mhaf_attn_mask.to(torch.bool)
                #     else:
                #         assert mhaf_attn_mask.is_floating_point() or mhaf_attn_mask.dtype == torch.bool, \
                #             f"Only float, byte, and bool types are supported for attn_mask, not {mhaf_attn_mask.dtype}"
                #     # ensure mhaf_attn_mask's dim is 3
                #     if mhaf_attn_mask.dim() == 2:
                #         correct_2d_size = (tgt_len, src_len)
                #         if mhaf_attn_mask.shape != correct_2d_size:
                #             raise RuntimeError(
                #                 f"The shape of the 2D attn_mask is {mhaf_attn_mask.shape}, but should be {correct_2d_size}.")
                #         mhaf_attn_mask = mhaf_attn_mask.unsqueeze(0)
                #     elif mhaf_attn_mask.dim() == 3:
                #         correct_3d_size = (bsz * mhaf_num_heads, tgt_len, src_len)
                #         if mhaf_attn_mask.shape != correct_3d_size:
                #             raise RuntimeError(
                #                 f"The shape of the 3D attn_mask is {mhaf_attn_mask.shape}, but should be {correct_3d_size}.")
                #     else:
                #         raise RuntimeError(f"attn_mask's dimension {mhaf_attn_mask.dim()} is not supported")

                # prep key padding mask
                # if mhaf_key_padding_mask is not None and mhaf_key_padding_mask.dtype == torch.uint8:
                #     warnings.warn(
                #         "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                #     mhaf_key_padding_mask = mhaf_key_padding_mask.to(torch.bool)

                # add bias along batch dimension (currently second)
                if mhaf_bias_k is not None and mhaf_bias_v is not None:
                    # assert static_k is None, "bias cannot be added to static key."
                    # assert static_v is None, "bias cannot be added to static value."
                    # k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                    # v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                    # if mhaf_attn_mask is not None:
                    #     mhaf_attn_mask = pad(mhaf_attn_mask, (0, 1))
                    # if mhaf_key_padding_mask is not None:
                    #     mhaf_key_padding_mask = pad(mhaf_key_padding_mask, (0, 1))
                    pass
                else:
                    assert mhaf_bias_k is None
                    assert mhaf_bias_v is None

                #
                # reshape q, k, v for multihead attention and make em batch first
                #
                query = query.contiguous().view(tgt_len, bsz * mhaf_num_heads, head_dim).transpose(0, 1)  # [900, 1, 256] -> [900, 8, 32] -> [8, 900, 32]
                if mhaf_static_k is None:
                    mhaf_key = mhaf_key.contiguous().view(-1, bsz * mhaf_num_heads, head_dim).transpose(0, 1)  # [900, 8, 32] -> [8, 900, 32]
                # else:
                #     # TODO finish disentangling control flow so we don't do in-projections when statics are passed
                #     assert mhaf_static_k.size(0) == bsz * mhaf_num_heads, \
                #         f"expecting static_k.size(0) of {bsz * mhaf_num_heads}, but got {mhaf_static_k.size(0)}"
                #     assert mhaf_static_k.size(2) == head_dim, \
                #         f"expecting static_k.size(2) of {head_dim}, but got {mhaf_static_k.size(2)}"
                #     mhaf_key = mhaf_static_k
                if mhaf_static_v is None:
                    mhaf_value = mhaf_value.contiguous().view(-1, bsz * mhaf_num_heads, head_dim).transpose(0, 1)  # [900, 8, 32] -> [8, 900, 32]
                # else:
                #     # TODO finish disentangling control flow so we don't do in-projections when statics are passed
                #     assert mhaf_static_v.size(0) == bsz * mhaf_num_heads, \
                #         f"expecting static_v.size(0) of {bsz * mhaf_num_heads}, but got {mhaf_static_v.size(0)}"
                #     assert mhaf_static_v.size(2) == head_dim, \
                #         f"expecting static_v.size(2) of {head_dim}, but got {mhaf_static_v.size(2)}"
                #     mhaf_value = mhaf_static_v

                # add zero attention along batch dimension (now first)
                # if mhaf_add_zero_attn:
                #     zero_attn_shape = (bsz * mhaf_num_heads, 1, head_dim)
                #     mhaf_key = torch.cat([mhaf_key, torch.zeros(zero_attn_shape, dtype=mhaf_key.dtype, device=mhaf_key.device)], dim=1)
                #     mhaf_value = torch.cat([mhaf_value, torch.zeros(zero_attn_shape, dtype=mhaf_value.dtype, device=mhaf_value.device)], dim=1)
                #     if mhaf_attn_mask is not None:
                #         mhaf_attn_mask = pad(mhaf_attn_mask, (0, 1))
                #     if mhaf_key_padding_mask is not None:
                #         mhaf_key_padding_mask = pad(mhaf_key_padding_mask, (0, 1))

                # update source sequence length after adjustments
                src_len = mhaf_key.size(1)

                # merge key padding and attention masks
                # if mhaf_key_padding_mask is not None:
                #     assert mhaf_key_padding_mask.shape == (bsz, src_len), \
                #         f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {mhaf_key_padding_mask.shape}"
                #     mhaf_key_padding_mask = mhaf_key_padding_mask.view(bsz, 1, 1, src_len). \
                #         expand(-1, mhaf_num_heads, -1, -1).reshape(bsz * mhaf_num_heads, 1, src_len)
                #     if mhaf_attn_mask is None:
                #         mhaf_attn_mask = mhaf_key_padding_mask
                #     elif mhaf_attn_mask.dtype == torch.bool:
                #         mhaf_attn_mask = mhaf_attn_mask.logical_or(mhaf_key_padding_mask)
                #     else:
                #         mhaf_attn_mask = mhaf_attn_mask.masked_fill(mhaf_key_padding_mask, float("-inf"))

                # convert mask to float
                # if mhaf_attn_mask is not None and mhaf_attn_mask.dtype == torch.bool:
                #     new_attn_mask = torch.zeros_like(mhaf_attn_mask, dtype=torch.float)
                #     new_attn_mask.masked_fill_(mhaf_attn_mask, float("-inf"))
                #     mhaf_attn_mask = new_attn_mask

                # adjust dropout probability
                # if not mhaf_training:
                #     dropout_p = 0.0

                #
                # (deep breath) calculate attention and out projection
                #
                # attn_output, attn_output_weights = _scaled_dot_product_attention(query, mhaf_key, mhaf_value, mhaf_attn_mask, mhaf_dropout)
                # ------------_scaled_dot_product_attention start------------
                # q: Tensor,
                # k: Tensor,
                # v: Tensor,
                # attn_mask: Optional[Tensor] = None,
                # dropout_p: float = 0.0,

                B, Nt, E = query.shape  # torch.Size([8, 900, 32]), mhaf_key and mhaf_value is same shape.
                query = query / math.sqrt(E)
                # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
                attn = torch.bmm(query, mhaf_key.transpose(-2, -1))  # [8, 900, 32] @ [8, 32, 900] -> [8, 900, 900]
                # if mhaf_attn_mask is not None:
                #     attn += mhaf_attn_mask
                attn = F.softmax(attn, dim=-1)
                if mhaf_dropout > 0.0:
                    attn = F.dropout(attn, p=mhaf_dropout)
                # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
                output = torch.bmm(attn, mhaf_value)  # [8, 900, 900] @ [8, 900, 32] -> # torch.Size([8, 900, 32])
                # return output, attn
                attn_output, attn_output_weights = output, attn
                # -------------_scaled_dot_product_attention end-------------
                # tgt_len: 900  # [8, 900, 32]->[900, 8, 32]->[900, 1, 256]
                attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # torch.Size([900, 1, 256])
                attn_output = F.linear(attn_output, mhaf_out_proj_weight, mhaf_out_proj_bias)  # nn.Linear

                # if mhaf_need_weights:
                #     # average attention weights over heads
                #     attn_output_weights = attn_output_weights.view(bsz, mhaf_num_heads, tgt_len, src_len)
                #     # return attn_output, attn_output_weights.sum(dim=1) / mhaf_num_heads
                #     attn_output_weights = attn_output_weights.sum(dim=1) / mhaf_num_heads
                # else:
                #     return attn_output, None
                # --------------F.multi_head_attention_forward end--------------
            # if mhaf_batch_first:
            #     return attn_output.transpose(1, 0), attn_output_weights
            # else:
            #     return attn_output, attn_output_weights
            out = attn_output
            # ------------------------------self.attn end------------------------------

            # if mha_batch_first:
            #     out = out.transpose(0, 1)

            # return mha_identity + self.dropout_layer(self.proj_drop(out))
            query = mha_identity + mha_dropout_layer(mha_proj_drop(out))
            #       # torch.Size([900, 1, 256]) + # torch.Size([900, 1, 256])
            # ------------------------------self.attentions end------------------------------
            dtdl_attn_index += 1
            dtdl_identity = query

        elif layer == 'norm':
            # query = self.norms[norm_index](query)
            # -----------------------------self.norms start-----------------------------
            mha_norm = nn.LayerNorm(query.shape[-1]).cuda()
            query = mha_norm(query)
            # ------------------------------self.norms end------------------------------
            dtdl_norm_index += 1

        elif layer == 'cross_attn':
            # query = self.attentions[attn_index](
            #     query,
            #     key,
            #     value,
            #     identity if self.pre_norm else None,
            #     query_pos=query_pos,
            #     key_pos=key_pos,
            #     attn_mask=attn_masks[attn_index],
            #     key_padding_mask=key_padding_mask,
            #     **kwargs)
            # -----------------------------cross_attn start-----------------------------
            # ca_key = dtdl_key
            ca_value = dtdl_value  # bev_embed # torch.Size([50*50, 1, 256])
            ca_identity = None
            ca_query_pos = dc_query_pos  # torch.Size([900, 1, 256]) # pt_query_pos
            ca_batch_first = False
            ca_im2col_step = 64
            ca_embed_dims = 256
            ca_num_levels = 1
            ca_num_heads = 8
            ca_num_points = 4
            ca_sampling_offsets = nn.Linear(ca_embed_dims, ca_num_heads * ca_num_levels * ca_num_points * 2).cuda()
            ca_attention_weights = nn.Linear(ca_embed_dims, ca_num_heads * ca_num_levels * ca_num_points).cuda()
            ca_value_proj = nn.Linear(ca_embed_dims, ca_embed_dims).cuda()
            ca_output_proj = nn.Linear(ca_embed_dims, ca_embed_dims).cuda()
            ca_dropout = nn.Dropout(0.1)
            ca_spatial_shapes = dtdl_spatial_shapes
            ca_key_padding_mask = dtdl_key_padding_mask

            # if ca_value is None:
            #     ca_value = query
            if ca_identity is None:
                ca_identity = query
            if ca_query_pos is not None:
                query = query + ca_query_pos
            if not ca_batch_first:
                # change to (bs, num_query ,ca_embed_dims)
                query = query.permute(1, 0, 2)  # torch.Size([1, 900, 256])
                ca_value = ca_value.permute(1, 0, 2)  # torch.Size([1, 50*50, 256])

            bs, num_query, _ = query.shape  # torch.Size([1, 900, 256])
            bs, num_value, _ = ca_value.shape  # torch.Size([1, 50*50, 256])
            assert (ca_spatial_shapes[:, 0] * ca_spatial_shapes[:, 1]).sum() == num_value

            ca_value = ca_value_proj(ca_value)
            # if ca_key_padding_mask is not None:
            #     ca_value = ca_value.masked_fill(ca_key_padding_mask[..., None], 0.0)
            ca_value = ca_value.view(bs, num_value, ca_num_heads, -1)  # torch.Size([1, 50*50, 8, 32])

            sampling_offsets = ca_sampling_offsets(query).view(
                bs, num_query, ca_num_heads, ca_num_levels, ca_num_points, 2)
            #    1,    900,          8,            1,             4,       2
            attention_weights = ca_attention_weights(query).view(
                bs, num_query, ca_num_heads, ca_num_levels * ca_num_points)
            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query, ca_num_heads, ca_num_levels, ca_num_points)
            #                                          1,    900,           8,             1,             4
            if dtdl_reference_points.shape[-1] == 2:  # torch.Size([1, 900, 1, 2])
                offset_normalizer = torch.stack(
                    [ca_spatial_shapes[..., 1], ca_spatial_shapes[..., 0]], -1)  # torch.Size([1, 2])
                sampling_locations = dtdl_reference_points[:, :, None, :, None, :] \
                                     + sampling_offsets \
                                     / offset_normalizer[None, None, None, :, None, :]
                # torch.Size([1, 900, 8, 1, 4, 2])
            # elif dtdl_reference_points.shape[-1] == 4:
            #     sampling_locations = dtdl_reference_points[:, :, None, :, None, :2] \
            #                          + sampling_offsets / ca_num_points \
            #                          * dtdl_reference_points[:, :, None, :, None, 2:] \
            #                          * 0.5
            # else:
            #     raise ValueError(
            #         f'Last dim of reference_points must be'
            #         f' 2 or 4, but get {dtdl_reference_points.shape[-1]} instead.')
            if torch.cuda.is_available() and ca_value.is_cuda:

                # using fp16 deformable attention is unstable because it performs many sum operations
                if ca_value.dtype == torch.float16:
                    MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                else:
                    MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                output = MultiScaleDeformableAttnFunction.apply(
                    ca_value, ca_spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, ca_im2col_step)
            else:
                output = multi_scale_deformable_attn_pytorch(
                    ca_value, ca_spatial_shapes, sampling_locations, attention_weights)

            output = ca_output_proj(output)  # torch.Size([1, 900, 256])

            if not ca_batch_first:
                # (num_query, bs ,ca_embed_dims)
                output = output.permute(1, 0, 2)  # torch.Size([900, 1, 256])
            query = ca_dropout(output) + ca_identity
            # return ca_dropout(output) + ca_identity
            # ------------------------------cross_attn end------------------------------
            dtdl_attn_index += 1
            dtdl_identity = query

        elif layer == 'ffn':
            # query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
            # -----------------------------ffns start-----------------------------
            ffn_embed_dims = 256
            ffn_feedforward_channels = 512
            ffn_num_fcs = 2
            ffn_layers = []
            ffn_in_channels = ffn_embed_dims
            ffn_drop = 0.1
            for _ in range(ffn_num_fcs - 1):
                ffn_layers.append(
                    Sequential(
                        nn.Linear(ffn_in_channels, ffn_feedforward_channels),
                        nn.ReLU(inplace=True),
                        nn.Dropout(ffn_drop)))
                ffn_in_channels = ffn_feedforward_channels
            ffn_layers.append(nn.Linear(ffn_feedforward_channels, ffn_embed_dims))
            ffn_layers.append(nn.Dropout(ffn_drop))
            ffn_layers = Sequential(*ffn_layers).cuda()
            ffn_dropout_layer = nn.Identity().cuda()
            ffn_add_identity = True
            ffn_identity = None

            ffn_out = ffn_layers(query)
            # if not ffn_add_identity:
            #     return self.dropout_layer(out)
            if ffn_identity is None:
                ffn_identity = query
            # return identity + self.dropout_layer(out)
            query = ffn_identity + ffn_dropout_layer(ffn_out)
            # ------------------------------ffns end------------------------------
            dtdl_ffn_index += 1

    # bev_embed: # torch.Size([1, 50*50, 256])
    # return query
    output = query  # torch.Size([900, 1, 256])
    # ---------------------------------layer end---------------------------------
    for lid in range(6):
        output = output + torch.randn_like(output)
        output = output.permute(1, 0, 2)  # torch.Size([1, 900, 256])
        if dc_reg_branches is not None:
            tmp = dc_reg_branches[lid](output)  # torch.Size([1, 900, 10])

            assert dc_reference_points.shape[-1] == 3  # torch.Size([1, 900, 3])

            new_reference_points = torch.zeros_like(dc_reference_points)
            new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(dc_reference_points[..., :2])
            new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(dc_reference_points[..., 2:3])

            new_reference_points = new_reference_points.sigmoid()

            dc_reference_points = new_reference_points.detach()

        output = output.permute(1, 0, 2)
        if dc_return_intermediate:
            intermediate.append(output)
            intermediate_reference_points.append(dc_reference_points)

    if dc_return_intermediate:
        # return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        inter_states, inter_references = torch.stack(intermediate), torch.stack(intermediate_reference_points)

    # return output, dc_reference_points
    # ------------------------------------self.decoder end------------------------------------

    inter_references_out = inter_references

    # return bev_embed, inter_states, init_reference_out, inter_references_out
    outputs = bev_embed, inter_states, init_reference_out, inter_references_out
    #      [2500,1,256], [6,900,1,256], [1,900,3]        , [6,1,900,3]
    # --------------------------------------self.transformer end--------------------------------------

bev_embed, inter_feats, init_reference_points, inter_references_points = outputs
inter_feats = inter_feats.permute(0, 2, 1, 3)  # torch.Size([6, 1, 900, 256])
outputs_classes = []
outputs_coords = []
for lvl in range(inter_feats.shape[0]):
    if lvl == 0:
        reference = init_reference_points
    else:
        reference = inter_references_points[lvl - 1]
    reference = inverse_sigmoid(reference)
    outputs_class = cls_branches[lvl](inter_feats[lvl])  # torch.Size([1, 900, 10])
    tmp = reg_branches[lvl](inter_feats[lvl])  # torch.Size([1, 900, 10])

    # TODO: check the shape of reference
    assert reference.shape[-1] == 3
    tmp[..., 0:2] += reference[..., 0:2]
    tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
    tmp[..., 4:5] += reference[..., 2:3]
    tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
    tmp[..., 0:1] = (tmp[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    tmp[..., 1:2] = (tmp[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    tmp[..., 4:5] = (tmp[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2])

    # TODO: check if using sigmoid
    outputs_coord = tmp
    outputs_classes.append(outputs_class)
    outputs_coords.append(outputs_coord)

# outputs_classes = torch.stack(outputs_classes)  # torch.Size([6, 1, 900, 10])
# outputs_coords = torch.stack(outputs_coords)  # torch.Size([6, 1, 900, 10])
# TODO: for convenience
outputs_classes = torch.stack([outputs_classes[0]])  # torch.Size([1, 1, 900, 10])
outputs_coords = torch.stack([outputs_coords[0]])  # torch.Size([1, 1, 900, 10])

outs = {
    'bev_embed': bev_embed,
    'all_cls_scores': outputs_classes,
    'all_bbox_preds': outputs_coords,
    'enc_cls_scores': None,
    'enc_bbox_preds': None,
}

# return outs
# -------------------------------------------pts_bbox_head end-------------------------------------------


loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
# losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
# ----------------------------------------pts_bbox_head.loss start----------------------------------------
bh_num_classes = 10
bh_cls_out_channels = 10
bh_bg_cls_weight = 0
bh_sync_cls_avg_factor = True
bh_code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
bh_code_weights = nn.Parameter(torch.tensor(bh_code_weights, requires_grad=False, device='cuda'), requires_grad=False)

gt_bboxes_list = gt_bboxes_3d  # [LiDARInstance3DBoxes] # LiDARInstance3DBoxes.tensor.shape: torch.Size([29, 9])
gt_labels_list = gt_labels_3d  # [torch.Size([29])]
preds_dicts = outs
gt_bboxes_ignore = None
# img_metas = img_metas

# assert gt_bboxes_ignore is None, \
#     f'{self.__class__.__name__} only supports ' \
#     f'for gt_bboxes_ignore setting to None.'

all_cls_scores = preds_dicts['all_cls_scores']  # torch.Size([1, 1, 900, 10])
all_bbox_preds = preds_dicts['all_bbox_preds']  # torch.Size([1, 1, 900, 10])
enc_cls_scores = preds_dicts['enc_cls_scores']  # None
enc_bbox_preds = preds_dicts['enc_bbox_preds']  # None

num_dec_layers = len(all_cls_scores)
device = gt_labels_list[0].device


def get_bboxes_gravity_center(boxes: LiDARInstance3DBoxes) -> torch.Tensor:
    bottom_center = boxes.tensor[:, :3]
    gravity_center = torch.zeros_like(bottom_center)
    gravity_center[:, :2] = bottom_center[:, :2]
    gravity_center[:, 2] = bottom_center[:, 2] + boxes.tensor[:, 5] * 0.5
    return gravity_center


# gt_bboxes_list = [torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
#                   for gt_bboxes in gt_bboxes_list]
gt_bboxes_list = [torch.cat((get_bboxes_gravity_center(gt_bboxes), gt_bboxes.tensor[:, 3:]), dim=1).to(device)
                  for gt_bboxes in gt_bboxes_list]

all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
# all_cls_scores: torch.Size([1, 1, 900, 10])
# all_bbox_preds: torch.Size([1, 1, 900, 10])
# all_gt_bboxes_list: [[torch.Size([29, 9])]]
# all_gt_labels_list: [[torch.Size([29])]]
# all_gt_bboxes_ignore_list: [None]
# losses_cls, losses_bbox = multi_apply(
#     self.loss_single, all_cls_scores, all_bbox_preds,
#     all_gt_bboxes_list, all_gt_labels_list,
#     all_gt_bboxes_ignore_list)
# ----------------------------------------loss_single start----------------------------------------
cls_scores = all_cls_scores[0]  # torch.Size([1, 900, 10])
bbox_preds = all_bbox_preds[0]  # torch.Size([1, 900, 10])
gt_bboxes_list = all_gt_bboxes_list[0]  # [torch.Size([29, 9])]
gt_labels_list = all_gt_labels_list[0]  # [torch.Size([29])]
gt_bboxes_ignore_list = None

num_imgs = cls_scores.size(0)
cls_scores_list = [cls_scores[i] for i in range(num_imgs)]  # [torch.Size([900, 10])]
bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]  # [torch.Size([900, 10])]
# cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
#                                    gt_bboxes_list, gt_labels_list,
#                                    gt_bboxes_ignore_list)
# -------------------------------------get_targets start-------------------------------------
assert gt_bboxes_ignore_list is None, 'Only supports for gt_bboxes_ignore setting to None.'
num_imgs = len(cls_scores_list)
gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]  # [None]

# (labels_list, label_weights_list, bbox_targets_list,
#  bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
#     self._get_target_single, cls_scores_list, bbox_preds_list,
#     gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
# -------------------------------_get_target_single start-------------------------------
cls_score = cls_scores_list[0]  # torch.Size([900, 10])
bbox_pred = bbox_preds_list[0]  # torch.Size([900, 10])
gt_labels = gt_labels_list[0]  # torch.Size([29])
gt_bboxes = gt_bboxes_list[0]  # torch.Size([29, 9])
gt_bboxes_ignore = None

num_bboxes = bbox_pred.size(0)
# assigner and sampler
gt_c = gt_bboxes.shape[-1]

# assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore)
# ------------------------------assigner.assign start------------------------------
bbox_pred = bbox_pred  # torch.Size([900, 10])
cls_pred = cls_score  # torch.Size([900, 10])
gt_bboxes = gt_bboxes  # torch.Size([29, 9])
gt_labels = gt_labels  # torch.Size([29])
gt_bboxes_ignore = gt_bboxes_ignore  # None
assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

# 1. assign -1 by default
assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
if num_gts == 0 or num_bboxes == 0:
    # No ground truth or boxes, return empty assignment
    if num_gts == 0:
        # No ground truth, assign all to background
        assigned_gt_inds[:] = 0
    # return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
    assign_result = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

# 2. compute the weighted costs
# classification and bboxcost.
# cls_cost = self.cls_cost(cls_pred, gt_labels)
# -------------------------------cls_cost start-------------------------------
cls_weight = 2.0
cls_alpha = 0.25
cls_gamma = 2
cls_eps = 1e-12

# gt_labels: torch.Size([29])
cls_pred = cls_pred.sigmoid()  # torch.Size([900, 10])
neg_cost = -(1 - cls_pred + cls_eps).log() * (1 - cls_alpha) * cls_pred.pow(cls_gamma)
pos_cost = -(cls_pred + cls_eps).log() * cls_alpha * (1 - cls_pred).pow(cls_gamma)
# TODO: ?
cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
# return cls_cost * cls_weight
cls_cost = cls_cost * cls_weight  # torch.Size([900, 29])
# --------------------------------cls_cost end--------------------------------
# regression L1 cost

# normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
# ----------------------------normalize_bbox start----------------------------
cx = gt_bboxes[..., 0:1]
cy = gt_bboxes[..., 1:2]
cz = gt_bboxes[..., 2:3]
w = gt_bboxes[..., 3:4].log()
l = gt_bboxes[..., 4:5].log()
h = gt_bboxes[..., 5:6].log()

rot = gt_bboxes[..., 6:7]
if gt_bboxes.size(-1) > 7:
    vx = gt_bboxes[..., 7:8]
    vy = gt_bboxes[..., 8:9]
    normalized_bboxes = torch.cat(
        (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
    )
# else:
#     normalized_bboxes = torch.cat(
#         (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
#     )
# return normalized_bboxes
normalized_gt_bboxes = normalized_bboxes.cuda()  # torch.Size([29, 10])
# -----------------------------normalize_bbox end-----------------------------
# reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
# -------------------------------reg_cost start-------------------------------
reg_weight = 0.25

# bbox_pred: torch.Size([900, 10])
bbox_cost = torch.cdist(bbox_pred[:, :8], normalized_gt_bboxes[:, :8], p=1)  # torch.Size([900, 29])
# return bbox_cost * self.weight
reg_cost = bbox_cost * reg_weight
# --------------------------------reg_cost end--------------------------------
# weighted sum of above two costs
cost = cls_cost + reg_cost  # torch.Size([900, 29])

# 3. do Hungarian matching on CPU using linear_sum_assignment
cost = cost.detach().cpu()
if linear_sum_assignment is None:
    raise ImportError('Please run "pip install scipy" to install scipy first.')
matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)  # torch.Size([29])
matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)  # torch.Size([29])

# 4. assign backgrounds and foregrounds
# assign all indices to backgrounds first
assigned_gt_inds[:] = 0  # torch.Size([900])
# assign foregrounds based on matching results
assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]  # torch.Size([900])
# return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
assign_result = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
# -------------------------------assigner.assign end-------------------------------
# sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
# -------------------------------sampler.sample start-------------------------------
assign_result, bboxes, gt_bboxes = assign_result, bbox_pred, gt_bboxes
# bboxes: # torch.Size([900, 10])  # gt_bboxes: # torch.Size([29, 9])
# pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
# neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
# TODO: Is .unique() necessary?
pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()  # torch.Size([29])
neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()  # torch.Size([871])
gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)  # torch.Size([900])
sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)

# return sampling_result
sampling_result = sampling_result
# --------------------------------sampler.sample end--------------------------------
# pos_inds = sampling_result.pos_inds
# neg_inds = sampling_result.neg_inds

# label targets
labels = gt_bboxes.new_full((num_bboxes,), bh_num_classes, dtype=torch.long)  # torch.Size([900])
# labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1  # pos_assigned_gt_inds == matched_col_inds
labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
label_weights = gt_bboxes.new_ones(num_bboxes)

# bbox targets
# bbox_pred: # torch.Size([900, 10])
# gt_c = gt_bboxes.shape[-1] # torch.Size([29, 9])
bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
bbox_weights = torch.zeros_like(bbox_pred)  # torch.Size([900, 10])
bbox_weights[pos_inds] = 1.0

# DETR
# bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
# gt_bboxes: # torch.Size([29, 9])
bbox_targets[pos_inds] = gt_bboxes[pos_assigned_gt_inds, :]
# return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)
labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list, neg_inds_list = [labels], [label_weights], [bbox_targets], [bbox_weights], [pos_inds], [neg_inds]
# --------------------------------_get_target_single end--------------------------------
num_total_pos = sum((inds.numel() for inds in pos_inds_list))
num_total_neg = sum((inds.numel() for inds in neg_inds_list))
# return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)
# --------------------------------------get_targets end--------------------------------------
# labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
labels = torch.cat(labels_list, 0)
label_weights = torch.cat(label_weights_list, 0)
bbox_targets = torch.cat(bbox_targets_list, 0)
bbox_weights = torch.cat(bbox_weights_list, 0)

# classification loss
cls_scores = cls_scores.reshape(-1, bh_cls_out_channels)
# construct weighted avg_factor to match with the official DETR repo
cls_avg_factor = num_total_pos * 1.0 + num_total_neg * bh_bg_cls_weight
if bh_sync_cls_avg_factor:
    # cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
    cls_avg_factor = cls_scores.new_tensor([cls_avg_factor])

cls_avg_factor = max(cls_avg_factor, 1)
# loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
focal_loss = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=2.0)
loss_cls = focal_loss(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

# Compute the average number of gt boxes accross all gpus, for
# normalization purposes
num_total_pos = loss_cls.new_tensor([num_total_pos])
# num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
num_total_pos = torch.clamp(num_total_pos, min=1).item()

# regression L1 loss
bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
normalized_bbox_targets = normalize_bbox(bbox_targets, pc_range)
isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
bbox_weights = bbox_weights * bh_code_weights

# loss_bbox = self.loss_bbox(bbox_preds[isnotnan, :10],
#                            normalized_bbox_targets[isnotnan, :10],
#                            bbox_weights[isnotnan, :10],
#                            avg_factor=num_total_pos)
l1_loss = L1Loss(reduction='mean', loss_weight=0.25)
loss_bbox = l1_loss(bbox_preds[isnotnan, :10],
                    normalized_bbox_targets[isnotnan, :10],
                    bbox_weights[isnotnan, :10],
                    avg_factor=num_total_pos)

if digit_version(TORCH_VERSION) >= digit_version('1.8'):
    loss_cls = torch.nan_to_num(loss_cls)
    loss_bbox = torch.nan_to_num(loss_bbox)
# return loss_cls, loss_bbox
# -----------------------------------------loss_single end-----------------------------------------
losses_cls, losses_bbox = [loss_cls], [loss_bbox]
loss_dict = dict()
# loss of proposal generated from encode feature map.
# if enc_cls_scores is not None:
#     binary_labels_list = [
#         torch.zeros_like(gt_labels_list[i])
#         for i in range(len(all_gt_labels_list))
#     ]
#     enc_loss_cls, enc_losses_bbox = \
#         self.loss_single(enc_cls_scores, enc_bbox_preds,
#                          gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
#     loss_dict['enc_loss_cls'] = enc_loss_cls
#     loss_dict['enc_loss_bbox'] = enc_losses_bbox

# loss from the last decoder layer
loss_dict['loss_cls'] = losses_cls[-1]
loss_dict['loss_bbox'] = losses_bbox[-1]

# loss from other decoder layers
num_dec_layer = 0
for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
    loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
    loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
    num_dec_layer += 1
# return loss_dict
# -----------------------------------------pts_bbox_head.loss end-----------------------------------------
# return losses
# ----------------------------------------------forward_pts_train end----------------------------------------------
losses.update(loss_dict)
# return losses
# --------------------------------------------------forward train end--------------------------------------------------

