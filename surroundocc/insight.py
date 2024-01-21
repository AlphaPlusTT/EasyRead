import copy
import pickle
import torch
import torch.nn as nn
import numpy as np
from mmcv.utils import TORCH_VERSION, digit_version
import torch.nn.functional as F
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import geo_scal_loss, sem_scal_loss
from insight_vis import occ_plot_points


img = torch.load('img.pt').cuda()
gt_occ = torch.load('gt_occ.pt').cuda()
with open('head_inputs.pkl', 'rb') as file:
    mlvl_feats, img_metas = pickle.load(file)
mlvl_feats = [tensor.cuda() for tensor in mlvl_feats]
# ------------------------------------------------OccHead forward start------------------------------------------------
conv_input = [512, 256, 256, 128, 128, 64, 64]
conv_output = [256, 256, 128, 128, 64, 64, 32]
num_classes = 17
volume_h = [100, 50, 25]
volume_w = [100, 50, 25]
volume_z = [8, 4, 2]
img_channels = [512, 512, 512]
use_semantic = True
embed_dims = [128, 256, 512]
fpn_level = len(embed_dims)  # 3
upsample_strides = [1, 2, 1, 2, 1, 2, 1]
out_indices = [0, 2, 4, 6]

volume_embedding = nn.ModuleList()
for i in range(fpn_level):
    volume_embedding.append(nn.Embedding(volume_h[i] * volume_w[i] * volume_z[i], embed_dims[i]))
volume_embedding = volume_embedding.cuda()

transfer_conv = nn.ModuleList()
# conv_cfg = dict(type='Conv2d', bias=True)
for i in range(fpn_level):
    # transfer_layer = build_conv_layer(
    #     conv_cfg,
    #     in_channels=img_channels[i],
    #     out_channels=embed_dims[i],
    #     kernel_size=1,
    #     stride=1)
    transfer_layer = nn.Conv2d(in_channels=img_channels[i],
                               out_channels=embed_dims[i],
                               kernel_size=1,
                               stride=1,
                               bias=True)
    transfer_block = nn.Sequential(transfer_layer,
                                   nn.ReLU(inplace=True))

    transfer_conv.append(transfer_block)
transfer_conv = transfer_conv.cuda()


deblocks = nn.ModuleList()
upsample_strides = upsample_strides
out_channels = conv_output
in_channels = conv_input
# norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
# upsample_cfg = dict(type='deconv3d', bias=False)
# conv_cfg = dict(type='Conv3d', bias=False)
# for i, out_channel in enumerate(out_channels):
#     stride = upsample_strides[i]
#     if stride > 1:
#         upsample_layer = build_upsample_layer(
#             upsample_cfg,
#             in_channels=in_channels[i],
#             out_channels=out_channel,
#             kernel_size=upsample_strides[i],
#             stride=upsample_strides[i])
#     else:
#         upsample_layer = build_conv_layer(
#             conv_cfg,
#             in_channels=in_channels[i],
#             out_channels=out_channel,
#             kernel_size=3,
#             stride=1,
#             padding=1)
#     deblock = nn.Sequential(upsample_layer,
#                             build_norm_layer(norm_cfg, out_channel)[1],  # [name, layer]
#                             nn.ReLU(inplace=True))
for i, out_channel in enumerate(out_channels):  # [256, 256, 128, 128, 64, 64, 32]
    stride = upsample_strides[i]
    if stride > 1:
        upsample_layer = nn.ConvTranspose3d(in_channels=in_channels[i], out_channels=out_channel,
                                            kernel_size=upsample_strides[i], stride=upsample_strides[i], bias=False)  # upsample*2
    else:
        upsample_layer = nn.Conv3d(in_channels=in_channels[i], out_channels=out_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
    deblock = nn.Sequential(upsample_layer,
                            nn.GroupNorm(num_groups=16, num_channels=out_channel),
                            nn.ReLU(inplace=True))
    deblocks.append(deblock)
deblocks = deblocks.cuda()


# conv_cfg = dict(type='Conv3d', bias=False)
occ = nn.ModuleList()
# for i in out_indices:
#     if use_semantic:
#         occ = build_conv_layer(
#             conv_cfg,
#             in_channels=out_channels[i],
#             out_channels=num_classes,
#             kernel_size=1,
#             stride=1,
#             padding=0)
#         occ.append(occ)
#     else:
#         occ = build_conv_layer(
#             conv_cfg,
#             in_channels=out_channels[i],
#             out_channels=1,
#             kernel_size=1,
#             stride=1,
#             padding=0)
#         occ.append(occ)
for i in out_indices:
    if use_semantic:
        occ_ = nn.Conv3d(in_channels=out_channels[i], out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=False)
    else:
        occ_ = nn.Conv3d(in_channels=out_channels[i], out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
    occ.append(occ_)
occ = occ.cuda()


bs, num_cam, _, _, _ = mlvl_feats[0].shape
# [torch.Size([1, 6, 512, 116, 200]), torch.Size([1, 6, 512, 58, 100]), torch.Size([1, 6, 512, 29, 50])]
dtype = mlvl_feats[0].dtype

volume_embed = []
for i in range(fpn_level):
    volume_queries = volume_embedding[i].weight.to(dtype)  # 0: torch.Size([100*100*8, 128])

    oh_volume_h = volume_h[i]  # 0: 100
    oh_volume_w = volume_w[i]  # 0: 100
    oh_volume_z = volume_z[i]  # 0: 8

    _, _, C, H, W = mlvl_feats[i].shape
    view_features = transfer_conv[i](mlvl_feats[i].reshape(bs * num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W)
    # 0: torch.Size([1, 6, 128, 116, 200])

    # volume_embed_i = transformer[i](
    #     [view_features],
    #     volume_queries,
    #     volume_h=volume_h,
    #     volume_w=volume_w,
    #     volume_z=volume_z,
    #     img_metas=img_metas
    # )
    # ---------------------------------------------transformer start---------------------------------------------
    # PerceptionTransformer
    use_cams_embeds = True
    num_feature_levels = 4
    num_cams = 6
    tf_embed_dims = embed_dims[i]
    level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, tf_embed_dims)).cuda()
    cams_embeds = nn.Parameter(torch.Tensor(num_cams, tf_embed_dims)).cuda()
    pc_range = [-50, -50, -5.0, 50, 50, 3.0]


    bs = view_features.size(0)  # 0: torch.Size([1, 6, 128, 116, 200])
    volume_queries = volume_queries.unsqueeze(1).repeat(1, bs, 1)  # 0: torch.Size([100*100*8, 1, 128])

    feat_flatten = []
    spatial_shapes = []
    for lvl, feat in enumerate([view_features]):
        bs, num_cam, c, h, w = feat.shape  # 0: torch.Size([1, 6, 128, 116, 200])
        spatial_shape = (h, w)
        feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam, bs, hw, c

        if use_cams_embeds:
            feat = feat + cams_embeds[:, None, None, :].to(feat.dtype)
        feat = feat + level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
        spatial_shapes.append(spatial_shape)
        feat_flatten.append(feat)

    feat_flatten = torch.cat(feat_flatten, 2)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
    # 0: tensor([[116, 200]], device='cuda:0')

    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # 0: tensor([0], device='cuda:0')

    feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, tf_embed_dims)
    # 0: torch.Size([6, 116*200, 1, 128])

    # volume_embed = encoder(
    #     volume_queries,
    #     feat_flatten,
    #     feat_flatten,
    #     volume_h=oh_volume_h,
    #     volume_w=oh_volume_w,
    #     volume_z=oh_volume_z,
    #     spatial_shapes=spatial_shapes,
    #     level_start_index=level_start_index,
    #     **kwargs  # img_metas=img_metas
    # )
    # -------------------------------------------encoder start-------------------------------------------
    # OccEncoder
    # volume_query = volume_queries
    key = feat_flatten
    value = feat_flatten
    # *args = []
    # volume_h = oh_volume_h
    # volume_w = oh_volume_w
    # volume_z = oh_volume_z
    spatial_shapes = spatial_shapes
    level_start_index = level_start_index
    # **kwargs = {img_metas: img_metas}
    output = volume_queries
    intermediate = []
    return_intermediate = False

    # ref_3d = self.get_reference_points(
    #     volume_h, volume_w, volume_z, bs=volume_queries.size(1), device=volume_queries.device, dtype=volume_queries.dtype)
    # ----------------------------------get_reference_points start----------------------------------
    H, W, Z = oh_volume_h, oh_volume_w, oh_volume_z  # 0: 100， 100， 8
    bs, device, dtype = volume_queries.size(1), volume_queries.device, volume_queries.dtype
    zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype, device=device).view(Z, 1, 1).expand(Z, H, W) / Z
    xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(Z, H, W) / W
    ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(Z, H, W) / H
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
    ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)  # 0: torch.Size([1, 1, 100*100*8, 3])
    # return ref_3d
    # occ_plot_points(ref_3d.cpu().numpy()[0, 0])
    # -----------------------------------get_reference_points end-----------------------------------

    # reference_points_cam, volume_mask = self.point_sampling(ref_3d, self.pc_range, kwargs['img_metas'])
    # -------------------------------------point_sampling start-------------------------------------
    reference_points, pc_range, img_metas = ref_3d, pc_range, img_metas
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # 0: torch.Size([1, 6, 4, 4])  # (B, N, 4, 4)
    reference_points = reference_points.clone()  # 0: torch.Size([1, 1, 100*100*8, 3])

    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    # xyz -> xyz1

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    # 0: torch.Size([1, 1, 6, 100*100*8, 4, 1])

    lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
    # 0: torch.Size([1, 1, 6, 100*100*8, 4, 4])

    reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
    # 0: torch.Size([1, 1, 6, 100*100*8, 4]) u'v's1
    eps = 1e-5

    # temp = reference_points_cam.cpu().numpy()[0, 0, 3, :, :3]
    # # mask = (temp[:, 0] > -500) & (temp[:, 0] < 500) & (temp[:, 1] > -500) & (temp[:, 1] < 500)
    # # temp = temp[mask]
    # temp = temp.reshape(-1, 3)
    # points = copy.deepcopy(temp)
    # # mask = temp[:, 2] > eps
    # # temp = temp[:, :2] / np.maximum(temp[:, 2:3], np.ones_like(temp[:, 2:3]) * eps)
    # # temp[:, 0] /= img_metas[0]['img_shape'][0][1]
    # # temp[:, 1] /= img_metas[0]['img_shape'][0][0]
    # # mask = mask & (temp[:, 1] > 0) & (temp[:, 1] < 1) & (temp[:, 0] > 0) & (temp[:, 0] < 1)
    # # points = points[mask]
    # occ_plot_points(points, flag=0)

    volume_mask = (reference_points_cam[..., 2:3] > eps)  # 0: torch.Size([1, 1, 6, 100*100*8, 1])  # > 0
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)  # u'v' / max(s, eps)

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]  # u / h
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]  # v / w

    volume_mask = (volume_mask & (reference_points_cam[..., 1:2] > 0.0)
                   & (reference_points_cam[..., 1:2] < 1.0)
                   & (reference_points_cam[..., 0:1] < 1.0)
                   & (reference_points_cam[..., 0:1] > 0.0))  # 0 < u/h, v/w < 1
    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        volume_mask = torch.nan_to_num(volume_mask)
    else:
        volume_mask = volume_mask.new_tensor(np.nan_to_num(volume_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)  # 0: torch.Size([6, 1, 100*100*8, 1, 2])  # xyz <-> uv
    volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1)  # 0: torch.Size([6, 1, 100*100*8, 1])
    # return reference_points_cam, volume_mask
    # --------------------------------------point_sampling end--------------------------------------

    # (num_query, bs, tf_embed_dims) -> (bs, num_query, tf_embed_dims)
    volume_queries = volume_queries.permute(1, 0, 2)  # 0: torch.Size([1, 100*100*8, 128])

    # for lid, layer in enumerate(self.layers):
    #     output = layer(
    #         volume_queries,
    #         key,
    #         value,
    #         *args,  # []
    #         ref_3d=ref_3d,
    #         volume_h=oh_volume_h,
    #         volume_w=oh_volume_w,
    #         volume_z=oh_volume_z,
    #         spatial_shapes=spatial_shapes,
    #         level_start_index=level_start_index,
    #         reference_points_cam=reference_points_cam,
    #         bev_mask=volume_mask,
    #         **kwargs)  # img_metas=img_metas
    #
    #     volume_queries = output
    #     if self.return_intermediate:
    #         intermediate.append(output)

    # [OccLayer]
    # [OccLayer, OccLayer, OccLayer]
    # [OccLayer, OccLayer, OccLayer, OccLayer, OccLayer, OccLayer]
    # -----------------------------------------layers start-----------------------------------------
    query = volume_queries  # 0: torch.Size([1, 100*100*8, 128])
    key = feat_flatten  # 0: torch.Size([6, 116*200, 1, 128])
    value = feat_flatten  # 0: torch.Size([6, 116*200, 1, 128])
    query_pos = None
    key_pos = None
    attn_masks = None
    query_key_padding_mask = None
    key_padding_mask = None
    ref_3d = ref_3d  # 0: torch.Size([1, 1, 100*100*8, 3])
    # volume_h = oh_volume_h
    # volume_w = oh_volume_w
    # volume_z = oh_volume_z
    reference_points_cam = reference_points_cam  # 0: torch.Size([6, 1, 100*100*8, 1, 2])
    mask = None
    spatial_shapes = spatial_shapes  # 0: tensor([[116, 200]], device='cuda:0')
    level_start_index = level_start_index  # 0: tensor([0], device='cuda:0')
    bev_mask = volume_mask  # 0: torch.Size([6, 1, 100*100*8, 1])


    operation_order = ('cross_attn', 'norm', 'ffn', 'norm', 'conv')
    num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
    deblock = nn.ModuleList()
    conv_num = 2
    # conv_cfg = dict(type='Conv3d', bias=False)
    # norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
    # for ii in range(conv_num):
    #     conv_layer = build_conv_layer(
    #         conv_cfg,
    #         in_channels=embed_dims,
    #         out_channels=embed_dims,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1)
    #     deblock = nn.Sequential(conv_layer,
    #                             build_norm_layer(norm_cfg, embed_dims)[1],  # [name, layer]
    #                             nn.ReLU(inplace=True))
    for ii in range(conv_num):
        conv_layer = nn.Conv3d(in_channels=tf_embed_dims, out_channels=tf_embed_dims,
                               kernel_size=3, stride=1, padding=1, bias=False)
        temp = nn.Sequential(conv_layer,
                             nn.GroupNorm(num_groups=16, num_channels=tf_embed_dims),
                             nn.ReLU(inplace=True))
        deblock.append(temp)
    deblock = deblock.cuda()


    norm_index = 0
    attn_index = 0
    ffn_index = 0
    identity = query
    if attn_masks is None:
        attn_masks = [None for _ in range(num_attn)]
    # elif isinstance(attn_masks, torch.Tensor):
    #     attn_masks = [copy.deepcopy(attn_masks) for _ in range(num_attn)]
    #     warnings.warn(f'Use same attn_mask in all attentions in 'f'{__class__.__name__} ')
    # else:
    #     assert len(attn_masks) == num_attn, f'The length of ' \
    #                                              f'attn_masks {len(attn_masks)} must be equal ' \
    #                                              f'to the number of attention in ' \
    #                                              f'operation_order {num_attn}'

    for layer in operation_order:
        # temporal self attention
        if layer == 'conv':
            bs = query.shape[0]
            identity = query
            query = query.reshape(bs, oh_volume_z, oh_volume_h, oh_volume_w, -1).permute(0, 4, 3, 2, 1)
            for ii in range(len(deblock)):
                query = deblock[ii](query)
            query = query.permute(0, 4, 3, 2, 1).reshape(bs, oh_volume_z * oh_volume_h * oh_volume_w, -1)
            query = query + identity

        elif layer == 'norm':
            norm = nn.LayerNorm(tf_embed_dims).cuda()
            # query = norms[norm_index](query)
            query = norm(query)
            norm_index += 1

        # spaital cross attention
        elif layer == 'cross_attn':
            # query = attentions[attn_index](
            #     query,
            #     key,
            #     value,
            #     identity if pre_norm else None,
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
            # --------------------------------cross_attn start--------------------------------
            query = volume_queries  # 0: torch.Size([1, 100*100*8, 128])
            key = feat_flatten  # 0: torch.Size([6, 116*200, 1, 128])
            value = feat_flatten  # 0: torch.Size([6, 116*200, 1, 128])
            residual = None
            query_pos = None
            key_pos = None
            key_padding_mask = None
            # reference_points = ref_3d  # 0: torch.Size([1, 1, 100*100*8, 3])
            spatial_shapes = spatial_shapes  # 0: tensor([[116, 200]], device='cuda:0')
            reference_points_cam = reference_points_cam  # 0: torch.Size([6, 1, 100*100*8, 1, 2])
            bev_mask = volume_mask  # 0: torch.Size([6, 1, 100*100*8, 1])
            level_start_index = level_start_index  # 0: tensor([0], device='cuda:0')
            flag = 'encoder'
            mask = None
            attn_mask = None
            img_metas = img_metas

            dropout = nn.Dropout(0.1).cuda()
            output_proj = nn.Linear(tf_embed_dims, tf_embed_dims).cuda()

            # if key is None:
            #     key = query
            # if value is None:
            #     value = key

            if residual is None:
                inp_residual = query  # 0: torch.Size([1, 100*100*8, 128])
                slots = torch.zeros_like(query)  # 0: torch.Size([1, 100*100*8, 128])
            # if query_pos is not None:
            #     query = query + query_pos

            bs, num_query, _ = query.size()

            D = reference_points_cam.size(3)  # 0: torch.Size([6, 1, 100*100*8, 1, 2])
            indexes = []
            for ii, mask_per_img in enumerate(bev_mask):  # 0: torch.Size([6, 1, 100*100*8, 1])
                index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                indexes.append(index_query_per_img)
            max_len = max([len(each) for each in indexes])

            # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
            queries_rebatch = query.new_zeros([bs, num_cams, max_len, tf_embed_dims])
            # 0: torch.Size([1, 6, 18919, 128])
            reference_points_rebatch = reference_points_cam.new_zeros([bs, num_cams, max_len, D, 2])
            # 0: torch.Size([1, 6, 18919, 1, 2])

            for j in range(bs):
                for ii, reference_points_per_img in enumerate(reference_points_cam):
                    index_query_per_img = indexes[ii]
                    queries_rebatch[j, ii, :len(index_query_per_img)] = query[j, index_query_per_img]
                    reference_points_rebatch[j, ii, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

            # num_cams, l, bs, embed_dims = key.shape
            num_cams, l, bs, _ = key.shape  # 0: torch.Size([6, 116*200, 1, 128])

            key = key.permute(2, 0, 1, 3).reshape(bs * num_cams, l, tf_embed_dims)
            # 0: torch.Size([6, 116*200, 128])
            value = value.permute(2, 0, 1, 3).reshape(bs * num_cams, l, tf_embed_dims)
            # 0: torch.Size([6, 116*200, 128])

            # queries = deformable_attention(
            #     query=queries_rebatch.view(bs * num_cams, max_len, tf_embed_dims), key=key, value=value,
            #     reference_points=reference_points_rebatch.view(bs * num_cams, max_len, D, 2),
            #     spatial_shapes=spatial_shapes,
            #     level_start_index=level_start_index).view(bs, num_cams, max_len, tf_embed_dims)
            # ------------------------deformable_attention start------------------------
            query = queries_rebatch.view(bs * num_cams, max_len, tf_embed_dims)  # torch.Size([6, 18919, 128])
            key = key  # 0: torch.Size([6, 116*200, 128])
            value = value  # 0: torch.Size([6, 116*200, 128])
            identity = None
            query_pos = None
            key_padding_mask = None
            reference_points = reference_points_rebatch.view(bs * num_cams, max_len, D, 2)
            # 0: torch.Size([6, 18919, 1, 2])
            spatial_shapes = spatial_shapes  # 0: tensor([[116, 200]], device='cuda:0')
            level_start_index = level_start_index  # 0: tensor([0], device='cuda:0')

            batch_first = True
            value_proj = nn.Linear(tf_embed_dims, tf_embed_dims).cuda()
            num_levels = 1
            num_heads = 8
            num_points = 2
            sampling_offsets = nn.Linear(tf_embed_dims, num_heads * num_levels * num_points * 2).cuda()
            attention_weights = nn.Linear(tf_embed_dims, num_heads * num_levels * num_points).cuda()
            im2col_step = 64

            # if value is None:
            #     value = query
            if identity is None:
                identity = query  # torch.Size([6, 18919, 128])
            # if query_pos is not None:
            #     query = query + query_pos

            # if not batch_first:
            #     # change to (bs, num_query ,tf_embed_dims)
            #     query = query.permute(1, 0, 2)
            #     value = value.permute(1, 0, 2)

            da_bs, num_query, _ = query.shape  # 0: torch.Size([6, 18919, 128])
            da_bs, num_value, _ = value.shape  # 0: torch.Size([6, 116*200, 128])
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

            value = value_proj(value)
            # if key_padding_mask is not None:
            #     value = value.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.view(da_bs, num_value, num_heads, -1)  # 0: torch.Size([6, 116*200, 8, 16])
            sampling_offsets = sampling_offsets(query).view(da_bs, num_query, num_heads, num_levels, num_points, 2)
            # 0: torch.Size([6, 18919, 8, 1, 2, 2])

            attention_weights = attention_weights(query).view(da_bs, num_query, num_heads, num_levels * num_points)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(da_bs, num_query, num_heads, num_levels, num_points)
            # 0: torch.Size([6, 18919, 8, 1, 2])

            if reference_points.shape[-1] == 2:
                """
                For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
                After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
                For each referent point, we sample `num_points` sampling points.
                For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
                """
                offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                da_bs, num_query, num_Z_anchors, xy = reference_points.shape  # 0: torch.Size([6, 18919, 1, 2])
                reference_points = reference_points[:, :, None, None, None, :, :]
                # 0: torch.Size([6, 18919, 1, 1, 1, 1, 2])
                sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                da_bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
                # 0: torch.Size([6, 18919, 8, 1, 2, 2])
                sampling_offsets = sampling_offsets.view(
                    da_bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
                # 0: torch.Size([6, 18919, 8, 1, 2, 1, 2])
                sampling_locations = reference_points + sampling_offsets
                da_bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
                # 0: torch.Size([6, 18919, 8, 1, 2, 1, 2])
                assert num_all_points == num_points * num_Z_anchors

                sampling_locations = sampling_locations.view(da_bs, num_query, num_heads, num_levels, num_all_points, xy)
                # 0: torch.Size([6, 18919, 8, 1, 2, 2])

            # elif reference_points.shape[-1] == 4:
            #     assert False
            # else:
            #     raise ValueError(
            #         f'Last dim of reference_points must be'
            #         f' 2 or 4, but get {reference_points.shape[-1]} instead.')

            #  sampling_locations.shape: da_bs, num_query, num_heads, num_levels, num_all_points, 2
            #  attention_weights.shape: da_bs, num_query, num_heads, num_levels, num_all_points

            use_cuda = False
            if torch.cuda.is_available() and value.is_cuda and use_cuda:
                # if value.dtype == torch.float16:
                #     MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                # else:
                #     MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                # output = MultiScaleDeformableAttnFunction.apply(
                #     value, spatial_shapes, level_start_index, sampling_locations,
                #     attention_weights, im2col_step)
                raise NotImplementedError
            else:
                # output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
                # ------------multi_scale_deformable_attn_pytorch start------------
                da_bs, _, num_heads, da_embed_dims = value.shape  # 0: torch.Size([6, 116*200, 8, 16])
                _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
                # 0: torch.Size([6, 18919, 8, 1, 2, 2])
                value_list = value.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)  # value_list = [value]
                sampling_grids = 2 * sampling_locations - 1  # 0: torch.Size([6, 18919, 8, 1, 2, 2])
                sampling_value_list = []
                for level, (H_, W_) in enumerate(spatial_shapes):
                    value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(da_bs * num_heads, da_embed_dims, H_, W_)
                    # da_bs, H_*W_, num_heads, da_embed_dims ->
                    # da_bs, H_*W_, num_heads*da_embed_dims ->
                    # da_bs, num_heads*da_embed_dims, H_*W_ ->
                    # da_bs*num_heads, da_embed_dims, H_, W_
                    # torch.Size([6*8, 16, 116, 200])

                    sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
                    # da_bs, num_queries, num_heads, num_points, 2 ->
                    # da_bs, num_heads, num_queries, num_points, 2 ->
                    # da_bs*num_heads, num_queries, num_points, 2
                    # torch.Size([48, 18919, 2, 2])

                    sampling_value_l_ = F.grid_sample(
                        value_l_,
                        sampling_grid_l_,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False)
                    # da_bs*num_heads, da_embed_dims, num_queries, num_points
                    # torch.Size([6*8, 16, 18919, 2])

                    sampling_value_list.append(sampling_value_l_)
                attention_weights = attention_weights.transpose(1, 2).reshape(
                    da_bs * num_heads, 1, num_queries, num_levels * num_points)
                # (da_bs, num_queries, num_heads, num_levels, num_points) ->
                # (da_bs, num_heads, num_queries, num_levels, num_points) ->
                # (da_bs, num_heads, 1, num_queries, num_levels*num_points)
                # 0: torch.Size([6*8, 1, 18919, 2])

                output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
                          ).sum(-1).view(da_bs, num_heads * da_embed_dims, num_queries)
                # return output.transpose(1, 2).contiguous()
                output = output.transpose(1, 2).contiguous()  # torch.Size([6, 18919, 128])
                # -------------multi_scale_deformable_attn_pytorch end-------------

            # if not batch_first:
            #     output = output.permute(1, 0, 2)

            # return output
            queries = output.view(bs, num_cams, max_len, tf_embed_dims)
            #                      1     6       18919      128
            # -------------------------deformable_attention end-------------------------
            for j in range(bs):
                for ii, index_query_per_img in enumerate(indexes):
                    slots[j, index_query_per_img] += queries[j, ii, :len(index_query_per_img)]

            count = bev_mask.sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots = slots / count[..., None]
            slots = output_proj(slots)

            # return self.dropout(slots) + inp_residual
            query = dropout(slots) + inp_residual
            # ---------------------------------cross_attn end---------------------------------
            attn_index += 1
            identity = query

        elif layer == 'ffn':
            # query = ffns[ffn_index](
            #     query, identity if pre_norm else None)
            num_fcs = 2
            feedforward_channels = 256
            ffn_drop = 0.1
            layers = []
            in_channels = tf_embed_dims
            for _ in range(num_fcs - 1):
                layers.append(nn.Sequential(nn.Linear(in_channels, feedforward_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(ffn_drop)))
                in_channels = feedforward_channels
            layers.append(nn.Linear(feedforward_channels, tf_embed_dims))
            layers.append(nn.Dropout(ffn_drop))
            layers = nn.Sequential(*layers).cuda()
            # dropout_layer = build_dropout(
            #     dropout_layer) if dropout_layer else torch.nn.Identity()
            dropout_layer = nn.Identity().cuda()
            ffn_index += 1
            add_identity = True
            identity = None

            out = layers(query)
            # if not add_identity:
            #     return self.dropout_layer(out)
            if identity is None:
                identity = query
            # return identity + self.dropout_layer(out)
            query = identity + dropout_layer(out)
    # return query
    output = query
    # ------------------------------------------layers end------------------------------------------
    volume_queries = output

    # if return_intermediate:
    #     intermediate.append(output)

    # if return_intermediate:
    #     return torch.stack(intermediate)

    # return output
    # --------------------------------------------encoder end--------------------------------------------
    # return volume_embed
    volume_embed_i = output
    # ----------------------------------------------transformer end----------------------------------------------
    volume_embed.append(volume_embed_i)
    # [torch.Size([1, 100*100*8, 128]), torch.Size([1, 50*50*4, 256]), torch.Size([1, 25*25*2, 512])]

volume_embed_reshape = []
for i in range(fpn_level):
    oh_volume_h = volume_h[i]
    oh_volume_w = volume_w[i]
    oh_volume_z = volume_z[i]

    volume_embed_reshape_i = volume_embed[i].reshape(bs, oh_volume_z, oh_volume_h, oh_volume_w, -1).permute(0, 4, 3, 2, 1)

    volume_embed_reshape.append(volume_embed_reshape_i)
    # [torch.Size([1, 128, 100, 100, 8]), torch.Size([1, 256, 50, 50, 4]), torch.Size([1, 512, 25, 25, 2])]

outputs = []
result = volume_embed_reshape.pop()
for i in range(len(deblocks)):
    result = deblocks[i](result)

    if i in out_indices:
        outputs.append(result)
    elif i < len(deblocks) - 2:  # we do not add skip connection at level 0
        volume_embed_temp = volume_embed_reshape.pop()
        result = result + volume_embed_temp
# outputs: # [torch.Size([1, 256, 25, 25, 2]),
#             torch.Size([1, 128, 50, 50, 4]),
#             torch.Size([1, 64, 100, 100, 8]),
#             torch.Size([1, 32, 200, 200, 16])]

occ_preds = []
for i in range(len(outputs)):
    occ_pred = occ[i](outputs[i])
    occ_preds.append(occ_pred)

outs = {
    'volume_embed': volume_embed,
    'occ_preds': occ_preds,
}
# volume_embed: [torch.Size([1, 80000, 128]),
#                torch.Size([1, 10000, 256]),
#                torch.Size([1, 1250, 512])]
# occ_preds: [torch.Size([1, 17, 25, 25, 2]),
#             torch.Size([1, 17, 50, 50, 4]),
#             torch.Size([1, 17, 100, 100, 8]),
#             torch.Size([1, 17, 200, 200, 16])]
# return outs
# -------------------------------------------------OccHead forward end-------------------------------------------------
# -----------------------------------------------------loss start-----------------------------------------------------
gt_occ = gt_occ  # torch.Size([1, 47876, 4]) 4:xyz,label
preds_dicts = outs

if not use_semantic:
    # loss_dict = {}
    # for i in range(len(preds_dicts['occ_preds'])):
    #     pred = preds_dicts['occ_preds'][i][:, 0]
    #
    #     ratio = 2 ** (len(preds_dicts['occ_preds']) - 1 - i)
    #
    #     gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
    #
    #     # gt = torch.mode(gt, dim=-1)[0].float()
    #
    #     loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))
    #
    #     loss_occ_i = loss_occ_i * ((0.5) ** (len(preds_dicts['occ_preds']) - 1 - i))  # * focal_weight
    #
    #     loss_dict['loss_occ_{}'.format(i)] = loss_occ_i
    raise NotImplementedError

else:
    pred = preds_dicts['occ_preds']

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")

    loss_dict = {}

    # occ_preds: [torch.Size([1, 17, 25, 25, 2]),
    #             torch.Size([1, 17, 50, 50, 4]),
    #             torch.Size([1, 17, 100, 100, 8]),
    #             torch.Size([1, 17, 200, 200, 16])]
    for i in range(len(preds_dicts['occ_preds'])):
        pred = preds_dicts['occ_preds'][i]  # 0: torch.Size([1, 17, 25, 25, 2])
        ratio = 2 ** (len(preds_dicts['occ_preds']) - 1 - i)  # 0: 8

        # gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
        # -------------------------------------multiscale_supervision start-------------------------------------
        gt_occ, ratio, gt_shape = gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape
        gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).type(torch.float)
        for ii in range(gt.shape[0]):  # 0: torch.Size([1, 25, 25, 2])
            # gt_occ: torch.Size([1, 47876, 4])
            coords = gt_occ[ii][:, :3].type(torch.long) // ratio
            gt[ii, coords[:, 0], coords[:, 1], coords[:, 2]] = gt_occ[ii][:, 3]
        # return gt
        # --------------------------------------multiscale_supervision end--------------------------------------

        loss_occ_i = (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))

        loss_occ_i = loss_occ_i * ((0.5) ** (len(preds_dicts['occ_preds']) - 1 - i))

        loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

# return loss_dict
# ------------------------------------------------------loss end------------------------------------------------------







