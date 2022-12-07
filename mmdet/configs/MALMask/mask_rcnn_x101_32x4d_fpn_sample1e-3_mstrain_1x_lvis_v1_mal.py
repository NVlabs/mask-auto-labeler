_base_ = './mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))

data = dict(
    train=dict(
        dataset=dict(ann_file='/discobox/AutoLabeler/pseudo_labels/lvis/vit-mae-base-crfloss6.json',
                     img_prefix='data/coco')),
    val=dict(ann_file="data/lvis/lvis_v1_val.json",
             img_prefix='data/coco'),
    test=dict(ann_file="data/lvis/lvis_v1_val.json",
             img_prefix='data/coco'))
evaluation = dict(interval=12)