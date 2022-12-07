_base_ = './mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

data = dict(
    train=dict(
        dataset=dict(ann_file='/discobox/AutoLabeler/pseudo_labels/lvis/vit-mae-base-crfloss6.json',
                     img_prefix='data/coco')),
    val=dict(ann_file="data/lvis/lvis_v1_val.json",
             img_prefix='data/coco'),
    test=dict(ann_file="data/lvis/lvis_v1_val.json",
             img_prefix='data/coco'))
evaluation = dict(interval=12)