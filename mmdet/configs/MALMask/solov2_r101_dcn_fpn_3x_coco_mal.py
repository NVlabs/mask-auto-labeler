_base_ = '../GTMask/solov2_r50_fpn_3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(checkpoint='torchvision://resnet101'),
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    mask_head=dict(
        mask_feature_head=dict(conv_cfg=dict(type='DCNv2')),
        dcn_cfg=dict(type='DCNv2'),
        dcn_apply_to_all_conv=True))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 10,
    step=[27, 33])

data=dict(
    train=dict(ann_file='../pseudo_labels/coco/instances_train2017_mal.json'),
    test=dict(
        ann_file='data/coco/annotations/image_info_test-dev2017.json',
        img_prefix='data/coco/test2017/'))
