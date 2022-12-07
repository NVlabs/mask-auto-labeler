_base_ = '../GTMask/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py'  # noqa

# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128-noema_in1k_20220222-dba4f95f.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[128, 256, 512, 1024]))

optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12
    })

data = dict(train=dict(ann_file='../pseudo_labels/coco/instances_train2017_mal.json'),
            test=dict(
                ann_file='data/coco/annotations/image_info_test-dev2017.json',
                img_prefix='data/coco/test2017/'))
