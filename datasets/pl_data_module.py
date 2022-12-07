# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .voc import BoxLabelVOC, BoxLabelVOCLMDB, InstSegVOC, InstSegVOCLMDB, InstSegVOCwithBoxInput
from .coco import BoxLabelCOCO, InstSegCOCO, BoxLabelCOCOLMDB, InstSegCOCOLMDB, InstSegCOCOwithBoxInput
from .objects365 import InstSegObjects365LMDB, BoxLabelObjects365LMDB, BoxLabelObjects365COCOScheduleLMDB
from .data_aug import data_aug_pipelines, custom_collate_fn
from .lvis import InstSegLVIS, BoxLabelLVIS, BoxLabelLVISLMDB, InstSegLVISLMDB, InstSegLVISwithBoxInput
from .ytvis import BoxLabelYTVIS, InstSegYTVIS


num_class_dict = {
    'coco': 81,
    'coco_original': 81,
    'cocolmdb': 81,
    'objects365lmdb': 365,
    'voc': 21,
    'lvis': 1302,
    'ytvis': 41,
    'astro': 81
}


datapath_configs = dict(
    coco=dict(
        training_config=dict(
            train_img_data_dir='data/coco/train2017', 
            val_img_data_dir='data/coco/val2017', 
            test_img_data_dir='data/coco/test2017',
            dataset_type='coco',
            train_ann_path="data/coco/annotations/boxes_train2017.json",
            val_ann_path="data/coco/annotations/instances_val2017.json",
        ),
        generating_pseudo_label_config=dict(
            train_img_data_dir='data/coco/train2017', 
            train_ann_path="data/coco/annotations/boxes_train2017.json",
            val_img_data_dir='data/coco/train2017', 
            dataset_type='coco',
            val_ann_path="data/coco/annotations/boxes_train2017.json",
        )
    ),
    coco_original=dict(
        training_config=dict(
            train_img_data_dir='data/coco/train2017', 
            val_img_data_dir='data/coco/train2017', 
            test_img_data_dir='data/coco/test2017',
            dataset_type='coco',
            train_ann_path="data/coco/annotations/boxes_train2017.json",
            val_ann_path="data/coco/annotations/instances_val2017.json",
        ),
        generating_pseudo_label_config=dict(
            train_img_data_dir='data/coco/train2017', 
            train_ann_path="data/coco/annotations/instances_train2017.json",
            #val_img_data_dir='data/coco/train2017', 
            val_img_data_dir='data/coco/val2017', 
            dataset_type='coco',
            #val_ann_path="data/coco/annotations/instances_train2017.json",
            val_ann_path="data/coco/annotations/instances_val2017.json",
        )
    ),
    coco_lmdb=dict(
        training_config=dict(
            train_img_data_dir='data/coco/train2017', 
            val_img_data_dir='data/coco/val2017', 
            test_img_data_dir='data/coco/test2017',
            dataset_type='cocolmdb',
            train_ann_path="data/coco/lmdb/train2017",
            val_ann_path="data/coco/lmdb/val2017"
        ),
        generating_pseudo_label_config=dict(
            train_img_data_dir='data/coco/train2017', 
            train_ann_path="data/coco/annotations/boxes_train2017.json",
            val_img_data_dir='data/coco/train2017', 
            dataset_type='coco',
            val_ann_path="data/coco/annotations/boxes_train2017.json",
        )
    ),
    lvis=dict(
        training_config=dict(
            train_img_data_dir='data/coco', 
            val_img_data_dir='data/coco', 
            test_img_data_dir='data/coco/test2017',
            dataset_type='lvis',
            train_ann_path="/discobox/zcxu/lvis/lvis_v1_box_train.json",
            val_ann_path="/discobox/zcxu/lvis/lvis_v1_val.json"
        ),
        generating_pseudo_label_config=dict(
            train_img_data_dir='data/coco', 
            val_img_data_dir='data/coco', 
            test_img_data_dir='data/coco/test2017',
            dataset_type='lvis',
            train_ann_path="/DDN_ROOT/data/lvis/lvis_v1_train.json",
            val_ann_path="/DDN_ROOT/data/lvis/lvis_v1_train.json"
        ),
    ),
    objects365lmdb=dict(
        training_config=dict(
            train_img_data_dir='', 
            val_img_data_dir='data/coco/val2017', 
            test_img_data_dir='data/coco/val2017',
            dataset_type='objects365lmdb',
            train_ann_path="data/Objects365/lmdb",
            val_ann_path="data/coco/annotations/instances_val2017.json"
        )
    ),
    ytvis=dict(
        training_config=dict(
            train_img_data_dir='data/ytvis/train_all_frames/JPEGImages', 
            val_img_data_dir='data/ytvis/train_all_frames/JPEGImages', 
            test_img_data_dir='data/ytvis/train_all_frames/JPEGImages',
            dataset_type='ytvis',
            train_ann_path="data/ytvis/train.json",
            val_ann_path="data/ytvis/subtrain.json"
        ),
        generating_pseudo_label_config=dict(
            train_img_data_dir='data/ytvis/train_all_frames/JPEGImages', 
            val_img_data_dir='data/ytvis/train_all_frames/JPEGImages', 
            test_img_data_dir='data/ytvis/train_all_frames/JPEGImages',
            dataset_type='ytvis',
            train_ann_path="data/ytvis/train.json",
            val_ann_path="data/ytvis/train.json"
        ),
    ),
    astro=dict(
        training_config=dict(
            train_img_data_dir='/', 
            val_img_data_dir='data/coco/val2017', 
            test_img_data_dir='data/coco/test2017',
            dataset_type='coco',
            train_ann_path="/astro_coco_json/ext_arms_rw_only_worot.json",
            val_ann_path="data/coco/annotations/instances_val2017.json"
        ),
        generating_pseudo_label_config=dict(
            train_img_data_dir='/', 
            val_img_data_dir='/', 
            test_img_data_dir='data/coco/test2017',
            dataset_type='coco',
            train_ann_path="/astro_coco_json/ext_arms_rw_only_worot.json",
            val_ann_path="/astro_coco_json/ext_arms_rw_only_worot.json"
        ),
    )
)

class WSISDataModule(pl.LightningDataModule):

    def __init__(self, 
                 num_workers,
                 load_train=False,
                 load_val=False,
                 load_test=False,
                 args=None):
        super().__init__()
        self.args = args
        self.num_workers = num_workers

        if isinstance(args.train_transform, str):
            self.train_transform = data_aug_pipelines[args.train_transform](args)
        else:
            raise NotImplementedError
    
        if isinstance(args.test_transform, str):
            self.test_transform = data_aug_pipelines[args.test_transform](args)
        else:
            raise NotImplementedError

        if load_train:
            transform = self.train_transform
            if self.args.dataset_type == 'voc':
                build_dataset = BoxLabelVOC
            elif self.args.dataset_type in ['coco', 'coco_original', 'astro']:
                build_dataset = BoxLabelCOCO
            elif self.args.dataset_type == 'cocolmdb':
                build_dataset = BoxLabelCOCOLMDB
            elif self.args.dataset_type == 'objects365lmdb':
                build_dataset = BoxLabelObjects365COCOScheduleLMDB
            elif self.args.dataset_type == 'lvis':
                transform = self.train_transform
                build_dataset = BoxLabelLVIS
            elif self.args.dataset_type == 'ytvis':
                build_dataset = BoxLabelYTVIS
            else:
                raise NotImplementedError
            datapath_config = datapath_configs[self.args.dataset_type]
            dataset = build_dataset(datapath_config["training_config"]["train_ann_path"],
                                      datapath_config["training_config"]["train_img_data_dir"],
                                      min_obj_size=self.args.min_obj_size,
                                      max_obj_size=self.args.max_obj_size,
                                      transform=transform, args=args)
            data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
            self._train_data_loader = data_loader
        else:
            self._train_data_loader = None
        
        if load_val:
            if self.args.box_inputs == None:
                transform = self.test_transform
                if self.args.dataset_type == 'voc':
                    build_dataset = InstSegVOC
                elif self.args.dataset_type in ['coco', 'coco_original', 'astro']:
                    build_dataset = InstSegCOCO
                elif self.args.dataset_type == 'cocolmdb':
                    build_dataset = InstSegCOCOLMDB
                elif self.args.dataset_type == 'objects365lmdb':
                    build_dataset = InstSegCOCO
                elif self.args.dataset_type == 'lvis':
                    build_dataset = InstSegLVIS
                elif self.args.dataset_type == 'ytvis':
                    build_dataset = InstSegYTVIS
                else:
                    raise NotImplementedError
                datapath_config = datapath_configs[self.args.dataset_type]
                config_type = 'training_config' if self.args.label_dump_path is None else 'generating_pseudo_label_config'
                dataset = build_dataset(datapath_config[config_type]["val_ann_path"],
                                          datapath_config[config_type]["val_img_data_dir"],
                                          min_obj_size=0, 
                                          max_obj_size=1e9,
                                          load_mask=not self.args.not_eval_mask, transform=transform)
                data_loader = DataLoader(dataset, collate_fn=custom_collate_fn,
                                         batch_size=self.args.batch_size, num_workers=self.num_workers)
                self._val_data_loader = data_loader
            else:
                transform = self.test_transform
                if self.args.dataset_type == 'voc':
                    build_dataset = InstSegVOCwithBoxInput
                elif self.args.dataset_type == 'coco':
                    build_dataset = InstSegCOCOwithBoxInput
                elif self.args.dataset_type == 'cocolmdb':
                    # LMDB not implemented
                    raise NotImplementedError
                elif self.args.dataset_type == 'objects365lmdb':
                    # LMDB not implemented
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                datapath_config = datapath_configs[self.args.dataset_type]
                dataset = build_dataset(datapath_config["training_config"]["val_ann_path"],
                                          datapath_config["training_config"]["val_img_data_dir"],
                                          min_obj_size=0, 
                                          max_obj_size=1e9,
                                          load_mask=not self.args.not_eval_mask, 
                                          transform=transform,
                                          box_inputs=self.args.box_inputs)
                data_loader = DataLoader(dataset, collate_fn=custom_collate_fn, 
                                         batch_size=self.args.batch_size, num_workers=self.num_workers)
                self._val_data_loader = data_loader
                
    def train_dataloader(self):
        return self._train_data_loader

    def val_dataloader(self):
        return self._val_data_loader
        
    
    def test_dataloader(self):
        # TODO:
        raise NotImplementedError
