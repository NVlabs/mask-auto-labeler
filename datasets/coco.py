# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import torch
from torch.utils.data import Dataset
from .voc import (InstSegVOC, BoxLabelVOC, BoxLabelVOCLMDB, InstSegVOCLMDB, 
                  InstSegVOCwithBoxInput)

import numpy as np

things = ["person", "bicycle", "car", "motorcycle", "airplane", "airplane",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
          "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
          "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "potted plant", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "toaster", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]
semistuff = ["bus", "train", "truck", "bench", "couch", "bed", "dining table", "toilet",
          "oven", "sink", "refrigerator"]

categories = [{'supercategory': 'person', 'id': 1, 'name': 'person'},
 {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
 {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
 {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
 {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
 {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
 {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
 {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
 {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
 {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
 {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
 {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
 {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
 {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
 {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
 {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
 {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
 {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
 {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
 {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
 {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
 {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
 {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
 {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
 {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
 {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
 {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
 {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
 {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
 {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
 {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
 {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
 {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
 {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
 {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
 {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
 {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
 {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
 {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
 {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
 {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
 {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
 {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
 {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
 {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
 {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
 {'supercategory': 'food', 'id': 52, 'name': 'banana'},
 {'supercategory': 'food', 'id': 53, 'name': 'apple'},
 {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
 {'supercategory': 'food', 'id': 55, 'name': 'orange'},
 {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
 {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
 {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
 {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
 {'supercategory': 'food', 'id': 60, 'name': 'donut'},
 {'supercategory': 'food', 'id': 61, 'name': 'cake'},
 {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
 {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
 {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
 {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
 {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
 {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
 {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
 {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
 {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
 {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
 {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
 {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
 {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
 {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
 {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
 {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
 {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
 {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
 {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
 {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
 {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
 {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
 {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
 {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]

cat_mapping = dict([ (cat['id'], idx+1) for idx, cat in enumerate(categories)])

training_config = {
    'train_img_data_dir': 'data/coco/train2017', 
    'val_img_data_dir': 'data/coco/val2017', 
    'test_img_data_dir': 'data/coco/test2017',
    'dataset_type': 'coco',
    'train_ann_path': "/saccadenet/Saccadenet/data/coco/annotations/boxes_train2017.json",
    'val_ann_path': "data/coco/annotations/instances_val2017.json"
}

generating_pseudo_label_config = {
    'train_img_data_dir': 'data/coco/train2017', 
    'train_ann_path': "/saccadenet/Saccadenet/data/coco/annotations/boxes_train2017.json",
    'val_img_data_dir': 'data/coco/train2017', 
    'dataset_type': 'coco',
    'val_ann_path': "/saccadenet/Saccadenet/data/coco/annotations/boxes_train2017.json",
}

class BoxLabelCOCO(BoxLabelVOC):

    def get_category_mapping(self):
        self.cat_mapping = dict([ (cat['id'], idx+1) for idx, cat in enumerate(categories)])


class InstSegCOCO(InstSegVOC):

    def get_category_mapping(self):
        self.cat_mapping = dict([ (cat['id'], idx+1) for idx, cat in enumerate(categories)])


class BoxLabelCOCOLMDB(BoxLabelVOCLMDB):

    def get_category_mapping(self):
        self.cat_mapping = dict([ (cat['id'], idx+1) for idx, cat in enumerate(categories)])


class InstSegCOCOLMDB(InstSegVOCLMDB):

    def get_category_mapping(self):
        self.cat_mapping = dict([ (cat['id'], idx+1) for idx, cat in enumerate(categories)])


class InstSegCOCOwithBoxInput(InstSegVOCwithBoxInput):

    def get_category_mapping(self):
        self.cat_mapping = dict([ (cat['id'], idx+1) for idx, cat in enumerate(categories)])
