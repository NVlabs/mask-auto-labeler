# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


from .voc import BoxLabelVOC, InstSegVOC, InstSegVOCwithBoxInput, BoxLabelVOCLMDB, DataWrapper
import json
from PIL import Image
import os
import numpy as np
import torch
import pycocotools.mask as maskUtils
from torch.utils.data import Dataset


categories = [{'supercategory': 'object', 'id': 1, 'name': 'person'},
 {'supercategory': 'object', 'id': 2, 'name': 'giant_panda'},
 {'supercategory': 'object', 'id': 3, 'name': 'lizard'},
 {'supercategory': 'object', 'id': 4, 'name': 'parrot'},
 {'supercategory': 'object', 'id': 5, 'name': 'skateboard'},
 {'supercategory': 'object', 'id': 6, 'name': 'sedan'},
 {'supercategory': 'object', 'id': 7, 'name': 'ape'},
 {'supercategory': 'object', 'id': 8, 'name': 'dog'},
 {'supercategory': 'object', 'id': 9, 'name': 'snake'},
 {'supercategory': 'object', 'id': 10, 'name': 'monkey'},
 {'supercategory': 'object', 'id': 11, 'name': 'hand'},
 {'supercategory': 'object', 'id': 12, 'name': 'rabbit'},
 {'supercategory': 'object', 'id': 13, 'name': 'duck'},
 {'supercategory': 'object', 'id': 14, 'name': 'cat'},
 {'supercategory': 'object', 'id': 15, 'name': 'cow'},
 {'supercategory': 'object', 'id': 16, 'name': 'fish'},
 {'supercategory': 'object', 'id': 17, 'name': 'train'},
 {'supercategory': 'object', 'id': 18, 'name': 'horse'},
 {'supercategory': 'object', 'id': 19, 'name': 'turtle'},
 {'supercategory': 'object', 'id': 20, 'name': 'bear'},
 {'supercategory': 'object', 'id': 21, 'name': 'motorbike'},
 {'supercategory': 'object', 'id': 22, 'name': 'giraffe'},
 {'supercategory': 'object', 'id': 23, 'name': 'leopard'},
 {'supercategory': 'object', 'id': 24, 'name': 'fox'},
 {'supercategory': 'object', 'id': 25, 'name': 'deer'},
 {'supercategory': 'object', 'id': 26, 'name': 'owl'},
 {'supercategory': 'object', 'id': 27, 'name': 'surfboard'},
 {'supercategory': 'object', 'id': 28, 'name': 'airplane'},
 {'supercategory': 'object', 'id': 29, 'name': 'truck'},
 {'supercategory': 'object', 'id': 30, 'name': 'zebra'},
 {'supercategory': 'object', 'id': 31, 'name': 'tiger'},
 {'supercategory': 'object', 'id': 32, 'name': 'elephant'},
 {'supercategory': 'object', 'id': 33, 'name': 'snowboard'},
 {'supercategory': 'object', 'id': 34, 'name': 'boat'},
 {'supercategory': 'object', 'id': 35, 'name': 'shark'},
 {'supercategory': 'object', 'id': 36, 'name': 'mouse'},
 {'supercategory': 'object', 'id': 37, 'name': 'frog'},
 {'supercategory': 'object', 'id': 38, 'name': 'eagle'},
 {'supercategory': 'object', 'id': 39, 'name': 'earless_seal'},
 {'supercategory': 'object', 'id': 40, 'name': 'tennis_racket'}]

CLASS_NAMES = [obj['name'] for obj in categories]


class BoxLabelYTVIS(Dataset):

    def __init__(self, ann_path, img_data_dir, min_obj_size=0, max_obj_size=1e10, transform=None, args=None):
        self.args = args
        self.ann_path = ann_path
        self.img_data_dir = img_data_dir
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.transform = transform
        self._plain_anns = []
        with open(ann_path, "r") as f:
            self.dataset = json.load(f)
        self._filter_imgs()
        self.get_category_mapping()
        # TODO: classes
        self.cat_mapping = dict([i,i] for i in range(1, 21))
        for inst_idx, inst_ann in enumerate(self.dataset['annotations']):
            for frame_idx, frame_ann in enumerate(inst_ann['bboxes']):
                if inst_ann['bboxes'][frame_idx] is not None:
                    self._plain_anns.append((inst_idx, frame_idx))

        self.id2video = {}
        for video in self.dataset['videos']:
            self.id2video[video['id']] = video
    
        self.get_category_mapping()

    def get_category_mapping(self):
        self.cat_mapping = dict([(cat['id'], cat['id']) for cat in categories])

    def _filter_imgs(self):
        anns = self.dataset['annotations']
        filtered_anns = []
        for inst_ann in anns:
            for frame_idx in range(len(inst_ann['bboxes'])):
                bbox = inst_ann['bboxes'][frame_idx]
                if bbox is None:
                    continue
                if bbox[2] * bbox[3] < self.min_obj_size or bbox[2] * bbox[3] > self.max_obj_size:
                    inst_ann['bboxes'][frame_idx] = None
                    inst_ann['segmentations'][frame_idx] = None

    def __len__(self):
        return len(self._plain_anns)
    
    def __getitem__(self, idx):
        inst_idx, frame_idx = self._plain_anns[idx]
        anns = self.dataset['annotations'][inst_idx]
        video_info = self.id2video[anns['video_id']]
        h, w, file_name = video_info['height'], video_info['width'], video_info['file_names'][frame_idx]

        img = self.get_image(file_name)

        # box mask
        mask = np.zeros((h, w))
        bbox = anns['bboxes'][frame_idx]
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        mask[y0:y1+1, x0:x1+1] = 1

        data = {'image': img, 'mask': mask, 'height': h, 'width': w, 
                'category_id': anns['category_id'], 'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
                'compact_category_id': self.cat_mapping[int(anns['category_id'])],
                'id': anns['id']}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_image(self, file_name):
        try:
            image = Image.open(os.path.join(self.img_data_dir, file_name)).convert('RGB')
        except FileNotFoundError:
            return None
        return image


class BoxLabelYTVISLMDB(BoxLabelVOCLMDB):

    #TODO:
    # This class is not implemented

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filter_imgs()
        self.get_category_mapping()

    def get_category_mapping(self):
        self.cat_mapping = dict([i,i] for i in range(1, 21))

    def _filter_imgs(self):
        anns = self.dataset['annotations']
        filtered_anns = []
        for inst_ann in anns:
            for frame_idx, frame_ann in inst_ann:
                bbox = inst_ann['bboxes'][frame_idx]
                if bbox[2] * bbox[3] > self.min_obj_size and bbox[2] * bbox[3] < self.max_obj_size and\
                    bbox[2] > 2 and bbox[3] > 2:
                    inst_ann['bboxes'][frame_idx] = None
                    inst_ann['segmentations'][frame_idx] = None


    def __len__(self):
        return self.__length__


    def __getitem__(self, _):
        idx = torch.randint(0, self.__length__, (1,))[0]
        with self.ann_env.begin(write=False) as txn:
            ann = json.loads(txn.get(str(self.mapping_ids[idx]).encode()))
        with self.img_env.begin(write=False) as txn:
            raw_img = txn.get(str(ann['image_id']).encode())
            if raw_img is None:
                return self[torch.randint(0, self.__length__, (1,))[0]]
            img = cv2.imdecode(
                np.fromstring(
                    pickle.loads(raw_img), np.uint8
                ), cv2.IMREAD_COLOR)
            img_info = pickle.loads(txn.get(str(ann['image_id']).encode()))

        h, w = img.shape[:2]

        # box mask
        mask = np.zeros((h, w))
        bbox = self.dataset['annotations'][inst_idx]['bboxes'][frame_idx]
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        mask[y0:y1+1, x0:x1+1] = 1

        data = {'image': img, 'mask': mask, 'height': h, 'width': w, 
                'category_id': anns['category_id'], 'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]),
                'compact_category_id': self.cat_mapping[int(anns['category_id'])],
                'id': anns['id']}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_image(self, file_name):
        try:
            image = Image.open(os.path.join(self.img_data_dir, file_name)).convert('RGB')
        except FileNotFoundError:
            return None
        return image




class InstSegYTVIS(BoxLabelYTVIS):

    def __init__(self, *args, load_mask=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.load_mask = load_mask

    def __getitem__(self, idx):
        inst_idx, frame_idx = self._plain_anns[idx]
        anns = self.dataset['annotations'][inst_idx]
        video_info = self.id2video[anns['video_id']]
        h, w, file_name = video_info['height'], video_info['width'], video_info['file_names'][frame_idx]
        img = self.get_image(file_name)
        if img is None:
            return self[torch.randint(0, len(self), (1,))[0]]

        # box mask
        boxmask = np.zeros((h, w))
        bbox = anns['bboxes'][frame_idx]
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        boxmask[y0:y1+1, x0:x1+1] = 1


        data = {'image': img, 'boxmask': boxmask, 'height': h, 'width': w, 
                'category_id': anns['category_id'], 'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
                'compact_category_id': self.cat_mapping[int(anns['category_id'])],
                'id': anns['id'], 'video_id': anns['video_id'], 'ytvis_idx': np.array([inst_idx, frame_idx])}

        if self.load_mask:
            mask = np.ascontiguousarray(maskUtils.decode(maskUtils.frPyObjects(anns['segmentations'][frame_idx], h, w)))
            if len(mask.shape) > 2:
                mask = mask.transpose((2, 0, 1)).sum(0) > 0
            mask = mask.astype(np.uint8)

            data['gtmask'] = DataWrapper(mask)
            data['mask'] = mask

        if self.transform is not None:
            data = self.transform(data)

        return data



class InstSegVOCwithBoxInput(InstSegVOC):
    # TODO:
    # This class is not implemented
    pass