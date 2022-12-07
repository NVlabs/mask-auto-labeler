# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import random
from copy import deepcopy

from PIL import ImageFilter, ImageOps, Image
import torch
import numpy as np
from torchvision import transforms
from torch._six import string_classes
import collections

from torch.utils.data._utils.collate import default_collate
from torch.utils.data._utils.collate import (default_collate_err_msg_format, 
                                             np_str_obj_array_pattern)

from .voc import DataWrapper


def custom_crop_image(img, box):
    # I implement this special `crop image` function
    # that aims at getting `no padding` cropped image.
    # Implementation Details:
    # If the target box goes beyond one of the borderlines,
    # the function will crop the content from the opposite
    # side of the image

    # Examples:
    # An image of HxW, if we crop the image using box
    # [W-10, H-10, W+10, H+10] 
    # Top-left corner: (W-10, H-10);
    # Bottom-right corner: (W+10, H+10).

    # Motivation:
    # Since the CRF algorithm uses the original pixels
    # for generating pseudo-labels, each pixels matters a lot here.
    # A fact that synthetic padding pixels (mean color of ImageNet)
    # do sereve damage to the refined image


    # box [x0, y0, x1 y1] [top left x, top left y, bottom right x, bottom right y]

    ret_shape = list(img.shape)
    ret_shape[:2] = box[3] - box[1], box[2] - box[0]
    h, w = img.shape[:2]

    ret_img = np.zeros(ret_shape)

    # top left
    if box[0] < 0 and box[1] < 0:
        ret_img[:-box[1], :-box[0]] = img[box[1]:, box[0]:]

    # middle top
    if (box[0] < w and box[2] > 0) and box[1] < 0:
        ret_img[:-box[1], max(-box[0], 0): min(w, box[2]) - box[0]] = img[box[1]:, max(0, box[0]):min(w, box[2])]

    # top right
    if box[2] > w and box[1] < 0:
        ret_img[:-box[1], -(box[2] - w):] = img[box[1]:, :box[2] - w]

    # middle left
    if box[0] < 0 and (box[1] < h and box[3] > 0):
        ret_img[max(0, -box[1]): min(h, box[3]) - box[1], :-box[0]] = img[max(0, box[1]):min(h, box[3]), box[0]:]

    # middle right
    if box[2] > w and (box[1] < h and box[3] > 0):
        ret_img[max(0, -box[1]): min(h, box[3]) - box[1], -(box[2]-w):] = img[max(0, box[1]):min(h, box[3]), :(box[2]-w)]

    # bottom left
    if box[0] < 0 and box[3] > h:
        ret_img[-(box[3] - h):, :-box[0]] = img[:box[3] - h, box[0]:]

    # middle bottom
    if (box[0] < w and box[2] > 0) and box[3] > h:
        ret_img[-(box[3] - h):, max(-box[0], 0): min(w, box[2]) - box[0]] = img[:box[3] - h, max(0, box[0]):min(w, box[2])]

    # bottom right
    if box[2] > w and box[3] > h:
        ret_img[-(box[3] - h):, -(box[2] - w):] = img[:(box[3] - h), :(box[2] - w)]

    # middle
    ret_img[max(0, -box[1]): min(h, box[3]) - box[1], max(0, -box[0]): min(w, box[2]) - box[0]] = \
        img[max(box[1], 0): min(h, box[3]), max(box[0], 0): min(w, box[2])]

    return ret_img


def custom_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: custom_collate_fn([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: custom_collate_fn([d[key] for d in batch]) for key in elem}
    if isinstance(elem, DataWrapper):
        return batch
    else:
        return default_collate(batch)
        

class RandomCropV2:
    
    def __init__(self, max_size=512, margin_rate=[0.05, 0.15], 
                 mean=(0.485, 0.456, 0.406), random=True,
                 crop_fields=['image', 'mask']):
        self._max_size = max_size
        self._margin_rate = np.array(margin_rate)
        self._mean = np.array(mean) * 255
        self._random = random
        self._crop_fields = crop_fields

    def _expand_box(self, box, margins):
        ctr = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
        box = ctr[0] - (ctr[0] - box[0]) * (1 + margins[0]), \
              ctr[1] - (ctr[1] - box[1]) * (1 + margins[1]), \
              ctr[0] + (box[2] - ctr[0]) * (1 + margins[2]) + 1, \
              ctr[1] + (box[3] - ctr[1]) * (1 + margins[3]) + 1
        return box

    def __call__(self, data):
        # obtain more info
        img = np.array(data['image'])
        box = np.array(data['bbox'])
        h, w = img.shape[0], img.shape[1]

        if self._random:
            margins = np.random.rand(4) * (self._margin_rate[1] - self._margin_rate[0]) + self._margin_rate[0]
            gates = np.random.rand(2) 
            #print(margins, gates)
            gates = np.array([gates[0], gates[1], 1 - gates[0], 1 - gates[1]])
            margins = margins * gates
            extbox = self._expand_box(box, margins)
            extbox = np.array([np.floor(extbox[0]), np.floor(extbox[1]), np.ceil(extbox[2]), np.ceil(extbox[3])]).astype(np.int)
            ext_h, ext_w = extbox[3] - extbox[1], extbox[2] - extbox[0] 
        else:
            margins = np.ones(4) * self._margin_rate[0] * 0.5
            extbox = self._expand_box(box, margins)
            extbox = np.array([np.floor(extbox[0]), np.floor(extbox[1]), np.ceil(extbox[2]), np.ceil(extbox[3])]).astype(np.int)
            ext_h, ext_w = extbox[3] - extbox[1], extbox[2] - extbox[0] 
    
        # extended box size
        data['ext_h'], data['ext_w'] = ext_h, ext_w

        # crop image
        if 'image' in self._crop_fields:
            ret_img = custom_crop_image(img, extbox)
            ret_img = Image.fromarray(ret_img.astype(np.uint8)).resize((self._max_size, self._max_size))
            data['image'] = ret_img

        # crop mask
        if 'mask' in self._crop_fields and 'mask' in data.keys():
            mask = np.array(data['mask'])
            ret_mask = custom_crop_image(mask, extbox)
            ret_mask = Image.fromarray(ret_mask.astype(np.uint8)).resize((self._max_size, self._max_size))
            ret_mask = np.array(ret_mask)
            data['mask'] = ret_mask

        # crop box mask (during test)
        if 'boxmask' in  self._crop_fields:
            boxmask = data['boxmask']
            ret_boxmask = np.zeros((ext_h, ext_w))
            ret_boxmask[max(0-extbox[1], 0):ext_h+min(0, h-extbox[3]), \
                max(0-extbox[0], 0):ext_w+min(0, w-extbox[2])] = \
                boxmask[max(extbox[1], 0):min(extbox[3], h), \
                    max(extbox[0], 0):min(extbox[2], w)]
            ret_boxmask = np.array(Image.fromarray(ret_boxmask.astype(np.uint8)).resize((self._max_size, self._max_size)))
            data['boxmask'] = ret_boxmask

        data['ext_boxes'] = extbox
        data['margins'] = margins

        return data


class RandomCropV3(RandomCropV2):

    def __call__(self, data):
        # obtain more info
        img = np.array(data['image'])
        box = np.array(data['bbox'])
        h, w = img.shape[0], img.shape[1]

        if self._random:
            margins = np.random.rand(4) * (self._margin_rate[1] - self._margin_rate[0]) + self._margin_rate[0]
            gates = np.random.rand(2) 
            gates = np.array([gates[0], gates[1], 1 - gates[0], 1 - gates[1]])
            margins = margins * gates
            extbox = self._expand_box(box, margins)
            extbox = np.array([np.floor(extbox[0]), np.floor(extbox[1]), np.ceil(extbox[2]), np.ceil(extbox[3])]).astype(np.int)
            ext_h, ext_w = extbox[3] - extbox[1], extbox[2] - extbox[0] 
        else:
            margins = np.ones(4) * self._margin_rate[0] * 0.5
            extbox = self._expand_box(box, margins)
            extbox = np.array([np.floor(extbox[0]), np.floor(extbox[1]), np.ceil(extbox[2]), np.ceil(extbox[3])]).astype(np.int)
            ext_h, ext_w = extbox[3] - extbox[1], extbox[2] - extbox[0] 
    
        # extended box size
        data['ext_h'], data['ext_w'] = ext_h, ext_w

        # crop image
        if 'image' in self._crop_fields:
            ret_img = custom_crop_image(img, extbox)
            ret_img = Image.fromarray(ret_img.astype(np.uint8)).resize((self._max_size, self._max_size))
            data['image'] = ret_img

        # crop mask
        if 'mask' in self._crop_fields:
            mask = np.array(data['mask'])
            ret_mask = custom_crop_image(mask, extbox)
            ret_mask = Image.fromarray(ret_mask.astype(np.uint8)).resize((self._max_size, self._max_size))
            ret_mask = np.array(ret_mask)
            data['mask'] = ret_mask

        # crop box mask (during test)
        if 'boxmask' in  self._crop_fields:
            boxmask = data['boxmask']
            ret_boxmask = np.zeros((ext_h, ext_w))
            ret_boxmask[max(0-extbox[1], 0):ext_h+min(0, h-extbox[3]), \
                max(0-extbox[0], 0):ext_w+min(0, w-extbox[2])] = \
                boxmask[max(extbox[1], 0):min(extbox[3], h), \
                    max(extbox[0], 0):min(extbox[2], w)]
            ret_boxmask = np.array(Image.fromarray(ret_boxmask.astype(np.uint8)).resize((self._max_size, self._max_size)))
            data['boxmask'] = ret_boxmask

        data['ext_boxes'] = extbox
        data['margins'] = margins

        return data
        
        


#### Weakly Supervised Instance Segmentation

class RandomCropv1:
    def __init__(self, crop_sizes=[448], crop_fields=['image', 'mask']):
        self._crop_sizes = crop_sizes

    def __call__(self, data):
        img = np.array(data['image'])
        mask = np.array(data['mask'])
        h, w = img.shape[0], img.shape[1]
        ret_images = []
        ret_masks = []
        ret_boxmask = []
        if 'boxmask' in data.keys():
            boxmask = data['boxmask']
        else:
            boxmask = data['mask']
        y_ids, x_ids = np.where(boxmask > 0)
        for crop_size in self._crop_sizes:
            max_y, max_x = min(y_ids.min(), h - crop_size), min(x_ids.min(), w - crop_size)
            min_y, min_x = max(0, y_ids.max() - crop_size + 1), max(0, x_ids.max() - crop_size + 1)
            max_y, max_x = max(max_y, min_y), max(max_x, min_x)
            offx, offy = torch.randint(min_x, max_x + 1, (1,)), torch.randint(min_y, max_y + 1, (1,))
            cropped_image = np.ascontiguousarray(img[offy:offy + crop_size, offx:offx + crop_size])
            cropped_mask  = np.ascontiguousarray(mask[offy:offy + crop_size, offx:offx + crop_size])
            if 'boxmask' in data.keys():
                boxmask = data['boxmask']
                cropped_boxmask = np.ascontiguousarray(boxmask[offy:offy + crop_size, offx:offx + crop_size])
                ret_boxmask.append(cropped_boxmask)
            ret_images.append(Image.fromarray(cropped_image))
            ret_masks.append(cropped_mask)
        if 'boxmask' in data.keys():
            data['boxmask'] = ret_boxmask
        data['image'] = ret_images
        data['mask'] = ret_masks
        return data


class ObjectCentricCropV2:

    def __init__(self, crop_size, mean, margin_rate=None, 
                pad_size=None, crop_fields=['image', 'mask']):
        self._crop_size = crop_size
        self._mean = np.array(mean) * 255
        self._margin_rate = margin_rate
        self._pad_size = pad_size
        self._crop_fields = crop_fields
    
    def extend_box(self, box, margin_rate_h=None, margin_rate_w=None, pad_size=None):
        if margin_rate_h is not None:
            center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            expanded_box = []
            expanded_box.append(int(center[0] - np.ceil((center[0] - box[0]) * (1 + margin_rate_w))))
            expanded_box.append(int(center[1] - np.ceil((center[1] - box[1]) * (1 + margin_rate_h))))
            expanded_box.append(int(center[0] + np.ceil((box[2] - center[0]) * (1 + margin_rate_w))))
            expanded_box.append(int(center[1] + np.ceil((box[3] - center[1]) * (1 + margin_rate_h))))
        elif pad_size is not None:
            center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            expanded_box = []
            expanded_box.append(int(center[0] - (center[0] - box[0] + pad_size // 2)))
            expanded_box.append(int(center[1] - (center[1] - box[1] + pad_size // 2)))
            expanded_box.append(int(center[0] + (box[2] - center[0] + pad_size // 2)))
            expanded_box.append(int(center[1] + (box[3] - center[1] + pad_size // 2)))
        return expanded_box


    def __call__(self, data):
        # Obtain all required info
        box = data['bbox']
        img = np.array(data['image'])
        h, w = img.shape[0], img.shape[1]
        # Compute the target bounding box
        # According to the margin and original boxes
        if isinstance(self._margin_rate, list):
            margin_rate_h = (self._margin_rate[1] - self._margin_rate[0]) * torch.rand((1,)) + self._margin_rate[0]
            margin_rate_w = (self._margin_rate[1] - self._margin_rate[0]) * torch.rand((1,)) + self._margin_rate[0]
        elif isinstance(self._margin_rate, float):
            margin_rate_h, margin_rate_w = self._margin_rate, self._margin_rate
        else:
            raise NotImplementedError
        extended_box = self.extend_box(box, margin_rate_h, margin_rate_w, self._pad_size)
        while (extended_box[3] - extended_box[1]) * (extended_box[2] - extended_box[0]) < 30000 and np.random.random((1,))[0] > 0.3:
            extended_box = self.extend_box(box, margin_rate_h, margin_rate_w, self._pad_size)
        ext_h, ext_w = extended_box[3] - extended_box[1], extended_box[2] - extended_box[0] 

        # crop image
        if 'image' in self._crop_fields:
            ret_img = np.ones((ext_h, ext_w, 3)) * self._mean
            ret_img[max(0-extended_box[1], 0):ext_h+min(0, h-extended_box[3]), \
                    max(0-extended_box[0], 0):ext_w+min(0, w-extended_box[2])] = \
                    img[max(extended_box[1], 0):min(extended_box[3], h), \
                        max(extended_box[0], 0):min(extended_box[2], w)]
            ret_img = Image.fromarray(ret_img.astype(np.uint8)).resize((self._crop_size, self._crop_size))
            data['image'] = ret_img

        # crop mask
        if 'mask' in self._crop_fields:
            mask = np.array(data['mask'])
            ret_mask = np.zeros((ext_h, ext_w))
            ret_mask[max(0-extended_box[1], 0):ext_h+min(0, h-extended_box[3]), \
                    max(0-extended_box[0], 0):ext_w+min(0, w-extended_box[2])] = \
                    mask[max(extended_box[1], 0):min(extended_box[3], h), \
                        max(extended_box[0], 0):min(extended_box[2], w)]
            ret_mask = Image.fromarray(ret_mask.astype(np.uint8)).resize((self._crop_size, self._crop_size))
            ret_mask = np.array(ret_mask)
            data['mask'] = ret_mask

        # crop box mask (during test)
        if 'boxmask' in self._crop_fields:
            boxmask = data['boxmask']
            ret_boxmask = np.zeros((ext_h, ext_w))
            ret_boxmask[max(0-extended_box[1], 0):ext_h+min(0, h-extended_box[3]), \
                max(0-extended_box[0], 0):ext_w+min(0, w-extended_box[2])] = \
                boxmask[max(extended_box[1], 0):min(extended_box[3], h), \
                    max(extended_box[0], 0):min(extended_box[2], w)]
            ret_boxmask = np.array(Image.fromarray(ret_boxmask.astype(np.uint8)).resize((self._crop_size, self._crop_size)))
            data['boxmask'] = ret_boxmask

        data['ext_boxes'] = extended_box
        data['margin_rate_h'] = margin_rate_h
        data['margin_rate_w'] = margin_rate_w
        return data
        

class RandomFlipV2:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if float(torch.rand(1)) > self.p:
            x['flip_records'] = 1
            x['image'] = ImageOps.mirror(x['image'])
            x['mask'] = x['mask'][:,::-1]
        else:
            x['flip_records'] = 0

        return x


class RandomFlip:

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x):
        if 'aug_images' in x.keys():
            x['flip_records'] = []
            for idx in range(len(x['aug_images'])):
                x['flip_records'].append([])
                for jdx in range(len(x['aug_images'][idx])):
                    if float(torch.rand(1)) > self.p:
                        x['aug_images'][idx][jdx] = ImageOps.mirror(x['aug_images'][idx][jdx])
                        x['flip_records'][idx].append(1)
                    else:
                        x['flip_records'][idx].append(0)
        elif 'image' in x.keys():
            if float(torch.rand(1)) > self.p:
                x['flip_records'] = 1
                x['image'] = ImageOps.mirror(x['image'])
                x['mask'] = x['mask'][:,::-1]
            else:
                x['flip_records'] = 0
        else:
            raise NotImplementedError

        return x


class Normalize(transforms.Normalize):

    def forward(self, data):
        if 'image' in data.keys():
            data['image'] = super().forward(data['image'])
            if 'timage' in data.keys():
                data['timage'] = super().forward(data['timage'])
        elif 'aug_images' in data.keys():
            for idx in range(len(data['aug_images'])):
                for jdx in range(len(data['aug_images'][idx])):
                    data['aug_images'][idx][jdx] = super().forward(data['aug_images'][idx][jdx])
        else:
            raise NotImplementedError

        return data


class Denormalize:

    def __init__(self, mean, std, inplace=False):
        self._mean = mean
        self._std = std
        self._inplace = inplace

    def __call__(self, img):
        img = (img * self._std + self._mean) * 255
        return img


class ToTensor(transforms.ToTensor):


    def __call__(self, data):
        if 'image' in data.keys():
            if isinstance(data['image'], list) or isinstance(data['image'], tuple):
                img_list = []
                for img in data['image']:
                    img_list.append(super().__call__(img))
                data['image'] = torch.cat(img_list)
            else:
                data['image'] = super().__call__(data['image'])
            if 'flip_records' in data.keys():
                data['flip_records'] = torch.tensor([data['flip_records']])
        elif 'aug_images' in data.keys():
            for idx in range(len(data['aug_images'])):
                for jdx in range(len(data['aug_images'][idx])):
                    data['aug_images'][idx][jdx] = super().__call__(data['aug_images'][idx][jdx])
                    data['aug_ranges'][idx][jdx] = torch.tensor(data['aug_ranges'][idx][jdx])
                if 'flip_records' in data.keys():
                    data['flip_records'][idx] = torch.tensor(data['flip_records'][idx])
        else:
            raise NotImplementedError
        
        if 'timage' in data.keys():
            if isinstance(data['timage'], list) or isinstance(data['timage'], tuple):
                img_list = []
                for img in data['timage']:
                    img_list.append(super().__call__(img))
                data['timage'] = torch.cat(img_list)
            else:
                data['timage'] = super().__call__(data['timage'])
        
        if 'mask' in data.keys():
            if isinstance(data['mask'], list) or isinstance(data['mask'], tuple):
                mask_list = []
                for mask in data['mask']:
                    mask_list.append(torch.tensor(mask, dtype=torch.float)[None,...])
                data['mask'] = torch.cat(mask_list)
            else:
                data['mask'] = torch.tensor(data['mask'], dtype=torch.float)[None,...]
        
        if 'boxmask' in data.keys():
            if isinstance(data['boxmask'], list) or isinstance(data['boxmask'], tuple):
                mask_list = []
                for mask in data['boxmask']:
                    mask_list.append(torch.tensor(mask, dtype=torch.float)[None,...])
                data['boxmask'] = torch.cat(mask_list)
            else:
                data['boxmask'] = torch.tensor(data['boxmask'], dtype=torch.float)[None,...]

        if 'ann' in data.keys():
            data['ann'] = torch.tensor(data['ann'])
        
        return data
            

class ColorJitter(transforms.ColorJitter):

    def single_forward(self, img):
        if isinstance(img, list):
            return [self.single_forward(_img) for _img in img]
        else:
            return super().forward(img)

    def forward(self, data):
        if 'image' in data.keys():
            data['image'] = self.single_forward(data['image'])
        elif 'aug_images' in data.keys():
            for idx in range(len(data['aug_images'])):
                for jdx in range(len(data['aug_images'][idx])):
                    data['aug_images'][idx][jdx] = super().forward(data['aug_images'][idx][jdx])
        else:
            raise NotImplementedError
        return data


class RandomGrayscale(transforms.RandomGrayscale):

    def single_forward(self, img):
        if isinstance(img, list):
            return [self.single_forward(_img) for _img in img]
        else:
            return super().forward(img)

    def forward(self, data):
        if 'image' in data.keys():
            data['image'] = self.single_forward(data['image'])
        elif 'aug_images' in data.keys():
            for idx in range(len(data['aug_images'])):
                for jdx in range(len(data['aug_images'][idx])):
                    data['aug_images'][idx][jdx] = super().forward(data['aug_images'][idx][jdx])
        else:
            raise NotImplementedError
        return data


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def single_forward(self, img):
        if isinstance(img, list):
            return [self.single_forward(img_) for img_ in img]
        else:
            do_it = random.random() <= self.prob
            if not do_it:
                return img
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        
    def __call__(self, data):
        if 'image' in data.keys():
            data['image'] = self.single_forward(data['image'])
        elif 'aug_images' in data.keys():
            for idx in range(len(data['aug_images'])):
                for jdx in range(len(data['aug_images'][idx])):
                    data['aug_images'][idx][jdx] = self.single_forward(data['aug_images'][idx][jdx])
        else:
            raise NotImplementedError
        return data


class DropAllExcept:

    def __init__(self, keep_keys):
        self.keep_keys = keep_keys 

    def __call__(self, data):
        data_keys = list(data.keys())
        for key in data_keys:
            if key not in self.keep_keys:
                del data[key]
        return data


class ChangeNames:

    def __init__(self, kv_dic):
        self.kv_dic = kv_dic

    def __call__(self, data):
        data_keys = list(data.keys())
        for key, value in self.kv_dic.items():
            if key in data_keys:
                data[value] = data[key]
                del data[key]
        return data


class Solarization:
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def single_forward(self, img):
        if isinstance(img, list):
            return [self.single_forward(img_) for img_ in img]
        else:
            if random.random() < self.p:
                return ImageOps.solarize(img)
            else:
                return img
    
    def __call__(self, data):
        if 'image' in data.keys():
            data['image'] = self.single_forward(data['image'])
        elif 'aug_images' in data.keys():
            for idx in range(len(data['aug_images'])):
                for jdx in range(len(data['aug_images'][idx])):
                    data['aug_images'][idx][jdx] = self.single_forward(data['aug_images'][idx][jdx])
        else:
            raise NotImplementedError
        return data


class ImageSizeAlignment:

    def __init__(self, max_size, mean, random_offset=False):
        self._max_size = max_size
        self._mean = (np.array(mean) * 255).astype(np.uint8)
        self._random_offset = random_offset
    
    def __call__(self, data):
        assert 'image' in data.keys()
        padded_image = np.ones((self._max_size, self._max_size, 3), dtype=np.uint8) * self._mean
        image = np.array(data['image'])
        h, w = image.shape[0], image.shape[1]
        if self._random_offset:
            offy, offx = torch.randint(0, self._max_size - h + 1, (1,)), torch.randint(0, self._max_size - w + 1, (1,))
        else:
            offy, offx = 0, 0
        padded_image[offy:offy+h, offx:offx+w] = image
        data['image'] = Image.fromarray(padded_image)
        if 'mask' in data.keys():
            padded_mask = np.ones((self._max_size, self._max_size))
            padded_mask[offy:offy+h, offx:offx+w] = np.array(data['mask'])
            data['mask'] = Image.fromarray(padded_mask)
        return data
        

class RandomScale:

    def __init__(self, min_size, max_size, mean=(0.485, 0.456, 0.406)):
        self._min_size = min_size
        self._max_size = max_size
        self._mean = mean

    def __call__(self, data):
        if 'image' in data.keys():
            for i in range(len(data['image'])):
                img = np.array(data['image'][i])
                w, h = img.shape[:2]
                mask = data['mask'][i]
                rw, rh = torch.randint(self._min_size, self._max_size + 1, (2, ))
                offw, offh = torch.randint(0, w - rw + 1, (1, )), torch.randint(0, h - rh + 1, (1, ))

                ret_img = (np.ones(img.shape) * np.array(self._mean)).astype(img.dtype) 
                ret_mask = np.zeros(mask.shape, img.dtype)

                img = np.array(Image.fromarray(img).resize((rw, rh)))
                mask = np.array(Image.fromarray(mask).resize((rw, rh)))

                ret_img[offh: offh + rh, offw: offw + rw] = img
                ret_mask[offh: offh + rh, offw: offw + rw] = mask
                data['image'][i] = Image.fromarray(ret_img)
                data['mask'][i] = ret_mask
        else:
            raise NotImplementedError
        
        return data

class SplitAndMerge:

    def __init__(self, branch1, branch2):
        self.branch1 = branch1
        self.branch2 = branch2

    def __call__(self, data):

        data_clone = deepcopy(data)
        data1 = self.branch1(data_clone)

        data_clone = deepcopy(data)
        data2 = self.branch2(data_clone)

        data1.update(data2)
        return data1



data_aug_pipelines = {
    'test': lambda args: transforms.Compose([
        RandomCropV2(args.crop_size, 
                     margin_rate=args.test_margin_rate,
                     random=False,
                     crop_fields=['image', 'boxmask', 'mask']),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'train': lambda args: transforms.Compose([
        RandomCropV3(args.crop_size, margin_rate=args.margin_rate),
        RandomFlip(0.5),
        SplitAndMerge(
            transforms.Compose([
                transforms.RandomApply(
                    [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.5
                ),
                RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [GaussianBlur(1.0)],
                    p=0.5
                )
            ]), 
            transforms.Compose([
                DropAllExcept(['image']),
                ChangeNames({'image': 'timage'})
            ])
        ),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
}