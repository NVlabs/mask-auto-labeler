# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by: Shiyi Lan


import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from models import vision_transformer
from utils import dino_utils as utils

import importlib.util

from datasets.coco import InstSegCOCO
from datasets.voc import InstSegVOC
from datasets.data_aug import *


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def visualize_one_image(img, args, image_idx=None):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    if image_idx is not None:
        attentions = model.get_selfattention(img.to(device), args.attention_layer_idx)
    else:
        attentions = model.get_selfattention(img.to(device), -1)

    bs = attentions.shape[0] # batch size
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)

    for layer_idx in range(len(attentions)):
        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions[layer_idx])
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(-1, nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions, scale_factor=args.patch_size, mode="nearest").cpu().numpy()

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img{}.png".format(image_idx)))
    if image_idx is None:
        for j in range(nh):
            fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[0, j], format='png')
            print(f"{fname} saved.")
    else:
        avg_attentions = attentions.sum(1)
        for layer_idx in range(len(avg_attentions)):
            folder_path = os.path.join(args.output_dir, str(image_idx))
            if not os.path.exists(folder_path):
                os.system("mkdir -p {}".format(folder_path))
            fname = os.path.join(args.output_dir, str(image_idx), str(layer_idx) + '.png')
            plt.imsave(fname=fname, arr=avg_attentions[layer_idx], format='png')

        print(f"{image_idx}-th image saved")

    if args.threshold is not None:
        if not os.path.exists(os.path.join(args.output_dir, "imgs")):
            os.mkdir(os.path.join(args.output_dir, "imgs"))
        image = skimage.io.imread(os.path.join(args.output_dir, "imgs", "{}.png".format(image_idx)))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--method', default='dino', type=str)
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit-mae-base/16', 'vit-mae-large/16', 'vit-mae-large/16'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./att_vis', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--vit_dpr', type=float, default=0)
    parser.add_argument('--vit_use_conv', default=False, action='store_true')
    parser.add_argument('--freeze_vit_pos_embed', default=False, action='store_true')
    parser.add_argument('--frozen_stages', default="-1", type=str)
    parser.add_argument('--use_vit_decoder', default=False, action='store_true')
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--val_ann_path', default="/mnt/data/voc/voc_2012_val_cocostyle.json", type=str)
    parser.add_argument("--config", type=str, help="config file", default='configs/coco.py')
    parser.add_argument('--test_transform', default='padding_test2', type=str)
    parser.add_argument('--vis_cnt', type=int, default=50)
    parser.add_argument('--attention_layer_idx', type=int, default=-1)
    args = parser.parse_args()

    # load config
    spec = importlib.util.spec_from_file_location("module.name", args.config)
    config_name = args.config[args.config.rfind("/")+1:args.config.rfind(".")]
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    for k, v in config.config.items():
        args.__dict__[k] = v

    args.frozen_stages = list(map(int, args.frozen_stages.split(",")))
    if len(args.frozen_stages) == 1:
        args.frozen_stages = [0, args.frozen_stages[0]]
    assert len(args.frozen_stages) == 2


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    if args.method == 'dino':
        args.arch = args.arch + '/16'
    print(args.arch)

    model = vision_transformer.get_vit(backbone_type=args.arch, pretrained_weight_type=None, frozen_stages=args.frozen_stages, args=args)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        state_dict = {k.replace("student.", ""): v for k, v in state_dict.items()}
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.method == 'dino':
            if args.arch == "vit_small" and args.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif args.arch == "vit_small" and args.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif args.arch == "vit_base" and args.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif args.arch == "vit_base" and args.patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                model.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")
        elif method == 'mae':
            if args.arch == "vit_small" and args.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif args.arch == "vit_small" and args.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif args.arch == "vit_base" and args.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif args.arch == "vit_base" and args.patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                model.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")
        else:
            raise NotImplementedError

    # open image
    # TODO:
    if args.dataset is None:
        if args.image_path is None:
            # user has not specified any image - we use our own image
            print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
            print("Since no image path have been provided, we take the first image in our paper.")
            response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
        elif os.path.isfile(args.image_path):
            with open(args.image_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            print(f"Provided image path {args.image_path} is non valid.")
            sys.exit(1)
        transform = pth_transforms.Compose([
            pth_transforms.Resize(args.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = transform(img)
        visualize_one_image(img, args)
    elif args.dataset in ['cocoval', 'cocotrain', 'voc']:
        # TODO dataset inference
        transform = transforms.Compose([
            RandomCropV2(args.image_size[0], 
                        margin_rate=[0.4, 0.4],
                        random=False,
                        crop_fields=['image', 'boxmask', 'mask']),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        if args.dataset == 'cocoval':
                dataset = InstSegCOCO(args.val_ann_path, args.val_img_data_dir, 
                                min_obj_size=0, 
                                max_obj_size=1e9,
                                load_mask=False, transform=transform)
        else:
            raise NotImplementedError
        
        for i in range(min(len(dataset), args.vis_cnt)):
            data = dataset[i]
            img = data['image']
            # TODO: img should be tensor
            visualize_one_image(img, args, i)

    else:
        raise NotImplementedError
