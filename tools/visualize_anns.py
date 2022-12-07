# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


from pycocotools.coco import COCO
import argparse
import json
import re
import pickle
import parser
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("annfile", type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument("--root_dir", type=str, default='data/coco')
    parser.add_argument('--split', type=str, default='train2017')
    parser.add_argument('--filter_list_file', type=str, default=None)
    parser.add_argument('--verbose_mode', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    coco = COCO(args.annfile)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.verbose_mode:
        raw_folder = os.path.join(args.output_dir, "raw")
        segmonly_folder = os.path.join(args.output_dir, "segm_only")
        img_and_segm_folder = os.path.join(args.output_dir, "img_and_segm")
        if not os.path.exists(raw_folder):
            os.mkdir(raw_folder)
        if not os.path.exists(segmonly_folder):
            os.mkdir(segmonly_folder)
        if not os.path.exists(img_and_segm_folder):
            os.mkdir(img_and_segm_folder)

    if args.filter_list_file is not None:
        with open(args.filter_list_file, "rb") as f:
            filter_list = pickle.load(f)
    else:
        filter_list = None

    for img_idx in coco.getImgIds():
        img_ann = coco.loadImgs(img_idx)[0]
        if filter_list is not None and os.path.basename(img_ann['file_name']) not in filter_list:
            continue
        if 'file_name' in img_ann.keys():
            # coco
            img = io.imread(os.path.join(args.root_dir, args.split, img_ann['file_name']))
            filename = img_ann['file_name']
        else:
            # lvis
            coco_url = img_ann['coco_url']
            idx = coco_url.rfind('/')
            idx = coco_url[:idx].rfind('/')
            coco_url = coco_url[idx+1:]
            filename = coco_url
            img = io.imread(os.path.join(args.root_dir, coco_url))
            
        annIds = coco.getAnnIds(imgIds=img_idx)
        anns = coco.loadAnns(annIds)
        if args.verbose_mode:
            # raw image
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(args.output_dir, "raw", os.path.basename(filename)), dpi=300)
            plt.clf()
            # segm
            plt.axis('off')
            plt.imshow(np.zeros(img.shape))
            coco.showAnns(anns)
            plt.savefig(os.path.join(args.output_dir, "segm_only", os.path.basename(filename)), dpi=300)
            plt.clf()
            # raw+segm
            plt.axis('off')
            plt.imshow(img)
            coco.showAnns(anns)
            plt.savefig(os.path.join(args.output_dir, 'img_and_segm', os.path.basename(filename)), dpi=300)
            plt.clf()
        else:
            plt.imshow(img)
            coco.showAnns(anns)
            plt.savefig(os.path.join(args.output_dir, os.path.basename(filename)), dpi=300)
            plt.clf()





    
