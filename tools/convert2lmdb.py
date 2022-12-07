# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import os
import argparse
import json
import lmdb
from PIL import Image
import pickle
import pyarrow as pa


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_list", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--map_size_exp", type=int, default=10)
    parser.add_argument("--img_start_from", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ann_list = args.ann_list.split(",")
    args.max_size = int(float('1e{}'.format(args.map_size_exp)))
    ann_list = []
    for ann in args.ann_list:
        with open(ann, "r") as f:
            ann_list.append(json.load(f))

    ann_env = lmdb.open(os.path.join(args.output_path, "lmdb_ann"), map_size=args.max_size, 
                        readonly=False, meminit=False, map_async=True)
    image_env = lmdb.open(os.path.join(args.output_path, "lmdb_image"), map_size=args.max_size,
                        readonly=False, meminit=False, map_async=True)
    

    # per dataset
    for ann_d in ann_list:
        # per image
        cnt, total_cnt = args.img_start_from, len(ann_d['images'])

        txn = image_env.begin(write=True)
        for image_ann in ann_d['images'][cnt:]:
            #img = cv2.imread(os.path.join(args.img_dir, image_ann['file_name']))
            try:
                with open(os.path.join(args.img_dir, image_ann['file_name']), "rb") as f:
                    txn.put(str(image_ann['id']).encode(), pickle.dumps(f.read(), protocol=pickle.HIGHEST_PROTOCOL))
            except IOError as e:
                continue
            cnt += 1
            if cnt % 100 == 1:
                print("{}/{} finished.".format(cnt, total_cnt))
                txn.commit()
                txn = image_env.begin(write=True)
        image_env.sync()
        image_env.close()

        # per object
        cnt, total_cnt = 0, len(ann_d['annotations'])
        with ann_env.begin(write=True, buffers=True) as txn:
            for ann in ann_d['annotations']:
                txn.put(str(cnt).encode(), json.dumps(ann).encode())
                cnt += 1
                if cnt % 1000 == 1:
                    print("{}/{} finished.".format(cnt, total_cnt))