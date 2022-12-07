# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import os, glob

if __name__ == '__main__':
    os.system('mkdir /dev/shm/coco')
    os.system('cp /coco/*.zip /dev/shm/coco')
    files = glob.glob('/dev/shm/coco/*.zip')
    for filename in files:
        os.system('unzip {} -d /dev/shm/coco'.format(filename))