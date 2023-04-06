[![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](LICENSE)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# Mask Auto-Labeler: The Official Implementation

![teaser](https://user-images.githubusercontent.com/6581457/208230608-3f346724-9cc7-47d9-acf9-ef0582aa897c.gif)



[Vision Transformers are Good Mask Auto-Labelers](https://arxiv.org/abs/2301.03992)

[Shiyi Lan](https://voidrank.github.io/), [Xitong Yang](https://scholar.google.com/citations?user=k0qC-7AAAAAJ&hl=en), [Zhiding Yu](https://chrisding.github.io/), [Zuxuan Wu](https://zxwu.azurewebsites.net/), [Jose M. Alvarez](https://alvarezlopezjosem.github.io/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/)

Accepted by Conference on Computer Vision and Pattern Recognition (**CVPR**) 2023.

## Installation

* Please refer to the dockerfile in the root directory for environment specs. We also provide the docker image [here](https://hub.docker.com/repository/docker/voidrank/mal).

## Training

### Phase 1: Mask Auto-labeling
```
python main.py
```

### Phase 2: Instance Segmentation Models
We copy the training scripts from [mmdet](https://github.com/open-mmlab/mmdetection/tree/5fb38fa4fc4a822ba6ced3b8c2e3dcefa6efacec).

To train a model, e.g. ResNet-50/SOLOv2, with 8 GPUs
```
cd mmdet;
bash tools/dist_train.sh configs/MALMask/solov2_r50_fpn_3x_coco_mal.py 8
```

For more detail, please refer the [documentation](https://mmdetection.readthedocs.io/en/latest/) or [github repo](https://github.com/open-mmlab/mmdetection/tree/5fb38fa4fc4a822ba6ced3b8c2e3dcefa6efacec) of mmdetection.

## Inference and Evaluation

### Phase 1: Generating Mask Psuedo-labels

```
python main.py --resume PATH/TO/WEIGHTS --label_dump_path PATH/TO/PSUEDO_LABELS_OUTPUT --not_eval_mask
```

### Phase 2: Evaluation and Inference of Instance Segmentation Models

To evaluate an instance segmentation model, e.g. ResNet-50/SOLOv2, with 8 GPUs:
```
bash tools/dist_test.sh configs/MALMask/solov2_r50_fpn_3x_coco_mal.py solov2_r50_fpn_3x_coco_essenco/latest.pth 8 --eval segm
```

To generate results of instance segmentation models, e.g. ResNet-50/SOLOv2, with 8 GPUs:
```
bash tools/dist_test.sh configs/MALMask/solov2_r50_fpn_3x_coco_mal.py solov2_r50_fpn_3x_coco_essenco/latest.pth 8 --format-only --options "jsonfile_prefix=work_dirs/solov2_r50_fpn_3x_coco_essenco/test-dev.json"
```

For more detail, please refer the [documentation](https://mmdetection.readthedocs.io/en/latest/) or [github repo](https://github.com/open-mmlab/mmdetection/tree/5fb38fa4fc4a822ba6ced3b8c2e3dcefa6efacec) of mmdetection.



### Phase 1: Mask Auto-labeling
#### Trained Weights

|  ViT-MAE-base (COCO)  |  MAL-ViT-base (LVIS v1.0)  |
|:---------------------:|:---------------------:|
|   [download](https://drive.google.com/file/d/1VEhlZV-McaizuPBQxNj6x-Ip0xemeGdB/view?usp=sharing)  |  [download](https://drive.google.com/file/d/1rScy9rg-2RFEQS_ggtZR1dY9Fy6kAm4K/view?usp=sharing) |

#### Mask Pseudo-labels

| MAL-ViT-base (COCO train2017) | MAL-ViT-base (LVIS v1.0 train) |
|:------------:|:-----------:|
|   [download](https://drive.google.com/file/d/1rF9GfHw9nYDZiVWqv9hSXIfh0PQSYmT-/view?usp=sharing) | [download](https://drive.google.com/file/d/18pr4zT23bJCMsNfDTgfxdZjq8QUBaqp2/view?usp=share_link) |

### Phase 2: Instance Segmentation Models

#### COCO

|   Encoder  |  Decoder  |    weights   |
|:----------:|:--------:|:------------:|
| ResNet-50  |  SOLOv2  | [download](https://drive.google.com/file/d/1dWptOj0se_P4o1V3ve8Bc8VabQTJLJlQ/view?usp=share_link) |
| ResNet-101-DCN |  SOLOv2   | [download](https://drive.google.com/file/d/12mTkFvMVQmt4C-tX1XZNzRtO_iRp_YhB/view?usp=share_link) |
| ResNeXt-101-DCN | SOLOv2 | [download](https://drive.google.com/file/d/1uy-ZL1s28B1v-H2q7_apGAXPHeOWcZ4Q/view?usp=share_link) |
| ConvNeXt-s | Cascade MR-CNN  | [download](https://drive.google.com/file/d/1U0ImyYX_mrKHEllrV8xakyAp7evNd4nt/view?usp=share_link) |
| ConvNeXt-b | Cascade MR-CNN  | [download](https://drive.google.com/file/d/14JBCJV3VFB4WeFCh_tZk_GXMEwkUoKRU/view?usp=sharing) |
| Swin-s     | Mask2Former    | [download](https://drive.google.com/file/d/1Yfuw7i1amO_KQb51g40HsAgFxtyUawcf/view?usp=share_link) |

## F.A.Q.

#### It seems like MIL loss is using mask labels for training?

No, we do not use mask. Check [this](https://github.com/NVlabs/mask-auto-labeler/issues/2)
   
#### I met errors during training/testing and *MMCV exists in the error log*, how do I do?

You have to rebuild your own docker since your nvidia driver version is different from mine and there are some customized operators in MMCV.

## LICENSE

Copyright © 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Click here to view a copy of this license.

The pre-trained models are shared under CC-BY-NC-SA-4.0. If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For business inquiries, please visit our website and submit the form: NVIDIA Research Licensing

## Acknowledgement

This repository is partly based on Pytorch-image-models (timm), MMDetection, and DINO. We leverage PyTorch Lightning.



