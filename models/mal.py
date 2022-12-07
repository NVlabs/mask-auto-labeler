# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import time
import itertools
import json
import os
import math

import cv2
import numpy as np

import torch
from torch import nn, optim

from PIL import Image

from pycocotools.coco import COCO
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval

from mmcv.cnn import ConvModule

import torchmetrics
import pytorch_lightning as pl

from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.beta import Beta as BetaDist
from torch.distributions.kl import kl_divergence
from torch.distributed import all_reduce, ReduceOp
import torch.distributed as dist
from torchvision import transforms

from . import vision_transformers
from datasets.data_aug import Denormalize
from datasets.pl_data_module import datapath_configs, num_class_dict
from utils.optimizers.adamw import AdamWwStep



class MeanField(nn.Module):

    def __init__(self, args=None):
        super(MeanField, self).__init__()
        self.kernel_size = args.crf_kernel_size
        assert self.kernel_size % 2 == 1
        self.zeta = args.crf_zeta
        self.num_iter = args.crf_num_iter
        self.high_thres = args.crf_value_high_thres
        self.low_thres = args.crf_value_low_thres
        self.args = args

    def trunc(self, seg):
        seg = torch.clamp(seg, min=self.low_thres, max=self.high_thres)
        return seg

    @torch.no_grad()
    def forward(self, feature_map, seg, targets=None):

        feature_map = feature_map.float()
        kernel_size = self.kernel_size
        B, H, W = seg.shape
        C = feature_map.shape[1]
        
        self.unfold = torch.nn.Unfold(kernel_size, stride=1, padding=self.kernel_size // 2)
        # feature_map [B, C, H, W]
        feature_map = feature_map + 10
        # unfold_feature_map [B, C, kernel_size ** 2, H*W]
        unfold_feature_map = self.unfold(feature_map).reshape(B, C, kernel_size**2, H * W)
        # B, kernel_size**2, H*W
        kernel = torch.exp(-(((unfold_feature_map - feature_map.reshape(B, C, 1, H*W)) ** 2) / (2 * self.zeta ** 2)).sum(1))
        
        if targets is not None:
            t = targets.reshape(B, H, W)
            seg = seg * t
        else:
            t = None
        
        seg = self.trunc(seg)

        for it in range(self.num_iter):
            seg = self.single_forward(seg, kernel, t, B, H, W, it)

        return (seg > 0.5).float()

    def single_forward(self, x, kernel, targets, B, H, W, it):
        x = x[:, None]
        # x [B 2 H W]
        B, C, H, W = x.shape
        x = torch.cat([1-x, x], 1)
        kernel_size = self.kernel_size
        # unfold_x [B, 2, kernel_size**2, H * W]
        # kernel   [B,    kennel_size**2, H * W]
        unfold_x = self.unfold(-torch.log(x)).reshape(B, 2, kernel_size ** 2, H * W)
        # aggre, x [B, 2, H * W]
        aggre = (unfold_x * kernel[:, None]).sum(2) 
        aggre = torch.exp(-aggre)
        if targets is not None:
            aggre[:, 1:] = aggre[:, 1:] * targets.reshape(B, 1, H * W)
        out = aggre
        out = out / (1e-6 + out.sum(1, keepdim=True))
        out = self.trunc(out)
        return out[:, 1].reshape(B, H, W)


class MaskHead(nn.Module):

    def __init__(self, in_channels=2048, args=None):
        super().__init__()
        self.num_convs                  = args.mask_head_num_convs
        self.in_channels                = in_channels
        self.mask_head_hidden_channel   = args.mask_head_hidden_channel
        self.mask_head_out_channel      = args.mask_head_out_channel
        self.mask_scale_ratio           = args.mask_scale_ratio

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.mask_head_hidden_channel
            out_channels = self.mask_head_hidden_channel if i < self.num_convs - 1 else self.mask_head_out_channel
            self.convs.append(ConvModule(in_channels, out_channels, 3, padding=1))

    def forward(self, x):
        for idx, conv in enumerate(self.convs):
            if idx == 3:
                h, w = x.shape[2:]
                th, tw = int(h * self.mask_scale_ratio), int(w * self.mask_scale_ratio)
                x = F.interpolate(x, (th, tw), mode='bilinear', align_corners=False)
            x = conv(x)
        return x


class RoIHead(nn.Module):

    def __init__(self, in_channels=2048, args=None):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, args.mask_head_out_channel)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(args.mask_head_out_channel, args.mask_head_out_channel)
    
    def forward(self, x, boxmask=None):
        x = x.mean((2, 3))
        x = self.mlp2(self.relu(self.mlp1(x)))
        return x



class MALStudentNetwork(pl.LightningModule):

    def __init__(self, in_channels=2048, args=None):
        super().__init__()
        self.args = args
        self.backbone               = vision_transformers.get_vit(args=args)
        mask_head_num_convs         = args.mask_head_num_convs
        mask_head_hidden_channel    = args.mask_head_hidden_channel
        mask_head_out_channel       = args.mask_head_out_channel

        # K head
        self.roi_head = RoIHead(in_channels, args=args)
        # V head
        self.mask_head = MaskHead(in_channels, args=args)

        # make student sharded on multiple gpus
        self.configure_sharded_model()

    def configure_sharded_model(self):
        self.backbone = auto_wrap(self.backbone)


    def forward(self, x, boxmask, bboxes):
        x = x.half()
        feat = self.backbone.base_forward(x)
        spatial_feat_ori = self.backbone.get_spatial_feat(feat)
        h, w = spatial_feat_ori.shape[2:]
        mask_scale_ratio_pre = int(self.args.mask_scale_ratio_pre)
        if not self.args.not_adjust_scale:
            spatial_feat_list = []
            masking_list = []
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            for idx, (scale_low, scale_high) in enumerate([(0, 32**2), (32**2, 96**2), (96**2, 1e5**2)]):
                masking = (areas < scale_high) * (areas > scale_low)
                if masking.sum() > 0:
                    spatial_feat = F.interpolate(spatial_feat_ori[masking], 
                                            size=(int(h*2**(idx-1)), int(w*2**(idx-1))),
                                            mode='bilinear', align_corners=False)
                    boxmask = None
                else:
                    spatial_feat = None
                    boxmask = None
                spatial_feat_list.append(spatial_feat)
                masking_list.append(masking)
            roi_feat = self.roi_head(spatial_feat_ori)
            n, maxh, maxw = roi_feat.shape[0], h * 4, w * 4 
            seg_list = []
            seg_all = torch.zeros(n, 1, maxh, maxw).to(roi_feat)
            for idx, (spatial_feat, masking) in enumerate(zip(spatial_feat_list, masking_list)):
                if masking.sum() > 0:
                    mn = masking.sum()
                    mh, mw = int(h * mask_scale_ratio_pre * 2**(idx-1)), int(w * mask_scale_ratio_pre * 2**(idx-1))
                    seg_feat = self.mask_head(spatial_feat)
                    c = seg_feat.shape[1]
                    masked_roi_feat = roi_feat[masking]
                    seg = (masked_roi_feat[:, None, :] @ seg_feat.reshape(mn, c, mh * mw * 4)).reshape(mn, 1, mh * 2, mw * 2)
                    seg = F.interpolate(seg, size=(maxh, maxw), mode='bilinear', align_corners=False)
                    seg_all[masking] = seg
            
            ret_vals = {'feat': feat, 'seg': seg_all, 'spatial_feat': spatial_feat_ori, 'masking_list': masking_list}
        else:
            spatial_feat    = F.interpolate(spatial_feat_ori, size=(int(h*self.args.mask_scale_ratio_pre), int(w*self.args.mask_scale_ratio_pre)),
                                        mode='bilinear', align_corners=False)
            boxmask         = F.interpolate(boxmask, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False)
            seg_feat        = self.mask_head(spatial_feat)
            roi_feat        = self.roi_head(spatial_feat_ori, boxmask)
            n, c, h, w      = seg_feat.shape
            seg             = (roi_feat[:,None,:] @ seg_feat.reshape(n, c, h * w)).reshape(n, 1, h, w)
            seg             = F.interpolate(seg, (h * 4, w * 4), mode='bilinear', align_corners=False)
            ret_vals = {'feat': feat, 'seg': seg, 'spatial_feat': spatial_feat_ori}
        return ret_vals
    

class MALTeacherNetwork(MALStudentNetwork):

    def __init__(self, in_channels, args=None):
        super().__init__(in_channels, args=args)
        self.eval()
        self.momentum = args.teacher_momentum

    @torch.no_grad()
    def update(self, student):
        for param_student, param_teacher in zip(student.parameters(), self.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + param_student.data * (1 - self.momentum)

    @torch.no_grad()
    def forward(self, *x):
        return super().forward(*x)


class MIoUMetrics(torchmetrics.Metric):

    def __init__(self, dist_sync_on_step=True, num_classes=20):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("cnt", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, label, iou):
        self.cnt[label-1] += 1
        self.total[label-1] += iou

    def update_with_ious(self, labels, ious):
        for iou, label in zip(ious, labels):
            self.cnt[label-1] += 1
            self.total[label-1] += float(iou)
        return ious
    

    def cal_intersection(self, seg, gt):
        B = seg.shape[0]
        inter_cnt = (seg * gt).reshape(B, -1).sum(1)
        return inter_cnt
    
    def cal_union(self, seg, gt, inter_cnt=None):
        B = seg.shape[0]
        if inter_cnt is None:
            inter_cnt = self.cal_intersection(seg, gt)
        union_cnt = seg.reshape(B, -1).sum(1) + gt.reshape(B, -1).sum(1) - inter_cnt
        return union_cnt
    
    def cal_iou(self, seg, gt):
        inter_cnt = self.cal_intersection(seg, gt)
        union_cnt = self.cal_union(seg, gt, inter_cnt)
        return 1.0 * inter_cnt / (union_cnt + 1e-6)

    def compute(self):
        mIoUs = self.total / (1e-6 + self.cnt)
        mIoU = mIoUs.sum() / (self.cnt > 0).sum()
        return mIoU

    def compute_with_ids(self, ids=None):
        if ids is not None:
            total = self.total[torch.tensor(np.array(ids)).long()]
            cnt = self.cnt[torch.tensor(np.array(ids)).long()]
        else:
            total = self.total
            cnt = self.cnt
        mIoUs = total / (1e-6 + cnt)
        mIoU = mIoUs.sum() / (cnt > 0).sum()
        return mIoU



class MAL(pl.LightningModule):
    
    def __init__(self, args=None, num_iter_per_epoch=None):

        super().__init__()

        # mIoU torchmetrics

        # loss term hyper parameters
        self.num_convs = args.mask_head_num_convs
        self.loss_mil_weight = args.loss_mil_weight
        self.loss_crf_weight = args.loss_crf_weight
        self.loss_crf_step = args.loss_crf_step
        self.args = args

        self.mask_thres = args.mask_thres

        self.num_classes = num_class_dict[args.dataset_type]

        self.mIoUMetric = MIoUMetrics(num_classes=self.num_classes)
        self.areaMIoUMetrics = nn.ModuleList([MIoUMetrics(num_classes=self.num_classes) for _ in range(3)])
        if self.args.comp_clustering:
            self.clusteringScoreMetrics = torchmetrics.MeanMetric()

        backbone_type = args.arch


        if 'tiny' in backbone_type.lower():
            in_channel = 192
        if 'small' in backbone_type.lower():
            in_channel = 384
        elif 'base' in backbone_type.lower():
            in_channel = 768
        elif 'large' in backbone_type.lower():
            in_channel = 1024
        elif 'huge' in backbone_type.lower():
            in_channel = 1280


        self.mean_field = MeanField(args=self.args)

        self.student = MALStudentNetwork(in_channel, args=args)

        self.teacher = MALTeacherNetwork(in_channel, args=args)

        self.denormalize = Denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # optimizer parameters
        self._optim_type = args.optim_type
        self._lr = args.lr
        self._wd = args.wd
        self._momentum = args.optim_momentum
        if num_iter_per_epoch is not None:
            self._num_iter_per_epoch = num_iter_per_epoch // len(self.args.gpus)

        self.args = args

        self.vis_cnt = 0
        self.local_step = 0

        # Enable manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = AdamWwStep(self.parameters(), eps=self.args.optim_eps, 
                                betas=self.args.optim_betas,
                                lr=self._lr, weight_decay=self._wd)
        return optimizer 

    def crf_loss(self, img, seg, tseg, boxmask):
        refined_mask = self.mean_field(img, tseg, targets=boxmask) 
        return self.dice_loss(seg, refined_mask).mean(), refined_mask

    def dice_loss(self, input, target):
        input = input.contiguous().view(input.size()[0], -1).float()
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)

        return 1-d
        
    def mil_loss(self, pred, target):
        row_labels = target.max(1)[0]
        column_labels = target.max(2)[0]

        row_input = pred.max(1)[0]
        column_input = pred.max(2)[0]

        loss_func = self.dice_loss

        loss = loss_func(column_input, column_labels) +\
               loss_func(row_input, row_labels)
        
        return loss

    def training_step(self, x):
        optimizer = self.optimizers()
        loss = {}
        image = x['image']

        local_step = self.local_step
        self.local_step += 1

        if 'timage' in x.keys():
            timage = x['timage']
        else:
            timage = image
        student_output = self.student(image, x['mask'], x['bbox'])
        teacher_output = self.teacher(timage, x['mask'], x['bbox'])
        B, oh, ow = student_output['seg'].shape[0], student_output['seg'].shape[2], student_output['seg'].shape[3]
        mask  = F.interpolate(x['mask'], size=(oh, ow), mode='bilinear', align_corners=False).reshape(-1, oh, ow)
        
        args = self.args


        if 'image' in x:
            student_seg_sigmoid = torch.sigmoid(student_output['seg'])[:,0].float()
            teacher_seg_sigmoid = torch.sigmoid(teacher_output['seg'])[:,0].float()

            # Multiple instance learning Loss
            loss_mil = self.mil_loss(student_seg_sigmoid, mask)
            # Warmup loss weight for multiple instance learning loss
            if self.current_epoch > 0:
                step_mil_loss_weight = 1
            else:
                step_mil_loss_weight = min(1, 1. * local_step / self.args.loss_mil_step)
            loss_mil *= step_mil_loss_weight
            loss_mil = loss_mil.sum() / (loss_mil.numel() + 1e-4) * self.loss_mil_weight
            loss.update({'mil': loss_mil})
            # Tensorboard logs
            self.log("train/loss_mil", loss_mil, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

            # Conditional Random Fields Loss
            th, tw = oh * args.crf_size_ratio, ow * args.crf_size_ratio
            # resize image
            scaled_img = F.interpolate(image, size=(th, tw), mode='bilinear', align_corners=False).reshape(B, -1, th, tw)
            # resize student segmentation
            scaled_stu_seg = F.interpolate(student_seg_sigmoid[None, ...], size=(th, tw), mode='bilinear', align_corners=False).reshape(B, th, tw)
            # resize teacher segmentation
            scaled_tea_seg = F.interpolate(teacher_seg_sigmoid[None, ...], size=(th, tw), mode='bilinear', align_corners=False).reshape(B, th, tw)
            # resize mask
            scaled_mask = F.interpolate(x['mask'], size=(th, tw), mode='bilinear', align_corners=False).reshape(B, th, tw)
            loss_crf, pseudo_label = self.crf_loss(scaled_img, scaled_stu_seg, (scaled_stu_seg + scaled_tea_seg)/2, scaled_mask)
            if self.current_epoch > 0:
                step_crf_loss_weight = 1
            else:
                step_crf_loss_weight = min(1. * local_step / self.loss_crf_step, 1.)
            loss_crf *= self.loss_crf_weight * step_crf_loss_weight
            loss.update({'crf': loss_crf})
            self.log("train/loss_crf", loss_crf, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        else:
            raise NotImplementedError

        total_loss = sum(loss.values())
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/bs", image.shape[0], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()
        if self._optim_type == 'adamw':
            self.set_lr_per_iteration(optimizer, 1. * local_step)
        self.teacher.update(self.student)
    
    def set_lr_per_iteration(self, optimizer, local_step):
        args = self.args
        epoch = 1. * local_step / self._num_iter_per_epoch + self.current_epoch
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch / args.warmup_epochs)
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.max_epochs - args.warmup_epochs) * args.num_wave))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr


    def training_epoch_end(self, outputs):
        optimizer = self.optimizers()
        self.local_step = 0

    def validation_step(self, batch, batch_idx, return_mask=False):
        if not self.args.not_eval_mask:
            imgs, gt_masks, masks, labels, ids, boxmasks, boxes, ext_boxes, ext_hs, ext_ws =\
                batch['image'], batch['gtmask'], batch['mask'], batch['compact_category_id'], \
                batch['id'], batch['boxmask'], batch['bbox'], batch['ext_boxes'], batch['ext_h'], batch['ext_w']
        else:
            imgs, gt_masks, masks, labels, ids, boxmasks, boxes, ext_boxes, ext_hs, ext_ws =\
                batch['image'], batch['boxmask'], batch['boxmask'], batch['compact_category_id'], \
                batch['id'], batch['boxmask'], batch['bbox'], batch['ext_boxes'], batch['ext_h'], batch['ext_w']
        #imgs = imgs.reshape(imgs.shape[0], imgs.shape[2], imgs.shape[3], imgs.shape[4])
        B, C, H, W = imgs.shape
        denormalized_images = self.denormalize(imgs.cpu().numpy().transpose(0, 2, 3, 1)).astype(np.uint8) 
        labels = labels.cpu().numpy()

        if self.args.use_mixed_model_test:
            s_outputs = self.student(imgs, batch['boxmask'], batch['bbox'])
            t_outputs = self.teacher(imgs, batch['boxmask'], batch['bbox'])
            segs = (s_outputs['seg'] + t_outputs['seg']) / 2
        else:
            if self.args.use_teacher_test:
                outputs = self.teacher(imgs, batch['boxmask'], batch['bbox'])
            else:
                outputs = self.student(imgs, batch['boxmask'], batch['bbox'])
            segs = outputs['seg']

        if self.args.use_flip_test:
            if self.args.use_mixed_model_test:
                s_outputs = self.student(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                t_outputs = self.teacher(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                flipped_segs = torch.flip((s_outputs['seg'] + t_outputs['seg']) / 2, [3])
                segs = (flipped_segs + segs) / 2
            else:
                if self.args.use_teacher_test:
                    flip_outputs = self.teacher(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                else:
                    flip_outputs = self.student(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                segs = (segs + torch.flip(flip_outputs['seg'], [3])) / 2

        segs = F.interpolate(segs, (H, W), align_corners=False, mode='bilinear')
        segs = segs.sigmoid()
        thres_list = [0, 32**2, 96 ** 2, 1e5**2]

        segs = segs * boxmasks
        areas = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        binseg = segs.clone()
        for idx, (lth, hth) in enumerate(zip(thres_list[:-1], thres_list[1:])):
            obj_ids = ((lth < areas) * (areas <= hth)).cpu().numpy()
            if obj_ids.sum() > 0:
                binseg[obj_ids] = (binseg[obj_ids] > self.args.mask_thres[idx]).float()

        tb_logger = self.logger.experiment
        epoch_count = self.current_epoch 

        batch_ious = []

        img_pred_masks = []

        for idx, (img_h, img_w, ext_h, ext_w, ext_box, seg, gt_mask, area, label) in enumerate(zip(batch['height'], batch['width'], ext_hs, ext_ws, ext_boxes, segs, gt_masks, areas, labels)):
            roi_pred_mask = F.interpolate(seg[None, ...], (ext_h, ext_w), mode='bilinear', align_corners=False)[0][0]
            h, w = int(img_h), int(img_w)
            img_pred_mask_shape = h, w

            img_pred_mask = np.zeros(img_pred_mask_shape).astype(np.float)

            img_pred_mask[max(ext_box[1], 0):min(ext_box[3], h), \
                            max(ext_box[0], 0):min(ext_box[2], w)] = \
                            roi_pred_mask[max(0-ext_box[1], 0):ext_h+min(0, h-ext_box[3]), \
                                max(0-ext_box[0], 0):ext_w+min(0, w-ext_box[2])].cpu().numpy()


            for idx, (lth, hth) in enumerate(zip(thres_list[:-1], thres_list[1:])):
                if lth < area <= hth:
                    img_pred_mask = (img_pred_mask > self.args.mask_thres[idx]).astype(np.float)

            img_pred_masks.append(img_pred_mask[None, ...])
            if not self.args.not_eval_mask:
                iou = self.mIoUMetric.cal_iou(img_pred_mask[np.newaxis,...], gt_mask.data[np.newaxis,...])
                # overall mask IoU
                self.mIoUMetric.update(int(label), iou[0])
                batch_ious.extend(iou)
                # Small/Medium/Large IoU
                for jdx, (lth, hth) in enumerate(zip(thres_list[:-1], thres_list[1:])):
                    obj_ids = ((lth < area) * (area <= hth)).cpu().numpy()
                    if obj_ids.sum() > 0:
                        self.areaMIoUMetrics[jdx].update_with_ious(labels[obj_ids], iou[obj_ids])
        
        # Tensorboard vis
        if not self.args.not_eval_mask:
            for idx, batch_iou, img, seg, label, gt_mask, mask, box, area in zip(ids, batch_ious, denormalized_images, segs, labels, gt_masks, masks, boxes, areas):
                if area > 64**2 and batch_iou < 0.78 and self.vis_cnt <= 100:
                    seg = seg.cpu().numpy().astype(np.float32)[0]
                    mask = mask.data
                    
                    seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_LINEAR)
                    #seg = (seg > self.mask_thres).astype(np.uint8)
                    seg = (seg * 255).astype(np.uint8)
                    seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
                    tseg = cv2.applyColorMap((mask[0] > 0.5).cpu().numpy().astype(np.uint8) * 255, cv2.COLORMAP_JET)

                    vis = cv2.addWeighted(img, 0.5, seg, 0.5, 0)
                    tvis = cv2.addWeighted(img, 0.5, tseg, 0.5, 0)

                    tb_logger.add_image('val/vis_{}'.format(int(idx)), vis, epoch_count, dataformats="HWC")
                    tb_logger.add_image('valgt/vis_{}'.format(int(idx)), tvis, epoch_count, dataformats="HWC")
                self.vis_cnt += 1

        ret_dict = dict()
        if return_mask:
            ret_dict['img_pred_masks'] = img_pred_masks
        if not self.args.not_eval_mask:
            ret_dict['ious'] = batch_ious
        return ret_dict




    def get_parameter_groups(self, print_fn=print):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'backbone' in name:
                if 'weight' in name:
                    # print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    # print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)
                    
            # scracthed weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[2].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[3].append(value)
        return groups

    def validation_epoch_end(self, validation_step_outputs):
        mIoU = self.mIoUMetric.compute()
        self.log("val/mIoU", mIoU, on_epoch=True, prog_bar=True, sync_dist=True)
        if dist.get_rank() == 0:
            print("val/mIoU: {}".format(mIoU))
        if "coco" in self.args.dataset_type:
            from datasets import coco
            cat_kv = dict([(cat["name"], cat["id"]) for cat in coco.categories])
            things_ids = []
            for thing in coco.things:
                things_ids.append(coco.cat_mapping[cat_kv[thing]])
            semistuff_ids = []
            for ss in coco.semistuff:
                semistuff_ids.append(coco.cat_mapping[cat_kv[ss]])
            mIoU_things = self.mIoUMetric.compute_with_ids(things_ids)
            self.log("val/mIoU_things", mIoU_things, on_epoch=True, prog_bar=True, sync_dist=True)
            mIoU_semistuff = self.mIoUMetric.compute_with_ids(semistuff_ids)
            self.log("val/mIoU_stuff", mIoU_semistuff, on_epoch=True, prog_bar=True, sync_dist=True)
            if self.args.comp_clustering:
                clustering_score = self.clusteringScoreMetrics.compute()
                self.log("val/cluster_score", clustering_score, on_epoch=True, prog_bar=True, sync_dist=True)
            if dist.get_rank() == 0:
                print("val/mIoU_things", mIoU_things)
                print("val/mIoU_semistuff", mIoU_semistuff)
                if self.args.comp_clustering:
                    print("val/cluster_score", clustering_score)
        self.mIoUMetric.reset()
        self.vis_cnt = 0

        for i, name in zip(range(len(self.areaMIoUMetrics)), ["small", "medium", "large"]):
            area_mIoU = self.areaMIoUMetrics[i].compute()
            self.log("val/mIoU_{}".format(name), area_mIoU, on_epoch=True, sync_dist=True)
            if dist.get_rank() == 0:
                print("val/mIoU_{}: {}".format(name, area_mIoU))
            self.areaMIoUMetrics[i].reset()


class MALPseudoLabels(MAL):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs['args']
        assert args is not None
        assert args.label_dump_path is not None

    def validation_step(self, batch, batch_idx):

        pred_dict = super().validation_step(batch, batch_idx, return_mask=True)
        pred_seg = pred_dict['img_pred_masks']
        if not self.args.not_eval_mask:
            ious = pred_dict['ious']

        ret = []

        cnt = 0
        t = time.time()
        for seg, (ex0, ey0, ex1, ey1), (x0, y0, x1, y1), w, h, idx, image_id, category_id \
            in zip(pred_seg, batch['ext_boxes'], batch['bbox'], batch['width'], \
                batch['height'], batch['id'], batch.get('image_id', batch.get('video_id', None)), batch['category_id']):
            sh, sw = ey1 - ey0, ex1 - ex0
            '''
            oseg = np.array(Image.fromarray(seg[0].cpu().numpy()).resize((sw, sh)))
            seg_label = np.zeros((h, w), dtype=np.uint8)
            seg_label[max(0, ey0): min(h, ey1), max(0, ex0): min(w, ex1)] = \
                oseg[max(0, -ey0): sh - max(ey1 - h, 0), \
                     max(0, -ex0): sw - max(ex1 - w, 0)]
            '''
            encoded_mask = encode(np.asfortranarray(seg[0].astype(np.uint8)))
            encoded_mask['counts'] = encoded_mask['counts'].decode('ascii')
            labels = {
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "id": int(idx),
                "category_id": int(category_id),
                "segmentation": encoded_mask,
                "iscrowd": 0,
                "area": float(x1 - x0) * float(y1 - y0),
                "image_id": int(image_id)
            }
            if 'score' in batch.keys():
                labels['score'] = float(batch['score'][cnt].cpu().numpy())
            if not self.args.not_eval_mask:
                labels['iou'] = float(ious[cnt])
            cnt += 1
            ret.append(labels)
        
        if batch.get('ytvis_idx', None) is not None:
            for ytvis_idx, labels in zip(batch['ytvis_idx'], ret):
                labels['ytvis_idx'] = list(map(int, ytvis_idx))

        return ret

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        ret = list(itertools.chain.from_iterable(validation_step_outputs))
        if self.trainer.strategy.root_device.index > 0:
            with open("{}.part{}".format(self.args.label_dump_path, self.trainer.strategy.root_device.index), "w") as f:
                json.dump(ret, f)
            torch.distributed.barrier()
        else:
            val_ann_path = datapath_configs[self.args.dataset_type]['generating_pseudo_label_config']['val_ann_path']
            with open(val_ann_path, "r") as f:
                anns = json.load(f)
            torch.distributed.barrier()
            for i in range(1, len(self.args.gpus)):
                with open("{}.part{}".format(self.args.label_dump_path, i), "r") as f:
                    obj = json.load(f)
                ret.extend(obj)
                os.remove("{}.part{}".format(self.args.label_dump_path, i))
            
            if ret[0].get('ytvis_idx', None) is None:
                # for COCO format
                _ret = []
                _ret_set = set()
                for ann in ret:
                    if ann['id'] not in _ret_set:
                        _ret_set.add(ann['id'])
                        _ret.append(ann)
                anns['annotations'] = _ret
            else:
                # for YouTubeVIS format
                for inst_ann in anns['annotations']:
                    len_video = len(inst_ann['bboxes'])
                    inst_ann['segmentations'] = [None for _ in range(len_video)]
                
                for seg_ann in ret:
                    inst_idx, frame_idx = seg_ann['ytvis_idx']
                    anns['annotations'][inst_idx]['segmentations'][frame_idx] = seg_ann['segmentation']
                
        
            with open(self.args.label_dump_path, "w") as f:
                json.dump(anns, f)
            with open(self.args.label_dump_path + ".result", "w") as f:
                json.dump(anns['annotations'], f)


            if self.args.box_inputs is not None:
                print("Start evaluating the results...")
                cocoGt = COCO(self.args.val_ann_path)
                cocoDt = cocoGt.loadRes(self.args.label_dump_path + ".result")

                for iou_type in ['bbox', 'segm']:
                    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()

                print("Evaluation finished.")
