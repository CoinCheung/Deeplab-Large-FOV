#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import logging
import importlib
import argparse
import os.path as osp
import sys
import math

from lib.model import DeepLabLargeFOV
from lib.pascal_voc import PascalVoc
from lib.pascal_voc_aug import PascalVoc_Aug
#  from utils.crf import crf


def get_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument(
            '--cfg',
            dest = 'cfg',
            type = str,
            default = 'config/pascal_voc_aug_multi_scale.py',
            help = 'config file used in training'
            )
    return parser.parse_args()



def compute_iou(mask, lb, ignore_lb = (255, )):
    assert mask.shape == lb.shape, 'prediction and gt do not agree in shape'
    classes = set(np.unique(lb).tolist())
    for cls in ignore_lb:
        if cls in classes:
            classes.remove(cls)

    iou_cls = []
    for cls in classes:
        gt = lb == cls
        pred = mask == cls
        intersection = np.logical_and(gt, pred)
        union = np.logical_or(gt, pred)
        iou = float(np.sum(intersection)) / float(np.sum(union))
        iou_cls.append(iou)
    return sum(iou_cls) / len(iou_cls)


def eval_model(net, cfg):
    logger = logging.getLogger(__name__)
    ## dataset
    dsval = PascalVoc(cfg, mode='val')
    ## evaluator
    evaluator = MscEval(
            dsval = dsval,
            scales = cfg.test_scales,
            n_classes = cfg.n_classes,
            lb_ignore = cfg.ignore_label,
            flip = cfg.flip,
            crop_size = cfg.crop_size,
            n_workers = 4,
            )
    ## inference
    logger.info('evaluating on standard voc2012 val set')
    mIOU = evaluator(net)

    return mIOU


class MscEval(object):
    def __init__(self,
            dsval,
            scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            n_classes = 19,
            lb_ignore = 255,
            flip = True,
            crop_size = 321,
            n_workers = 2,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.crop_size = crop_size
        ## dataloader
        self.dsval = dsval
        self.net = None


    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        ## TODO: use zeros
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, im):
        cropsize = self.crop_size
        stride_rate = 5/6.
        N, C, H, W = im.size()
        long_size, short_size = (H,W) if H>W else (W,H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize*stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W-cropsize)/stride)+1
            n_y = math.ceil((H-cropsize)/stride)+1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride*iy+cropsize), min(W, stride*ix+cropsize)
                    hst, wst = hed-cropsize, wed-cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob


    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    def compute_hist(self, pred, lb, lb_ignore=255):
        n_classes = self.n_classes
        keep = np.logical_not(lb==lb_ignore)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def __call__(self, net):
        self.net = net
        ## evaluate
        hist = np.zeros((self.n_classes, self.n_classes), dtype=np.float32)
        for i, (imgs, label) in enumerate(tqdm(self.dsval)):
            N, _, H, W = imgs.size()
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label)
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(args):
    ## set up logger and parse cfg
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    spec = importlib.util.spec_from_file_location('mod_cfg', args.cfg)
    mod_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_cfg)
    cfg = mod_cfg.cfg

    ## initialize model
    net = DeepLabLargeFOV(3, cfg.n_classes)
    net.eval()
    net.cuda()
    model_pth = osp.join(cfg.res_pth, 'model_final.pkl')
    net.load_state_dict(torch.load(model_pth))

    ## evaluate
    mIOU = eval_model(net, cfg)
    logger.info('iou in whole is: {}'.format(mIOU))


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
