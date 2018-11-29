#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import logging
import importlib
import argparse
import os.path as osp
import sys

from lib.model import DeepLabLargeFOV
from lib.pascal_voc import PascalVoc
from utils.crf import crf


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


def eval_model(net, use_crf = True):
    logger = logging.getLogger(__name__)
    ## dataset
    ds = PascalVoc('./data/VOCdevkit/', mode = 'val')

    ## inference
    logger.info('evaluating on standard voc2012 val set')
    ious = []
    for i, (im, lb) in enumerate(tqdm(ds)):
        im_org = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        im = ds.trans(im)
        im = im.cuda().unsqueeze(0)
        with torch.no_grad():
            scores = net(im)
            scores = F.interpolate(scores, im.size()[2:], mode = 'bilinear')
            scores = F.softmax(scores, 1)
        scores = scores.detach().cpu().numpy()
        if use_crf:
            mask = crf(im_org, scores)
        else:
            mask = np.argmax(scores, axis = 1).squeeze(0).astype(np.uint8)
        lb = np.asarray(lb)

        iou = compute_iou(mask, lb)
        ious.append(iou)

    mIOU = sum(ious) / len(ious)
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

    ## evaluating
    mIOU = eval_model(net, cfg.use_crf)
    logger.info('iou in whole is: {}'.format(mIOU))


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
