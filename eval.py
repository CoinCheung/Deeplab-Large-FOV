#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from model import DeepLabLargeFOV
from pascal_voc import PascalVoc
from crf import crf
from logger import *


## TODO: combine this eval to train and print whole logger info to the log file


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


def evaluate():
    ## network
    net = DeepLabLargeFOV(3, 21)
    net.eval()
    net.cuda()
    model_pth = './res/voc2012/model_final.pkl'
    net.load_state_dict(torch.load(model_pth))

    ## dataset
    ds = PascalVoc('./data/VOCdevkit/', mode = 'val')

    ## inference
    logger.info('evaluating on val set')
    ious = []
    for i, (im, lb) in enumerate(tqdm(ds)):
        im_org = im
        im = ds.trans(im)
        im = im.cuda().unsqueeze(0)
        scores = net(im)
        scores = F.interpolate(scores, im.size()[2:], mode = 'bilinear')
        scores = scores.detach().cpu().numpy()
        mask = np.argmax(scores, axis = 1).squeeze(0).astype(np.uint8)
        lb = np.asarray(lb)

        iou = compute_iou(mask, lb)
        ious.append(iou)
        logger.info('image {}, iou is: {}'.format(i, iou))

    mIOU = sum(ious) / len(ious)
    logger.info('iou in whole is: {}'.format(mIOU))



if __name__ == "__main__":
    evaluate()
