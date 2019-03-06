#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os.path as osp
import cv2
from PIL import Image
import argparse
import importlib

from lib.model import DeepLabLargeFOV
#  from utils.crf import crf
from utils.colormap import color_map


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument(
            '--cfg',
            dest = 'cfg',
            type = str,
            default = 'config/pascal_voc_aug_multi_scale.py',
            help = 'config file associated with the model'
            )
    parser.add_argument(
            '--img',
            dest = 'impth',
            type = str,
            default = './example.jpg',
            help = 'image to be tested'
            )
    return parser.parse_args()



def infer(args):
    spec = importlib.util.spec_from_file_location('mod_cfg', args.cfg)
    mod_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_cfg)
    cfg = mod_cfg.cfg

    ## set up
    cm = color_map(N = cfg.n_classes)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    ## image
    args.impth = './example.jpg'
    assert osp.exists(args.impth), '{} not exists !!'.format(args.impth)
    im = Image.open(args.impth)
    im_org = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    im = trans(im).cuda().unsqueeze(0)

    ## network
    net = DeepLabLargeFOV(3, cfg.n_classes)
    net.eval()
    net.cuda()
    model_pth = osp.join(cfg.res_pth, 'model_final.pkl')
    net.load_state_dict(torch.load(model_pth))

    ## inference
    scores = net(im)
    scores = F.interpolate(scores, im.size()[2:], mode='bilinear', align_corners=True)
    scores = F.softmax(scores, 1)
    scores = scores.detach().cpu().numpy()
    if cfg.use_crf:
        mask = crf(im_org, scores)
    else:
        mask = np.argmax(scores, axis = 1)
        mask = np.squeeze(mask, axis = 0)
    H, W = mask.shape
    pic = np.empty((H, W, 3))
    for i in range(cfg.n_classes):
        pic[mask == i] = cm[i]

    ## show
    cv2.imshow('org', im_org)
    cv2.imshow('pred', pic)
    cv2.waitKey(0)


if __name__ == "__main__":
    args = parse_args()
    infer(args)
