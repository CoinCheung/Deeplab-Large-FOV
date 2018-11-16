#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os.path as osp
import cv2
from PIL import Image

from model import DeepLabLargeFOV
from crf import crf
from colormap import color_map


## TODO: use argparse


def infer():
    ## set up
    cm = color_map(N = 21)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    ## image
    impth = './example.jpg'
    assert osp.exists(impth), '{} not exists !!'.format(impth)
    im = Image.open(impth)
    im_org = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    im = trans(im).cuda().unsqueeze(0)

    ## network
    net = DeepLabLargeFOV(3, 21)
    net.eval()
    net.cuda()
    model_pth = './res/model.pkl'
    net.load_state_dict(torch.load(model_pth))

    ## inference
    scores = net(im)
    scores = F.interpolate(scores, im.size()[2:], mode = 'bilinear')
    scores = scores.detach().cpu().numpy()
    mask = np.argmax(scores, axis = 1)
    mask = np.squeeze(mask, axis = 0)
    H, W = mask.shape
    pic = np.empty((H, W, 3))
    for i in range(21):
        pic[mask == i] = cm[i]

    ## show
    cv2.imshow('org', im_org)
    cv2.imshow('pred', pic)
    cv2.waitKey(0)



if __name__ == "__main__":
    infer()
