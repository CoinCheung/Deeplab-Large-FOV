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


    #  for i, (im, lb) in enumerate(ds):
    #      #  print(im.size)
    #      #  print(lb.size)
    #      #  im.show()
    #      im_org = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    #      #  print(im_org.shape)
    #      #  print(im_org.dtype)
    #      #  cv2.imshow('cv', im_org)
    #      #  cv2.waitKey(0)
    #
    #      im = ds.trans(im_org)
    #      im = im.cuda().unsqueeze(0)
    #      #  print(im.shape)
    #      #  print(torch.max(im))
    #      #  print(torch.min(im))
    #
    #      scores = net(im)
    #      tmp = scores.detach().cpu().numpy()
    #      tmp = np.argmax(tmp, axis = 1)
    #      tmp = np.squeeze(tmp, axis = 0)
    #      print(np.max(tmp))
    #      print(np.min(tmp))
    #      #  print(scores.shape)
    #      scores = F.interpolate(scores, im.size()[2:], mode = 'bilinear')
    #      print(torch.max(scores))
    #      print(torch.min(scores))
    #
    #      scores = scores.detach().cpu().numpy()
    #      print(scores.shape)
    #      mask = np.argmax(scores, axis = 1)
    #      print(np.max(mask))
    #      print(np.min(mask))
    #      mask = np.squeeze(mask, axis = 0)
    #      H, W = mask.shape
    #      pic = np.empty((H, W, 3))
    #      for i in range(21):
    #          pic[mask == i] = cm[i]
    #      print(cm.shape)
    #      print(mask.shape)
    #      print(mask.dtype)
    #      print(np.max(mask))
    #      print(np.min(mask))
    #      from PIL import Image
    #      #  pic = scores.transpose(0, 2, 3, 1)
    #      #  pic = np.argmax(pic, axis = 3).astype(np.uint8)[0] * 20
    #      #  pic = np.argmax(pic, axis = 3)[0] * 20
    #      print(np.max(pic))
    #      print(np.min(pic))
    #      print(scores[0, :, 167, 89])
    #      #  cv2.imshow('pic', pic[:, :, np.newaxis])
    #      cv2.imshow('pic', pic)
    #      cv2.imshow('org', im_org)
    #      cv2.waitKey(0)
    #
    #      #  print(im.shape)
    #      #  print(scores.shape)
    #      #  print(type(im))
    #      #  print(type(scores))
    #      #  print(im.dtype)
    #      #  print(scores.dtype)
    #      mask = crf(im_org, scores)
    #      break



if __name__ == "__main__":
    infer()
