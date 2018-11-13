#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
import cv2
import numpy as np


## TODO: see what augs are used here


class PascalVoc(Dataset):
    def __init__(self, root_pth, mode = 'train', *args, **kwargs):
        super(PascalVoc, self).__init__(*args, **kwargs)
        self.mode =mode
        rootpath = osp.join(root_pth, 'VOC2012/')
        if not osp.exists(rootpath): assert(False)
        if not self.mode in ('train', 'val', 'trainval'): assert(False)
        if mode == 'train':
            txtfile = osp.join(rootpath, 'ImageSets/Segmentation/train.txt')
        elif mode == 'val':
            txtfile = osp.join(rootpath, 'ImageSets/Segmentation/val.txt')
        elif mode == 'trainval':
            txtfile = osp.join(rootpath, 'ImageSets/Segmentation/trainval.txt')
        else: assert(False)
        jpgpth = osp.join(rootpath, 'JPEGImages')
        lbpth = osp.join(rootpath, 'SegmentationClass')

        with open(txtfile, 'r') as fr:
            fns = fr.read().splitlines()
            self.len = len(fns)
            fns_img = ['{}.jpg'.format(el) for el in fns]
            self.fns_img = [osp.join(jpgpth, el) for el in fns_img]
            fns_lbs = ['{}.png'.format(el) for el in fns]
            self.fns_lbs = [osp.join(lbpth, el) for el in fns_lbs]

        if self.mode in ('train', 'trainval'):
            self.trans = transforms.Compose([
                transforms.Resize((321, 321)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        else:
            self.trans = transforms.Compose([
                transforms.Resize((321, 321)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])


    def __getitem__(self, idx):
        iname = self.fns_img[idx]
        lname = self.fns_lbs[idx]
        img = Image.open(iname)
        img = self.trans(img)
        label = Image.open(lname)
        label = np.array(label)
        label = cv2.resize(label, (321, 321),
                interpolation = cv2.INTER_NEAREST).astype(np.int)
        #  print(np.max(label))
        #  print(np.min(label))
        #  print(np.max(label))
        #  print(np.min(label))
        #  print(img.shape)
        #  print(label.shape)
        #  cv2.imshow('img', img)
        cv2.imshow('label', label)
        cv2.waitKey(0)
        return img, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    ds = PascalVoc('./data/VOCdevkit')
    ds[15]

