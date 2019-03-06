#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
import cv2
import numpy as np

import lib.transform as T

'''
0: background
1: aeroplane
2: bicycle
3: bird
4: boat
5: bottle
6: bus
7: car
8: cat
9: chair
10: cow
11: diningtable
12: dog
13: horse
14: motorbike
15: person
16: pottedplant
17: sheep
18: sofa
19: train
20: tvmonitor
'''


class PascalVoc(Dataset):
    def __init__(self, cfg, mode='train', *args, **kwargs):
        super(PascalVoc, self).__init__(*args, **kwargs)
        self.cfg = cfg
        self.mode = mode
        rootpath = osp.join(cfg.datapth, 'VOCdevkit/VOC2012/')
        if not osp.exists(rootpath): assert(False)
        if not self.mode in ('train', 'val', 'trainval', 'test'): assert(False)
        if mode == 'train':
            txtfile = osp.join(rootpath, 'ImageSets/Segmentation/train.txt')
        elif mode == 'val':
            txtfile = osp.join(rootpath, 'ImageSets/Segmentation/val.txt')
        elif mode == 'trainval':
            txtfile = osp.join(rootpath, 'ImageSets/Segmentation/trainval.txt')
        elif mode == 'test':
            txtfile = osp.join(rootpath, 'ImageSets/Segmentation/test.txt')
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

        self.trans = T.Compose([
            T.RandomScale(cfg.train_scales),
            T.RandomCrop((cfg.crop_size, cfg.crop_size)),
            T.HorizontalFlip(),
            T.ColorJitter(
                brightness = cfg.color_brightness,
                contrast = cfg.color_contrast,
                saturation = cfg.color_saturation
                ),
            ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __getitem__(self, idx):
        iname = self.fns_img[idx]
        lname = self.fns_lbs[idx]
        img = Image.open(iname)
        label = Image.open(lname)
        im_lb = dict(im = img, lb = label)
        if self.mode in ('train', 'trainval'):
            im_lb = self.trans(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        if self.mode == 'val':
            img = torch.unsqueeze(img, 0)

        return img, label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config.pascal_voc_2012_multi_scale import cfg


    ds = PascalVoc(cfg, mode='val')
    im, lb = ds[110]
    print(type(lb))
    print(lb.shape)
    dl = DataLoader(ds,
                    batch_size = 20,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)

    for im, label in ds:
        if not im.size() == (20, 3, 513, 513):
            print(im.size())
        if not label.shape == (20, 1, 513, 513):
            print(label.shape)
        label[label==255] = 0
        if label.max() > 20:
            print(label.max())
    print(len(ds))
