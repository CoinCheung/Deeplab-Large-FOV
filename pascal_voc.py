#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
import cv2
import numpy as np

from transform import HorizontalFlip, RandomCrop



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

        self.random_crop = RandomCrop((321, 321))
        self.horizon_flip = HorizontalFlip()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __getitem__(self, idx):
        iname = self.fns_img[idx]
        lname = self.fns_lbs[idx]
        img = Image.open(iname)
        label = Image.open(lname)
        if self.mode in ('train', 'trainval'):
            im_lb = dict(im = img, lb = label)
            im_lb = self.random_crop(im_lb)
            im_lb = self.horizon_flip(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.trans(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]

        #  import cv2
        #  lbb = label.astype(np.uint8).reshape(321, 321)
        #  print(lbb.shape)
        #  cv2.imshow('before', lbb)
        #  cv2.waitKey(0)
        #  print('before')
        #  lbb[lbb == 255] = 2
        #  lbb = lbb * 10
        #  print(np.max(lbb))
        #  print(np.min(lbb))

        return img, label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = PascalVoc('./data/VOCdevkit')
    im, lb = ds[110]
    print(type(lb))
    print(lb.shape)
    dl = DataLoader(ds,
                    batch_size = 20,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    #  print(im.shape)
    #  print(im.size())

    for im, label in dl:
        if not im.size() == (20, 3, 321, 321):
            print(im.size())
        if not label.size() == (20, 321, 321):
            print(label.size)
        #  label[label == 255] = 3
        #  print(torch.max(label))
        #  print(torch.min(label))
    print(len(ds))
