#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

from transform import HorizontalFlip, RandomCrop



class PascalVoc_Aug(Dataset):
    def __init__(self, root_pth, mode = 'train', *args, **kwargs):
        self.impth = osp.join(root_pth, 'images')
        self.lbpth = osp.join(root_pth, 'labels')
        self.mode = mode
        if mode == 'train':
            txtpth = osp.join(root_pth, 'train.txt')
        elif mode == 'val':
            txtpth = osp.join(root_pth, 'val.txt')
        else:
            raise(ValueError)

        with open(txtpth, 'r') as fr:
            fns = fr.read().splitlines()
            self.len = len(fns)
        self.imgs = ['{}.jpg'.format(el) for el in fns]
        self.imgs = [osp.join(self.impth, el) for el in self.imgs]
        self.lbs = ['{}.png'.format(el) for el in fns]
        self.lbs = [osp.join(self.lbpth, el) for el in self.lbs]

        self.random_crop = RandomCrop((321, 321))
        self.horizon_flip = HorizontalFlip()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __getitem__(self, idx):
        iname = self.imgs[idx]
        lname = self.lbs[idx]
        img = Image.open(iname)
        label = Image.open(lname)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.random_crop(im_lb)
            im_lb = self.horizon_flip(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            img = self.trans(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]

        return img, label


    def __len__(self):
        return self.len




if __name__ == '__main__':
    ds = PascalVoc_Aug('data/VOC_AUG/')
    im, lb = ds[11]
    for i, (im, lb) in enumerate(ds):
        print(im.size())
        print(lb.shape)
        if i == 10: break
