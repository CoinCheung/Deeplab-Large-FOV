#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import os.path as osp
import time
import sys
import logging

from model import DeepLabLargeFOV
from pascal_voc import PascalVoc
from transform import RandomCrop


## logging
if not osp.exists('./res/'): os.makedirs('./res/')
logfile = 'deeplab_lfov-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
logfile = osp.join('res', logfile)
FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def train():
    ## modules and losses
    ignore_label = 255
    net = DeepLabLargeFOV(3, 21)
    net.train()
    net.cuda()
    net = nn.DataParallel(net)
    Loss = nn.CrossEntropyLoss(ignore_index = ignore_label)
    Loss.cuda()

    ## dataset
    batchsize = 30
    ds = PascalVoc('./data/VOCdevkit/', mode = 'train', down_factor = 8)
    dl = DataLoader(ds,
            batch_size = batchsize,
            shuffle = True,
            num_workers = 4,
            drop_last = True)

    ## optimizer
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    lr_step_size = 2000
    lr_factor = 0.1
    optimizer = optim.SGD(net.parameters(),
            lr = lr,
            momentum = momentum,
            weight_decay = weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
            step_size = lr_step_size,
            gamma = lr_factor)

    ## train loop
    iter_num = 8000
    diter = iter(dl)
    for it in range(iter_num):
        try:
            im, lb = next(diter)
            if not im.size()[0] == batchsize: continue
        except StopIteration:
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()

        print(lb.dtype)
        print(lb.size())
        optimizer.zero_grad()
        out = net(im)
        print(out.shape)
        size = out.size()[2:]
        #  lb = F.interpolate(lb, size, mode = 'nearest')
        lb = F.upsample(lb, size, mode = 'nearest')
        loss = Loss(out, lb)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print(im.shape)
        print(lb.shape)
        print(lb.dtype)
        print(loss.detach().cpu().numpy())

        break


if __name__ == "__main__":
    train()
