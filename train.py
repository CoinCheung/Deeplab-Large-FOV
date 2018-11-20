#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import os.path as osp
import time
import sys
import logging
import numpy as np

from model import DeepLabLargeFOV
from pascal_voc import PascalVoc
from pascal_voc_aug import PascalVoc_Aug
from transform import RandomCrop
from optimizer import Optimizer
from logger import *


## TODO:
# 1. refactoring the codes to make files look neater
# move sources to directories: lib utils
# 2. use argparse, and config file
# 3. move F.interpolation to model.py


## hyper parameters
### VOC AUG parameters
#  ## model and loss
#  ignore_label = 255
#  n_classes = 21
#  ## dataset
#  dataset = 'PascalVoc_Aug'
#  datapth = './data/VOC_AUG/'
#  batchsize = 30
#  n_workers = 6
#  ## optimizer
#  warmup_start_lr = 1e-6
#  start_lr = 1e-3
#  warmup_iter = 1000
#  lr_steps = [5000, 7000]
#  lr_factor = 0.1
#  momentum = 0.9
#  weight_decay = 5e-4
#  ## training control
#  iter_num = 8000
#  log_iter = 20
#  ## logger path
#  alpha = 0.1
#  res_pth = './res/voc_aug'


### VOC parameters
## model and loss
ignore_label = 255
n_classes = 21
## dataset
dataset = 'PascalVoc'
datapth = './data/VOCdevkit/'
batchsize = 30
n_workers = 6
## optimizer
warmup_start_lr = 1e-6
start_lr = 1e-3
warmup_iter = 1000
lr_steps = [5000, 7000]
lr_factor = 0.1
momentum = 0.9
weight_decay = 5e-4
## training control
iter_num = 8000
log_iter = 20
alpha = 0.1
res_pth = './res/voc2012'


## setup
if not osp.exists(res_pth): os.makedirs(res_pth)
setup_logger(res_pth)
torch.multiprocessing.set_sharing_strategy('file_system')


def train():
    ## modules and losses
    logger.info('creating model and loss module')
    net = DeepLabLargeFOV(3, n_classes)
    net.train()
    net.cuda()
    net = nn.DataParallel(net)
    Loss = nn.CrossEntropyLoss(ignore_index = ignore_label)
    Loss.cuda()

    ## dataset
    logger.info('creating dataset and dataloader')
    ds = eval(dataset)(datapth, mode = 'train')
    dl = DataLoader(ds,
            batch_size = batchsize,
            shuffle = True,
            num_workers = 6,
            drop_last = True)

    ## optimizer
    logger.info('creating optimizer')
    optimizer = Optimizer(
            params = net.parameters(),
            warmup_start_lr = warmup_start_lr,
            warmup_iter = warmup_iter,
            start_lr = start_lr,
            lr_steps = lr_steps,
            lr_factor = lr_factor,
            momentum = momentum,
            weight_decay = weight_decay)

    ## train loop
    loss_avg = []
    st = time.time()
    diter = iter(dl)
    logger.info('start training')
    for it in range(iter_num):
        try:
            im, lb = next(diter)
            if not im.size()[0] == batchsize: continue
        except StopIteration:
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()

        #  lam = np.random.beta(alpha, alpha)
        #  idx = torch.randperm(batchsize)
        #  mix_im = im * lam + (1. - lam) * im[idx, :]
        #  mix_lb = lb[idx, :]
        #  optimizer.zero_grad()
        #  out = net(mix_im)
        #  out = F.interpolate(out, im.size()[2:], mode = 'bilinear')
        #  lb = torch.squeeze(lb)
        #  mix_lb = torch.squeeze(mix_lb)
        #  loss = lam * Loss(out, lb) + (1. - lam) * Loss(out, mix_lb)
        #  loss.backward()
        #  optimizer.step()

        optimizer.zero_grad()
        out = net(im)
        out = F.interpolate(out, im.size()[2:], mode = 'bilinear')
        lb = torch.squeeze(lb)
        loss = Loss(out, lb)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_avg.append(loss)
        ## log message
        if it % log_iter == 0 and not it == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            ed = time.time()
            t_int = ed - st
            lr = optimizer.get_lr()
            msg = 'iter: {}/{}, loss: {:3f}'.format(it, iter_num, loss_avg)
            msg = '{}, lr: {:4f}, time: {:3f}'.format(msg, lr, t_int)
            logger.info(msg)
            st = ed
            loss_avg = []

    ## dump model
    model_pth = osp.join(res_pth, 'model_final.pkl')
    torch.save(net.module.state_dict(), model_pth)
    logger.info('training done, model saved to: {}'.format(model_pth))



if __name__ == "__main__":
    train()
