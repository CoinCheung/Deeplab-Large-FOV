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

from model import DeepLabLargeFOV
from pascal_voc import PascalVoc
from transform import RandomCrop
from optimizer import Optimizer
#  from logger import *


## setup
if not osp.exists('./res/'): os.makedirs('./res/')
#  logger.setup('./res/')
logfile = 'deeplab_lfov-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
logfile = osp.join('./res/', logfile)
FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())



def resize_lb(out, lb):
    ## infer shape of the label tensor, and implement nearest interpolate
    H, W = lb.size()[2:]
    h, w = out.size()[2:]
    ih, iw = torch.linspace(0, H - 1, h).long(), torch.linspace(0, W - 1, w).long()
    lb = lb[:, :, ih[:, None], iw]
    lb = lb.contiguous().view(-1, h, w)
    return lb


def train():
    ## modules and losses
    logger.info('creating model and loss module')
    ignore_label = 255
    net = DeepLabLargeFOV(3, 21)
    net.train()
    net.cuda()
    net = nn.DataParallel(net)
    Loss = nn.CrossEntropyLoss(ignore_index = ignore_label)
    Loss.cuda()

    ## dataset
    logger.info('creating dataset and dataloader')
    batchsize = 30
    ds = PascalVoc('./data/VOCdevkit/', mode = 'train')
    dl = DataLoader(ds,
            batch_size = batchsize,
            shuffle = True,
            num_workers = 6,
            drop_last = True)

    ## optimizer
    logger.info('creating optimizer')
    warmup_start_lr = 1e-6
    warmup_iter = 1000
    start_lr = 1e-3
    lr_step_size = 2000
    lr_factor = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    optimizer = Optimizer(
            params = net.parameters(),
            warmup_start_lr = warmup_start_lr,
            warmup_iter = warmup_iter,
            start_lr = start_lr,
            lr_step_size = lr_step_size,
            lr_factor = lr_factor,
            momentum = momentum,
            weight_decay = weight_decay)

    ## train loop
    iter_num = 8000
    log_iter = 20
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

        #  print(im.size())
        #  print(lb.size())
        #  break
        #  lb = lb.squeeze(1).cuda()

        optimizer.zero_grad()
        out = net(im)
        out = F.interpolate(out, im.size()[2:], mode = 'bilinear')
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, 21)
        lb = lb.permute(0, 2, 3, 1).contiguous().view(-1,)
        #TODO: see if this interpolation is correct
        #  lb = resize_lb(out, lb)

        loss = Loss(out, lb)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_avg.append(loss)

        ## logger
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
    model_pth = './res/model.pkl'
    torch.save(net.module.state_dict(), model_pth)
    logger.info('training done, model saved to: {}'.format(model_pth))



if __name__ == "__main__":
    train()
