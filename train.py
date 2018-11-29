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
import argparse
import importlib
import json

from lib.model import DeepLabLargeFOV
from lib.pascal_voc import PascalVoc
from lib.pascal_voc_aug import PascalVoc_Aug
from lib.transform import RandomCrop
from lib.optimizer import Optimizer
from utils.logger import setup_logger
from evaluate import eval_model



torch.multiprocessing.set_sharing_strategy('file_system')

def get_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument(
            '--cfg',
            dest = 'cfg',
            type = str,
            default = 'config/pascal_voc_aug_multi_scale.py',
            help = 'config file for training'
            )
    return parser.parse_args()


def train(args):
    ## setup cfg and logger
    spec = importlib.util.spec_from_file_location('mod_cfg', args.cfg)
    mod_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_cfg)
    cfg = mod_cfg.cfg
    cfg_str = json.dumps(cfg, ensure_ascii = False, indent = 2)
    if not osp.exists(cfg.res_pth): os.makedirs(cfg.res_pth)
    setup_logger(cfg.res_pth)
    logger = logging.getLogger(__name__)
    logger.info(cfg_str)

    ## modules and losses
    logger.info('creating model and loss module')
    net = DeepLabLargeFOV(3, cfg.n_classes)
    net.train()
    net.cuda()
    net = nn.DataParallel(net)
    Loss = nn.CrossEntropyLoss(ignore_index = cfg.ignore_label)
    Loss.cuda()

    ## dataset
    logger.info('creating dataset and dataloader')
    ds = eval(cfg.dataset)(cfg.datapth, crop_size = cfg.crop_size,  mode = 'train')
    dl = DataLoader(ds,
            batch_size = cfg.batchsize,
            shuffle = True,
            num_workers = 6,
            drop_last = True)

    ## optimizer
    logger.info('creating optimizer')
    optimizer = Optimizer(
            params = net.parameters(),
            warmup_start_lr = cfg.warmup_start_lr,
            warmup_iter = cfg.warmup_iter,
            start_lr = cfg.start_lr,
            lr_steps = cfg.lr_steps,
            lr_factor = cfg.lr_factor,
            momentum = cfg.momentum,
            weight_decay = cfg.weight_decay)

    ## train loop
    loss_avg = []
    st = time.time()
    diter = iter(dl)
    logger.info('start training')
    for it in range(cfg.iter_num):
        try:
            im, lb = next(diter)
            if not im.size()[0] == cfg.batchsize: continue
        except StopIteration:
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()

        #  if use_mixup:
        #      lam = np.random.beta(alpha, alpha)
        #      idx = torch.randperm(batchsize)
        #      mix_im = im * lam + (1. - lam) * im[idx, :]
        #      mix_lb = lb[idx, :]
        #      optimizer.zero_grad()
        #      out = net(mix_im)
        #      out = F.interpolate(out, lb.size()[2:], mode = 'bilinear') # upsample to original size
        #      lb = torch.squeeze(lb)
        #      mix_lb = torch.squeeze(mix_lb)
        #      loss = lam * Loss(out, lb) + (1. - lam) * Loss(out, mix_lb)
        #      loss.backward()
        #      optimizer.step()
        #  else:
        #      optimizer.zero_grad()
        #      out = net(im)
        #      out = F.interpolate(out, lb.size()[2:], mode = 'bilinear') # upsample to original size
        #      lb = torch.squeeze(lb)
        #      loss = Loss(out, lb)
        #      loss.backward()
        #      optimizer.step()

        optimizer.zero_grad()
        H, W = im.size()[2:]
        for s in cfg.scales:
            h, w = int(H * s), int(W * s)
            im_s = F.interpolate(im, (h, w), mode = 'bilinear')
            out = net(im_s)
            out = F.interpolate(out, lb.size()[2:], mode = 'bilinear') # upsample to original size
            lb = torch.squeeze(lb)
            loss = Loss(out, lb)
            loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_avg.append(loss)
        ## log message
        if it % cfg.log_iter == 0 and not it == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            ed = time.time()
            t_int = ed - st
            lr = optimizer.get_lr()
            msg = 'iter: {}/{}, loss: {:3f}'.format(it, cfg.iter_num, loss_avg)
            msg = '{}, lr: {:4f}, time: {:3f}'.format(msg, lr, t_int)
            logger.info(msg)
            st = ed
            loss_avg = []

    ## dump model
    model_pth = osp.join(cfg.res_pth, 'model_final.pkl')
    torch.save(net.module.state_dict(), model_pth)
    logger.info('training done, model saved to: {}'.format(model_pth))

    ## test after train
    if cfg.test_after_train:
        mIOU = eval_model(net, cfg.use_crf)
        logger.info('iou in whole is: {}'.format(mIOU))


if __name__ == "__main__":
    args = get_args()
    train(args)
