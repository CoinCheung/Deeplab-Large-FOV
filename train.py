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
from lib.loss import *
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
    cfg_str = json.dumps(cfg, ensure_ascii=False, indent=2)
    if not osp.exists(cfg.res_pth): os.makedirs(cfg.res_pth)
    setup_logger(cfg.res_pth)
    logger = logging.getLogger(__name__)
    logger.info(cfg_str)

    ## modules and losses
    logger.info('creating model and loss module')
    net = DeepLabLargeFOV(3, cfg.n_classes)
    net.train()
    net.cuda()
    if not torch.cuda.device_count() == 0: net = nn.DataParallel(net)
    n_min = (cfg.crop_size**2) * cfg.batchsize // 16
    criteria = OhemCELoss(0.7, n_min)
    criteria.cuda()

    ## dataset
    logger.info('creating dataset and dataloader')
    ds = eval(cfg.dataset)(cfg, mode='train')
    dl = DataLoader(ds,
            batch_size = cfg.batchsize,
            shuffle = True,
            num_workers = cfg.n_workers,
            drop_last = True)

    ## optimizer
    logger.info('creating optimizer')
    optimizer = Optimizer(
            params = net.parameters(),
            warmup_start_lr = cfg.warmup_start_lr,
            warmup_steps = cfg.warmup_iter,
            lr0 = cfg.start_lr,
            max_iter = cfg.iter_num,
            momentum = cfg.momentum,
            wd = cfg.weight_decay,
            power = cfg.power)

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
        out = net(im)
        lb = torch.squeeze(lb)
        loss = criteria(out, lb)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_avg.append(loss)
        ## log message
        if it%cfg.log_iter==0 and not it==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            ed = time.time()
            t_int = ed - st
            lr = optimizer.get_lr()
            msg = 'iter: {}/{}, loss: {:.4f}'.format(it, cfg.iter_num, loss_avg)
            msg = '{}, lr: {:4f}, time: {:.4f}'.format(msg, lr, t_int)
            logger.info(msg)
            st = ed
            loss_avg = []

    ## dump model
    model_pth = osp.join(cfg.res_pth, 'model_final.pkl')
    net.cpu()
    state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state_dict, model_pth)
    logger.info('training done, model saved to: {}'.format(model_pth))

    ## test after train
    if cfg.test_after_train:
        net.cuda()
        mIOU = eval_model(net, cfg)
        logger.info('iou in whole is: {}'.format(mIOU))


if __name__ == "__main__":
    args = get_args()
    train(args)
