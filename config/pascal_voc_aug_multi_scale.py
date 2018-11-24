#!/usr/bin/python
# -*- encoding: utf-8 -*-

from utils.AttrDict import AttrDict


cfg = AttrDict(
    ## model and loss
    ignore_label = 255,
    n_classes = 21,
    ## dataset
    dataset = 'PascalVoc_Aug',
    datapth = './data/VOC_AUG/',
    crop_size = (497, 497),
    batchsize = 20,
    n_workers = 6,
    ## optimizer
    warmup_iter = 1000,
    warmup_start_lr = 1e-6,
    start_lr = 1e-3,
    lr_steps = [12500, 17500],
    iter_num = 20000,
    lr_factor = 0.1,
    momentum = 0.9,
    weight_decay = 5e-4,
    ## training control
    scales = (0.5, 0.75, 1),
    log_iter = 20,
    use_mixup = False,
    alpha = 0.1,
    res_pth = './res/voc_aug',
    ## test
    test_after_train = True,
    use_crf = False,
)
