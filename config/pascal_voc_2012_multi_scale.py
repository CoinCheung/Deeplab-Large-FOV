#!/usr/bin/python
# -*- encoding: utf-8 -*-

from utils.AttrDict import AttrDict


cfg = AttrDict(
    ## model and loss
    ignore_label = 255,
    n_classes = 21,
    ## dataset
    dataset = 'PascalVoc',
    datapth = './data/VOCdevkit/',
    crop_size = (321, 321),
    batchsize = 30,
    n_workers = 6,
    ## optimizer
    warmup_iter = 1000,
    warmup_start_lr = 1e-6,
    start_lr = 1e-3,
    lr_steps = [4000, 5500],
    iter_num = 6000,
    lr_factor = 0.1,
    momentum = 0.9,
    weight_decay = 5e-4,
    ## training control
    scales = (1, ),
    log_iter = 20,
    use_mixup = False,
    alpha = 0.1,
    res_pth = './res/voc2012',
    ## test
    test_after_train = True,
    use_crf = False,
)
