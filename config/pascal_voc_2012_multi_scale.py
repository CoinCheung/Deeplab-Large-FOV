#!/usr/bin/python
# -*- encoding: utf-8 -*-

#  import sys
#  print(sys.path)
from utils.AttrDict import AttrDict


cfg = AttrDict(
    ## model and loss
    ignore_label = 255,
    n_classes = 21,
    ## dataset
    dataset = 'PascalVoc',
    datapth = './data/',
    crop_size = 321,
    batchsize = 16,
    n_workers = 4,
    ## optimizer
    warmup_iter = 1000,
    warmup_start_lr = 1e-6,
    start_lr = 1e-3,
    iter_num = 7500,
    power = 0.9,
    momentum = 0.9,
    weight_decay = 5e-4,
    ## training control
    train_scales = (0.75, 1, 1.25, 1.5, 1.75, 2.),
    color_brightness = 0.5,
    color_contrast = 0.5,
    color_saturation = 0.5,
    log_iter = 20,
    use_mixup = False,
    alpha = 0.1,
    res_pth = './res/voc2012',
    ## test
    test_after_train = True,
    test_scales = (0.5, 0.75, 1, 1.25, 1.5, 1.75),
    flip = True,
    use_crf = False,
)
