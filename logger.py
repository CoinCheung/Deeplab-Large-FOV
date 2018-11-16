#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import sys
import logging


def setup_logger(logpth):
    logfile = 'deeplab_lfov-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)


def get_logger():
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger

logger = get_logger()
