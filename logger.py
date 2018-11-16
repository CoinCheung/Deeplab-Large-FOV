#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import sys
import logging


logger = Logger()


class Logger(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def setup_logger(self, logpth):
        logfile = 'deeplab_lfov-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
        logfile = osp.join(logpth, logfile)
        FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)


    def get_logger(self):
        self.logger.addHandler(logging.StreamHandler())
        return self.logger

    def into(self, msg):
        self.logger.info(msg)

