#!/usr/bin/python
# -*- encoding: utf-8 -*-


from pydensecrf import densecrf
import numpy as np


def crf(img, scores):
    h, w, _ = img.shape

