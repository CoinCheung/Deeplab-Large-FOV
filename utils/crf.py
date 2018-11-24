#!/usr/bin/python
# -*- encoding: utf-8 -*-


from pydensecrf import densecrf as dcrf
import numpy as np


def crf(img, scores):
    '''
    scores: prob numpy array after softmax with shape (_, C, H, W)
    img: image of shape (H, W, C)
    CRF parameters: bi_w = 4, bi_xy_std = 121, bi_rgb_std = 5, pos_w = 3, pos_xy_std = 3
    '''
    pos_w = 3
    pos_xy_std = 3
    bi_w = 4
    bi_xy_std = 121
    bi_rgb_std = 5

    scores = np.ascontiguousarray(scores[0])
    img = np.ascontiguousarray(img)
    n_classes, h, w = scores.shape

    d = dcrf.DenseCRF2D(w, h, n_classes)
    U = -np.log(scores)
    U = U.reshape((n_classes, -1))
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)
    d.addPairwiseBilateral(sxy=bi_xy_std, srgb=bi_rgb_std, rgbim=img, compat=bi_w)

    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
