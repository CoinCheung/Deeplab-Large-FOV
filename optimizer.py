#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch.optim as optim
from logger import *


class Optimizer(object):
    def __init__(self,
            params,
            warmup_start_lr,
            warmup_iter,
            start_lr,
            lr_steps,
            lr_factor,
            momentum,
            weight_decay,
            *args, **kwargs):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_iter = warmup_iter
        self.start_lr = start_lr
        self.lr_steps = lr_steps
        self.lr_factor = lr_factor
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.optim = optim.SGD(
                params,
                lr = start_lr,
                momentum = momentum,
                weight_decay = weight_decay)

        self.iter = 0
        self.lr = warmup_start_lr
        self.warmup_factor = (start_lr / warmup_start_lr) ** (1. / warmup_iter)


    def step(self):
        lr = self.get_lr()
        if self.iter in self.lr_steps:
            logger.info('\t===> learning rate is set to: {}'.format(self.lr))
        for pg in self.optim.param_groups:
            pg['lr'] = lr
        self.optim.defaults['lr'] = lr
        self.optim.step()
        self.iter += 1


    def get_lr(self):
        if self.iter <= self.warmup_iter:
            self.lr = self.warmup_start_lr * self.warmup_factor ** self.iter
        else:
            for i, ms in enumerate(self.lr_steps):
                if ms == self.iter: self.lr = self.start_lr * (0.1 ** (i + 1))
        return self.lr

    def zero_grad(self):
        self.optim.zero_grad()



if __name__ == '__main__':
    import torchvision
    net = torchvision.models.vgg16()

    warmup_start_lr = 1e-6
    warmup_iter = 10
    start_lr = 1e-3
    lr_steps = [20, 40]
    lr_factor = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    optimizer = Optimizer(
            params = net.parameters(),
            warmup_start_lr = warmup_start_lr,
            warmup_iter = warmup_iter,
            start_lr = start_lr,
            lr_steps = lr_steps,
            lr_factor = lr_factor,
            momentum = momentum,
            weight_decay = weight_decay
            )
    for i in range(8000):
        if i <= 30:
            print(i, ": ", optimizer.get_lr())
            optimizer.step()


