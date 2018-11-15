#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch.optim as optim
import logging


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Optimizer(object):
    def __init__(self,
            params,
            warmup_start_lr,
            warmup_iter,
            start_lr,
            lr_step_size,
            lr_factor,
            momentum,
            weight_decay,
            *args, **kwargs):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_iter = warmup_iter
        self.start_lr = start_lr
        self.lr_step_size = lr_step_size
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
        if self.iter % self.lr_step_size == 0 and self.iter > self.warmup_iter:
            #  logger.info('===> learning rate is set to: {}'.format(self.lr))
            print('===> learning rate is set to: {}'.format(self.lr))
        lr = self.get_lr()
        for pg in self.optim.param_groups:
            pg['lr'] = lr
        self.optim.defaults['lr'] = lr
        self.optim.step()
        self.iter += 1


    def get_lr(self):
        if self.iter <= self.warmup_iter:
            self.lr = self.warmup_start_lr * self.warmup_factor ** self.iter
        elif self.iter % self.lr_step_size == 0:
            index = self.iter / self.lr_step_size
            self.lr = self.start_lr * self.lr_factor ** index
        return self.lr

    def zero_grad(self):
        self.optim.zero_grad()



if __name__ == '__main__':
    import torchvision
    net = torchvision.models.vgg16()

    warmup_start_lr = 1e-6
    warmup_iter = 10
    start_lr = 1e-3
    lr_step_size = 20
    lr_factor = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    optimizer = Optimizer(
            params = net.parameters(),
            warmup_start_lr = warmup_start_lr,
            warmup_iter = warmup_iter,
            start_lr = start_lr,
            lr_step_size = lr_step_size,
            lr_factor = lr_factor,
            momentum = momentum,
            weight_decay = weight_decay
            )
    for i in range(8000):
        if i <= 30:
            print(i, ": ", optimizer.get_lr())
            optimizer.step()


