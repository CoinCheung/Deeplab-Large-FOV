#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torchvision
import torch
import torch.nn as nn


class DeepLabLargeFOV(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super(DeepLabLargeFOV, self).__init__(*args, **kwargs)
        vgg16 = torchvision.models.vgg16()

        layers = []
        layers.append(nn.Conv2d(in_dim, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))

        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))
        self.features = nn.Sequential(*layers)

        classifier = []
        classifier.append(nn.AvgPool2d(3, stride = 1, padding = 1))
        classifier.append(nn.Conv2d(512,
            1024,
            kernel_size = 3,
            stride = 1,
            padding = 12,
            dilation = 12))
        classifier.append(nn.ReLU(inplace = True))
        classifier.append(nn.Conv2d(1024, 1024, kernel_size = 1, stride = 1, padding = 0))
        classifier.append(nn.ReLU(inplace = True))
        classifier.append(nn.Dropout(p = 0.5))
        classifier.append(nn.Conv2d(1024, out_dim, kernel_size = 1))
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = DeepLabLargeFOV(3, 10)
    print(net)
    in_ten = torch.randn(1, 3, 224, 224)
    out = net(in_ten)
    print(out.size())

    in_ten = torch.randn(1, 3, 64, 64)
    mod = nn.Conv2d(3,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2)
    out = mod(in_ten)
    print(out.shape)

