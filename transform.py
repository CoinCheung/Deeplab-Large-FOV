#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import random


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im = im, lb = lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(im = im.crop(crop),
                    lb = lb.crop(crop))




class HorizontalFlip(object):
    def __init__(self, p = 0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )

if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
    #  img.show('img')
    #  lb.show('lb')
    im_lb = dict(im=img, lb=lb)
    #  im_fl_lb = flip(im_lb)
    im_fl_lb = crop(im_lb)
    #  im_fl_lb['im'].show()
    #  im_fl_lb['im'].close()
    im_fl_lb['lb'].show()
    #  im_fl_lb['lb'].close()




