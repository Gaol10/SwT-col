import numpy as np
import cv2

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""

    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def read_img_grey(filename):
    img = cv2.imread(filename, 3)
    img = cv2.cvtColor(cv2.resize(img, (300, 300)), cv2.COLOR_BGR2Lab)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img = img[:, :, 0]
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, img, img), axis=2)
    return img.astype('float32') / 255.0


def read_img(filename):
    img1 = cv2.imread(filename, 3)
    img = cv2.cvtColor(cv2.resize(img1, (300, 300)), cv2.COLOR_BGR2Lab)
    # img = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    return img.astype('float32') / 255.0


def write_img(filename, img):
    # img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
    cv2.imwrite(filename, img)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()
