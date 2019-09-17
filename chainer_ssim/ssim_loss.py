#coding:utf-8
import chainer
import chainer.functions as F
import numpy as np
from math import exp

def gaussian(window_size, sigma, xp):
    gauss = chainer.Variable(xp.array([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], dtype=xp.float32))
    # print(gauss)
    return gauss/F.sum(gauss)

def create_2d_window(window_size, channel, xp):
    _1D_window = F.reshape(gaussian(window_size, 1.5, xp), (-1, 1))

    _2D_window = F.reshape(F.tensordot(_1D_window, _1D_window.transpose(), axes=1), (1, 1, -1, window_size))
    window = F.repeat(_2D_window, channel, axis=0)

    return window

def _2d_ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.convolution_2d(img1, window, pad=window_size//2, groups=channel)
    mu2 = F.convolution_2d(img2, window, pad=window_size//2, groups=channel)

    mu1_sq = F.square(mu1)
    mu2_sq = F.square(mu2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.convolution_2d(img1*img1, window, pad=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.convolution_2d(img2*img2, window, pad=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.convolution_2d(img1*img2, window, pad=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return F.mean(ssim_map)
    return NotImplementedError()

def calc_2d_ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.shape
    xp = chainer.backends.cuda.get_array_module(img1)
    window = create_2d_window(window_size, channel, xp)

    return _2d_ssim(img1, img2, window, window_size, channel, size_average)
