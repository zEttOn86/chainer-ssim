# coding:utf-8
import os, sys, time
import chainer
import chainer.functions as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from chainer_ssim.structural_similarity2d_loss import structural_similarity2d_loss

class OptimizeEinstein(chainer.Chain):
    def __init__(self, img, window_size=11, size_average=True):
        super(OptimizeEinstein, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        with self.init_scope():
            self.img = chainer.Parameter(initializer=img)

    def __call__(self, x):
        (_, channel, _, _) = x.shape

        return structural_similarity2d_loss(x, self.img, self.window_size, self.size_average)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = "{}/image".format(base_dir)

    print("----- Read image -----")
    npImg1 = cv2.imread("{}/einstein.png".format(image_dir))
    plt.figure()
    plt.imshow(npImg1)
    plt.title("einstein")
    plt.show()

    img1 = (np.rollaxis(npImg1, 2)[None,...]/255.0).astype(np.float32)
    img2 = np.random.rand(img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]).astype(np.float32)
    plt.imshow(np.squeeze(img2, axis=0).transpose(1,2,0))
    plt.title("random image")
    plt.show()

    print("----- Start optimization -----")
    ssim_loss = OptimizeEinstein(img=img2)
    ssim_loss.to_gpu()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(ssim_loss)
    xp = ssim_loss.xp
    img1 = chainer.Variable(xp.array(img1, dtype=xp.float32))

    ssim_value = 0.
    iter = 0
    while ssim_value < 0.95:
        ssim_out = -ssim_loss(img1)
        optimizer.target.cleargrads()
        ssim_out.backward()
        optimizer.update()
        ssim_value = - chainer.backends.cuda.to_cpu(ssim_out.array)
        if iter % 10 == 0:
            print(ssim_value)
            # plt.figure()
            # plt.imshow(chainer.backends.cuda.to_cpu(F.squeeze(ssim_loss.img, axis=0).data).transpose(1,2,0))
            # plt.text(10, 30, 'SSIM = {:.3f}'.format(ssim_value), fontsize=18, color="white")
            # plt.title("random image")
            # plt.show()
        # print(ssim_value)
        iter += 1

    plt.figure()
    plt.imshow(chainer.backends.cuda.to_cpu(F.squeeze(ssim_loss.img, axis=0).data).transpose(1,2,0))
    plt.text(10, 30, 'SSIM = {:.3f}'.format(ssim_value), fontsize=18, color="white")
    plt.title("random image")
    plt.show()
