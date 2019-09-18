# coding:utf-8
import os, sys, time
import chainer
import chainer.functions as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk

from chainer_ssim.structural_similarity3d_loss import structural_similarity3d_loss

class OptimizeBrain(chainer.Chain):
    def __init__(self, img, window_size=11, size_average=True):
        super(OptimizeBrain, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        with self.init_scope():
            self.img = chainer.Parameter(initializer=img)

    def __call__(self, x):
        return structural_similarity3d_loss(x, self.img, self.window_size, self.size_average)

def read_image(path, numpyFlag=True):
    """
    This function use sitk
    path : Meta data path
    ex. /hogehoge.mhd
    numpyFlag : Return numpyArray or sitkArray
    return : numpyArray(numpyFlag=True)
    Note ex.3D :numpyArray axis=[z,y,x], sitkArray axis=(z,y,x)
    """
    img = sitk.ReadImage(path)
    if not numpyFlag:
        return img

    nda = sitk.GetArrayFromImage(img) #(img(x,y,z)->numpyArray(z,y,x))
    return nda

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = "{}/image".format(base_dir)

    print("----- Read image -----")
    npImg1 = read_image("{}/brain.nii.gz".format(image_dir))
    D, H, W = npImg1.shape
    plt.figure()
    plt.imshow(npImg1[D//2], cmap="gray")
    plt.title("brain")
    plt.show()

    img1 = min_max(npImg1[None,None,...]).astype(np.float32)
    img2 = np.random.rand(*img1.shape).astype(np.float32)
    batchsize, ch, D, H, W = img2.shape
    plt.imshow(img2[0,0,D//2,:,:], cmap="gray")
    plt.title("random image")
    plt.show()

    print("----- Start optimization -----")
    ssim_loss = OptimizeBrain(img=img2)
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
    plt.imshow(chainer.backends.cuda.to_cpu(ssim_loss.img.data[0,0,D//2,:,:]), cmap="gray")
    plt.text(10, 30, 'SSIM = {:.3f}'.format(ssim_value), fontsize=18, color="white")
    plt.title("random image")
    plt.show()
