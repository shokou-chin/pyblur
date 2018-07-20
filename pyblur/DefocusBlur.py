# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import convolve2d
from skimage.draw import circle

defocusKernelDims = [3,5,7,9]

def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))    
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)

def DefocusBlur(img, dim):
    kernel = DiskKernel(dim)
    r_ = img[:,:,0]
    g_ = img[:,:,1]
    b_ = img[:,:,2]
    convolved_r = convolve2d(r_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    convolved_g = convolve2d(g_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    convolved_b = convolve2d(b_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    return np.stack([convolved_r, convolved_g, convolved_b], axis=2)


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = dim // 2
    circleRadius = circleCenterCoord +1
    
    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr,cc]=1
    
    if(dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)
        
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0,0] = 0
    kernel[0,kernelwidth-1]=0
    kernel[kernelwidth-1,0]=0
    kernel[kernelwidth-1, kernelwidth-1] =0 
    return kernel