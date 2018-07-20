import numpy as np
from PIL import Image
from scipy.signal import convolve2d

boxKernelDims = [3,5,7,9]

def BoxBlur_random(img):
    kernelidx = np.random.randint(0, len(boxKernelDims))    
    kerneldim = boxKernelDims[kernelidx]
    return BoxBlur(img, kerneldim)

def BoxBlur(img, dim):
    kernel = BoxKernel(dim)
    r_ = img[:, :, 0]
    g_ = img[:, :, 1]
    b_ = img[:, :, 2]
    convolved_r = convolve2d(r_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    convolved_g = convolve2d(g_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    convolved_b = convolve2d(b_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    return np.stack([convolved_r, convolved_g, convolved_b], axis=2)

def BoxKernel(dim):
    kernelwidth = dim
    kernel = np.ones((kernelwidth, kernelwidth), dtype=np.float32)        
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel