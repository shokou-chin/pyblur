# -*- coding: utf-8 -*-
import numpy as np
import pickle
from PIL import Image
from scipy.signal import convolve2d
import os.path

pickledPsfFilename =os.path.join(os.path.dirname( __file__),"psf.pkl")

with open(pickledPsfFilename, 'rb') as pklfile:
    psfDictionary = pickle.load(pklfile, encoding='latin1')


def PsfBlur(img, psfid):
    kernel = psfDictionary[psfid]
    r_ = img[:, :, 0]
    g_ = img[:, :, 1]
    b_ = img[:, :, 2]
    convolved_r = convolve2d(r_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    convolved_g = convolve2d(g_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    convolved_b = convolve2d(b_, kernel, mode='same', fillvalue=255.0).astype("uint8")
    return np.stack([convolved_r, convolved_g, convolved_b], axis=2)
    
def PsfBlur_random(img):
    psfid = np.random.randint(0, len(psfDictionary))
    return PsfBlur(img, psfid)
    
    
