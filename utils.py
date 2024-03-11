import numpy as np
import cv2 as cv

def get_center(x, y, w, h):
    return x + w//2, y + h//2

def get_midpoint(a, b):
    return int((a + b) / 2)

def get_lbp_grids(img):
    assert img.shape == (64,64)
    return [
        img[:26, :26],
        img[:26, 19:45],
        img[:26, 38:],

        img[19:45, :26],
        img[19:45, 19:45],
        img[19:45, 38:],

        img[38:, :26],
        img[38:, 19:45],
        img[38:, 38:],
    ]

def dist(x1:np.ndarray, x2:np.ndarray):
    if len(x2.shape) == 1: x2 = x2[None, ...]
    return ((x1 - x2)**2).sum(axis=1)

def imread(filepath):
    img = cv.imread(filepath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img
