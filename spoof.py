import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.transform import resize
import pickle
from utils import get_lbp_grids

def get_features(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = resize(img, (64,64), preserve_range=True).astype(np.uint8)

    lbp81 = local_binary_pattern(img, 8, 1, 'uniform')
    hist_grids = [
        np.histogram(grid, 9, (0, 9))[0]
        for grid in get_lbp_grids(lbp81)
    ]
    hist81 = np.concatenate(hist_grids)

    lbp82 = local_binary_pattern(img, 8, 2, 'uniform')
    hist82 = np.histogram(lbp82, 9, (0, 9))[0]

    lbp16 = local_binary_pattern(img, 16, 2, 'uniform')
    hist16 = np.histogram(lbp16, 17, (0, 17))[0]

    hist = np.concatenate([hist81, hist82, hist16])
    return hist

with open('model/anti_spoof_svc', 'rb') as f:
    SVC = pickle.load(f)

def isreal(img):
    features = get_features(img)
    return SVC.predict([features])[0]