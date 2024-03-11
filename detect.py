import cv2 as cv
import numpy as np
from numpy.linalg import norm
from utils import get_center, get_midpoint

FACE_CASCADE = cv.CascadeClassifier('model/haarcascade_face.xml')
EYE_CASCADE = cv.CascadeClassifier('model/haarcascade_eye.xml')

def detect_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    return FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

def detect_eye(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    return EYE_CASCADE.detectMultiScale(gray)

def align_face(face, eyes):
    left_eye, right_eye = eyes
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye

    center_left = xleft, yleft = get_center(*left_eye)
    center_right = xright, yright = get_center(*right_eye)

    diagon = norm(np.array(center_left) - np.array(center_right))
    height = yleft - yright
    angle = -np.arcsin(height/diagon) * 180/np.pi

    xm = get_midpoint(xleft, xright)
    ym = get_midpoint(yleft, yright)
    anchor = xm, ym

    M = cv.getRotationMatrix2D(anchor, angle, 1.1)
    face = cv.warpAffine(face, M, face.shape[:-1])
    return face