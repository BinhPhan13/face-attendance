import torch
from torchvision.io import read_image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import cv2 as cv
import numpy as np
import glob
import os
import re
from utils import dist, imread

MODEL = InceptionResnetV1(classify=False, pretrained="casia-webface")
MODEL.eval()
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
    transforms.ToTensor(),
])
def get_embed(face:np.ndarray):
    if face.dtype == np.float32:
        face = (face * 255).astype(np.uint8)

    face = TRANSFORM(face)
    face = face.float().unsqueeze(0)
    with torch.no_grad():
        return MODEL(face).squeeze(0).numpy()

try: NAMES = np.load('names.npy').tolist()
except: NAMES = []

try: EMBEDS = np.load('embeds.npy')
except: EMBEDS = None

THRESHOLD = 1.0
def add_data(img):
    global EMBEDS

    embed = get_embed(img)
    if EMBEDS is None: EMBEDS = [embed]
    else:
        distance = dist(embed, EMBEDS)
        min_idx = np.argmin(distance)
        if distance[min_idx] < THRESHOLD:
            print(f"Similar to {NAMES[min_idx]} !")
            return
        else:
            EMBEDS = EMBEDS.tolist()
            EMBEDS.append(embed)

    name = input("Username: ")
    NAMES.append(name)

    np.save('embeds.npy', EMBEDS)
    np.save('names.npy', NAMES)

def recognize(img):
    if EMBEDS is None: return
    embed = get_embed(img)

    distance = dist(embed, EMBEDS)
    min_idx = np.argmin(distance)
    if distance[min_idx] < THRESHOLD:
        return NAMES[min_idx]

if __name__ == '__main__':
    breal = imread('images/binh-real.jpg')
    bfake = imread('images/binh-fake.jpg')
    treal = imread('images/trung-real.jpg')
    tfake = imread('images/trung-fake.jpg')

    ebreal = get_embed(breal)
    ebfake = get_embed(bfake)
    etreal = get_embed(treal)
    etfake = get_embed(tfake)

    print(dist(ebfake, etfake))