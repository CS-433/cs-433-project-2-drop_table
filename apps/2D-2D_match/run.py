import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import KDTree


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the log directory')
args = parser.parse_args()

logdir = args.logdir

fname = 'data/3dmatch/metadata/test.txt'
scenes = [line.strip() for line in open(fname, 'r')]

root = 'data/'
step = 10
num_samples = 1000
image_size = 64
half_patch_size = 32
threshold = 0.03
cx = 320.0
cy = 240.0
fx = 585.0
fy = 585.0


def load_log(fname):
    fp = open(fname, 'r')
    result = []
    while True:
        line = fp.readline()
        if not line: break
        i, j, n = [int(x) for x in line.split(' ')]
        transformation = np.loadtxt(fp, max_rows=4)
        result.append((i, j, transformation))
    fp.close()
    return result

bf = cv.BFMatcher()

correct = 0
total = 0
for scn in scenes:
    fname = os.path.join(logdir, '{}.npz'.format(scn))
    if not os.path.exists(fname): continue
    print(fname)
    data = np.load(fname)
    src = data['src']
    dst = data['dst']

    for i in tqdm(range(src.shape[0]), desc='[{}]'.format(scn)):
        matches = bf.knnMatch(src[i], dst[i], k=1)
        for m in matches:
            for k in m:
                if k.queryIdx == k.trainIdx:
                    correct += 1
                    break
        total += src.shape[1]

print(step, correct / total)
