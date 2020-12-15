import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the log directory')
args = parser.parse_args()

logdir = args.logdir

fname = 'data/scenenn/metadata/test.txt'
scenes = [line.strip() for line in open(fname, 'r')]

root = '/run/media/hieu/data/scenenn/'
step = 30
num_samples = 1000
image_size = 64
half_patch_size = 32
threshold = 0.03
cx = 320.0
cy = 240.0
fx = 585.0
fy = 585.0


sift = cv.xfeatures2d.SIFT_create()
surf = cv.xfeatures2d.SURF_create()
orb = cv.ORB_create()


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


np.random.seed(42)
for scn in scenes[:5]:
    traj = os.path.join(root, scn, 'trajectory.log')
    traj = load_log(traj)[::step]

    features0 = []
    features1 = []
    pairs = list(zip(traj[:-1], traj[1:]))
    for start, end in tqdm(pairs, desc='[{}]'.format(scn)):
        fname = os.path.join(root, scn, 'depth', '{:06d}.png'.format(start[0]))
        depth0 = np.array(Image.open(fname)) * 0.001
        fname = os.path.join(root, scn, 'color', '{:06d}.png'.format(start[0]))
        color0 = cv.imread(fname, cv.IMREAD_GRAYSCALE)
        T0 = start[2]

        fname = os.path.join(root, scn, 'depth', '{:06d}.png'.format(end[0]))
        depth1 = np.array(Image.open(fname)) * 0.001
        fname = os.path.join(root, scn, 'color', '{:06d}.png'.format(end[0]))
        color1 = cv.imread(fname, cv.IMREAD_GRAYSCALE)
        T1 = np.linalg.inv(end[2])

        keypts0 = []
        keypts1 = []
        while len(keypts0) < num_samples:
            u0 = np.random.choice(depth0.shape[1])
            v0 = np.random.choice(depth0.shape[0])
            if depth0[v0, u0] <= 0.0: continue

            z = depth0[v0, u0]
            x = (u0 - cx) * z / fx
            y = (v0 - cy) * z / fy
            p0 = np.array([x, y, z])
            q = T0[0:3, 0:3] @ p0 + T0[0:3, 3]

            p1 = T1[0:3, 0:3] @ q + T1[0:3, 3]
            u1 = int(p1[0] * fx / p1[2] + cx)
            v1 = int(p1[1] * fy / p1[2] + cy)
            if u1 < 0 or u1 >= depth1.shape[1]: continue
            if v1 < 0 or v1 >= depth1.shape[0]: continue
            if depth1[v1, u1] <= 0.0: continue
            if abs(depth1[v1, u1] - p1[2]) > threshold: continue

            kp0 = (u0, v0)
            keypts0.append(kp0)
            kp1 = (u1, v1)
            keypts1.append(kp1)

        keypts0 = cv.KeyPoint_convert(keypts0)
        keypts1 = cv.KeyPoint_convert(keypts1)
        _, desc0 = orb.compute(color0, keypts0)
        _, desc1 = orb.compute(color1, keypts1)

        features0.append(desc0)
        features1.append(desc1)

    features0 = np.stack(features0, axis=0)
    features1 = np.stack(features1, axis=0)
    results = {'src': features0, 'dst': features1}
    fname = os.path.join(logdir, '{}.npz'.format(scn))
    print('> Saving descriptors to {}...'.format(fname))
    np.savez(fname, **results)
