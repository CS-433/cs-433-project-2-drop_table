import os
import sys
import json
import argparse
from demo_superpoint import SuperPointFrontend
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append('.')
from models import *


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


def extract_color_patch(color, u, v):
    xmin = max(int(u) - half_patch_size, 0)
    xmax = min(int(u) + half_patch_size, color.shape[1])
    ymin = max(int(v) - half_patch_size, 0)
    ymax = min(int(v) + half_patch_size, color.shape[0])
    image = Image.fromarray(color[ymin:ymax, xmin:xmax])
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def extract_features(patches, model, batch_size=32, device='cuda'):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    features = []
    with torch.no_grad():
        for i, x in enumerate(batches):
            x = x.to(device)
            z = model.encode(x)
            z = z.transpose(0, 1)
            z = z.cpu().numpy()
            features.append(z)
    return np.concatenate(features, axis=1)


def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the log directory')
args = parser.parse_args()

logdir = args.logdir
config = os.path.join(logdir, 'config.json')
args = json.load(open(config))
device = args['device']

fname = os.path.join(logdir, 'model.pth')
print('==> Loading model from {}....'.format(fname))
model = PatchNetAutoencoder(
    args['embedding_size'],
    args['normalize'])
model.load_state_dict(torch.load(fname)['patchnet'])
model.to(device)
model.eval()

root = '/run/media/hieu/data/scenenn/'
step = 10
weights_path = 'superpoint_v1.pth'
nms_dist = 4
conf_thresh = 0.015
nn_thresh = 0.7
image_size = 64
half_patch_size = 32
threshold = 0.03
cuda = True
cx = 320.0
cy = 240.0
fx = 585.0
fy = 585.0

fe = SuperPointFrontend(weights_path=weights_path,
                        nms_dist=nms_dist,
                        conf_thresh=conf_thresh,
                        nn_thresh=nn_thresh,
                        cuda=cuda)
print('==> Successfully loaded pre-trained network.')

fname = 'data/scenenn/metadata/test.txt'
scenes = [line.strip() for line in open(fname, 'r')]

correct = 0
total = 0

for scn in scenes:
    traj = os.path.join(root, scn, 'trajectory.log')
    traj = load_log(traj)[::step]

    pairs = list(zip(traj[:-1], traj[1:]))
    for start, end in tqdm(pairs):
        fname = os.path.join(root, scn, 'depth', '{:06d}.png'.format(start[0]))
        depth0 = np.array(Image.open(fname)) * 0.001
        fname = os.path.join(root, scn, 'color', '{:06d}.png'.format(start[0]))
        color0 = np.array(Image.open(fname))
        gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
        gray0 = gray0.astype(np.float32) / 255.0
        T0 = start[2]
        pts0, desc0, _ = fe.run(gray0)

        fname = os.path.join(root, scn, 'depth', '{:06d}.png'.format(end[0]))
        depth1 = np.array(Image.open(fname)) * 0.001
        fname = os.path.join(root, scn, 'color', '{:06d}.png'.format(end[0]))
        color1 = np.array(Image.open(fname))
        gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
        gray1 = gray1.astype(np.float32) / 255.0
        T1 = np.linalg.inv(end[2])
        pts1, desc1, _ = fe.run(gray1)

        patches0 = []
        for i in range(pts0.shape[1]):
            patch = extract_color_patch(color0, pts0[0, i], pts0[1, i])
            patches0.append(patch)

        patches1 = []
        for i in range(pts1.shape[1]):
            patch = extract_color_patch(color1, pts1[0, i], pts1[1, i])
            patches1.append(patch)

        desc0 = extract_features(patches0, model, args['batch_size'], device)
        desc1 = extract_features(patches1, model, args['batch_size'], device)

        matches = nn_match_two_way(desc0, desc1, nn_thresh)

        for k in range(matches.shape[1]):
            i = int(matches[0, k])
            j = int(matches[1, k])
            u0 = int(pts0[0, i])
            v0 = int(pts0[1, i])
            z = depth0[v0, u0]
            x = (u0 - cx) * z / fx
            y = (v0 - cy) * z / fy
            p0 = np.array([x, y, z])
            q = T0[0:3, 0:3] @ p0 + T0[0:3, 3]

            p1 = T1[0:3, 0:3] @ q + T1[0:3, 3]
            u1_p = int(p1[0] * fx / p1[2] + cx)
            v1_p = int(p1[1] * fy / p1[2] + cy)

            u1 = int(pts1[0, j])
            v1 = int(pts1[1, j])
            if abs(u1 - u1_p) > 4: continue
            if abs(v1 - v1_p) > 4: continue

            correct += 1

        total += matches.shape[1]

print(correct / total)
