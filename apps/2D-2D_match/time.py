import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
from lcd.models import *


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the log directory')
args = parser.parse_args()

logdir = args.logdir
config = os.path.join(logdir, 'config.json')
args = json.load(open(config))
device = args['device']

fname = os.path.join(logdir, 'model.pth')
print('> Loading model from {}....'.format(fname))
model = PatchNetAutoencoder(
    args['embedding_size'],
    args['normalize']
)
model.load_state_dict(torch.load(fname)['patchnet'])
model.to(device)
model.eval()

def extract_color_patch(color, u, v):
    xmin = max(u - half_patch_size, 0)
    xmax = min(u + half_patch_size, color.shape[1])
    ymin = max(v - half_patch_size, 0)
    ymax = min(v + half_patch_size, color.shape[0])
    image = Image.fromarray(color[ymin:ymax, xmin:xmax])
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def extract_features(patches, model, device='cuda'):
    import time
    x = torch.tensor(patches, dtype=torch.float32)
    with torch.no_grad():
        x = x.to(device)
        start = time.time()
        z = model.encode(x)
        elapsed = time.time() - start
    print("Elapsed time: {}".format(elapsed * 1e3))


orb = cv.ORB_create()
num_samples = 2048
image_size = 64
half_patch_size = 32

keypts = []
patches = []

# color = cv.imread("data/scenenn/query/011/000000.png", cv.IMREAD_GRAYSCALE)
color = np.array(Image.open("samples/frame-000430.color.png"))

while len(keypts) < num_samples:
    u = np.random.choice(color.shape[1])
    v = np.random.choice(color.shape[0])
    keypts.append((u, v))
    patch = extract_color_patch(color, u, v)
    patches.append(patch)

keypts = cv.KeyPoint_convert(keypts)
extract_features(patches, model, device)

import time
start = time.time()
_, descriptors = orb.compute(color, keypts)
elapsed = time.time() - start
print("Elapsed time: {}".format(elapsed * 1e3))
