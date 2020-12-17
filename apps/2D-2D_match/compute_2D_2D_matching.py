import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image

sys.path.append('.')
from lcd.models import *
from lcd import *

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

if(device == "cuda"):
    model.load_state_dict(torch.load(fname)['patchnet'])
    model.to(device)
else:
    model.load_state_dict(torch.load(fname, map_location=torch.device(device))["patchnet"])
model.eval()


num_samples = 1024
image_size = 64
half_patch_size = 32
threshold = 0.03
cx = 320.0
cy = 240.0
fx = 585.0
fy = 585.0


def extract_color_patch(color, u, v):
    xmin = max(u - half_patch_size, 0)
    xmax = min(u + half_patch_size, color.shape[1])
    ymin = max(v - half_patch_size, 0)
    ymax = min(v + half_patch_size, color.shape[0])
    image = Image.fromarray(color[ymin:ymax, xmin:xmax])
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image



#color0 = cv.resize(cv.imread('samples/DJI_0029.JPG'), (720, 480))
color0 = cv.imread('samples/EPFL_2020-09-17-piloted_14_DJI_0046_img.png')
color1 = cv.imread('samples/EPFL_2020-09-17-piloted_15_DJI_0047_img.png')

keypts0 = []
keypts1 = []
patches0 = []
patches1 = []

while len(keypts0) < num_samples:
    u0 = np.random.choice(color0.shape[1])
    v0 = np.random.choice(color0.shape[0])

    u1 = np.random.choice(color1.shape[1])
    v1 = np.random.choice(color1.shape[0])

    kp0 = (u0, v0)
    keypts0.append(kp0)
    kp1 = (u1, v1)
    keypts1.append(kp1)

    patch0 = extract_color_patch(color0, u0, v0)
    patches0.append(patch0)
    patch1 = extract_color_patch(color1, u1, v1)
    patches1.append(patch1)

keypts0 = cv.KeyPoint_convert(keypts0)
keypts1 = cv.KeyPoint_convert(keypts1)

def compute_lcd_descriptors(patches, model, batch_size, device):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    with torch.no_grad():
        for i, x in enumerate(batches):
            if(device == "cuda"):
                x = x.to(device)
            z = model.encode(x)
            if(device == "cuda"):
                z = z.cpu()
            z = z.numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)

# src = cv.drawKeypoints(color0, keypts0, color0)
# dst = cv.drawKeypoints(color1, keypts1, color1)
desc0 = compute_lcd_descriptors(patches0, model, args['batch_size'], device)
desc1 = compute_lcd_descriptors(patches1, model, args['batch_size'], device)

bf = cv.BFMatcher()
matches = bf.match(desc0, desc1)
matches = sorted(matches, key=lambda x:x.distance)

image = cv.drawMatches(color0, keypts0, color1, keypts1, matches[0:50], None, singlePointColor=None, flags=2)

plt.figure(figsize=(8,4))
plt.imshow(image[:,:,::-1])
plt.axis('off')
plt.tight_layout()
plt.savefig("dcdd.pdf", bbox_inches='tight')
plt.show()
plt.close()
