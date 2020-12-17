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
from lcd import *


parser = argparse.ArgumentParser()
parser.add_argument('source', help="path to the source point cloud")
parser.add_argument('target', help="path to the target point cloud")
parser.add_argument('--logdir', help='path to the log directory')
parser.add_argument('--number_of_matches', default=15, help='Number of matches to display')

parse_args = parser.parse_args()

logdir = parse_args.logdir
config = os.path.join(logdir, 'config.json')
args = json.load(open(config))
device = args['device']

# Import the Patchnet Auto Encoder
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




def extract_color_patch(color, u, v):
    xmin = max(u - half_patch_size, 0)
    xmax = min(u + half_patch_size, color.shape[1])
    ymin = max(v - half_patch_size, 0)
    ymax = min(v + half_patch_size, color.shape[0])
    image = Image.fromarray(color[ymin:ymax, xmin:xmax])
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image

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

# Parameters
num_samples = 1024
image_size = 64
half_patch_size = 32

# Import both images
color0 = cv.imread(parse_args.source)
color1 = cv.imread(parse_args.target)

keypts0 = []
keypts1 = []
patches0 = []
patches1 = []

# Pick at random num_samples pixels on the images
while len(keypts0) < num_samples:
    u0 = np.random.choice(color0.shape[1])
    v0 = np.random.choice(color0.shape[0])

    u1 = np.random.choice(color1.shape[1])
    v1 = np.random.choice(color1.shape[0])

    # The coordinate of the pixel become a keypoint
    kp0 = (u0, v0)
    keypts0.append(kp0)
    kp1 = (u1, v1)
    keypts1.append(kp1)

    # We create a patch around each pixels
    patch0 = extract_color_patch(color0, u0, v0)
    patches0.append(patch0)
    patch1 = extract_color_patch(color1, u1, v1)
    patches1.append(patch1)

keypts0 = cv.KeyPoint_convert(keypts0)
keypts1 = cv.KeyPoint_convert(keypts1)

# Compute the descriptors for each patches
desc0 = compute_lcd_descriptors(patches0, model, args['batch_size'], device)
desc1 = compute_lcd_descriptors(patches1, model, args['batch_size'], device)


bf = cv.BFMatcher()
# Create a ordered list of the best matches
matches = bf.match(desc0, desc1)
matches = sorted(matches, key=lambda x:x.distance)

image = cv.drawMatches(color0, keypts0, color1, keypts1, matches[0:int(parse_args.number_of_matches)], None, singlePointColor=None, flags=2)

# Display it using matplotlib
plt.figure(figsize=(8,4))
plt.imshow(image[:,:,::-1])
plt.axis('off')
plt.tight_layout()
plt.savefig("dcdd.pdf", bbox_inches='tight')
plt.show()
plt.close()
