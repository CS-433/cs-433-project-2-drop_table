import os
import json
import open3d
import torch
import argparse
import numpy as np
from PIL import Image
from skimage.util.shape import view_as_windows
import scipy.spatial
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

from lcd.models import *


parser = argparse.ArgumentParser()
parser.add_argument("source", help="path to the source image")
parser.add_argument("target", help="path to the target image")
parser.add_argument("--logdir", help="path to the log directory")
args = parser.parse_args()

logdir = args.logdir
config = os.path.join(logdir, "config.json")
config = json.load(open(config))

device = config["device"]

fname = os.path.join(logdir, "model.pth")
print("> Loading model from {}".format(fname))
model = PatchNetAutoencoder(
    config["embedding_size"],
    config["normalize"],
)
model.load_state_dict(torch.load(fname)["patchnet"])
model.to(device)
model.eval()


def compute_lcd_descriptors(patches, model, batch_size, device):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    with torch.no_grad():
        for i, x in enumerate(batches):
            x = x.to(device)
            z = model.encode(x)
            z = z.cpu().numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)

source_path = args.source
target_path = args.target

# source = Image.open(source_path)
# target = Image.open(target_path)

# source_array = np.array(source)
# target_array = np.array(target)

img1 = cv2.imread(source_path) # queryImage
img2 = cv2.imread(target_path) # trainImage

source_array = np.array(img1)
target_array = np.array(img2)

step = 8

#We compute the patches for the two images

def extract_color_patch(color, u, v):
    xmin = max(u - 32, 0)
    xmax = min(u + 32, color.shape[1])
    ymin = max(v - 32, 0)
    ymax = min(v + 32, color.shape[0])
    image = Image.fromarray(color[ymin:ymax, xmin:xmax])
    image = image.resize((64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    return image

keypts0 = []
keypts1 = []

source_patches = []
target_patches = []


for j in range(32, source_array.shape[0]-24, step):
    for i in range(32, source_array.shape[1]-24, step):
        keypts0.append((i, j))
        keypts1.append((i, j))

        source_patches.append(extract_color_patch(img1, i, j))
        target_patches.append(extract_color_patch(img2, i, j))

#We prepare the data for the model and compute the descriptors for the two images

source_patches = np.array(source_patches)
target_patches = np.array(target_patches)

source_descriptors = compute_lcd_descriptors(
    source_patches, model, batch_size=128, device=device
)

print("> Computed {} descriptors from the source".format(len(source_descriptors)))

target_descriptors = compute_lcd_descriptors(
    target_patches, model, batch_size=128, device=device
)

print("> Computed {} descriptors from the target".format(len(target_descriptors)))


bf = cv2.BFMatcher()
matches = bf.match(source_descriptors, target_descriptors)
matches = sorted(matches, key=lambda x:x.distance)

print("> Found the {} best matches".format(30))

#We display the keypoints on the images using OpenCV with our custom parameters

keypts0 = cv2.KeyPoint_convert(keypts0)
keypts1 = cv2.KeyPoint_convert(keypts1)

image = cv2.drawMatches(img1, keypts0, img2, keypts1, matches[:50], None, singlePointColor=None, flags=2)

plt.figure(figsize=(8,4))
plt.imshow(image[:,:,::-1])
plt.axis('off')
plt.tight_layout()
plt.savefig("dcdd.pdf", bbox_inches='tight')
plt.show()
plt.close()