import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
from PIL import Image
import glob
sys.path.append('.')
from lcd.models import *
from lcd import *

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the log directory')
parser.add_argument('--imagesdir', help='path to the images directory')
parser.add_argument('--save_file', help='name of the file in which saving the ')

parse_args = parser.parse_args()

logdir = parse_args.logdir
config = os.path.join(logdir, 'config.json')
args = json.load(open(config))
device = args['device']
fname = os.path.join(logdir, 'model.pth')

# Import the Patchnet Auto Encoder
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


imagesdir = parse_args.imagesdir

if (parse_args.imagesdir == "epfl-trajectory"):
    images_pairs = list(zip(range(0, 100), range(1, 101)))
else:
    images_pairs = list(zip(range(1, 100), range(2, 101)))


all_matches = np.empty((0,6), int)
# For all consecutive pair of file in the folder
for image_nb0, image_nb1 in tqdm(images_pairs, desc='[Computation of 3D matches]'):

    if (parse_args.imagesdir == "epfl-trajectory"):
        image_path0 = glob.glob(imagesdir + "/EPFL_2020-09-17_{}_*.png".format(image_nb0))[0]
        image_path1 = glob.glob(imagesdir + "/EPFL_2020-09-17_{}_*.png".format(image_nb1))[0]
    else:
        image_path0 = glob.glob(imagesdir + "/*_{:04d}_f2_img.png".format(image_nb0))[0]
        image_path1 = glob.glob(imagesdir + "/*_{:04d}_f2_img.png".format(image_nb1))[0]
    
    # Import both images
    color0 = cv.imread(image_path0)
    color1 = cv.imread(image_path1)

    # Import both point clouds
    pc0 = np.load(image_path0.replace("img.png", "pc.npy"))
    pc1 = np.load(image_path1.replace("img.png", "pc.npy"))

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

    # Create a ordered list of the best matches
    bf = cv.BFMatcher()
    matches = bf.match(desc0, desc1)
    matches = sorted(matches, key=lambda x:x.distance)

    # Append the matches to all_matches
    for match in matches:
        index_query0, index_query1 = keypts0[match.queryIdx].pt
        index_train0, index_train1 = keypts1[match.trainIdx].pt

        pair = np.hstack((pc0[int(index_query1), int(index_query0)], pc1[int(index_train1), int(index_train0)]))

        all_matches = np.vstack((all_matches, pair))

# Export the matches int a file
save_file = "-"+parse_args.save_file

print("> Saving matches to {}".format(imagesdir+save_file))
np.save(imagesdir+save_file, all_matches)