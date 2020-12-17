import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image
import glob
import open3d as o3d

sys.path.append('.')
from lcd.models import *
from lcd import *

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the log directory')
parser.add_argument('--imagesdir', help='path to the images directory')
parser.add_argument('--save_file', help='name of the file in which saving the ')
parser.add_argument("--voxel_size", default=100, type=float)
parser.add_argument("--radius", default=540, type=float)
parser.add_argument("--num_points", default=1024, type=int)

parse_args = parser.parse_args()

logdir = parse_args.logdir
config = os.path.join(logdir, 'config.json')
config = json.load(open(config))
device = config['device']
fname = os.path.join(logdir, 'model.pth')
print('> Loading model from {}....'.format(fname))
patchnet = PatchNetAutoencoder(
    config['embedding_size'],
    config['normalize']
)
if (device == "cuda"):
    patchnet.load_state_dict(torch.load(fname)['patchnet'])
    patchnet.to(device)
else:
    patchnet.load_state_dict(torch.load(fname, map_location=torch.device(device))["patchnet"])
patchnet.eval()

print("> Loading pointnet from {}".format(fname))
pointnet = PointNetAutoencoder(
    config["embedding_size"],
    config["input_channels"],
    config["output_channels"],
    config["normalize"],
)

if (device == "cuda"):
    pointnet.load_state_dict(torch.load(fname)['pointnet'])
    pointnet.to(device)
else :
    pointnet.load_state_dict(torch.load(fname, map_location=torch.device(device))["pointnet"])
pointnet.eval()

# Parameters
num_samples = 1024
image_size = 64
half_patch_size = 32

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
            
            if (device == "cuda"):
                x = x.to(device)
            z = model.encode(x)
            if (device == "cuda"):
                z = z.cpu()
            z = z.numpy()

            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)

def extract_uniform_patches(pcd, voxel_size, radius, num_points):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    downsampled = o3d.geometry.voxel_down_sample(pcd,voxel_size)
    points = np.asarray(downsampled.points)
    patches = []
    for i in tqdm(range(points.shape[0])):
        k, index, _ = kdtree.search_hybrid_vector_3d(points[i], radius, num_points)
        if k < num_points:
            index = np.random.choice(index, num_points, replace=True)
        xyz = np.asarray(pcd.points)[index]
        rgb = np.asarray(pcd.colors)[index]
        xyz = (xyz - points[i]) / radius  # normalize to local coordinates
        patch = np.concatenate([xyz, rgb], axis=1)
        patches.append(patch)
    patches = np.stack(patches, axis=0)
    return downsampled, patches


def encode_3D(patches, model, batch_size, device):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    print("Encoding the 3D patches in descriptors ...")
    with torch.no_grad():
        for i, x in enumerate(batches):
            print("   > Batch : ", i, "/" , len(batches))
            if (device == "cuda"):
                x = x.to(device)
            z = model.encode(x)
            if (device == "cuda"):
                z = z.cpu()
            z = z.numpy()

            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)


def open_image(rgb_file):
    source = Image.open(rgb_file)
    source_array = np.array(source)
    img1 = cv.imread(rgb_file)
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB) # queryImage
    return img1, source, source_array

imagesdir = parse_args.imagesdir

if (parse_args.imagesdir == "epfl-trajectory"):
    images_pairs = list(zip(range(0, 100), range(1, 101)))
else:
    images_pairs = list(zip(range(1, 100), range(2, 101)))

all_matches = np.empty((0,6), int)

for image_nb0, image_nb1 in tqdm(images_pairs, desc='[Computation of 3D matches]'):

    if (parse_args.imagesdir == "epfl-trajectory"):
        image_path0 = glob.glob(imagesdir + "/EPFL_2020-09-17_{}_*.png".format(image_nb0))[0]
        image_path1 = glob.glob(imagesdir + "/EPFL_2020-09-17_{}_*.png".format(image_nb1))[0]
    else:
        image_path0 = glob.glob(imagesdir + "/*_{:04d}_f2_img.png".format(image_nb0))[0]
        image_path1 = glob.glob(imagesdir + "/*_{:04d}_f2_img.png".format(image_nb1))[0]

    color0 = cv.imread(image_path0)
    color0 = cv.cvtColor(color0,cv.COLOR_BGR2RGB)
    color1 = cv.imread(image_path1)
    color1 = cv.cvtColor(color1,cv.COLOR_BGR2RGB)


    pc0 = np.load(image_path0.replace("img.png", "pc.npy"))
    pc1 = np.load(image_path1.replace("img.png", "pc.npy"))
    
    img1, source, source_array = open_image(image_path0)


    # Point Cloud
    pcd = o3d.geometry.PointCloud()
    pc1_flatten = pc1.reshape(-1, pc1.shape[-1])
    colors_flatten = color1.reshape(-1, color1.shape[-1])
    indices = np.where(pc1_flatten[:,0]!=-1)
    pcd.points = o3d.utility.Vector3dVector(pc1_flatten[indices])
    pcd.colors = o3d.utility.Vector3dVector(colors_flatten[indices]/255)
    
    downsampled, patches1 = extract_uniform_patches(pcd, parse_args.voxel_size, parse_args.radius, parse_args.num_points)
    keypts1 = downsampled.points

    desc1 = encode_3D(patches1, pointnet, batch_size=124, device=device)

    # Pick at random num_samples pixels on the images
    keypts0 = []
    patches0 = []
    while len(keypts0) < num_samples:
        u0 = np.random.choice(color0.shape[1])
        v0 = np.random.choice(color0.shape[0])

        # The coordinate of the pixel become a keypoint
        kp0 = (u0, v0)
        keypts0.append(kp0)

        # We create a patch around each pixels
        patch0 = extract_color_patch(color0, u0, v0)
        patches0.append(patch0)
    
    # Compute the descriptors for each patches
    desc0 = compute_lcd_descriptors(patches0, patchnet, batch_size=256, device=device)

    # Create a ordered list of the best matches
    bf = cv.BFMatcher()
    matches = bf.match(desc0, desc1)
    matches = sorted(matches, key=lambda x:x.distance)

    # Append the matches to all_matches
    for match in matches:
        index_query0, index_query1 = keypts0[match.queryIdx]
        key1 = keypts1[match.trainIdx]

        pair = np.hstack((pc0[int(index_query1), int(index_query0)], key1))

        all_matches = np.vstack((all_matches, pair))

# Export the matches int a file
save_file = "-"+parse_args.save_file

print("> Saving matches to {}".format(imagesdir+save_file))
np.save(imagesdir+save_file, all_matches)