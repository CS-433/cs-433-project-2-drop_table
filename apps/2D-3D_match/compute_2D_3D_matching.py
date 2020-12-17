import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import open3d as o3d
sys.path.append('.')
from lcd.models import *
from lcd import *

parser = argparse.ArgumentParser()
parser.add_argument("source_image", help="path to the source image")
parser.add_argument("destination_image", help="path to the destination image")
parser.add_argument("destination_point_cloud", help="path to the destination depth map")
parser.add_argument('--logdir', help='path to the log directory')
parser.add_argument("--voxel_size", default=20, type=float)
parser.add_argument("--radius", default=50, type=float)
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
if(device == "cuda"):
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

if(device == "cuda"):
    pointnet.load_state_dict(torch.load(fname)['pointnet'])
    pointnet.to(device)
else :
    pointnet.load_state_dict(torch.load(fname, map_location=torch.device(device))["pointnet"])
pointnet.eval()

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
    print("Encoding the 2D patches in descriptors ...")
    with torch.no_grad():
        for i, x in enumerate(batches):
            print("   > Batch : ", i, "/" , len(batches))
            if(device == "cuda"):
                x = x.to(device)
            z = model.encode(x)
            if(device == "cuda"):
                z = z.cpu().numpy()
            else:
                z = z.numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)

def extract_uniform_patches(pcd, voxel_size, radius, num_points):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    downsampled = o3d.geometry.voxel_down_sample(pcd,voxel_size)
    points = np.asarray(downsampled.points)
    patches = []
    for i in range(points.shape[0]):
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
            if(device == "cuda"):
                x = x.to(device)
            z = model.encode(x)
            if(device == "cuda"):
                z = z.cpu().numpy()
            else:
                z = z.numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)


def open_image(rgb_file):
    source = Image.open(rgb_file)
    source_array = np.array(source)
    img1 = cv.imread(rgb_file)
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB) # queryImage
    return img1, source, source_array

def from_2D_image_in_3D_space(source_array):
    img1_points = []
    img1_colors = []
    pcimg1 = o3d.geometry.PointCloud()
    for i in range(source_array.shape[0]):
        for j in range(source_array.shape[1]):
            img1_points.append([i,j,0])
            img1_colors.append(source_array[i][j]/255)
    img1_colors = np.delete(img1_colors,3,axis=1)
    pcimg1.points = o3d.utility.Vector3dVector(img1_points)
    pcimg1.colors = o3d.utility.Vector3dVector(img1_colors)
    pcimg1.translate((-source_array.shape[0]/2, -source_array.shape[1]/2, 0))
    pcimg1.scale(-1, center=True)
    pcimg1.rotate([0, -np.pi / 4, 0], center=True)
    pcimg1.scale(20, center=True)
    pcimg1.translate((4374723,540817, 4595214))
    pcimg1.translate((0,0,5000))
    return pcimg1


image_path0 = parse_args.source_image
image_path1 = parse_args.destination_image
pc_path1 = parse_args.destination_point_cloud

color0 = cv.imread(image_path0)
color0 = cv.cvtColor(color0,cv.COLOR_BGR2RGB)
color1 = cv.imread(image_path1)
color1 = cv.cvtColor(color1,cv.COLOR_BGR2RGB)

pc1 = np.load(pc_path1)

_, _, source_array = open_image(image_path0)
pc01 = from_2D_image_in_3D_space(source_array)

# Point Cloud
pcd = o3d.geometry.PointCloud()
pc1_flatten = pc1.reshape(-1, pc1.shape[-1])
colors_flatten = color1.reshape(-1, color1.shape[-1])
indices = np.where(pc1_flatten[:,0]!=-1)
pcd.points = o3d.utility.Vector3dVector(pc1_flatten[indices])
pcd.colors = o3d.utility.Vector3dVector(colors_flatten[indices]/255)

downsampled, patches1 = extract_uniform_patches(pcd, parse_args.voxel_size, parse_args.radius, parse_args.num_points)
keypts1 = downsampled.points

o3d.visualization.draw_geometries([downsampled])


desc1 = encode_3D(patches1, pointnet, batch_size=124, device=device)

# RGB
keypts0 = []
patches0 = []
while len(keypts0) < num_samples:
    u0 = np.random.choice(color0.shape[1])
    v0 = np.random.choice(color0.shape[0])

    kp0 = (u0, v0)
    keypts0.append(kp0)

    patch0 = extract_color_patch(color0, u0, v0)
    patches0.append(patch0)


desc0 = compute_lcd_descriptors(patches0, patchnet, batch_size=256, device=device)

bf = cv.BFMatcher()
matches = bf.match(desc0, desc1)
matches = sorted(matches, key=lambda x:x.distance)


correspondences = []
for elem in matches:
    train_real_idx = keypts0[elem.queryIdx][1]* source_array.shape[1] + keypts0[elem.queryIdx][0]
    correspondences.append((train_real_idx,elem.trainIdx))

lineset = o3d.geometry.create_line_set_from_point_cloud_correspondences(pc01,downsampled,correspondences[:15])
o3d.visualization.draw_geometries([pc01, downsampled, lineset])
