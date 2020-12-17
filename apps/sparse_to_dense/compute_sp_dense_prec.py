import os
import json
import open3d as o3d
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from matplotlib import cm
import h5py
import glob
from sklearn.neighbors import KDTree
import sys
sys.path.append('.')
from lcd.models import *

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the log directory')
parser.add_argument('--imagesdir', help='path to the images directory')
args = parser.parse_args()

logdir = args.logdir
config = os.path.join(logdir, "config.json")
config = json.load(open(config))
device = config["device"]
fname = os.path.join(logdir, "model.pth")

# Load model patchnet
print("Loading the LCD models ...")
print("   > Loading patchnet from {}".format(fname))
patchnet = PatchNetAutoencoder(
    config["embedding_size"],
    config["normalize"],
)
if(device == "cuda"):
    patchnet.load_state_dict(torch.load(fname)['patchnet'])
    patchnet.to(device)
else:
    patchnet.load_state_dict(torch.load(fname, map_location=torch.device(device))["patchnet"])
patchnet.eval()

# Load model pointnet
print("   > Loading pointnet from {}".format(fname))
pointnet = PointNetAutoencoder(
    config["embedding_size"],
    config["input_channels"],
    config["output_channels"],
    config["normalize"],
)
if(device == "cuda"):
    patchnet.load_state_dict(torch.load(fname)['pointnet'])
    patchnet.to(device)
else:
    patchnet.load_state_dict(torch.load(fname, map_location=torch.device(device))["pointnet"])
patchnet.eval()


# Define the number of patches we want from each image
num_samples = 3000

def encode_2D(patches, patchnet, batch_size, device):
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
                z = z.cpu()
            z = z.numpy()
            descriptors.append(z)
    np_desc = np.concatenate(descriptors, axis=0)
    return np_desc

def decode_3D(descriptors, pointnet, batch_size, device):
    batches = torch.tensor(descriptors, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    print("Decoding the descriptors into 3D patches ...")
    patches = []
    with torch.no_grad():
        for i, elem in enumerate(batches):
            print("   > Batch : ", i, "/" , len(batches))
            elem = elem.to(device)
            z = pointnet.decode(elem)
            z = z.detach().cpu().numpy()
            patches.append(z)
    return np.concatenate(patches, axis=0)

def extract_color_patch(color, u, v):
    xmin = max(u - 32, 0)
    xmax = min(u + 32, color.shape[1])
    ymin = max(v - 32, 0)
    ymax = min(v + 32, color.shape[0])
    image = Image.fromarray(color[ymin:ymax, xmin:xmax])
    image = image.resize((64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    return image

def assemble(pointcloud, keypoints, scale):
    pointcloud_global = []
    for i, kp in enumerate(keypoints):
        elem = pointcloud[i]
        elem[:,0:3] = scale * elem[:,0:3] + kp
        pointcloud_global.append(elem)
    tmp = np.array(pointcloud_global)
    tmp = tmp.reshape(tmp.shape[0] * 1024, 6)
    print("tmp shape :", tmp.shape)
    return tmp

def get_descriptors_from_PNG(source_patches):
    source_descriptorsv2 = encode_2D(source_patches, patchnet, batch_size=256, device=device)
    pointcloud = decode_3D(source_descriptorsv2, pointnet, batch_size=256, device=device)
    return pointcloud

def draw_point_cloud(pointcloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud[:,3:])
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    o3d.visualization.draw_geometries([pcd])
    return

imagesdir = args.imagesdir

all_matches = np.empty((0,12), int)

for image_nb0 in tqdm(range(0, 100, 20), desc='[Computation of 3D matches]'):

    image_path0 = glob.glob(imagesdir + "/EPFL_2020-09-17_{}_*.png".format(image_nb0))[0]

    # image_path0 = glob.glob(imagesdir + "/*_{:04d}_f2_img.png".format(image_nb0))[0]

    pc0 = np.load(image_path0.replace("img.png", "pc.npy"))

    color0 = cv2.imread(image_path0)
    color0 = cv2.cvtColor(color0, cv2.COLOR_BGR2RGB) # queryImage

    keypts0 = []
    patches0 = []

    # open3dcolors = []

    while len(keypts0) < num_samples:
        u0 = np.random.choice(color0.shape[1])
        v0 = np.random.choice(color0.shape[0])

        if -1 in pc0[v0, u0]:
            continue

        kp0 = pc0[v0, u0]
        keypts0.append(kp0)

        # open3dcolors.append(color0[v0, u0] / 255)

        patch0 = extract_color_patch(color0, u0, v0)
        patches0.append(patch0)

    # draw_point_cloud(np.hstack((np.array(keypts0), np.array(open3dcolors))))

    pointcloud = get_descriptors_from_PNG(patches0)

    pc = assemble(pointcloud, keypts0, 100)
    
    # draw_point_cloud(pc)

    pc0 = pc0.reshape(pc0.shape[0] * pc0.shape[1], pc0.shape[2])
    color0 = color0.reshape(color0.shape[0] * color0.shape[1], color0.shape[2]) / 255

    pc0 = np.hstack((pc0, color0))
    pc0_clean = np.delete(pc0, np.where(pc0[:,] == -1), axis=0)

    tree = KDTree(pc)
    distance, indices = tree.query(pc0_clean, k=1)
    
    pair = np.hstack((pc0_clean, pc[indices].reshape(pc0_clean.shape)))

    all_matches = np.vstack((all_matches, pair))

save_file = "-sparse-to-dense-EPFL-synth"

print("> Saving matches to {}".format(imagesdir+save_file))
np.save(imagesdir+save_file, all_matches)
