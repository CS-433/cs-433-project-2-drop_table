import os
import json
import open3d as o3d
import torch
import argparse
import numpy as np
from PIL import Image
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py
from lcd.models import *

MANUAL_INPUT = True

def manual_inputs():
    scene_dir = "samples/rgbd-scenes-v2-scene_13_sample"
    intrinsics = os.path.join(scene_dir, "camera-intrinsics.txt")
    frames = os.path.join(scene_dir,"seq-01")
    rgb = os.path.join(frames,"frame-000043.color.png")
    rgb2 = os.path.join(frames,"frame-000044.color.png")
    depth2 = os.path.join(frames,"frame-000044.depth.png")
    return rgb, rgb2, depth2, intrinsics

parser = argparse.ArgumentParser()
if (not MANUAL_INPUT) :
    parser.add_argument("source_image", help="path to the source image")
    parser.add_argument("destination_image", help="path to the destination image")
    parser.add_argument("destination_depth_map", help="path to the destination depth map")
    parser.add_argument("camera_intrinsics", help="path to the camera intrinsics file")
parser.add_argument("--logdir", help="path to the log directory")
parser.add_argument("--voxel_size", default=0.15, type=float)
parser.add_argument("--radius", default=0.15, type=float)
parser.add_argument("--num_points", default=1024, type=int)
parser.add_argument("--step", default=8, type=int)

args = parser.parse_args()

logdir = args.logdir
step = args.step
config = os.path.join(logdir, "config.json")
config = json.load(open(config))

if (not MANUAL_INPUT) :
    rgb = args.source_image
    rgb2 = args.destination_image
    depth2 = args.destination_depth_map
    intrinsics = args.camera_intrinsics

device = config["device"]

fname = os.path.join(logdir, "model.pth")
print("Loading the LCD models ...")
print("   > Loading patchnet from {}".format(fname))
patchnet = PatchNetAutoencoder(
    config["embedding_size"],
    config["normalize"],
)

if (device == "cuda"):
    patchnet.load_state_dict(torch.load(fname)['patchnet'])
    patchnet.to(device)
else:
    patchnet.load_state_dict(torch.load(fname, map_location=torch.device(device))["patchnet"])
patchnet.eval()

print("   > Loading pointnet from {}".format(fname))
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

def load_data(fname,i):
    h5 = h5py.File(fname, "r")
    return h5["points"][i], h5["images"][i]


def encode_2D(patches, patchnet, batch_size, device):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    print("Encoding the 2D patches in descriptors ...")
    with torch.no_grad():
        for i, x in enumerate(batches):
            print("   > Batch : ", i, "/" , len(batches))
            
            if (device == "cuda"):
                x = x.to(device)
            z = patchnet.encode(x)
            if (device == "cuda"):
                z = z.cpu()
            z = z.numpy()

            descriptors.append(z)
    np_desc = np.concatenate(descriptors, axis=0)
    return np_desc

def decode_2D(descriptors, patchnet, batch_size, device):
    batches = torch.tensor(descriptors, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    print("Decoding the descriptors into 2D patches ...")
    patches = []
    for i,elem in enumerate(batches):
        print("   > Batch : ", i, "/" , len(batches))
        z = patchnet.decode(elem)
        z = z.detach().numpy()
        patches.append(z)
    return np.concatenate(patches, axis=0)

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
                z = z.cpu()            
            z = z.numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)

def decode_3D(descriptors, pointnet, batch_size, device):
    batches = torch.tensor(descriptors, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    print("Decoding the descriptors into 3D patches ...")
    patches = []
    for i, elem in enumerate(batches):
        print("   > Batch : ", i, "/" , len(batches))
        z = pointnet.decode(elem)
        z = z.detach().numpy()
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

def extract_patches(img,source, step = 8):
    source_patches = []
    keypts0 = []
    for j in range(32, source.height-(24), step):
        for i in range(32, source.width-(24), step):
            keypts0.append((i, j))
            source_patches.append(extract_color_patch(img, i, j))
    return np.array(source_patches), keypts0, (len(range(32, source.height-(32-step), step)),len(range(32, source.width-(32-step), step)))

def open_image(rgb_file):
    source = Image.open(rgb_file)
    source_array = np.array(source)
    img1 = cv2.imread(rgb_file)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB) # queryImage
    return img1, source, source_array

def match(source_image, source_keypoints,source_descriptors, target_image, target_keypoints,target_descriptors):
    bf = cv2.BFMatcher()
    matches = bf.match(source_descriptors, target_descriptors)
    matches = sorted(matches, key=lambda x:x.distance)
    i = 15
    print("> Found the {} best matches".format(i))

    #We display the keypoints on the images using OpenCV with our custom parameters

    source_keypoints = cv2.KeyPoint_convert(source_keypoints)
    target_keypoints = cv2.KeyPoint_convert(target_keypoints)

    image = cv2.drawMatches(source_image, source_keypoints, target_image, target_keypoints, matches[:i], None, singlePointColor=None, flags=2)

    plt.figure(figsize=(8,4))
    plt.imshow(image[:,:,::-1])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("dcdd.pdf", bbox_inches='tight')
    plt.show()
    plt.close()

    return matches

def extract_uniform_patches(pcd, voxel_size, radius, num_points):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    downsampled, lst = o3d.geometry.voxel_down_sample_and_trace(pcd,voxel_size,pcd.get_min_bound(),pcd.get_max_bound())
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
    return downsampled, patches, lst

def depth_map_to_point_cloud(intrinsics, rgb_file, depth_file, display = False):
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file).convert('I')
    focalLength, centerX, centerY, scalingFactor = import_intrinsics(intrinsics)
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(rgb.size[0], rgb.size[1], focalLength, focalLength, centerX, centerY)

    color_raw = o3d.io.read_image(rgb_file)
    depth_raw = o3d.io.read_image(depth_file)

    rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity = False)

    pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, camera_intrinsics)
    # flip the orientation, so it looks upright, not upside-down
    #pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    x = np.asarray(depth)
    indices = []
    i = 0
    for u in range(rgb.size[1]):
        for v in range(rgb.size[0]):
            Z = np.asarray(rgbd.depth)[u][v]
            if Z==0: continue
            indices.append([v,u])

    if display :
        plt.subplot(1, 2, 1)
        plt.title('RGB')
        plt.imshow(rgbd.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth')
        plt.imshow(rgbd.depth)
        plt.show()

        o3d.visualization.draw_geometries([pcd])

    return indices, np.asarray(pcd.points), np.asarray(pcd.colors), pcd

def import_intrinsics(intrinsics):
    lst = []
    with open(intrinsics) as f:
        for line in f:
            line = line.split()
            lst+=line
    float_lst = list(map(float, lst))
    return float_lst[0], float_lst[2], float_lst[5], float_lst[1]

def open_and_encode_2D(rgb):
    img1, source, source_array = open_image(rgb)
    source_patches, keypoints, _ = extract_patches(img1, source, step)
    descriptors = encode_2D(source_patches, patchnet, batch_size=256, device=device)
    return img1, source_array, keypoints, descriptors

def from_2D_image_in_3D_space(source_array):
    img1_points = []
    img1_colors = []
    pcimg1 = o3d.geometry.PointCloud()
    for i in range(source_array.shape[0]):
        for j in range(source_array.shape[1]):
            img1_points.append([i,j,0])
            img1_colors.append(source_array[i][j]/255)
    pcimg1.points = o3d.utility.Vector3dVector(img1_points)
    pcimg1.colors = o3d.utility.Vector3dVector(img1_colors)
    pcimg1.translate((-source_array.shape[0]/2, -source_array.shape[1]/2, 0))
    pcimg1.rotate([0, 0, - np.pi / 2], center=True)
    pcimg1.scale(0.002, center=True)
    pcimg1.transform([[1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]])
    pcimg1.translate((-source_array.shape[0]*0.002/2,0, 0))
    #indices, points,colors, pcd = depth_map_to_point_cloud(intrinsics, rgb2, depth2, display = False)
    #o3d.visualization.draw_geometries([pcimg1,pcd])
    return pcimg1

def open_and_encode_depth_map(rgb, depth, intrinsics):
    img, source, source_array = open_image(rgb)
    indices, points,colors, pcd = depth_map_to_point_cloud(intrinsics, rgb, depth, display = False)
    downsampled, patches, lst = extract_uniform_patches(pcd, args.voxel_size,args.radius, args.num_points)
    keypoints2_3D = []
    for elem in lst:
        for i in elem:
            if i != -1:
                keypoints2_3D.append(indices[i])
                break
    keypoints2_3D_2 = []
    for elem in lst:
        for i in elem:
            if i != -1:
                keypoints2_3D_2.append(i)
                break

    descriptors = encode_3D(patches, pointnet, batch_size=128, device=device)
    return img, keypoints2_3D, keypoints2_3D_2, descriptors, pcd


#Manual Parameters
if(MANUAL_INPUT):
    rgb, rgb2, depth2, intrinsics = manual_inputs()

img1, source_array, keypoints, descriptors = open_and_encode_2D(rgb)
pcimg1 = from_2D_image_in_3D_space(source_array)

img2, keypoints2_3D, keypoints2_3D_2, descriptors2, pcd = open_and_encode_depth_map(rgb2,depth2,intrinsics)
matches = match(img1, keypoints, descriptors, img2, keypoints2_3D, descriptors2)

correspondences = []
for elem in matches:
    train_real_idx = keypoints[elem.queryIdx][1]* source_array.shape[0] + keypoints[elem.queryIdx][0]

    correspondences.append((train_real_idx,keypoints2_3D_2[elem.trainIdx]))

lineset = o3d.geometry.create_line_set_from_point_cloud_correspondences(pcimg1,pcd,correspondences[:15])
o3d.visualization.draw_geometries([pcimg1, pcd, lineset])
