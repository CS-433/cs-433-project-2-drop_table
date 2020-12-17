import os
import sys
import glob
import h5py
import datetime
import numpy as np
from PIL import Image
from itertools import chain


scenes = ["SURVEY"]

root = "C:/Users/Eva.DESKTOP-21K26HC/Desktop/"
dataset_size = 30000
batch_size = 10000
radius = 0.15
cutoff = 100000
cloud_size = 1024
image_size = 64
seed = 42

foc_x = 450
foc_y = 450
center_x = 360
center_y = 240

camera_intrinsincs = [[foc_x, 0, center_x], [0, foc_y, center_y], [0, 0, 1]]

def get_random_frame(scene):
    if scene == "phantom/":
        return np.random.randint(2, 1244)
    elif scene == "EPFL-LSH/":
        return np.random.randint(0, 15000)
    elif scene == "comballaz-air2":
        return np.random.randint(0, 1182)
    elif scene == "comballaz-phantom-piloted":
        return np.random.randint(0, 798)
    elif scene == "comballaz-phantom-survey":
        return np.random.randint(0, 447)
    elif scene == "EPFL_piloted_2020_09_17":
        return np.random.randint(0, 145)
    elif scene == "EPFL_2020-09-17" or scene == "SURVEY":
        return np.random.randint(0, 841)
    elif scene == "EPFL_2020-09-24":
        return np.random.randint(1, 1549)
    elif scene == "EPFL_2020-11-11":
        return np.random.randint(0, 623)
    else:
        return np.random.randint(2, 1181)

def compute_bounding_box(p):
    return np.array(
        [
            [p[0] - radius, p[1] - radius, p[2] - radius],
            [p[0] - radius, p[1] - radius, p[2] + radius],
            [p[0] - radius, p[1] + radius, p[2] - radius],
            [p[0] - radius, p[1] + radius, p[2] + radius],
            [p[0] + radius, p[1] - radius, p[2] - radius],
            [p[0] + radius, p[1] - radius, p[2] + radius],
            [p[0] + radius, p[1] + radius, p[2] - radius],
            [p[0] + radius, p[1] + radius, p[2] + radius],
        ]
    )


def extract_point_cloud(depth, color, w, h, origin, K):
    cloud = []
    for v in range(h[0], h[1]):
        for u in range(w[0], w[1]):
            z = depth[v, u]
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            r = color[v, u, 0] / 255.0
            g = color[v, u, 1] / 255.0
            b = color[v, u, 2] / 255.0
            if z <= 0.0:
                continue
            if abs(x - origin[0]) >= radius:
                continue
            if abs(y - origin[1]) >= radius:
                continue
            if abs(z - origin[2]) >= radius:
                continue
            cloud.append([x, y, z, r, g, b])

    # Subsample point cloud
    cloud = np.array(cloud, dtype=np.float32)
    indices = np.random.choice(cloud.shape[0], cloud_size, replace=True)
    cloud = cloud[indices, :]
    cloud[:, 0:3] = (cloud[:, 0:3] - origin) / radius
    return cloud


def extract_color_patch(color, w, h):
    image = Image.fromarray(color[h[0] : h[1], w[0] : w[1]])
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def sample_matching_pairs(scene):
    frame = get_random_frame(scene)
    K = np.array(camera_intrinsincs)
    path = root + scene + "/"

    # Pick a random depth frame
    depth = np.array(Image.open(path+"depth_{}.png".format(frame))) * 0.001
    img_path = glob.glob(path+("img{:04d}.png").format(frame))
    #color = np.array(Image.open(path+"EPFL-LHS_{}_LHS_img.png".format(frame)))
    color = np.array(Image.open(img_path[0]))
    #color = np.delete(color, 3, axis=2)
    depth[depth > cutoff] = 0.0

    # Pick a random point P
    u0 = np.random.choice(depth.shape[1])
    v0 = np.random.choice(depth.shape[0])

    if depth[v0, u0] <= 0.0:
        return None

    # Compute bounding box
    z = depth[v0, u0]
    x = (u0 - K[0, 2]) * z / K[0, 0]
    y = (v0 - K[1, 2]) * z / K[1, 1]
    p0 = np.array([x, y, z])
    b = compute_bounding_box(p0)
    if not np.all(b[:, 2]):
        print(scene)
        print(frame)
        print("Div by 0")
        return None
    b[:, 0] = np.round(b[:, 0] * K[0, 0] / b[:, 2] + K[0, 2])
    b[:, 1] = np.round(b[:, 1] * K[1, 1] / b[:, 2] + K[1, 2])

    # Get the depth patch
    x = np.array([np.min(b[:, 0]), np.max(b[:, 0])], dtype=np.int32)
    y = np.array([np.min(b[:, 1]), np.max(b[:, 1])], dtype=np.int32)
    if np.any(x < 0) or np.any(x >= depth.shape[1]):
        print("some")
        return None
    if np.any(y < 0) or np.any(y >= depth.shape[0]):
        print("some2")
        return None

    patch = {}
    patch["cloud"] = extract_point_cloud(depth, color, x, y, p0, K)
    patch["color"] = extract_color_patch(color, x, y)
    return patch


def save_batch_h5(fname, batch):
    #patches = list(chain(*batch))
    indices = [len(sample) for sample in batch]
    indices = np.array(indices, dtype=np.int32)
    points = np.stack([patch["cloud"] for patch in batch])
    images = np.stack([patch["color"] for patch in batch])
    fp = h5py.File(fname, "w")
    fp.create_dataset("points", data=points, compression="gzip")
    fp.create_dataset("images", data=images, compression="gzip")
    fp.close()


size = 0
batch = []
#scene = "matching/plannedflights/"
while size < dataset_size:
    scene = np.random.choice(scenes)
    sample = sample_matching_pairs(scene)
    if sample is None:
        continue

    size += 1
    batch += [sample]
    print("Sample matching patches [{}/{}]".format(size, dataset_size))

    # Save batch if needed
    if len(batch) == batch_size:
        i = size // batch_size
        fname = "C:/Users/Eva.DESKTOP-21K26HC/Desktop/survey-real-h5/10/{:04d}.h5".format(i)
        print("> Saving batch to {}...".format(fname))
        save_batch_h5(fname, batch)
        batch = []