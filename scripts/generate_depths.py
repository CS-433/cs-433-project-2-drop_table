import numpy as np
import ast
import numpngw
import glob
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", help="path to the data")
parser.add_argument("--pose_file", help="name of the  pose file npy of the directory")
parser.add_argument("--relative_pointcloud_path", help="Relative path to the pointcloud data")
parser.add_argument("--start_image", help="number of the first image")
parser.add_argument("--end_image", help="number of the image")

args = parser.parse_args()
data_path = args.data_path
pose_file = args.pose_file
relative_pointcloud_path = args.relative_pointcloud_path
start_image = args.start_image
end_image = args.end_image
def npy_to_py(path):
    a = np.load(path)
    return a

def file_to_py(path):
    with open(path,'r') as file:
        myfile = file.read()
        return txt_to_py(myfile)

def file_to_py_d(path):
    with open(path,'r') as file:
        myfile = file.read()
        return txt_to_py_d(myfile)

def txt_to_py(txt):
    index = txt.find(']')
    drone = ast.literal_eval(txt[1:index+1])
    pixels = ast.literal_eval("["+ txt[index+2:])
    return drone, pixels

def txt_to_py_d(txt):
    drone = ast.literal_eval(txt)
    return drone

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    m = 1000 * km
    return m

def compute_depth(drone, pixels):
    c,r = pixels[:,:,0].shape

    y_drone,x_drone,z_drone,_,_,_ = drone
    drone_coord = [(x_drone,y_drone)]

    pixels_x = pixels[:,:,0]
    pixels_y = pixels[:,:,1]
    
    pixels_invalid = np.equal(pixels_x,-1)
    pixels_alt = pixels[:,:,2]


    haversine_distance = haversine_np(x_drone,y_drone,pixels_x,pixels_y)
    print(haversine_distance.shape)
    distance = np.sqrt(haversine_distance**2+(pixels_alt-z_drone)**2)
    distance[np.where(pixels_invalid)] = 0
  
    return distance

def compute_depth_EPFL(drone, pixels):
    c,r = pixels[:,:,0].shape

    x_drone,y_drone,z_drone,_,_,_ = drone
    drone_coord = [(x_drone,y_drone,z_drone)]
    pixels_x = pixels[:,:,0]
    pixels_invalid = np.equal(pixels_x,-1)

    distance = np.sqrt(np.sum((drone_coord-pixels)**2,axis=2))
    distance[np.where(pixels_invalid)] = 0
  
    return distance

def py_to_png(depth_array, output_path):
    z = (65535*((depth_array - depth_array.min())/depth_array.ptp())).astype(np.uint16)
    numpngw.write_png(output_path, z)

def read_one_dir(start_image, end_image, data_path, relative_pointcloud_path):
    #The pose file of the directory
    drone_file_name = pose_file

    #The path to the data
    path_to_data = data_path
    drones = npy_to_py(os.path.join(path_to_data, drone_file_name))

    for j in tqdm(range(int(start_image), int(end_image)), desc='[Computation of depth]'):
        #The path of the point cloud
        relative_path = os.path.join(path_to_data, relative_pointcloud_path+str(j)+"*.npy")
        input_path = glob.glob(relative_path)
        
        pixels = npy_to_py(input_path[0])

        depth_array = compute_depth_EPFL(drones[j],pixels)

        output_path = os.path.join(path_to_data, "depth_{}.png".format(j))
        py_to_png(depth_array,output_path)

#The name of the directory containing the pose file and the point clouds, and the number of images
read_one_dir(start_image, end_image, data_path, relative_pointcloud_path)