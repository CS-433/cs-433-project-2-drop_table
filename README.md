# cs-433-project-2-drop_table

# Introduction

The original repositery for the LCD network can be found [here](https://github.com/hkust-vgd/lcd).

The network described in it correspond to the lcd folder, and the train.py file in our repositery.

We also have a modified version that uses pytorch-lightning to exploit multiple nodes on a cluster.

The bash scripts used to train the networks are in the "run scripts" folder.

To choose which version to train, simply rename the directory containing the source code to lcd and the corresponding training script as "train.py".

# Dependencies to install
	
	os
    sys
    torch
    sklearn
    h5py
    bisect
    json
    tqdm
    argparse
    cv2
    numpy
    matplotlib
    PIL
    glob
    open3d-python (version 0.7.0)
    

# How to train the Network

Depending on your configuration, you may needs to change the line 12 in the train.py:
1. If you are using a computer with less ram than the size of your training size :

        from lcd.dataset import CrossTripletDataset

2. If you have a lot of ram and are generating your data with regulars pairs of corresponding images/patches :

        from lcd.dataset_V1 import CrossTripletDataset

3. If you have a lot of ram are generating your data using LCD author's script as we explained in our report :

        from lcd.dataset_V1 import CrossTripletDataset

You will also need to specify the path to your data in the config.json.
You may also change the others parameters according to your needs.

Warning : With loading all data in the RAM and Nvidia v100 GPUs, the differents trainings took us 17 hours each, you may want to run the trainings on cluster or powerfull device.

We also provide a run.sh file designed to run the train.py script on EPFL's Izar cluster that run on slurm.

# How to use a pretrained network

Additionnaly to the authors's pretrained network, you can find in the folder logs/ the severals networks we trained as explained in our report. In all our application, the command line argument --logdir allows to choose which network to use.

# How to run the applications

## 2D-2D match
**compute_2D_2D_matching** : Compute and display the best matches between 2 images

Usage :
    
    python3 -m apps.2D-2D_match.compute_2D_2D_matching [path_to_src] [path_to_dst] --logdir [path_to_log] --number_of_matches [nb_of_matches]

Example :
    
    python3 -m apps.2D-2D_match.compute_2D_2D_matching samples/comballaz-air2_5_DJI_0003_f2_img.png samples/comballaz-air2_7_DJI_0004_f2_img.png --logdir logs/LCD-comballaz-mix --number_of_matches 20

    python3 -m apps.2D-2D_match.compute_2D_2D_matching samples/100_0039_0352.JPG samples/100_0039_0353.JPG --logdir logs/LCD-D256 --number_of_matches 50


**compute_2D_2D_matching_precision** : Compute matches for all 2 consecutives files in a specified folder and store all the found matches in a numpy persistant file.

Usage : 

    python3 -m apps.2D-2D_match.compute_2D_2D_matching_precision --logdir [path_to_log] --imagesdir [path_to_img_folder] --save_file [name_of_numpy_file] --dataset [EPFL or Comballaz]

Example : 

    python -m apps.2D-2D_match.compute_2D_2D_matching_precision --logdir logs\LCD-D256 --imagesdir comballaz-trajectory --save_file comballaz-D256

## 2D-3D match

**compute_2D_3D_matching_3D_match** : Compute matches between an image and a depth image both from the 3D match dataset, the configurations are donne inside the script

Usage :

    python -m apps.2D-3D_match.compute_2D_3D_matching_3D_match --logdir [log_dir]

Example :

    python -m apps.2D-3D_match.compute_2D_3D_matching_3D_match --logdir logs\LCD-D256

**compute_2D_3D_matching** : Compute matches beetween an image and a colored point cloud

Usage :

    python3 -m apps.2D-3D_match.compute_2D_3D_matching [source_img] [dest_img] [dest_point_cloud] --logdir [path_to_log_dir] --voxel_size [voxel_size] --radius [radius] --num_points [num_point]

Example :

    python3 -m apps.2D-3D_match.compute_2D_3D_matching samples/comballaz-air2_5_DJI_0003_f2_img.png samples/comballaz-air2_7_DJI_0004_f2_img.png samples/comballaz-air2_5_DJI_0003_f2_pc.npy --logdir logs/LCD-comballaz-mix --voxel_size 30 --radius 80 --num_points 1024

**compute_2D_3D_matching_precision** : Compute matches for all 2 consecutives images and pointcloud in a specified folder and store all the found matches in a numpy persistant file

Usage :

    python -m apps.2D-3D_match.compute_2D_3D_matching_precision --logdir [path_to_log_dir] --imagesdir [path_to_image_dir] --save_file [name_of_the_output_file]


Example :
    
    python -m apps.2D-3D_match.compute_2D_3D_matching_precision --logdir logs\LCD-comballaz_synth --imagesdir comballaz-trajectory --save_file comballaz-D256



## Sparse to dense point cloud

**compute_sp_dense** : Iterate over images and their corresponding point clouds, downsample the point cloud, reconstruct it using LCD and then store the matches between this reconstructed point cloud and the original one in a npy file.

Usage :
    
    python -m apps.sparse_to_dense.compute_sp_dense [path_to_image] [path_to_point_cloud] --logdir [path_to_logs]


Example :

    python -m apps.sparse_to_dense.compute_sp_dense samples/comballaz-air2_5_DJI_0003_f2_img.png samples/comballaz-air2_5_DJI_0003_f2_pc.npy --logdir logs/LCD-comballaz_synth


## Precisions Tools

**compute_precisions** : Compute the precision of the matches contained in a npy file

Usage :

    python3 -m apps.precision_tools.compute_precisions --pairfile [path_to_the_npy_file] --threshold [threshold_for_valid_distances]

Example :

    python3 -m apps.precision_tools.compute_precisions --pairfile precisions/2D-3D/comballaz/comballaz-prec-comballaz-mix.npy --threshold 200

**compute_precisions** : Compute the precision for all files in a folder and compare them in a figure

Usage :

    python3 -m apps.precision_tools.compare_precisions --dir [path_to_the_npy_file] --threshold_max [threshold_maximum for a range] --step [step for a range]

Example :

    python3 -m apps.precision_tools.compare_precisions --dir precisions/2D-3D/comballaz/ --threshold_max 2000 --step 1

## 3D-3D Match

The script was given by the LCD's paper author, it is detailled on his repository.