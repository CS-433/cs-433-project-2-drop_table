# cs-433-project-2-drop_table

# Introduction

The original repositery for the LCD network can be found [here](https://github.com/hkust-vgd/lcd).

The network described in it correspond to the lcd folder, and the train.py file in our repositery.

We also have a modified version that uses pytorch-lightning to exploit multiple nodes on a cluster.

The bash scripts used to train the networks are in the "run scripts" folder.

To choose which version to train, simply rename the directory containing the source code to lcd and the corresponding training script as "train.py".

# Dependencies to install
	
	opencv-python

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
    
    python3 apps.2D-2D_match.compute_2D_2D_matching.py [path_to_src] [path_to_dst] --logdir [path_to_log] --number_of_matches [nb_of_matches]

Example :
    
    python3 apps.2D-2D_match.compute_2D_2D_matching.py samples/comballaz-air2_5_DJI_0003_f2_img.png samples/comballaz-air2_7_DJI_0004_f2_img.png --logdir logs/LCD-comballaz-mix --number_of_matches 20

**compute_2D_2D_matching_precision** : Compute matches for all 2 consecutives files in a specified folder and store all the found matches in a numpy persistant file.

Usage : 

    python3 apps.2D-2D_match.compute_2D_2D_matching_precision.py --logdir [path_to_log] --imagesdir [path_to_img_folder] --save_file [name_of_numpy_file] --dataset [EPFL or Comballaz]

Example : 

## 2D-3D match

**compute_2D_3D_matching_3D_match** : Compute matches for all 2 consecutives images and pointcloud in a specified folder and store all the found matches in a numpy persistant file

## sparse to dense point cloud
