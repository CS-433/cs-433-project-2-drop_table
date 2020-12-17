# cs-433-project-2-drop_table

# Introduction

The original repositery for the LCD network can be found [here](https://github.com/hkust-vgd/lcd).

The network described in it correspond to the lcd folder, and the train.py file in our repositery.

We also have a modified version that uses pytorch-lightning to exploit multiple nodes on a cluster.

The bash scripts used to train the networks are in the "run scripts" folder.

To choose which version to train, simply rename the directory containing the source code to lcd and the corresponding training script as "train.py".

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

We also provide a run.sh file designed to run the train.py script on EPFL's Izar cluster that run on slurm.

# How to use a pretrained network

Additionnaly to the authors's pretrained network, you can find in the folder logs/ the severals networks we trained as explained in our report. In all our application, the command line argument --logdir allows to choose which network to use.

# How to run the applications

## 2D-2D match

## 2D-3D match

## sparse to dense point cloud
