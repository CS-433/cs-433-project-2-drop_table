import os
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from lcd.models import lcdnet
from lcd.losses import *

config = "config.json"
logdir = "logs/LCD"
args = json.load(open(config))

if not os.path.exists(logdir):
    os.mkdir(logdir)

fname = os.path.join(logdir, "config.json")
with open(fname, "w") as fp:
    json.dump(args, fp, indent=4)

criterion = {
    "mse": MSELoss(),
    "chamfer": ChamferLoss(args["output_channels"]),
    "triplet": HardTripletLoss(args["margin"], args["hardest"]),
}

lcdnet = lcdnet.LCDNet(args, criterion)

trainer = pl.Trainer(gpus=2, num_nodes=4, accelerator='ddp')

trainer.fit(lcdnet)