import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
import datetime
import torch.optim as optim
import os

from lcd.models import patchnet
from lcd.models import pointnet
from lcd.dataset import CrossTripletDataset

class LCDNet(pl.LightningModule):
    def __init__(self, args, criterion):
        super(LCDNet, self).__init__()
        self.patchnet = patchnet.PatchNetAutoencoder(args['embedding_size'], args['normalize'])
        self.pointnet = pointnet.PointNetAutoencoder(
            args['embedding_size'],
            args['input_channels'],
            args['output_channels'],
            args['normalize'],
        )
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.gamma = args['gamma']
        self.criterion = criterion
        self.learning_rate = args['learning_rate']
        self.momentum = args['momentum']
        self.weight_decay = args['weight_decay']
        self.parameters = list(self.patchnet.parameters()) + list(self.pointnet.parameters())
        self.args = args
        self.logdir = "logs/LCD"

    def training_step(self, batch, batch_idx):
        x, y = batch
        y0, z0 = self.pointnet(x[0])
        y1, z1 = self.patchnet(x[1])

        loss_r = 0
        loss_d = 0
        loss_r += self.alpha * self.criterion["mse"](x[1], y1)
        loss_r += self.beta * self.criterion["chamfer"](x[0], y0)
        loss_d += self.criterion * self.criterion["triplet"](z0, z1)
        loss = loss_d + loss_r

        now = datetime.datetime.now()
        log = "{} | Batch [{:04d}/{:04d}] | loss: {:.4f} |"
        log = log.format(now.strftime("%c"), batch_idx, self.total_data, loss.item())
        print(log)

        fname = os.path.join(self.logdir, "train.log")
        with open(fname, "a") as fp:
            fp.write(log + "\n")

        return pl.TrainResult(loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(
        self.parameters,
        lr=self.learning_rate,
        momentum=self.momentum,
        weight_decay=self.weight_decay
        )
        return optimizer
        
    def train_dataloader(self):
        dataset = CrossTripletDataset(self.args["root"], split="train")
        loader = data.DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            pin_memory=True,
            shuffle=True,
        )
        self.total_data = len(loader)
        return loader