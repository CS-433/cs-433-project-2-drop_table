import os
import h5py
import glob
import bisect
import numpy as np
import torch.utils.data as data


class CrossTripletDataset(data.Dataset):
    def __init__(self, root, split):
        self.points = []
        self.images = []
        self.flist = os.path.join(root, "*.h5")
        self.flist = sorted(glob.glob(self.flist))
        print("Loading the dataset into ram")
        for fname in self.flist:
            print(f'loading file {fname}', flush = True)
            pts, ims = self._load_data(fname)
            self.points += list(pts)
            self.images += list(ims)
        print(len(self.points), " elements loaded", flush=True)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i], self.images[i]

    def _load_data(self, fname):
        h5 = h5py.File(fname, "r")
        return h5["points"][:], h5["images"][:]