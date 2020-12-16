import os
import sys
import json
from tqdm import tqdm
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import KDTree


parser = argparse.ArgumentParser()
parser.add_argument('--pairfile', help='path to file containing the 3d matching')
args = parser.parse_args()

threshold = 5000

pairfile = args.pairfile

matchings = np.load(pairfile)
print(matchings.shape)

matchings = np.delete(matchings, np.where(matchings[:,] == -1), axis=0)
print(matchings.shape)
print(matchings)
predictions, true_value = matchings[:, :3], matchings[:, 3:]

distances = np.sqrt(np.sum((true_value - predictions)**2, axis=1))

print(distances)

isCorrect = np.count_nonzero(distances < threshold)

print("{} % of correct matching".format(100 * isCorrect / matchings.shape[0]))
