import argparse
import numpy as np

# Compute the obtained precision for a given sample of a dataset and a given threshold

parser = argparse.ArgumentParser()
parser.add_argument('--pairfile', help='path to file containing the 3d matching')
parser.add_argument('--threshold', help='path to file containing the 3d matching')

args = parser.parse_args()

threshold = int(args.threshold)

pairfile = args.pairfile

matchings = np.load(pairfile)
matchings = np.delete(matchings, np.where(matchings[:,] == -1), axis=0)
predictions, true_value = matchings[:, :3], matchings[:, 3:]
distances = np.sqrt(np.sum((true_value - predictions)**2, axis=1))

isCorrect = np.count_nonzero(distances < threshold)


print("{} % of correct matching".format(100 * isCorrect / matchings.shape[0]))
