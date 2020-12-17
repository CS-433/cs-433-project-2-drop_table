import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='path to the directory containing the files')
parser.add_argument('--threshold_max', help='Treshold maximum to display (in meter) for 2 points to be correctly matched')
parser.add_argument('--step', help='Step between 2 threshold')

args = parser.parse_args()
using_colors = False

matplotlib.rcParams.update({'font.size': 20})

files_path = glob.glob(args.dir+"/*.npy")

def compute_distances(file):
	matchings = np.load(elem)
	matchings = np.delete(matchings, np.where(matchings[:,] == -1), axis=0)
	if "sparse-dense" in args.dir:
		if using_colors:
			color_predictions, true_color = matchings[:,3:6], matchings[:,9:12]
			distances = np.sqrt(np.sum((true_color - color_predictions)**2, axis=1))
		else :		
			predictions, true_value = matchings[:,:3], matchings[:, 6:9]
			distances = np.sqrt(np.sum((predictions - true_value)**2, axis=1))
	else :
		predictions, true_value = matchings[:,:3], matchings[:, 3:]
		distances = np.sqrt(np.sum((predictions - true_value)**2, axis=1))
	return distances, matchings.shape[0]



x=np.arange(0, int(args.threshold_max), float(args.step))
plt.figure()
for elem in files_path:
	distances, size = compute_distances(elem)
	y = [np.count_nonzero(distances < i)/ size for i in x]
	last = elem.split(".")[-1]
	name = elem.replace("."+last,"")
	name = name.replace(last,"")
	rmv = "-".join(name.split("-")[:3])
	name = name.replace(rmv+"-","")
	name = name.replace("sparse-to-dense","")
	plt.plot(x, y, label = name, linewidth=2)

plt.tight_layout()
plt.xlabel('Treshold (in meter) for 2 points to be correctly matched')
plt.ylabel('Percentage of correctly matched points')

plt.legend()
plt.show()


#print("{} % of correct matching".format(100 * isCorrect / matchings.shape[0]))
