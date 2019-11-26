import pandas as pd
import numpy as np
import json
import sys
import pdb
import os
import re
from pprint import pprint

if len(sys.argv) < 3:
	print("input file with weights and folder with data")
	sys.exit(-1)
dataFile = sys.argv[1]
data = json.load(open(dataFile))

dataFolder = sys.argv[2]

# parameters
W = data["W"]
b = data["b"]

m = 0; n = 0;
for f in os.listdir(dataFolder):
	if "_clean.csv" in f:
		curNums = re.findall(r'\d+', f)
		if len(curNums) == 2:
			curNum = int(curNums[1])
			if curNum > m:
				m = curNum + 1

LABEL_TO_ACTIVATION = {
	"1": 1,
	"2": 0
}

ACTIVATION_TO_LABEL = {
	1: "1",
	0: "2"
}

xs = []
ys = []
for f in os.listdir(dataFolder):
	if "_clean.csv" in f:
		c_data = pd.read_csv(os.path.join(dataFolder, f))
		label = f.split("_")[0]
		ys.append(LABEL_TO_ACTIVATION[label])
		xs.append(c_data["data"])
n = len(xs[0])

print("n = {}, m = {}".format(n,m))

# input
Y = np.array(ys)
X = np.array(xs).T


def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

Z = np.dot(W, X) + b
A = sigmoid(Z)

def prob_to_binary(p):
	if p < 0.5:
		return 0
	else:
		return 1


pprint(np.round(A).tolist())
pprint(Y.tolist())


res = (np.abs(np.round(A)-Y)).tolist()
pprint(res)

print("accuracy = {}".format(res.count(0)/len(res)))

