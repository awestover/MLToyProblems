import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import sys
import pdb
import os
import re


# I may have the back prop wrong...

"""

This is a very simple neural network

we will have a single layer which is a sigmoid

For every itteration:
	For every training example (0, m):
		forward prop:
			X[n, 1] (input) -> 
			W[1, n] * X[n, 1] + b[1, 1] = Z[1, m] (linear component) -> 
			sigmoid(Z)[1,1] = A[1,1] (activation / answer)
		backward prop:
			J(theta) = J(W, b; X) = cross entropy = - ( Y * log(A) + (1 - Y) * log(1 - A) ) [1, 1]
			dJ / dA  = dA = ( Y - A ) / ((A)(A - 1)) [1, 1]
			dJ / dZ = dZ = Y - A [1, 1]
			dJ / db = db = Y - A [1, 1]
			dJ / dW = dW = ((Y - A)[1, 1] * (X.T) [1, n])[1, n] 

Or vectorized:
forward:
X[n, m] (input) -> 
W[1, n] * X[n, m] + b[1, m] = Z[1, m] (linear component) -> 
sigmoid(Z)[1, m] = A (activations)

backwards:
J = - ( Y * log(A) + (1 - Y) * log(1 - A) ) [1, m]
dJ / dA  = dA = ( Y - A ) / ((A)(A - 1)) [1, m]
dJ / dZ = dZ = Y - A [1, m]
dJ / db = db = Y - A [1, m]
dJ / dW = dW = ((Y - A)[1, m] * (X.T) [m, n])[1, n] 


"""

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))


def relu(Z):
	return max(0, Z)


if len(sys.argv) < 2:
    print("Trains the model. Input folder for data files to be found in")
    sys.exit(-1)

dataFolder = sys.argv[1]

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
# parameters
W = np.random.randn(1, n)*0.01
b = np.zeros((1, 1))

# hyperparameters
ITERATIONS = 100
learning_rate = 10**(-6)
l = 0.5  # weight for sum of squared errors

# to plot
costs = []

for i in range(0, ITERATIONS):
	# forward propogation
	Z = np.dot(W, X) + b
	A = sigmoid(Z)

	dZ = Y - A
	db = -np.sum(dZ)  # really iffy
	dW = -np.dot(dZ, X.T)

	b -= learning_rate*db
	W -= learning_rate*dW # + l*W


	c_cost = -1/m * np.sum( Y * np.log(A) + (1 - Y) * np.log(1 - A) ) 
	costs.append(c_cost)


	if i % 50 == 0:
		print("cost " + str(c_cost))

		plt.cla()
		plt.scatter(np.arange(0, i+1), costs)
		plt.pause(0.01)
	

plt.cla()
plt.plot(np.arange(0, ITERATIONS), costs)
plt.show()

weights = {
	"W": np.squeeze(W).tolist(),
	"b": np.squeeze(b).tolist()
}

outFile = os.path.join(dataFolder, "logisticRegressionWeights.json")
with open(outFile, 'w') as f:
    json.dump(weights, f)
