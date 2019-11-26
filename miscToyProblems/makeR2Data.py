# makes cool data in R2 to learn

# libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

# dimension of data, R3 would be cool...
n = 2
# number of exaples
m = 300

X = []
Y = []


"""
method 0
linear boundary
"""

name = "data/linear"
plt.cla()
for ex in range(0, m):
	xCur = np.random.randn(2)
	X.append(xCur)
	if xCur[0] + 2*xCur[1] > 1:
		Y.append(1)
		color="r"
	else:
		Y.append(0)
		color="b"
	plt.scatter(xCur[0], xCur[1], c=color)
	if abs(np.random.randn()) < 0.01:
		plt.pause(0.1)
plt.pause(1)
plt.savefig(name + ".png")

X = np.array(X)
Y = np.array(Y)

df = pd.DataFrame(X)
df[2] = Y
df.to_csv(name + ".csv", index=False)

pdb.set_trace()


"""
Data set1

A kinda disk in the middle
"""

# hyperparamers for data
rApprox = 1
error = 0.4
noise = 0.1

name = "data/disk2"

plt.cla()
for ex in range(0, m):
	xCur = np.random.randn(2)
	X.append(xCur)
	if abs(np.linalg.norm(xCur) + np.random.randn()*noise - rApprox) < error:
		Y.append(1)
		color="r"
	else:
		Y.append(0)
		color="b"
	plt.scatter(xCur[0], xCur[1], c=color)
	if abs(np.random.randn()) < 0.01:
		plt.pause(0.1)
plt.pause(1)
plt.savefig(name + ".png")

X = np.array(X)
Y = np.array(Y)

df = pd.DataFrame(X)
df[2] = Y
df.to_csv(name + ".csv", index=False)




