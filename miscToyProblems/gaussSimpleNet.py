# simple gaussian network
# radial bsis function network

# libraries
import matplotlib.patches as patches
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb


# functions

def gaussian(sq_error, sigma):
	return ((1/np.sqrt(2*np.pi*sigma**2))) * np.exp(-(sq_error)/(2*sigma**2))

def calc_X1(X0, Mx, My, m, sigma):
	X1 = [] # shape will be (10, m)
	for ex in range(0, m):
		sq_error = (X0[0][ex] - Mx) **2 + (X0[1][ex] - My) **2
		X1.append(gaussian(sq_error, sigma))
	X1 = np.array(X1)
	return X1.T

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def calc_X2(W2, X1, b2):
	return sigmoid(np.dot(W2, X1) + b2)

def cost(X2, Y, m):
	return -1/m * ( np.dot(Y, np.log(X2.T)) + np.dot(1-Y, np.log(1-X2.T))) [0] 

def calc_dZ2(X2, Y):
	return X2 - Y

def calc_dM(dZ2, W2, X1, sigma, M, m, xOrY, X0):
	cur_dM = np.zeros(M.shape)
	for i in range(0, m):
		# pdb.set_trace()
		cur_dM += dZ2[0][i] * float(np.dot(W2, X1.T[i])) * 1/sigma**2 * (X0[xOrY][i] - M)
	return cur_dM / m


def train_correct(X2, Y, m):
	ct = 0
	for i in range(0, m):
		if np.round(X2[0][i]) == Y[i]:
			ct += 1
	return ct / m


# graphing functions
def plot_train_data(X, Y, m, ax):
	for ex in range(0, m):
		xCur = X[0][ex]
		yCur = X[1][ex]
		if Y[ex] == 1:
			color=(1, 0, 0)
		else:
			color=(0,0,1)
		ax.scatter(xCur, yCur, c=color)

def probability_hash(pr):
	return (float(pr), float(np.round(pr)), float(1-pr))

def probability_hash_1d(pr):
	return float(pr)

def plot_boundary(Mx, My, sigma, W2, b2, ax):
	boundsx = [-5, 5]
	boundsy = [-5, 5]

	samples = [10, 10]

	width = (boundsx[1] - boundsx[0]) / samples[0]
	height = (boundsy[1] - boundsy[0]) / samples[1]	

	pt = np.zeros((2,1))
	for x in np.linspace(boundsx[0], boundsx[1], samples[0]):
		for y in np.linspace(boundsy[0], boundsy[1], samples[1]):
			pt[0][0] = x
			pt[1][0] = y
			X1_cur = calc_X1(pt, Mx, My, 1, sigma)
			X2_cur = calc_X2(W2, X1_cur, b2)
			# ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=probability_hash(X2_cur)))
			ax.scatter(x, y, c=probability_hash(X2_cur))


def cool_plot_boundary(Mx, My, sigma, W2, b2, ax):
	boundsx = [-2, 2]
	boundsy = [-2, 2]

	samples = [50, 50]

	width = (boundsx[1] - boundsx[0]) / samples[0]
	height = (boundsy[1] - boundsy[0]) / samples[1]	

	pt = np.zeros((2,1))

	heats = []

	xs = np.linspace(boundsx[0], boundsx[1], samples[0])
	ys = np.linspace(boundsy[0], boundsy[1], samples[1])

	for x in xs:
		heats.append([])
		for y in ys:
			pt[0][0] = x
			pt[1][0] = y
			X1_cur = calc_X1(pt, Mx, My, 1, sigma)
			X2_cur = calc_X2(W2, X1_cur, b2)
			heats[-1].append(probability_hash_1d(X2_cur))

	# xticks = []
	# yticks = []
	# for i in range(0, len(xs)):
	# 	if i % 3 == 0:
	# 		xticks.append(round(xs[i], 2))
	# for i in range(0, len(ys)):
	# 	if i % 3 == 0:
	# 		yticks.append(round(ys[i], 2))

	xticks = []
	yticks = []

	sns.heatmap(heats, ax=ax, cbar=True, xticklabels=xticks, yticklabels=yticks)

def plot_m(Mx, My, n1, ax):
	for i in range(0, n1):
		ax.scatter(Mx[i], My[i], c="k")


# initialize parameters
file = "data/linear.csv"
df = pd.read_csv(file)

sigma = 2
itterations = 10000
learning_rate = 0.9

n0 = 2  # DO NOT CHANGE, formality
X0 = np.row_stack((df["0"], df["1"]))  # shape is (2, m)
Y = np.array(df["2"])

m = len(Y)

n1 = 50
Mx = np.random.randn(n1)
My = np.random.randn(n1)
X1 = calc_X1(X0, Mx, My, m, sigma)

n2 = 1  # DO NOT CHANGE, formality
small_number = 0.01
W2 = np.random.randn(1, n1) * small_number
b2 = 0
X2 = calc_X2(W2, X1, b2)

J = cost(X2, Y, m)
Js = []
itters = []

fig = plt.figure()
plotGap = 200


for i in range(0, itterations):
	X1 = calc_X1(X0, Mx, My, m, sigma)
	X2 = calc_X2(W2, X1, b2)

	J = cost(X2, Y, m)

	if i % plotGap == 0:
		fig.clear()
		costAx = fig.add_subplot(311)
		plotAx = fig.add_subplot(312)
		pointsAx = fig.add_subplot(313)
		cool_plot_boundary(Mx, My, sigma, W2, b2, plotAx)
		# plot_boundary(Mx, My, sigma, W2, b2, plotAx)
		plot_train_data(X0, Y, m, pointsAx)
		Js.append(J)
		itters.append(i)
		costAx.plot(itters, Js, c="k")
		print("cost = " + str(J) + "\ttraining correct = " + str(train_correct(X2, Y, m)))
		plot_m(Mx, My, n1, pointsAx)
		plt.pause(0.1)

	dZ2 = calc_dZ2(X2, Y)
	dW2 = np.dot(dZ2, X1.T) / m
	db2 = np.sum(dZ2) / m
	dMx = calc_dM(dZ2, W2, X1, sigma, Mx, m, 0, X0)
	dMy = calc_dM(dZ2, W2, X1, sigma, My, m, 1, X0)

	b2 -= learning_rate * db2
	W2 -= learning_rate * dW2

	Mx -= learning_rate * dMx
	My -= learning_rate * dMy



