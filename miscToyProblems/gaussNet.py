# radial bsis function network

# libraries
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

# functions
def gauss(sigma, error):
	return 1/sigma * np.exp(- error / sigma**2)

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def calc_X1(X0, M, sigma, m, n1):
	# pdb.set_trace()
	error = np.zeros((n1, m))
	for i in range(0, n1):
		for j in range(0, m):
			error[i][j] = np.linalg.norm(M[i] - X0.T[j])**2
			# M is n1 by n0 and X0 is n0 by m
	return gauss(sigma, error)


def calc_X2(W2, X1, b2):
	return sigmoid(np.dot(W2, X1) + b2)

def cost(X2, Y, m):
	return -1/m * ( np.dot(Y, np.log(X2.T)) + np.dot(1-Y, np.log(1-X2.T))) [0] 

def calc_Z2(X2, Y):
	return X2 - Y

def calc_dW2(dZ2, X1, m):
	return np.dot(dZ2, X1.T) / m

def calc_db2(dZ2, m):
	return np.sum(dZ2) / m

def plot_m(M, n1, ax):
	for i in range(0, n1):
		ax.scatter(M[i][0], M[i][1], c="k")

def X0_M_diff_sum(X0, M, m):
	error = np.zeros((M.shape[0], M.shape[1]))
	for j in range(0, m):
		error += M - X0.T[j]
	return error

def calc_dM(dZ2, W2, X1, X0, M, sigma, m):
	# pdb.set_trace()
	# print(dZ2.shape)
	# print(W2.shape)
	# print(X1.shape)
	# print(X0.shape)
	# print(M.shape)
	return np.squeeze(np.dot(np.dot(dZ2, X1.T), W2.T)) * ( 2* X0_M_diff_sum(X0, M, m) / sigma**2 ) / m

def ez_m(W2, X1, X0, M, Y, X2, sigma, m, n1):
	dm1 = np.zeros((n1,1))
	dm2 = np.zeros((n1,1))

	pdb.set_trace()

	for ex in range(0, m):
		t1 = X2[0][ex] - Y[ex]
		t2 = np.dot(W2, X1[ex][0])
		t3 = X0[ex][0] - M[0]
		t4 = 2 / sigma**2
		t5 = t1*t2*t3*t4
		dm1 += t5

		t1 = X2[0][ex] - Y[ex]
		t2 = np.dot(W2, X1[ex][0])
		t3 = X0[ex][0] - M[1]
		t4 = 2 / sigma**2
		t5 = t1*t2*t3*t4		
		dm2 += t5

	return np.array(dm1, dm2)

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
	return (float(pr), 0, float(1-pr))

def plot_boundary(M, sigma, W2, b2, ax):
	boundsx = [-5, 5]
	boundsy = [-5, 5]

	samples = [10, 10]

	width = (boundsx[1] - boundsx[0]) / samples[0]
	height = (boundsy[1] - boundsy[0]) / samples[1]	

	pt = np.zeros((1,2))
	for x in np.linspace(boundsx[0], boundsx[1], samples[0]):
		for y in np.linspace(boundsy[0], boundsy[1], samples[1]):
			pt[0][0] = x
			pt[0][1] = y
			X1_cur = calc_X1(pt, M, sigma, 1, n1)
			X2_cur = calc_X2(W2, X1_cur, b2)
			# ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=probability_hash(X2_cur)))
			ax.scatter(x, y, c=probability_hash(X2_cur))


# def plot_boundary2(X2, ax):
# 	ax.imshow(X0, cmap='hot', interpolation='nearest')


# initialize parameters
file = "data/disk2.csv"
df = pd.read_csv(file)

# pdb.set_trace()

sigma = 1
itterations = 10000
learning_rate = 1.0

n0 = 2
X0 = np.row_stack((df["0"], df["1"]))  # shape is (n0, m)
Y = np.array(df["2"])

m = len(Y)

n1 = 100
M = np.random.randn(n1, n0)*2
X1 = calc_X1(X0, M, sigma, m, n1)

n2 = 1
small_number = 0.01
W2 = np.random.randn(1, n1) * small_number
b2 = 0
X2 = calc_X2(W2, X1, b2)

# pdb.set_trace()
J = cost(X2, Y, m)

fig = plt.figure()
costAx = fig.add_subplot(312)
plotAx = fig.add_subplot(311)
pointsAx = fig.add_subplot(313)

for i in range(0, itterations):
	# forward propogation
	X1 = calc_X1(X0, M, sigma, m, n1)
	X2 = calc_X2(W2, X1, b2)
	J = cost(X2, Y, m)

	# pdb.set_trace()

	# backward propogation
	dZ2 = calc_Z2(X2, Y)
	db2 = calc_db2(dZ2, m)
	dW2 = calc_dW2(dZ2, X1, m)
	dM = ez_m(W2, X1, X0, M, Y, X2, sigma, m, n1)
	# dM = calc_dM(dZ2, W2, X1, X0, M, sigma, m)

	b2 -= learning_rate * db2
	W2 -= learning_rate * dW2
	M  -= learning_rate * dM

	if i % 10 == 0:
		print("cost = " + str(J))
		costAx.scatter(i, J, c="k")
		pointsAx.clear()
		plotAx.clear()
		plot_train_data(X0, Y, m, pointsAx)
		plot_boundary(M, sigma, W2, b2, plotAx)
		plot_m(M, n1, pointsAx)
		plt.pause(0.1)

plot_boundary(M, sigma, W2, b2, plotAx)

plt.show()

