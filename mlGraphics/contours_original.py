# simple network to demonstrate what the contours of the cost
# function look like in various scenarios

import matplotlib.pyplot as plt
import numpy as np
import pdb

m = 200
X = np.random.randn(2, m)
Y = np.zeros(m)

p_range = [[-3, 3], [-3, 3]]

# generate the data
for i in range(0, m):
	if 1*X[0][i] - 1*X[1][i] < 0:
		Y[i] = 1

def display_data(m, X, Y, ax=None):
	for i in range(0, m):
		color = 'r'
		if Y[i] == 1:
			color = 'g'
		if ax == None:
			plt.scatter(X[0][i], X[1][i], c=color)
		else:
			ax.scatter(X[0][i], X[1][i], c=color)

def display_data_error(m, X, Y, A, ax=None):
	for i in range(0, m):
		color = 'r'
		if Y[i] == 1:
			color = 'g'
		if Y[i] != np.round(A[i]):
			color='k'
		if ax == None:
			plt.scatter(X[0][i], X[1][i], c=color)
		else:
			ax.scatter(X[0][i], X[1][i], c=color)

	
def display_error(W, b, ax=None):
	nxs = 10; nys = 10;
	xr = p_range[0];	yr = p_range[1];

	heats = [[0 for i in range(0, nxs)] for j in range(0, nys)]
	
	xs = np.linspace(xr[0], xr[1], nxs)
	ys = np.linspace(yr[0], yr[1], nys)
	
	for i in range(0, nys):
		for j in range(0, nxs):
			Z = W[0][0]*xs[j] + W[0][1]*ys[i] + b
			A = sigmoid(Z)
			heats[i][j] = A

	if ax == None:
		plt.imshow(heats, extent=[p_range[0][0],p_range[0][1],p_range[1][1],p_range[1][0]])  # wierd range to reverse from matrix indexing 
		# (top is smaller to grid indexing top is larger)
	else:
		ax.imshow(heats, extent=[p_range[0][0],p_range[0][1],p_range[1][1],p_range[1][0]])


def percent_error(Y, A, m):
	return np.sum(np.abs(Y-np.round(A))) / m


# what is the cost asociated with a certain choice of W and b, given the data
def J_Mb(Wx, Wy, b, X, Y):
	Z = np.dot(np.array([[Wx, Wy]]), X) + b
	A = sigmoid(Z)[0]
	return avg(cross_entropy(Y, A))


def J_contours(X, Y, b, ax=None):
	nxs = 200; nys = 200;
	xr = [-30,30];	yr = [-30,30];

	heats = [[0 for j in range(0, nxs)] for i in range(0, nys)]
	
	wxs = np.linspace(xr[0], xr[1], nxs)
	wys = np.linspace(yr[0], yr[1], nys)
	
	for i in range(0, nys):
		for j in range(0, nxs):
			heats[i][j] = J_Mb(wxs[j], wys[i], b, X, Y)

	return heats, wxs, wys

def plot_J_stuff(X, Y, b, ws=[], js=[], ax=None, ax3d=None):
	heats, wxs, wys = J_contours(X, Y, b, ax=ax)
	rxw, ryw = np.meshgrid(wxs, wys)

	if ax == None:
		# plt.contour(X, Y, Z)
		plt.imshow(heats, extent=[-30,30,-30,30])
	else:
		# ax.imshow(heats, extent=[-10,10,-10,10])
		ax.contour(rxw, ryw, np.array(heats))

	if ax3d != None:	
		ax3d.plot_surface(rxw, ryw, np.array(heats))
		ax3d.scatter([w[0][0] for w in ws], [w[0][1] for w in ws], js, c='r')
		ax3d.plot([w[0][0] for w in ws], [w[0][1] for w in ws], js, c='k')


def sigmoid(Z):
	return 1 / ( 1 + np.exp(-Z) )

def cross_entropy(Y, A):
	eps = 10**(-8)
	return -Y*np.log(A+eps) - (1-Y)*np.log(1-A+eps)

def avg(a):
	return np.sum(a) / len(a)

# we will use a linear decision boundary
# AKA take Z = np.dot(W, X) + b 
# and then do A = sigmoid(Z) element wise for predictions

# then compute the cost
# we use J(A; Y) = cross entropy = -Y*log(A) - (1-Y)*log(1-A)
# average this over all training examples for total cost

# the backpropogation is also trivial

# dJ/dA = dA = (A-Y) / ((A)(1-A))
# dZ = A - Y
# db = average(dZ)
# Note Z[i] = X[0][i]*W[0] + X[1][i]*W[1]
# thus dZ[i]/dW[j] = X[j][i]
# But those are already the dimensions of X
# dJ/dW[j] = dJ / dZ * dZ / dW[j] 
# the multipliction show above is done ELEMENT WISE
# every training example has an asociated dJ/dZ[i] and dZ[i]/dW[j]
# thus simply np.dot(dZ, X.T) gives dW



W = np.random.randn(1, 2)*0.01
W = np.array([[2.,1.3]])
b = 0

itterations = 10**6
alpha = 10**(0)*4

fig = plt.figure()
# note the args in add_subplot are grid size 1 grid size 2 and then which one it is
# not what I thought previously
cost_p = fig.add_subplot(2,2,1)
data_p = fig.add_subplot(2,2,2)
contour_p = fig.add_subplot(2,2,3)

fig3d = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
d3_p = Axes3D(fig3d)

plt_gap = 50

ws = []
js = []

for it in range(0, itterations):
	# pdb.set_trace()
	# forward propogation
	Z = np.dot(W, X) + b
	A = sigmoid(Z)[0]

	J = avg(cross_entropy(Y, A))

	# backwards propogation
	dZ = A - Y
	db = avg(dZ)
	dW = np.dot(dZ, X.T) / m

	# W[0][0] -= alpha * dW[0]
	W -= alpha * dW
	# b -= alpha * db

	# plot it (sometimes)
	if it % plt_gap == 0:

		# plt.cla()
		# plt.axis((-3,3,-3,3))
		data_p.clear()
		data_p.set_xlim(-3, 3)
		data_p.set_ylim(-3, 3)
		display_error(W, b, ax=data_p)
		display_data_error(m, X, Y, A, ax=data_p)
		
		cost_p.scatter(it, J)

		print(percent_error(Y, A, m), W, b)
		
		ws.append(np.copy(W))
		js.append(J)
		d3_p.clear()
		plot_J_stuff(X, Y, b, ws = ws, js = js, ax=contour_p, ax3d=d3_p)


		plt.pause(0.2)
		# pdb.set_trace()

plt.show()













"""
https://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
# Generate some test data
x = np.random.randn(8873)
y = np.random.randn(8873)

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()


import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)


# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')

"""