'''
ECE 172A, Homework 2 Robot Traversal
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import numpy as np
import matplotlib.pyplot as plt

initial_loc = np.array([0,0])
final_loc = np.array([100,100])
sigma = np.array([[50,0],[0,50]])
mu = np.array([[60, 50], [10, 40]])

def f(x, y):
	return ((final_loc[0]-x)**2 + (final_loc[1]-y)**2)/20000 + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[0,0], y-mu[0,1]]),np.matmul(np.linalg.pinv(sigma), np.atleast_2d(np.array([x-mu[0,0], y-mu[0,1]])).T)))[0]) + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[1,0], y-mu[1,1]]),np.matmul(np.linalg.pinv(sigma), np.array([x-mu[1,0], y-mu[1,1]])))))

x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
z = f(x[:,None], y[None,:])
z = np.rot90(np.fliplr(z))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, z, 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Contour')
plt.show()


##1.1 (i) plot the 2D contour plot of the vector field. 
##(ii) plot the gradient as quivers on the same plot as the contour plot. 
#determine X and Y positions
X, Y = np.meshgrid(x, y)
#determine dx and dy (notice order of parameters)
dy, dx = np.gradient(z)

#object initial position
xinit = [20]
yinit = [10]
#object final position
xfin = [90]
yfin = [90]

fig, ax = plt.subplots(figsize=(8,8))
ax.quiver(X, Y, -dx, -dy)
plt.contour(x,y,z, 12, angles='xy', scale_units='xy')
plt.title("2D Contour")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
#plt.plot(xinit, yinit, marker="*", markersize=15, markeredgecolor="black", markerfacecolor="blue")
#plt.plot(xfin, yfin, marker="*", markersize=15, markeredgecolor="black", markerfacecolor="red")

#implement the gradient descent algorithm to navigate the potential field.

plt.show()