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
#plt.show()


##1.1 (i) plot the 2D contour plot of the vector field. 
##(ii) plot the gradient as quivers on the same plot as the contour plot. 
#determine X and Y positions

#meshgrid returns coordinate matrices from coordinate vectors
X, Y = np.meshgrid(x, y)
#determine dx and dy (notice order of parameters)
dy, dx = np.gradient(z)


fig, ax = plt.subplots(figsize=(8,8))
ax.quiver(X, Y, dx, dy)
plt.contour(x,y,z, 12)
plt.title("2D Contour")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-10, 110])
plt.ylim([-10, 110])
plt.colorbar()
#add asterisk for initial/final loc in contour quiver plot
plt.plot(0,0, marker="*", markersize=20, markeredgecolor="black", markerfacecolor="blue")
plt.plot(100,100, marker="*", markersize=20, markeredgecolor="black", markerfacecolor="red")
#implement the gradient descent algorithm to navigate the potential field.
##Testing
x_old = initial_loc
x_new = initial_loc - 1*(dx[5, 5], dy[5, 5])
print("gradient at (0,0) is: ", dx[5, 5],dy[5,5])
print("x_new is: ", x_new)
plt.plot(x_new, marker="*", markersize=20)
plt.show()


#test print the the magnitude of the gradient at position (0,0) 
print("magnitude of (0,0) is:", np.linalg.norm([dx[0,0], dy[0, 0]]))

##note to self: this is the gradient at point (0,0)
#print("TEST dx is: ", dx[99, 99])
#print("TEST dy is: ", dy[99, 99])
#print("this is", i, j)
