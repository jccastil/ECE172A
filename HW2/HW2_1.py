'''
ECE 172A, Homework 2 Robot Traversal
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''
import sys 
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

#np.set_printoptions(threshold=np.inf)

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

##gradient for a point (x,y); (0,0) for initial_location
#gradx = dx[initial_loc[0], initial_loc[1]]
#grady = dy[initial_loc[0], initial_loc[1]]
##create an array for gradient of point x,y	
#gradloc = np.array([gradx, grady])

##obtain magnitude of gradient for point 0,0
#gradlocmag = np.linalg.norm(gradloc)
##Determine what epsilon is

#create next step
#Note: order of derivatives must be established since a has to be 1000+ in order to see significant change for next step.
nextstep = np.array([0,0]) #initialize next step
alpha = 100				#initialize alpha
nextstep[0] = nextstep[0]-alpha*dx[0,0]
nextstep[1] = nextstep[1]-alpha*dy[0,0]
np.set_printoptions(threshold=np.inf)
print(nextstep)
print(nextstep[0])
print(100*dx[0,0])

for i in range(0,10):
	for j in range(0,10):
		gradx = dx[i,j]
		grady = dy[i,j]
		gradloc = np.array([gradx, grady])
		gradlocmag = np.linalg.norm(gradloc)
		nextstep[0] = nextstep[0] - alpha*dx[i,j]
		nextstep[1] = nextstep[1] - alpha*dy[i,j]
		print("i is now: ", i, "j is now: ", j)
		plt.plot(nextstep[0],nextstep[1], marker="*", markersize=20, markeredgecolor="black", markerfacecolor="black")
		plt.show()



plt.show()

##note to self: this is the gradient at point (0,0)
#print("TEST dx is: ", dx[0, 0])
#print("TEST dy is: ", dy[0, 0])
#print("this is", i, j)
