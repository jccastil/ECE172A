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
plt.contour(x,y,z, 13)
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
alpha = 150 #initialize alpha

#For loop method
# nextstep = np.array([0,99]) #initialize next step to have starting point coordinates
# for test in range(500):
# 	nextstep[0] = nextstep[0]-round(alpha*dx[nextstep[1],nextstep[0]])
# 	nextstep[1] = nextstep[1]-round(alpha*dy[nextstep[1],nextstep[0]])
# 	plt.plot(nextstep[0], nextstep[1], marker="*", markersize=20, markeredgecolor="black", markerfacecolor="black")

#For loop method2
nextstep = np.array([0,0])
xnextgrad = dx[nextstep[1],nextstep[0]]
ynextgrad = dy[nextstep[1],nextstep[0]]
nextgrad = np.array([xnextgrad, ynextgrad])
gradmag = np.linalg.norm(nextgrad)
#create while loop
while(gradmag > 0.0001):
	print("befor next step is: ",nextstep)
	xnextgrad = dx[nextstep[1], nextstep[0]]
	ynextgrad = dy[nextstep[1], nextstep[0]]
	nextgrad = np.array([xnextgrad, ynextgrad])
	nextstep[0] = nextstep[0]-np.round(alpha*xnextgrad)
	nextstep[1] = nextstep[1]-np.round(alpha*ynextgrad)
	print("now nextstep is: ",nextstep)

	#attempt to plot begin
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.quiver(X, Y, dx, dy)
	plt.contour(x, y, z, 13)
	plt.title("2D Contour")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim([-10, 110])
	plt.ylim([-10, 110])
	plt.colorbar()
	# add asterisk for initial/final loc in contour quiver plot
	plt.plot(0, 0, marker="*", markersize=20, markeredgecolor="black", markerfacecolor="blue")
	plt.plot(100, 100, marker="*", markersize=20, markeredgecolor="black", markerfacecolor="red")
	#attempt to plot end
	plt.plot(nextstep[0], nextstep[1], marker="*", markersize=20, markeredgecolor="black", markerfacecolor="black")
	plt.show()







# While loop that needs to be fixed.
# nextgrad = np.array([dx[0,0],dy[0,0]]) #hold the gradient of initial location. Will get updated in while loop.
# location0 = np.array([0,0])
# print(np.linalg.norm(nextgrad))
# print(nextgrad[0], nextgrad[1])
# while np.linalg.norm(nextgrad) > 0.00001:
# 	print("location0 begin:",location0)
# 	location0[0] = location0[0]- np.round(alpha*(nextgrad[0]))
# 	location0[1] = nextgrad[1]- np.round(alpha*(nextgrad[1]))
# 	print("locatin0 middleL: ",location0)
# 	nextgrad[0] = dx[location0[1],location0[0]]
# 	nextgrad[1] = dy[location0[1],location0[0]]
# 	print("nextgrad end: ",nextgrad)







np.set_printoptions(threshold=np.inf)
plt.show()

##note to self: this is the gradient at point (0,0)
#print("TEST dx is: ", dx[0, 0])
#print("TEST dy is: ", dy[0, 0])
#print("this is", i, j)
