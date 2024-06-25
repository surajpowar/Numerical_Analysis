# Authors: Suraj Powar and Prof. Dr. Henri Schurz
# Date: 06/10/2024

#### Simple plot of just the integral in 2D and 3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D

# Parameters
p = 3
f = lambda x : x**p
a = 0
b = 5
N = 10
n = 10 # Use n*N+1 points to plot the function smoothly

# Integral function x^p dx
def integrand(x, p):
    return x**p

# Define the function to calculate the integral
def integral_function(a, b, p):
    result, error = quad(integrand, a, b, args=(p,))
    return result

integral_value = integral_function(a, b, p)
print("Integral value:", integral_value)

# 2D Plot
x = np.linspace(a, b, n*N+1)
y = f(x)

plt.figure()
plt.plot(x, y, label=f'$x^p$')
plt.fill_between(x, y, alpha=0.2)
plt.title('2D Plot of $x^p$')
plt.xlabel('x')
plt.ylabel('$x^p$')
plt.legend()
plt.show()

# 3D Plot
x = np.linspace(a, b, n*N+1)
y = np.linspace(a, b, n*N+1)
X, Y = np.meshgrid(x, y)
Z = X**p

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D Plot of $x^p$')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('$x^p$')

plt.show()


integral_value = integral_function(a, b, p)
print(integral_value)




#--------------------------------------------------------------------------------------------------------------------
### For 2D plot

import numpy as np
import matplotlib.pyplot as plt

p = 3
f = lambda x : x**p
a = 0; b = 5; N = 10
n = 10 # Use n*N+1 points to plot the function smoothly

x = np.linspace(a,b,N+1)
y = f(x)

X = np.linspace(a,b,n*N+1)
Y = f(X)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(X,Y,'b')
x_left = x[:-1] # Left endpoints
y_left = y[:-1]
plt.plot(x_left,y_left,'b.',markersize=10)
plt.bar(x_left,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='b')
plt.title('Left Riemann Sum, N = {}'.format(N))

plt.subplot(1,3,2)
plt.plot(X,Y,'b')
x_mid = (x[:-1] + x[1:])/2 # Midpoints
y_mid = f(x_mid)
plt.plot(x_mid,y_mid,'b.',markersize=10)
plt.bar(x_mid,y_mid,width=(b-a)/N,alpha=0.2,edgecolor='b')
plt.title('Midpoint Riemann Sum, N = {}'.format(N))

plt.subplot(1,3,3)
plt.plot(X,Y,'b')
x_right = x[1:] # Left endpoints
y_right = y[1:]
plt.plot(x_right,y_right,'b.',markersize=10)
plt.bar(x_right,y_right,width=-(b-a)/N,alpha=0.2,align='edge',edgecolor='b')
plt.title('Right Riemann Sum, N = {}'.format(N))

plt.show()


#--------------------------------------------------------------------------------------------------------------------
# def riemann_sum(f,a,b,N,method='midpoint'):
#     '''Compute the Riemann sum of f(x) over the interval [a,b].

#     Parameters
#     ----------
#     f : function
#         Vectorized function of one variable
#     a , b : numbers
#         Endpoints of the interval [a,b]
#     N : integer
#         Number of subintervals of equal length in the partition of [a,b]
#     method : string
#         Determines the kind of Riemann sum:
#         right : Riemann sum using right endpoints
#         left : Riemann sum using left endpoints
#         midpoint (default) : Riemann sum using midpoints

#     Returns
#     -------
#     float
#         Approximation of the integral given by the Riemann sum.
#     '''
#     dx = (b - a)/N
#     x = np.linspace(a,b,N+1)

#     if method == 'left':
#         x_left = x[:-1]
#         return np.sum(f(x_left)*dx)
#     elif method == 'right':
#         x_right = x[1:]
#         return np.sum(f(x_right)*dx)
#     elif method == 'midpoint':
#         x_mid = (x[:-1] + x[1:])*0.5
#         return np.sum(f(x_mid)*dx)
#     else:
#         raise ValueError("Method must be 'left', 'right' or 'midpoint'.")

# print(riemann_sum(x**0.5, 0, 5, 4))


#--------------------------------------------------------------------------------------------------------------------
#### For 3D Plot


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

p = 3
f = lambda x : x**p
a = 0
b = 5
N = 10
n = 10  # Use n*N+1 points to plot the function smoothly

x = np.linspace(a, b, N+1)
y = f(x)

X = np.linspace(a, b, n*N+1)
Y = f(X)

fig = plt.figure(figsize=(15,5))

# Left Riemann Sum
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(X, Y, zs=0, zdir='z', label='curve', color='b')
x_left = x[:-1]  # Left endpoints
y_left = y[:-1]
z_left = np.zeros_like(x_left)
ax1.bar3d(x_left, z_left, z_left, dx=(b-a)/N, dy=y_left, dz=0.1, alpha=0.2, color='b', edgecolor='b')
ax1.set_title('Left Riemann Sum, N = {}'.format(N))

# Midpoint Riemann Sum
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(X, Y, zs=0, zdir='z', label='curve', color='b')
x_mid = (x[:-1] + x[1:]) / 2  # Midpoints
y_mid = f(x_mid)
z_mid = np.zeros_like(x_mid)
ax2.bar3d(x_mid, z_mid, z_mid, dx=(b-a)/N, dy=y_mid, dz=0.1, alpha=0.2, color='b', edgecolor='b')
ax2.set_title('Midpoint Riemann Sum, N = {}'.format(N))

# Right Riemann Sum
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(X, Y, zs=0, zdir='z', label='curve', color='b')
x_right = x[1:]  # Right endpoints
y_right = y[1:]
z_right = np.zeros_like(x_right)
ax3.bar3d(x_right, z_right, z_right, dx=(b-a)/N, dy=y_right, dz=0.1, alpha=0.2, color='b', edgecolor='b')
ax3.set_title('Right Riemann Sum, N = {}'.format(N))

plt.show()

#--------------------------------------------------------------------------------------------------------------------

####Plot surface of absolute error
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

p = 3
f = lambda x : x**p
a = 0; b = 5; N = 10
n = 10 # Use n*N+1 points to plot the function smoothly

#Integral function x^p dx

def integrand(x, p):
    return x**p

# Define the function to calculate the integral
def integral_function(a, b, p):
    result, error = quad(integrand, a, b, args=(p,))
    return result

# Parameters
a = 0
b = 5
p = 3

# Calculate the integral
integral_value = integral_function(a, b, p)
print(f"The integral of x^{p} from {a} to {b} is: {integral_value}")
print(integral_value)

x = np.linspace(a,b,N+1)
y = f(x)

X = np.linspace(a,b,n*N+1)
Y = f(X)

plt.figure(figsize=(15,5))

plt.plot(X,Y,'b')
x_left = x[:-1] # Left endpoints
y_left = y[:-1]
plt.plot(x_left,y_left,'b.',markersize=10)
plt.bar(x_left,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='b')
plt.title('Left Riemann Sum, N = {}'.format(N))

plt.show()

absolute_error = abs(integral_value - x_left)
print(absolute_error)

plt.figure(figsize=(15,5))

plt.plot(absolute_error,y_left,'b')
x_left = x[:-1] # Left endpoints
y_left = y[:-1]
plt.plot(absolute_error,y_left,'g.',markersize=10)
plt.bar(absolute_error,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='g')
plt.title('Surface of Absolute Error, N = {}'.format(N))
plt.show()


#--------------------------------------------------------------------------------------------------------------------
####Plot surface of absolute error in 3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

p = 3
f = lambda x: x**p
a = 0
b = 5
N = 10
n = 10  # Use n*N+1 points to plot the function smoothly

# Integral function x^p dx
def integrand(x, p):
    return x**p

# Define the function to calculate the integral
def integral_function(a, b, p):
    result, error = quad(integrand, a, b, args=(p,))
    return result

# Parameters
a = 0
b = 5
p = 3

# Calculate the integral
integral_value = integral_function(a, b, p)
print(f"The integral of x^{p} from {a} to {b} is: {integral_value}")

x = np.linspace(a, b, N+1)
y = f(x)

X = np.linspace(a, b, n*N+1)
Y = f(X)

# Create 3D plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for 3D plot
X, Y = np.meshgrid(X, Y)
Z = f(X)

# Plot surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'3D Surface plot of x^{p} from {a} to {b}')

plt.show()

# Calculate absolute error for visualization
absolute_error = abs(integral_value - x[:-1])
print(absolute_error)

# Create 3D plot for absolute error
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for 3D plot of absolute error
X_err, Y_err = np.meshgrid(absolute_error, y[:-1])
Z_err = f(X_err)

# Plot surface
ax.plot_surface(X_err, Y_err, Z_err, cmap='plasma', edgecolor='none')

# Labels and title
ax.set_xlabel('Absolute Error')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'3D Surface plot of Absolute Error of x^{p} from {a} to {b}')

plt.show()























#--------------------------------------------------------------------------------------------------------------------

##### PLot of x**p

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import quad
# from mpl_toolkits.mplot3d import Axes3D

# p = 3
# # Parameters
# f = lambda x : x**p
# a =0
# b = 5
# N = 10
# n = 10 # Use n*N+1 points to plot the function smoothly

# # Integral function x^p dx
# def integrand(x):
#     return (np.exp(x) - 1)/x

# # Define the function to calculate the integral
# def integral_function(a, b):
#     result, error = quad(integrand, a, b)
#     return result

# integral_value = integral_function(a, b)
# print("Integral value:", integral_value)

# # 2D Plot
# x = np.linspace(a, b, n*N+1)
# y = f(x)

# plt.figure()
# plt.plot(x, y, label=f'$(e^x - 1)/x$')
# plt.fill_between(x, y, alpha=0.2)
# plt.title('2D Plot of $x^p$')
# plt.xlabel('x')
# plt.ylabel('$x^p$')
# plt.legend()
# plt.show()

# # 3D Plot
# x = np.linspace(a, b, n*N+1)
# y = np.linspace(a, b, n*N+1)
# X, Y = np.meshgrid(x, y)
# Z = (np.exp(X) - 1)/X

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# ax.set_title('3D Plot of $x^p$')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('$x^p$')

# plt.show()


# integral_value = integral_function(a, b)
# print(integral_value)



#--------------------------------------------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import quad
# from mpl_toolkits.mplot3d import Axes3D

# p = 3
# f = lambda x : np.exp(x)
# a = 0; b = 100; N = 10
# n = 10 # Use n*N+1 points to plot the function smoothly

# #Integral function x^p dx

# def integrand(x):
#     return 

# # Define the function to calculate the integral
# def integral_function(a, b, p):
#     result, error = quad(integrand, a, b, args=(p,))
#     return result

# # Parameters
# a = 0
# b = 100
# p = 3

# # Calculate the integral
# integral_value = integral_function(a, b, p)
# print(f"The integral of x^{p} from {a} to {b} is: {integral_value}")
# print(integral_value)

# x = np.linspace(a,b,N+1)
# y = f(x)

# X = np.linspace(a,b,n*N+1)
# Y = f(X)

# plt.figure(figsize=(15,5))

# plt.plot(X,Y,'b')
# x_left = x[:-1] # Left endpoints
# y_left = y[:-1]
# plt.plot(x_left,y_left,'b.',markersize=10)
# plt.bar(x_left,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='b')
# plt.title('Left Riemann Sum, N = {}'.format(N))

# plt.show()

# absolute_error = abs(integral_value - x_left)
# print(absolute_error)

# plt.figure(figsize=(15,5))

# plt.plot(absolute_error,y_left,'b')
# x_left = x[:-1] # Left endpoints
# y_left = y[:-1]
# plt.plot(absolute_error,y_left,'g.',markersize=10)
# plt.bar(absolute_error,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='g')
# plt.title('Surface of Absolute Error, N = {}'.format(N))
# plt.show()
