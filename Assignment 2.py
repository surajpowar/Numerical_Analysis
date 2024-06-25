# # Authors: Suraj Powar and Prof. Dr. Henri Schurz
# # Date: 06/12/2024

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Define the function and its Taylor series approximation
def func(x):
    return (np.exp(x) - 1)/x  # Example function: (e^x - 1)/x

def taylor_approx(x, n_terms):
    approx = sum((x**n) / math.factorial(n) for n in range(n_terms))
    return approx

# Define the interval for integration
a, b = -4, 4

# Compute the quadrature of the function
integral_func, _ = quad(func, a, b)

# Compute the error for different number of terms in the Taylor series
n_terms_list = np.arange(1, 50)
errors = []

for n_terms in n_terms_list:
    integral_taylor, _ = quad(lambda x: taylor_approx(x, n_terms), a, b)
    error = abs(integral_func - integral_taylor)
    errors.append(error)

# Plot the error in 2D
plt.figure(figsize=(10, 6))
plt.plot(n_terms_list, errors, marker='o')
plt.xlabel('Number of terms in Taylor series')
plt.ylabel('Error')
plt.title('Error of Taylor Series Approximation in Quadrature')
plt.grid(True)
#plt.show()

# Create 3D plot for the error
from mpl_toolkits.mplot3d import Axes3D

# Define the interval for the 3D plot
x_values = np.linspace(a, b, 100)
n_terms_3d = np.arange(1, 21)

X, Y = np.meshgrid(x_values, n_terms_3d)
Z = np.zeros_like(X)

for i, n in enumerate(n_terms_3d):
    Z[i, :] = abs(func(X[i, :]) - taylor_approx(X[i, :], n))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('b')
ax.set_ylabel('Number of terms')
ax.set_zlabel('Error')
ax.set_title('Error Surface of Taylor Series Approximation')

plt.show()




# #----------------------------------------------------------------------------------------------------------------------------
# #Calculating the error of approximation of (e^x - 1)/x

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import math

# Parameters
f = lambda x : (np.exp(x) - 1)/x 
a = -4
b = 4
N = 10
n = 6 # Use n*N+1 points to plot the function smoothly

# Integral function x^p dx
def integrand(x):
    return (np.exp(x) - 1)/x 

# Define the function to calculate the integral
def integral_function(a, b):
    result, error = quad(integrand, a, b)
    return result

def error_taylor_type(n):
    return np.exp(1)/(math.factorial(n+1)*(n+1))

print('The integral value of integrand is {}'.format(integral_function(0, 1)))
print('The error for taylor type for n = 6 is {}'.format(error_taylor_type(6)))



# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np

# a_values = np.linspace(0, np.pi/2, 20)
# b_values = np.linspace(np.pi/2, np.pi, 20)
# a_values, b_values = np.meshgrid(a_values, b_values)

# errors_riemannL_3d = np.zeros(a_values.shape)
# errors_riemannR_3d = np.zeros(a_values.shape)
# errors_mid_3d = np.zeros(a_values.shape)

# n = 6  # Fixed n

# for i in range(a_values.shape[0]):
#     for j in range(a_values.shape[1]):
#         a = a_values[i, j]
#         b = b_values[i, j]
#         h = (b - a) / (n - 1)
#         p = 2
#         x = np.linspace(a, b, n)
#         f = x**p
        
#         I_riemannL = h * sum(f[:n-1])
#         err_riemannL = 2 - I_riemannL
        
#         I_riemannR = h * sum(f[1:])
#         err_riemannR = 2 - I_riemannR
        
#         I_mid = h * sum(np.sin((x[:n-1] + x[1:])/2))
#         err_mid = 2 - I_mid
        
#         errors_riemannL_3d[i, j] = err_riemannL
#         errors_riemannR_3d[i, j] = err_riemannR
#         errors_mid_3d[i, j] = err_mid

# fig = plt.figure(figsize=(18, 6))

# ax1 = fig.add_subplot(131, projection='3d')
# ax1.plot_surface(a_values, b_values, errors_riemannL_3d, cmap='viridis')
# ax1.set_title('Left Riemann Sum Error')
# ax1.set_xlabel('a')
# ax1.set_ylabel('b')
# ax1.set_zlabel('Error')

# ax2 = fig.add_subplot(132, projection='3d')
# ax2.plot_surface(a_values, b_values, errors_riemannR_3d, cmap='viridis')
# ax2.set_title('Right Riemann Sum Error')
# ax2.set_xlabel('a')
# ax2.set_ylabel('b')
# ax2.set_zlabel('Error')

# ax3 = fig.add_subplot(133, projection='3d')
# ax3.plot_surface(a_values, b_values, errors_mid_3d, cmap='viridis')
# ax3.set_title('Midpoint Rule Error')
# ax3.set_xlabel('a')
# ax3.set_ylabel('b')
# ax3.set_zlabel('Error')

# plt.tight_layout()
# plt.show()
