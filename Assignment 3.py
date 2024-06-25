# Authors: Suraj Powar and Prof. Dr. Henri Schurz
# Date: 06/14/2024

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import math


# Integral function
p = 3
f = lambda x : x**p
a = 0; b = 5; N = 100
n = 100 # Use n*N+1 points to plot the function smoothly

#Integral function x^p dx

def integrand(x, p):
    return x**p

# Define the function to calculate the integral
def integral_function(a, b, p):
    result, error = quad(integrand, a, b, args=(p,))
    return result


integral_value = integral_function(a, b, p)
print("The integral value is ",integral_value)


dx = (b-a)/N
x_left = np.linspace(a,b-dx,N)
x_midpoint = np.linspace(dx/2,b - dx/2,N)
x_right = np.linspace(dx,b,N)

print("Partition with",N,"subintervals.")
left_riemann_sum = np.sum(f(x_left) * dx)
print("Left Riemann Sum:",left_riemann_sum)

midpoint_riemann_sum = np.sum(f(x_midpoint) * dx)
print("Midpoint Riemann Sum:",midpoint_riemann_sum)

right_riemann_sum = np.sum(f(x_right) * dx)
print("Right Riemann Sum:",right_riemann_sum)

print("Left Riemann Sum Error:",np.abs(left_riemann_sum - integral_value))
print("Midpoint Riemann Sum:",np.abs(midpoint_riemann_sum - integral_value))
print("Right Riemann Sum:",np.abs(right_riemann_sum - integral_value))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D

# Integral function
a = 0
N = 100

# Integral function x^p dx
def integrand(x, p):
    return x**p

# Define the function to calculate the integral
def integral_function(a, b, p):
    result, error = quad(integrand, a, b, args=(p,))
    return result

# Function to calculate Riemann sums and their errors
def riemann_sums_and_errors(a, b, p, N):
    dx = (b - a) / N
    x_left = np.linspace(a, b - dx, N)
    x_midpoint = np.linspace(dx / 2, b - dx / 2, N)
    x_right = np.linspace(dx, b, N)

    f = lambda x: x**p

    integral_value = integral_function(a, b, p)

    left_riemann_sum = np.sum(f(x_left) * dx)
    midpoint_riemann_sum = np.sum(f(x_midpoint) * dx)
    right_riemann_sum = np.sum(f(x_right) * dx)

    left_error = np.abs(left_riemann_sum - integral_value)
    midpoint_error = np.abs(midpoint_riemann_sum - integral_value)
    right_error = np.abs(right_riemann_sum - integral_value)

    return left_error, midpoint_error, right_error

# Define ranges for b and p
b_values = np.linspace(0, 5, 1000)
p_values = np.linspace(0, 3, 10)

# Create meshgrid for b and p
B, P = np.meshgrid(b_values, p_values)

# Calculate errors for each combination of b and p
left_errors = np.zeros_like(B)
midpoint_errors = np.zeros_like(B)
right_errors = np.zeros_like(B)

for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        left_errors[i, j], midpoint_errors[i, j], right_errors[i, j] = riemann_sums_and_errors(a, B[i, j], P[i, j], N)

# Plotting the 2D and 3D surfaces

fig = plt.figure(figsize=(12, 8))

# 3D plot for Left Riemann Sum Error
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(B, P, left_errors, cmap='viridis')
# ax.set_title('Left Riemann Sum Error', fontsize = 30)
# ax.set_xlabel('b', fontsize = 20)
# ax.set_ylabel('p', fontsize = 20)
# ax.set_zlabel('Error', fontsize = 20)
# ax.tick_params(axis='both', which='major', labelsize=15)

# 3D plot for Midpoint Riemann Sum Error
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(B, P, midpoint_errors, cmap='viridis')
# ax.set_title('Midpoint Riemann Sum Error', fontsize = 30)
# ax.set_xlabel('b', fontsize = 20)
# ax.set_ylabel('p', fontsize = 20)
# ax.set_zlabel('Error', fontsize = 20)
# ax.tick_params(axis='both', which='major', labelsize=15)

# # 3D plot for Right Riemann Sum Error
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, P, right_errors, cmap='viridis')
ax.set_title('Right Riemann Sum Error', fontsize = 30)
ax.set_xlabel('b', fontsize = 20)
ax.set_ylabel('p', fontsize = 20)
ax.set_zlabel('Error', fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)

plt.tight_layout()
plt.show()















































# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def riemann_error_plot(p_values, n_values, a=0, b=2):
#     """
#     Function to plot the error of the left-hand Riemann sum approximation for f(x) = x^p
#     for different values of p and n (number of intervals).

#     Parameters:
#     p_values (list or numpy array): Array of p values for f(x) = x^p
#     n_values (list or numpy array): Array of n values (number of intervals)
#     a (float): Lower limit of integration
#     b (float): Upper limit of integration
#     """
#     # Initialize arrays to store errors
#     errors = np.zeros((len(p_values), len(n_values)))

#     # Calculate errors for each combination of p and n
#     for i, p in enumerate(p_values):
#         for j, n in enumerate(n_values):
#             # Calculate the exact integral value
#             exact_value = (b**(p+1) - a**(p+1)) / (p+1)

#             # Calculate the left-hand Riemann sum
#             x = np.linspace(a, b, n, endpoint=False)
#             dx = (b - a) / n
#             riemann_sum = np.sum(x**p) * dx

#             # Calculate the error
#             errors[i, j] = abs(exact_value - riemann_sum)

#     # Plotting 2D error plot for a fixed p value (first p value)
#     print(p_values)
#     plt.figure()
#     plt.plot(n_values, errors[0, :], marker='o')
#     plt.title(f"Error of Left-hand Riemann Sum for $f(x) = x^{p_values[0]}$")
#     plt.xlabel("Number of intervals (n)")
#     plt.ylabel("Error")
#     plt.grid(True)
#     plt.show()

#     # Plotting 3D surface plot of errors
#     n_mesh, p_mesh = np.meshgrid(n_values, p_values)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(n_mesh, p_mesh, errors, cmap='viridis')
#     ax.set_title("Error Surface of Left-hand Riemann Sum")
#     ax.set_xlabel("Number of intervals (n)")
#     ax.set_ylabel("Power of x (p)")
#     ax.set_zlabel("Error")
#     plt.show()

# # Example usage
# p_values = np.linspace(0, 2, 10)  # Different values of p
# n_values = np.arange(10, 110, 10000)  # Different values of n
# riemann_error_plot(p_values, n_values)


## The Midpoint Riemann Method

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# p_values = 3
# def midpoint_riemann_error_plot(p_values, n_values, a=0, b=5):
#     """
#     Function to plot the error of the midpoint Riemann sum approximation for f(x) = x^p
#     for different values of p and n (number of intervals).

#     Parameters:
#     p_values (list or numpy array): Array of p values for f(x) = x^p
#     n_values (list or numpy array): Array of n values (number of intervals)
#     a (float): Lower limit of integration
#     b (float): Upper limit of integration
#     """
#     # Initialize arrays to store errors
#     errors = np.zeros((len(p_values), len(n_values)))

#     # Calculate errors for each combination of p and n
#     for i, p in enumerate(p_values):
#         for j, n in enumerate(n_values):
#             # Calculate the exact integral value
#             exact_value = (b**(p+1) - a**(p+1)) / (p+1)

#             # Calculate the midpoint Riemann sum
#             x = np.linspace(a + (b - a) / (2 * n), b - (b - a) / (2 * n), n)
#             dx = (b - a) / n
#             riemann_sum = np.sum(x**p) * dx

#             # Calculate the error
#             errors[i, j] = abs(exact_value - riemann_sum)

#     # Plotting 2D error plot for a fixed p value (first p value)
#     plt.figure()
#     plt.plot(n_values, errors[0, :], marker='o')
#     plt.title(f"Error of Midpoint Riemann Sum for $f(x) = x^{p_values[0]}$")
#     plt.xlabel("Number of intervals (n)")
#     plt.ylabel("Error")
#     plt.grid(True)
#     plt.show()

#     # Plotting 3D surface plot of errors
#     n_mesh, p_mesh = np.meshgrid(n_values, p_values)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(n_mesh, p_mesh, errors, cmap='viridis')
#     ax.set_title("Error Surface of Midpoint Riemann Sum")
#     ax.set_xlabel("Number of intervals (n)")
#     ax.set_ylabel("Power of x (p)")
#     ax.set_zlabel("Error")
#     plt.show()

# # Example usage
# p_values = np.linspace(1, 5, 10)  # Different values of p
# n_values = np.arange(10, 110, 10)  # Different values of n
# midpoint_riemann_error_plot(p_values, n_values)
