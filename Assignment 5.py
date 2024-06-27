# Authors: Suraj Powar and Prof. Dr. Henri Schurz
# Date: 06/24/2024


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Integral function
p = 0.5
f = lambda x: x**p
a = 0
b = 5
N = 10  # Number of subintervals
n = 10  # Use n*N+1 points to plot the function smoothly

# Integral function x^p dx
def integrand(x, p):
    return x**p

# Define the function to calculate the integral using quad
def integral_function(a, b, p):
    result, error = quad(integrand, a, b, args=(p,))
    return result

integral_value = integral_function(a, b, p)
print("The integral value using quad is:", integral_value)

# Boole's rule (m = 3, n = 4)
def booles_rule(f, a, b, N):
    h = (b - a) / (4 * N)
    result = 0
    for i in range(N):
        x0 = a + 4 * i * h
        x1 = x0 + h
        x2 = x0 + 2 * h
        x3 = x0 + 3 * h
        x4 = x0 + 4 * h
        result += 7 * f(x0) + 32 * f(x1) + 12 * f(x2) + 32 * f(x3) + 7 * f(x4)
    return (2 * h / 45) * result

# Weddle's rule (m = 6, n = 6)
def weddles_rule(f, a, b, N):
    h = (b - a) / (6 * N)
    result = 0
    for i in range(N):
        x0 = a + 6 * i * h
        x1 = x0 + h
        x2 = x0 + 2 * h
        x3 = x0 + 3 * h
        x4 = x0 + 4 * h
        x5 = x0 + 5 * h
        x6 = x0 + 6 * h
        result += (f(x0) + 5 * f(x1) + f(x2) + 6 * f(x3) +
                   f(x4) + 5 * f(x5) + f(x6))
    return (3 * h / 10) * result

# Calculate the integral using Boole's rule and Weddle's rule
booles_integral = booles_rule(f, a, b, N)
weddles_integral = weddles_rule(f, a, b, N)

print("The integral value using Boole's rule is:", booles_integral)
print("The integral value using Weddle's rule is:", weddles_integral)

# Errors for Boole's rule and Weddle's rule
booles_error = np.abs(booles_integral - integral_value)
weddles_error = np.abs(weddles_integral - integral_value)

print("Error using Boole's rule:", booles_error)
print("Error using Weddle's rule:", weddles_error)

# Plotting
b_values = np.linspace(0, 5, 1000)
p_values = np.linspace(0, 0.5, 10)

# Create meshgrid for b and p
B, P = np.meshgrid(b_values, p_values)

# Calculate errors for each combination of b and p
booles_errors = np.zeros_like(B)
weddles_errors = np.zeros_like(B)

for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        f_current = lambda x: x**P[i, j]
        integral_value_current = integral_function(a, B[i, j], P[i, j])
        booles_integral_current = booles_rule(f_current, a, B[i, j], N)
        weddles_integral_current = weddles_rule(f_current, a, B[i, j], N)
        booles_errors[i, j] = np.abs(booles_integral_current - integral_value_current)
        weddles_errors[i, j] = np.abs(weddles_integral_current - integral_value_current)

fig = plt.figure(figsize=(12, 8))

# 3D plot for Boole's rule Error
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, P, booles_errors, cmap='viridis')
ax.set_title("Boole's Rule Error")
ax.set_xlabel('b')
ax.set_ylabel('p')
ax.set_zlabel('Error')
ax.tick_params(axis='both', which='major', labelsize=15)

# 3D plot for Weddle's rule Error
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(B, P, weddles_errors, cmap='viridis')
# ax.set_title("Weddle's Rule Error")
# ax.set_xlabel('b')
# ax.set_ylabel('p')
# ax.set_zlabel('Error')
# ax.tick_params(axis='both', which='major', labelsize=15)

plt.tight_layout()
plt.show()
