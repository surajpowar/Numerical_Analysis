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

# Trapezoidal rule
def trapezoidal_rule(a, b, f, N):
    x = np.linspace(a, b, N+1)
    y = f(x)
    dx = (b - a) / N
    T = (dx / 2) * (y[0] + 2 * np.sum(y[1:N]) + y[N])
    return T

# Simpson's 1/3 rule
def simpsons_rule(a, b, f, N):
    if N % 2 == 1:
        N += 1  # N must be even
    x = np.linspace(a, b, N+1)
    y = f(x)
    dx = (b - a) / N
    S = (dx / 3) * (y[0] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N-1:2]) + y[N])
    return S

# Function to calculate errors for Trapezoidal and Simpson's rules
def trapezoidal_and_simpsons_errors(a, b, p, N):
    f = lambda x: x**p
    integral_value = integral_function(a, b, p)
    
    trapezoidal_result = trapezoidal_rule(a, b, f, N)
    simpsons_result = simpsons_rule(a, b, f, N)
    
    trapezoidal_error = np.abs(trapezoidal_result - integral_value)
    simpsons_error = np.abs(simpsons_result - integral_value)

    return trapezoidal_error, simpsons_error

# Print the values of the error for both Trapezoidal as well as Simpsons
print(trapezoidal_and_simpsons_errors(0, 5, 3, 100)) # Remember to change the values of a, b, p, N

# Define ranges for b and p
b_values = np.linspace(0, 5, 1000)
p_values = np.linspace(0, 3, 10)

# Create meshgrid for b and p
B, P = np.meshgrid(b_values, p_values)

# Calculate errors for each combination of b and p
trapezoidal_errors = np.zeros_like(B)
simpsons_errors = np.zeros_like(B)

for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        trapezoidal_errors[i, j], simpsons_errors[i, j] = trapezoidal_and_simpsons_errors(a, B[i, j], P[i, j], N)

# Plotting the 2D and 3D surfaces

fig = plt.figure(figsize=(12, 8))

# 3D plot for Trapezoidal Rule Error
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(B, P, trapezoidal_errors, cmap='viridis')
# ax.set_title('Trapezoidal Rule Error', fontsize = 30)
# ax.set_xlabel('b', fontsize = 20)
# ax.set_ylabel('p', fontsize = 20)
# ax.set_zlabel('Error', fontsize = 20)
# ax.tick_params(axis='both', which='major', labelsize=15)

# 3D plot for Simpson's 1/3 Rule Error
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, P, simpsons_errors, cmap='viridis')
ax.set_title("Simpson's Rule Error", fontsize = 30)
ax.set_xlabel('b', fontsize = 20)
ax.set_ylabel('p', fontsize = 20)
ax.set_zlabel('Error', fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)

plt.tight_layout()
plt.show()
