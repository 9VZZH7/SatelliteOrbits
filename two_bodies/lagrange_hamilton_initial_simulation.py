# First code, just compute position given the data and the explicit solution, seems to work:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initial conditions
r0 = 1.496  # Initial radius
theta0_dt = 0.002053 #radiants per day 

# Parameters
m1= 1   #5.972
m2=332.94604    #1989.0
G = 6.6743 * (10**(-11)) #how to scale?
mu = (m1 * m2) / (m1 + m2)  # Reduced mass
k = (G) * (m1 * m2)  # Gravitational factor
l = mu * r0**2 * theta0_dt  #
epsilon = 0.067  # Eccentricity

# Set up time grid
# Calculate the time for one full revolution
t_full_revolution = (2 * np.pi * mu * r0**2) / l
t_span = np.linspace(0, t_full_revolution, 10000000)

# Calculate theta values based on time
theta_values = (l / (mu * r0**2)) * t_span

# Calculate r values using the given formula
r_values = (l**2 / (mu * k)) / (1 + epsilon * np.cos(theta_values))
# Compute Cartesian coordinates from polar coordinates
x_values = r_values * np.cos(theta_values)
y_values = r_values * np.sin(theta_values)
z_values = np.zeros_like(x_values)  # z-coordinate remains zero for 2D motion

# Plot the orbit in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, label='Orbit')
ax.scatter(0, 0, 0, color='red', label='Center of mass (Sun)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_title('Orbit of Two Bodies (Kepler Case)')
plt.legend()
plt.grid(True)
plt.show()

####################################################################################################################
# Second code, trying to use a built in solver, but it doesn't really work as I would like it to..
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# Initial conditions
r0 = 1.496  # Initial radius
theta0_dt = 0.002053  # radiants per day

# Parameters
m1 = 1
m2 = 332.94604
G = 6.6743e-11
mu = (m1 * m2) / (m1 + m2)
k = G * (m1 * m2)
l = mu * r0**2 * theta0_dt

# Define the orbit equations
def orbit_equations(theta_values, y):
    u1,u2 = y
    du1 = u2
    du2 = ((l**2)/(mu*k)) - u1
    return [du1, du2]

# Set up theta grid
t_full_revolution = (2 * np.pi * mu * r0**2) / l
t_span = np.linspace(0, t_full_revolution, 10000000)

# Calculate theta values based on time
theta_values = (l / (mu * r0**2)) * t_span

# Calculate initial condition for u
u0 = 1 / r0
du0 = 0

# Solve the ODE
sol = solve_ivp(orbit_equations, [theta_values[0], theta_values[-1]], [u0, du0], t_eval=theta_values)

# Extract solutions
u1_values, u2_values = sol.y
r_values = 1 / u1_values

# Calculate Cartesian coordinates from polar coordinates
x_values = r_values * np.cos(theta_values)
y_values = r_values * np.sin(theta_values)
z_values = np.zeros_like(x_values)  # Z-component is 0

# Plot the orbit in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, label='Orbit')
ax.scatter(0, 0, 0, color='red', label='Center of mass (Sun)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Orbit of Earth around the Sun')
plt.legend()
plt.grid(True)
plt.show()
