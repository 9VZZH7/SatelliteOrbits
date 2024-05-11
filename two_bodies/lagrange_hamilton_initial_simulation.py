#MATLAB CODE1 with given solution from ODE Online Solver: https://www.wolframalpha.com/widgets/view.jsp?id=17adb9cfd6c67a920240e6db6fd8e8a1
r0 = 1.496;
theta0_dt = 0.002053;
m1 = 1;
m2 = 332.94604;
G = 6.6743e-8;
mu = (m1 * m2) / (m1 + m2);
k = G * (m1 * m2);
l = mu * (r0^2) * theta0_dt;
n = (mu*k)/(l^2)
% Calculate theta values based on time
slices = 100;
t_full_revolution = (2 * pi * mu * r0^2) / l;
t_span = linspace(0, t_full_revolution, slices);
theta_values = (l / (mu * r0^2)) * t_span;

%Computed solution Acos(x)+mu*k/l^2
r=zeros(slices,1);
s = (l^2)/(mu*k);
A = 0.667349;    %Update with correct value, if we could effectively solve the ODE it would work
epsilon = ((A*l^2)/(mu*k))
for i =1:slices
    r(i) = s*(1/(1+epsilon*cos(theta_values(i))));
end

X=zeros(slices,1);
Y=zeros(slices,1);
for i=1:slices
    X(i)=r(i)*cos(theta_values(i));
    Y(i)=r(i)*sin(theta_values(i));
end
figure
plot(X,Y)

######################################################################
#MATLAB CODE WITH SOLVER, DOES NOT WORK FOR SOME REASON (THE SOLUTION IS WRONG)
%initial data
r0 = 1.496;
theta0_dt = 0.002053;
m1 = 1;
m2 = 332.94604;
G = 6.6743e-8;
mu = (m1 * m2) / (m1 + m2);
k = G * (m1 * m2);
l = mu * (r0^2) * theta0_dt;

% Calculate theta values based on time
slices = 100;
t_full_revolution = 2*(2 * pi * mu * r0^2) / l;
t_span = linspace(0, t_full_revolution, slices);
theta_values = (l / (mu * r0^2)) * t_span;

%set up ODE;
s = (mu*k)/(l^2);
F=@(t,y) [y(2);s-y(1)];
[t,y] = ode45(F,theta_values,[1/r0; 0]);

figure
plot(t,y)

%extract solution 
r = zeros(slices,1);
for i = 1:slices-1
    r(i)=1/(y(i,1));
end
%back to cartesian coordinates
X = zeros(slices,1);
Y = zeros(slices,1);
Z = zeros(slices,1);
for i = 1:slices
    X(i)=r(i)*cos(theta_values(i));
    Y(i)=r(i)*sin(theta_values(i));
end
%plot
figure
plot(X,Y)
axis equal

    


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
