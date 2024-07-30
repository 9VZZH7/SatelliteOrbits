#SUN - EARTH SIMULATION - MATLAB
% Initial Data
r0 = 0.983; % AU at aphelion, where we can assume 0 radial velocity
theta0_dt = 0.018389172986805; % rad/day
m1 = 1; % Earth mass
m2 = 334448; % Sun mass in Earth mass units
G =  9.001123e-10; % AU^3/(day^2 * Earth mass)

% Gravitational parameter and other constants
mu = (m1 * m2) / (m1 + m2); % Reduced mass
k = G * (m1 * m2); % Gravitational parameter
l = mu * (r0^2) * theta0_dt; % Angular momentum per unit mass

% Calculate theta values based on time
slices = 1000;
t_full_revolution = 2*(2 * pi * mu * r0^2) / l;
t_span = linspace(0, t_full_revolution, slices);
theta_values = (l / (mu * r0^2)) .* t_span;   %This way theta0=0, so we expect the farthest spot from Sun to be oon the right

%set up ODE;
s = (mu*k)/(l^2);
F=@(t,y) [y(2);s-y(1)];
[t,y] = ode45(F,theta_values,[1/r0; 0]);

figure
plot(t,y)
legend('theta','r')

%extract solution 
r = zeros(slices,1);
for i = 1:slices
    r(i)=1/y(i,1);
end

%%Eccentricity
epsilon = abs((1-(s*r(5)))/(s*r(5)*cos(theta_values(5))))

%back to cartesian coordinates
X = zeros(slices,1);
Y = zeros(slices,1);
Z = zeros(slices,1);
for i = 1:slices
    X(i)=r(i)*cos(theta_values(i));
    Y(i)=r(i)*sin(theta_values(i));
end

%Print Aphelium/Perihelium
A = zeros(2,1);
A(1)=min(X);
A(2) = max(X);
A
a = abs(A(1)- A(2))/2;
b = mu/k;
T = 2*pi*a*sqrt(mu*a/k)

%plot
figure
plot3(X,Y,Z,'g')
hold on
radius = 0.109;
[W,K,T] = sphere;
W2 = W * radius;
K2 = K * radius;
T2 = T * radius;
plot3(W2,K2,T2,'y')
title("Earth's orbit around the Sun")
axis equal


#EART - MOON SIMULATION - MATLAB
% Initial Data
r0 = 0.002711093; % AU at aphelion, where we can assume 0 radial velocity
theta0_dt = 0.208077269; % rad/day
m1 = 1; % Earth mass
m2 = 0.0123 ; % Moon mass in Earth mass units
G =  9.001123e-10; % AU^3/(day^2 * Earth mass)

% Gravitational parameter and other constants
mu = (m1 * m2) / (m1 + m2); % Reduced mass
k = G * (m1 * m2); % Gravitational parameter
l = mu * (r0^2) * theta0_dt; % Angular momentum per unit mass

% Calculate theta values based on time
slices = 1000;
t_full_revolution = 2*(2 * pi * mu * r0^2) / l;
t_span = linspace(0, t_full_revolution, slices);
theta_values = (l / (mu * r0^2)) .* t_span;   %This way theta0=0, so we expect the farthest spot from Sun to be oon the right

%set up ODE;
s = (mu*k)/(l^2)
F=@(t,y) [y(2);s-y(1)];
[t,y] = ode45(F,theta_values,[1/r0; 0]);

figure
plot(t,y)
legend('theta','r')

%extract solution 
r = zeros(slices,1);
for i = 1:slices
    r(i)=1/y(i,1);
end

%%Eccentricity
epsilon = abs((1-(s*r(5)))/(s*r(5)*cos(theta_values(5))))

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
plot3(X,Y,Z,'k')
hold on
radius = 0.00003;
[W,K,T] = sphere;
W2 = W * radius;
K2 = K * radius;
T2 = T * radius;
plot3(W2,K2,T2,'g')
title("Moon's orbit around the Earth")
axis equal
