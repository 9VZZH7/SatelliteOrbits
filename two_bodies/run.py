import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from matplotlib.pyplot import axes
from scipy import integrate
import ipywidgets as widgets
from ipywidgets import interactive

mass_sun = 2e30
mass_1 = 5.976e24 # earth
mass_2 = 1.8987e27 # jupiter
mass_3 = 3.301e23 # mercury

# dist is given in km e8
pos_sun = np.array([0,0])
pos_1 = np.array([1.47e8,0])
pos_2 = np.array([7.784e8,0])
pos_3 = np.array([-4.6e7,0])

# unit is unknown, values are experimental
velo_sun = np.array([0,0])
velo_1 = np.array([0, 0.0000897])
velo_2 = np.array([0, 0.000035])
velo_3 = np.array([0, -0.0000897*2])

def dist(x, y):
    return 0

def odePlanet(t, rhs, mrel):

    lhs = np.zeros((4,))
    lhs[:2] = rhs[-2:]

    r = rhs[:2]
    r3 = np.sqrt(np.sum(r**2))**3

    lhs[-2:] = -(1+mrel)*r/r3

    return lhs

def odePlanetN(t, rhs, mass, N):
    lhs = np.zeros(((N) * 4,))
    lhs[:-2*N] = rhs[2*N:]

    mrel = mass[1:]/mass[0]

    r = rhs[:-2*N]
    interim = r**2
    r3 = np.sqrt(interim[::2]+interim[1::2])**3

    lhs[2*N:] = -(1+np.repeat(mrel, 2))*r/np.repeat(r3, 2)
    for i in range(1, N):
        r_diff = r - np.roll(r, i)
        interim_diff = r_diff**2
        r3_diff = np.sqrt(interim[::2]+interim[1::2])**3
        interim = np.roll(r, i)**2
        r3 = np.sqrt(interim[::2]+interim[1::2])**3
        lhs[2*N:] -= np.repeat(np.roll(mrel, i), 2) * ((r_diff)/np.repeat(r3_diff, 2) + np.roll(r, i)/np.repeat(r3, 2))

    return lhs

def movEarthSun(initial_state, mrel, T, steps):
    t_eval = np.linspace(0,T,steps,endpoint=True) # Time grid

    b = integrate.solve_ivp(odePlanet, t_span = (0,T), t_eval = t_eval,  y0 = initial_state, args= ([mrel]), method = 'Radau')
    return b

def movNPlanets(starting_x, starting_v, mass, T, steps, N):
    initial_state = np.concatenate((np.array(starting_x).flatten(), np.array(starting_v).flatten()))
    t_eval = np.linspace(0,T,steps,endpoint=True) # Time grid

    b = integrate.solve_ivp(odePlanetN, t_span = (0,T), t_eval = t_eval,  y0 = initial_state, args= ([mass, N]), method = 'Radau')
    return b


mass = np.array([mass_sun, mass_1, mass_2, mass_3])
T = 100_000_000_000_000
steps = 10_000
N = 3
mov = movNPlanets([pos_1, pos_2, pos_3], [velo_1, velo_2, velo_3], mass, T, steps, N)
for i in range(N):
    plt.plot(mov.y[(2*i)], mov.y[(2*i)+1])
plt.plot(0,0, marker = 'o', markersize=15, color = 'yellow', label = 'Sun')

# initial_state = np.concatenate((pos_1, velo_1))
# mrel = mass_1/mass_sun
# T = 100000000000000
# movement = movEarthSun(initial_state=initial_state, mrel=mrel, T=T, steps=10000)
# plt.plot(movement.y[0], movement.y[1])
# plt.plot(0,0, marker = 'o', markersize=15, color = 'yellow', label = 'Sun')
# plt.plot(initial_state[0], initial_state[1], color = 'blue', marker = 'o',markersize=(5+10*mrel), label = 'Start')
# plt.legend(loc = 'upper right')
# plt.xlim(-10e8, 1e9)
# plt.ylim(-1e9, 1e9)
# #plt.axis('auto')
# plt.show()
