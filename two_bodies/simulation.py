#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2024

@author: Jannik Hausladen
"""

import numpy as np
from matplotlib import pyplot as plt


class body:
    '''
    
    This class is used to represent all kinds of bodies: Sun, plantes, moons, 
    space crafts, etc...
    
    '''

    def __init__(self, mass, x_0, v_0, can_move = False, color = 'k'):
        '''
        Each body is initilaized accoring to some values

        Parameters
        ----------
        mass : double
            Mass of the planet, constant.
        x_0 : vector
            Initial position.
        v_0 : vector
            Initial velocity, shape confrming to x_0.
        can_move : bool, optional
            Can this body move on its own, i.e., by thrust. The default is False.
        color : char/string, optional
            For plotting reasons, choose color of body. The default is 'k'.

        Returns
        -------
        None.

        '''
        self.mass = mass
        self.x = np.array(x_0)
        self.v = np.array(v_0)
        self.x_old = np.array(x_0)
        self.v_old = np.array(v_0)
        self.can_move = can_move
        self.color = color

    def set_thrust(self, thrust_func):
        '''
        
        If required, one can add a thrust function that will be used if 
        self.can_move == 1.
        
        '''
        self.thrust = thrust_func

    def get_dist(self, _body):
        '''

        Parameters
        ----------
        _body : body
            Second body.

        Returns
        -------
        double
            Absolute distance between to bodies.

        '''
        return np.sqrt(sum((self.x - _body.x)**2))

    def get_force(self, _body):
        '''

        Parameters
        ----------
        _body : body
            Second body.

        Returns
        -------
        vector
            Force that the second body exerts one self.

        '''
        return G * _body.mass * (_body.x - self.x) / self.get_dist(_body)**3

    def get_forces(self, *bodies):
        '''

        Parameters
        ----------
        *bodies : list of bodies
            All bodies to evaluate.

        Returns
        -------
        vetor
            Combined force based on all the other given bodies and possibly the
            own acceleration.

        '''
        if not self.can_move:
            r = [self.get_force(_body) for _body in bodies if _body != self]
            return np.sum(r, 0)
        else:
            planets = np.sum([self.get_force(_body) for _body in bodies if _body != self], 0)
            self_forces = self.thrust()
            return planets + self_forces

    def get_energy(self):
        '''
        
        Return the kinetic eneergy of the body.
        
        '''
        return 0.5 * (self.mass * sum(self.v ** 2))

    def update(self, x, v, tau, use_old = False):
        '''

        Parameters
        ----------
        x : vector
            New value for x.
        v : vector
            New vlaue for v.
        tau : double
            Scaling factor, step size.
        use_old : bool, optional
            Include the old values of x and v in the calculation, used for 
            implicit or RK methods. The default is False.

        Returns
        -------
        None.

        '''
        self.x = self.x + np.array(x) * tau
        self.v = self.v + np.array(v) * tau
        if use_old:
            self.x = self.x_old + np.array(x) * tau
            self.v = self.v_old + np.array(v) * tau
            self.x_old = self.x.copy()
            self.v_old = self.v.copy()

    def martin(self, tau):
        '''
        
        Deprecated.
        
        '''
        alpha = mu / np.sqrt(sum(self.x ** 2)) ** 3
        x_old = self.x
        v_old = self.v
        self.x = 1/(1 + tau * alpha) * x_old + tau/(1 + tau * alpha) * v_old
        self.v = (1 - tau ** 2 * alpha) * v_old - tau * alpha/(1 + tau * alpha) * x_old

    def plot(self, fig = None):
        if fig is None:
            plt.plot(self.x[0], self.x[1], 'o', markersize = 2, color = self.color)
            plt.pause(0.0001)

class ode_algorithm:

    def __init__(self, approach, tau, steps, ode, plot = np.inf):
        self.approach = approach
        self.ode = ode
        self.tau = tau
        self.steps = steps
        self.plot = plot

    def __call__(self, *bodies):
        func = getattr(self, self.approach)
        func(*bodies)

    def fwd(self, *bodies):
        N = len(bodies)
        e = []
        for i in range(self.steps):
            ret = self.ode(*bodies)
            [_body.update(val[:2], val[2:], self.tau) for val, _body in zip(np.split(ret, N), bodies)]
            if not i % self.plot and i > 0:
                [_body.plot() for _body in bodies]
            e.append(np.sum([_body.get_energy() for _body in bodies]) - (G * np.prod([_body.mass for _body in bodies])/bodies[1].get_dist(bodies[0])))
        plt.figure()
        plt.plot(e)

    def bwd(self, *bodies):
        N = len(bodies)
        e = []
        for i in range(self.steps):
            ret_old = np.zeros(N*4)
            for _ in range(20):
                ret = self.ode(*bodies)
                [_body.update(val[:2] - old_val[:2], val[2:] - old_val[2:], self.tau) for val, old_val, _body in zip(np.split(ret, N), np.split(ret_old, N), bodies)]
                ret_old = ret
            if not i % self.plot:
                [_body.plot() for _body in bodies]
            e.append(np.sum([_body.get_energy() for _body in bodies]) - (G * np.prod([_body.mass for _body in bodies])/bodies[1].get_dist(bodies[0])))
        plt.figure()
        plt.plot(e)

    def martin(self, *bodies):
        for i in range(self.steps):
            bodies[1].martin(self.tau)
            if not i % self.plot:
                bodies[1].plot()

    def runge_kutta(self, a, b, c, *bodies):
        N = len(bodies)
        order = len(b)
        k = np.zeros(order, dtype = object)
        e = []
        for s in range(self.steps):
            for j in range(order):
                k[j] = self.ode(*bodies)
                new = np.zeros(k[0].shape)
                for i in range(j):
                    new = new + a[j, i] * k[i]
                [_body.update(val[:2], val[2:], self.tau) for val, _body in zip(np.split(new, N), bodies)]
            final = np.zeros_like(k[0])
            for i in range(order):
                final = final + b[i] * k[i]
            [_body.update(val[:2], val[2:], self.tau, True) for val, _body in zip(np.split(final, N), bodies)]
            if not s % self.plot and s > 0:
                [_body.plot() for _body in bodies]
            e.append(np.sum([_body.get_energy() for _body in bodies]) - (G * np.prod([_body.mass for _body in bodies])/bodies[1].get_dist(bodies[0])))
        # plt.figure()
        # plt.plot(e)

class solver:

    def __init__(self, _type, tau):
        self.type = _type
        self.tau = tau

    def __call__(self, approach, steps, *bodies, rk_params = None, plot = np.inf):
        if approach == 'fwd':
            ode = self.newton_ode_n
        elif approach == 'bwd':
            ode = self.newton_ode_n
        elif approach == 'martin':
            ode = None
        elif approach == 'rk':
            ode = self.newton_ode_n
            return ode_algorithm('runge_kutta', self.tau, steps, ode, plot = plot)(rk_params['a'], rk_params['b'], rk_params['c'], *bodies)
        return ode_algorithm(approach, self.tau, steps, ode, plot = plot)(*bodies)

    def newton_ode(self, *bodies):
        N = len(bodies)
        rhs = np.concatenate([np.concatenate([_body.x, _body.v]) for _body in bodies])
        lhs = np.roll(rhs, -2)

        for i in range(0, N):
            fixed_body = bodies[i]
            lhs[i * 4 + 2:i * 4 + 4] = np.sum([fixed_body.get_forces(_body) for _body in bodies if _body != fixed_body], 0)

        return lhs

    def newton_ode_n(self, *bodies):
        N = len(bodies)
        rhs = np.concatenate([np.concatenate([_body.x, _body.v]) for _body in bodies])
        lhs = np.roll(rhs, -2)

        for i in range(0, N):
            fixed_body = bodies[i]
            lhs[i * 4 + 2:i * 4 + 4] = fixed_body.get_forces(*bodies)

        return lhs

class engine:

    def linear():
        return np.array([0, 0])
    
    def hohmann(_body):
        return np.array([0, 0])

# Constants
au = 149.6e6
m_e = 5.976e24
mu = 1.33e20/((au * 1e3)**3) * 86400**2
G = 6.67e-11/((au * 1e3)**3) * 86400**2 * m_e

# Bodies
sun = body(2e30/m_e, [0,0], [0,0])
mercury = body(0.0552, [57.9e6/au * (1 - 0.206),0], [0, np.sqrt((mu * (1 + 0.206))/(57.9e6/au * (1 - 0.206)))])
venus = body(4.87e24/m_e, [108.2e6/au * (1 - 0.007), 0], [0, np.sqrt((mu * (1 + 0.007))/(108.2e6/au * (1 - 0.007)))])
earth = body(1, [1 * (1 - 0.0167), 0], [0, np.sqrt((mu * (1.0167))/(1 * (1 - 0.0167)))])

moon = body(0.073e24/m_e, [1 * (1 - 0.0167), 0.384e6/au], [np.sqrt(((3e7/((0.384e9)**3) * 86400**2) * (1.055))/(0.073e24/m_e * (1 - 0.055))), np.sqrt((mu * (1.0167))/(1 * (1 - 0.0167)))], color = 'r')

mars = body(0.642e24/m_e, [228e6/au * (1 - 0.094), 0], [0, np.sqrt((mu * (1 + 0.094))/(228e6/au * (1 - 0.094)))])
jupiter = body(1.8987e27/m_e, [5.19 * (1 - 0.049),0], [0, np.sqrt((mu * (1.049))/(5.19 * (1 - 0.049)))])

ganymed = body(14.82e22/m_e, [5.19 * (1 - 0.049), - 1.07e6/au], [np.sqrt(((155/((1.07e6)**3) * 86400**2) * (1))/(14.82e22/m_e * (1))), np.sqrt((mu * (1.049))/(5.19 * (1 - 0.049)))], color = 'r')

saturn = body(568e24/m_e, [1432e6/au * (1 - 0.052), 0], [0, np.sqrt((mu * (1 + 0.052))/(1432e6/au * (1 - 0.052)))])
uranus = body(86.8e24/m_e, [2867e6/au * (1 - 0.047), 0], [0, np.sqrt((mu * (1 + 0.047))/(2867e6/au * (1 - 0.047)))])
neptune = body(102e24/m_e, [4515e6/au * (1 - 0.010), 0], [0, np.sqrt((mu * (1 + 0.010))/(4515e6/au * (1 - 0.010)))])

# Spaceship
ship = body(-1, [(5.19 + 0.37) * (1 - 0.049),0], [0, np.sqrt((mu * (1.049))/((5.19 + 0.37) * (1 - 0.049)))], color = 'g')
# ship.set_thrust(engine.linear)

#%%
if __name__ == "__main__":
    plt.close('all')
    
    sun = body(2e30/m_e, [0,0], [0,0])
    earth = body(1, [1 * (1 - 0.0167), 0], [0, np.sqrt((mu * (1.0167))/(1 * (1 - 0.0167)))])
    jupiter = body(1.8987e27/m_e, [5.19 * (1 - 0.049),0], [0, np.sqrt((mu * (1.049))/(5.19 * (1 - 0.049)))])
    saturn = body(568e24/m_e, [1432e6/au * (1 - 0.052), 0], [0, np.sqrt((mu * (1 + 0.052))/(1432e6/au * (1 - 0.052)))])

    # Setup
    sol = solver('newton', 0.1)
    
    a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
    b = np.array([1/6,1/3,1/3,1/6])
    c = np.array([0, 0.5, 0.5, 1])
    
    
    rk = {'a': a, 'b': b, 'c': c}
    plot = 500 # np.inf
    

    # Run
    factor = 9.6/1.2
    if plot < np.inf:
        plt.figure(dpi = 200)
        plt.title("Parts of the solar system")
        plt.plot(0,0, 'rx')
        plt.pause(0.01)
        plt.xlim(-1.2 * factor, 1.2 * factor)
        plt.ylim(-1.2 * factor, 1.2 * factor)
    # ret = sol('rk', 7700, sun, earth, moon, rk_params = rk, plot = plot)
    ret = sol('rk', 7700, sun, earth, jupiter, saturn, rk_params = rk, plot = plot)
    
#%% 
if __name__ == "__main__":
    plt.close('all')

    sun = body(2e30/m_e, [0,0], [0,0])
    earth = body(1, [1 * (1 - 0.0167), 0], [0, np.sqrt((mu * (1.0167))/(1 * (1 - 0.0167)))])
    
    moon = body(0.073e24/m_e, [1 * (1 - 0.0167), 0.384e6/au], [np.sqrt(((3e7/((0.384e9)**3) * 86400**2) * (1.055))/(0.073e24/m_e * (1 - 0.055))), np.sqrt((mu * (1.0167))/(1 * (1 - 0.0167)))], color = 'r')

    # Setup
    sol = solver('newton', 0.01)
    
    a = np.array([[0,0,0,0],[2/5,0,0,0],[(-2889+1428*np.sqrt(5))/1024,(3785-1620*np.sqrt(5))/1024, 0,0],[(-3365+2094*np.sqrt(5))/6040, (-975-3046*np.sqrt(5))/2552, (467040+203968*np.sqrt(5))/240845,0]])
    b = np.array([(263+24*np.sqrt(5))/1812,(125-1000*np.sqrt(5))/3828,(3426304+1661952*np.sqrt(5))/5924787,(30-4*np.sqrt(5))/123])
    c = np.array([0, 2/5, (14-3*np.sqrt(5))/16, 1])
  
    rk = {'a': a, 'b': b, 'c': c}
    plot = 50 # np.inf
    

    # Run
    factor = 1.2/1.2
    if plot < np.inf:
        plt.figure(dpi = 200)
        plt.title("Earth - Moon system, projection along earth orbit")
        plt.plot(0,0, 'rx')
        plt.pause(0.01)
        plt.xlim(0.9, 1.07)
        plt.ylim(-0.1, 2)
    # ret = sol('rk', 7700, sun, earth, moon, rk_params = rk, plot = plot)
    ret = sol('rk', 7700, earth, moon, rk_params = rk, plot = plot)
