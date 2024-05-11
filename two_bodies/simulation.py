import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

def two_bodies():
    pass

class body:

    def __init__(self, mass, x_0, v_0):
        self.mass = mass
        self.x = np.array(x_0)
        self.v = np.array(v_0)

    def get_dist(self, body):
        return np.sqrt(sum((self.x - body.x)**2))

    def update(self, x, v, tau):
        self.x = self.x + np.array(x) * tau
        self.v = self.v + np.array(v) * tau

    def plot(self, fig = None):
        if fig is None:
            plt.plot(self.x[0], self.x[1], 'go')

class ode_algorithm:

    def __init__(self, approach, tau, steps, ode):
        self.approach = approach
        self.ode = ode
        self.tau = tau
        self.steps = steps

    def __call__(self, sun, *bodies):
        func = getattr(self, self.approach)
        func(sun, *bodies)

    def fwd(self, sun, *bodies):
        N = len(bodies)
        for i in range(self.steps):
            ret = self.ode(sun, *bodies)
            [body.update(val[:2], val[2:], self.tau) for val, body in zip(np.split(ret, N), bodies)]
            [body.plot() for body in bodies]

class solver:

    def __init__(self, _type, tau):
        self.type = _type
        self.tau = tau

    def __call__(self, approach, steps, sun, *bodies):
        if approach == 'fwd':
            ode = self.newton_ode
            fwd_euler = ode_algorithm('fwd', self.tau, steps, ode)
            return fwd_euler(sun, bodies[0])

    def newton_ode(self, sun, body):
        rhs = np.concatenate([body.x, body.v])
        lhs = np.zeros((4,))
        lhs[:2] = rhs[-2:]

        r3 = body.get_dist(sun)

        mrel = body.mass/sun.mass
        lhs[-2:] = -(1+mrel)*rhs[:2]/r3

        return lhs

    def newton_ode_n(self, sun, *bodies):
        N = len(bodies)
        rhs = [np.concatenate([body.x, body.v]) for body in bodies]
        rhs = np.concatenate(rhs)
        lhs = np.zeros(((N) * 4,))
        lhs[:-2*N] = rhs[2*N:]

        mrel = np.array([body.mass for body in bodies])/sun.mass

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


if __name__ == "__main__":
    earth = body(1, [1, 0], [0, 1])
    sun = body(339000, [0,0], [0,0])
    sol = solver('newton', 0.001)
    ret = sol('fwd', 1000, sun, earth)
