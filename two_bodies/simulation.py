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

    def get_forces(self, body):
        return (1 + 1/body.mass) * (body.x - self.x) / self.get_dist(body)**3

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
            ret_x = np.split(ret[:2*N], N)
            ret_v = np.split(ret[2*N:], N)
            ret = np.concatenate([np.concatenate([x, v]) for x,v in zip(ret_x, ret_v)])
            [body.update(val[:2], val[2:], self.tau) for val, body in zip(np.split(ret, N), bodies)]
            [body.plot() for body in bodies]

class solver:

    def __init__(self, _type, tau):
        self.type = _type
        self.tau = tau

    def __call__(self, approach, steps, sun, *bodies):
        if approach == 'fwd':
            ode = self.newton_ode_n
            fwd_euler = ode_algorithm('fwd', self.tau, steps, ode)
            return fwd_euler(sun, *bodies)

    def newton_ode(self, sun, body):
        rhs = np.concatenate([body.x, body.v])
        lhs = np.zeros((4,))
        lhs[:2] = rhs[-2:]

        r3 = body.get_dist(sun) ** 3

        mrel = body.mass/sun.mass
        lhs[-2:] = body.get_forces(sun) # (1+mrel)*rhs[:2]/r3

        return lhs

    def newton_ode_n(self, sun, *bodies):
        N = len(bodies)
        rhs = np.concatenate([np.concatenate([body.x for body in bodies]), np.concatenate([body.v for body in bodies])])
        lhs = np.zeros(((N) * 4,))
        lhs[:2*N] = rhs[-2*N:]

        mrel = np.array([body.mass for body in bodies])/sun.mass

        for i in range(0, N):
            fixed_body = bodies[i]

            lhs[2*N + 2*i:2*N + 2*(i+1)] = fixed_body.get_forces(sun) + np.sum([fixed_body.get_forces(body) for body in bodies if body != fixed_body], 0)

        return lhs


if __name__ == "__main__":
    earth = body(1, [1, 0], [0, 1])
    sun = body(333000, [0,0], [0,0])
    jupiter = body(1000, [20,0], [0,0.25])
    sol = solver('newton', 0.01)
    ret = sol('fwd', 10000, sun, earth)
    plt.plot(0,0, 'rx')
