import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


class body:

    def __init__(self, mass, x_0, v_0):
        self.mass = mass
        self.x = np.array(x_0)
        self.v = np.array(v_0)
        self.x_old = np.array(x_0)
        self.v_old = np.array(v_0)

    def get_dist(self, _body):
        return np.sqrt(sum((self.x - _body.x)**2))

    def get_forces(self, _body):
        return G * _body.mass * (_body.x - self.x) / self.get_dist(_body)**3

    def get_energy(self):
        return 0.5 * (self.mass * sum(self.v ** 2))

    def update(self, x, v, tau, use_old = False):
        self.x = self.x + np.array(x) * tau
        self.v = self.v + np.array(v) * tau
        if use_old:
            self.x = self.x_old + np.array(x) * tau
            self.v = self.v_old + np.array(v) * tau
            self.x_old = self.x.copy()
            self.v_old = self.v.copy()

    def martin(self, tau):
        alpha = mu / np.sqrt(sum(self.x ** 2)) ** 3
        x_old = self.x
        v_old = self.v
        self.x = 1/(1 + tau * alpha) * x_old + tau/(1 + tau * alpha) * v_old
        self.v = (1 - tau ** 2 * alpha) * v_old - tau * alpha/(1 + tau * alpha) * x_old

    def plot(self, fig = None):
        if fig is None:
            plt.plot(self.x[0], self.x[1], 'go', markersize = 2)
            plt.pause(0.01)

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
            ret_x = np.split(ret[:2*N], N)
            ret_v = np.split(ret[2*N:], N)
            ret = np.concatenate([np.concatenate([x, v]) for x,v in zip(ret_x, ret_v)])
            [_body.update(val[:2], val[2:], self.tau) for val, _body in zip(np.split(ret, N), bodies)]
            if not i % self.plot and i > 0:
                [_body.plot() for _body in bodies]
            e.append(np.sum([_body.get_energy() for _body in bodies]) - (G * np.prod([_body.mass for _body in bodies])/bodies[1].get_dist(bodies[0])))
        plt.figure()
        plt.plot(e)

    def bwd(self, *bodies):
        N = len(bodies)
        for i in range(self.steps):
            ret_old = np.zeros(N*4)
            for _ in range(20):
                ret = self.ode(*bodies)
                ret_x = np.split(ret[:2*N], N)
                ret_v = np.split(ret[2*N:], N)
                ret = np.concatenate([np.concatenate([x, v]) for x,v in zip(ret_x, ret_v)])
                [_body.update(val[:2] - old_val[:2], val[2:] - old_val[2:], self.tau) for val, old_val, _body in zip(np.split(ret, N), np.split(ret_old, N), bodies)]
                ret_old = ret
            if not i % self.plot:
                [_body.plot() for _body in bodies]

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
                ret_x = np.split(k[j][:2*N], N)
                ret_v = np.split(k[j][2*N:], N)
                k[j] = np.concatenate([np.concatenate([x, v]) for x,v in zip(ret_x, ret_v)])
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
        plt.figure()
        plt.plot(e)

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
        rhs = np.concatenate([bodies[1].x, bodies[1].v])
        lhs = np.zeros((4,))
        lhs[:2] = rhs[-2:]

        lhs[-2:] = bodies[1].get_forces(bodies[0])

        return lhs

    def newton_ode_n(self, *bodies):
        N = len(bodies)
        rhs = np.concatenate([np.concatenate([_body.x for _body in bodies]), np.concatenate([_body.v for _body in bodies])])
        lhs = np.zeros(((N) * 4,))
        lhs[:2*N] = rhs[-2*N:]

        for i in range(0, N):
            fixed_body = bodies[i]
            lhs[2*N + 2*i:2*N + 2*(i+1)] = np.sum([fixed_body.get_forces(_body) for _body in bodies if _body != fixed_body], 0)

        return lhs


if __name__ == "__main__":
    plt.close('all')
    mu = 1.33e20/((1.52e11)**3) * 86400**2
    G = 6.67e-11/((1.52e11)**3) * 86400**2 * 5.976e24
    earth = body(1, [1 * (1 - 0.0167), 0], [0, np.sqrt((mu * (1.0167))/(1 * (1 - 0.0167)))])
    sun = body(2e30/5.976e24, [0,0], [0,0])
    jupiter = body(1.8987e27/5.976e24, [5.19 * (1 - 0.0167),0], [0, np.sqrt((mu * (1.0167))/(5.19 * (1 - 0.0167)))])
    sol = solver('newton', 0.05)
    a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
    b = np.array([1/6,1/3,1/3,1/6])
    c = np.array([0, 0.5, 0.5, 1])
    rk = {'a': a, 'b': b, 'c': c}
    plot = 1000 # np.inf
    if plot < np.inf:
        plt.plot(0,0, 'rx')
        plt.pause(0.01)
    ret = sol('rk', 50000, sun, earth, rk_params = rk, plot = plot)
