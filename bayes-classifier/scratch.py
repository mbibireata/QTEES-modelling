import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sdeint import itoint

class LorenzAttractor:
    def __init__(self, params, init_state):
        self.params = params
        self.init_state = init_state
        self.state = self.init_state
        self.state_history = [self.state]

    def dtx(self, x, y, z=None):
        return self.params['sigma'] * (y - x)

    def dty(self, x, y, z):
        return x * (self.params['rho'] - z) - y

    def dtz(self, x, y, z):
        return x * y - self.params['beta'] * z

    def evolve_dt(self, t, state):
        x, y, z = state
        s, r, b = self.params['sigma'], self.params['rho'], self.params['beta']
        return np.array([s * (y - x), x * (r - z) - y, x * y - b * z])

    def f(self, r, t):
        x, y, z = r
        s, r, b = self.params['sigma'], self.params['rho'], self.params['beta']
        return np.array([s * (y - x), x * (r - z) - y, x * y - b * z])

    def G(self, r, t):
        return np.diag([1., 1., 1.])

    def simulate_stochastic(self):
        a, b = self.params['t_init'], self.params['t_finish']
        t = np.linspace(a, b, self.params['n_steps'])

        soln = itoint(self.f, self.G, self.init_state, t)


        self.state_history = soln

    def simulate(self):
        a, b = self.params['t_init'], self.params['t_finish']
        t = np.linspace(a, b, self.params['n_steps'])

        #self.state_history = solve_ivp(self.evolve_dt, [a, b], self.init_state, t_eval=t)

    def visualize(self):
        fig = plt.figure("Dynamical System")
        ax = plt.axes(projection='3d')
       
        #data = np.array(self.state_history)
        #data = self.state_history.y
        data = self.state_history

        ax.plot3D(data[:, 0], data[:, 1], data[:, 2], lw=1, color='r')
        #ax.plot3D(data[0], data[1], data[2], lw=1, color='r')

        plt.show()


def main():
    params = {
        'sigma'   : 10. ,
        'rho'     : 28. ,
        'beta'    : 8./3. ,
        't_init'  : 0 ,
        't_finish': 10 , 
        'n_steps' : 10001
    }

    init_state = np.array([0., .5, 0.5], dtype=float)

    system = LorenzAttractor(params, init_state)
    #system.simulate()
    system.simulate_stochastic()
    system.visualize()

if __name__ == "__main__":
    main()
