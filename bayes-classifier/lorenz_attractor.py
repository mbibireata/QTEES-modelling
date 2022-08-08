import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sdeint import itoint, deltaW

class LorenzAttractor:
    def __init__(self, params, init_state):
        self.params = params
        self.init_state = init_state
        self.state = self.init_state
        self.state_history = [self.state]

    def skopt_f(self, x, noise_level=0.):
        a, b = self.params['t_init'], self.params['t_finish']
        t = np.linspace(a, b, self.params['n_steps'])

        # Assign parameters according to dimensions in x
        self.params['sigma'] = x[0]
        self.params['rho'] =   x[1]
        self.params['beta'] =  x[2]

        soln = itoint(self.f, self.G, self.init_state, t)

        # Return final point in trajectory as scalar formed by
        # linear combination of xyz components
        return np.sum(np.array([2., 3., 5.]) * soln[-1])

    def scipy_f(self, t, r):
        ''' 
            scipy.integrate takes arguments as t, r instead of r, t, so call
            via a wrapper
        '''
        return self.f(r, t)

    def f(self, r, t):
        x, y, z = r
        s, r, b = self.params['sigma'], self.params['rho'], self.params['beta']
        return np.array([s * (y - x), x * (r - z) - y, x * y - b * z])

    def G(self, r, t):
        #return np.diag(deltaW(1, 3, 1)[0])
        #return np.array([[.5, .5, 0.],
        #                 [.5, .5, .5],
        #                 [.5, .5, .5]], dtype=float)
        return np.diag([.5, .5, .5])

    def simulate_stochastic(self):
        a, b = self.params['t_init'], self.params['t_finish']
        t = np.linspace(a, b, self.params['n_steps'])

        soln = itoint(self.f, self.G, self.init_state, t)

        self.state_history = soln

        return soln

    def simulate(self):
        a, b = self.params['t_init'], self.params['t_finish']
        t = np.linspace(a, b, self.params['n_steps'])

        soln = solve_ivp(self.scipy_f, [a, b], self.init_state, t_eval=t)

        self.state_history = soln.y.T

        return soln.y.T

        #self.state_history = solve_ivp(self.evolve_dt, [a, b], self.init_state, t_eval=t)

    def visualize(self):
        fig = plt.figure("Dynamical System")
        ax = plt.axes(projection='3d')
       
        data = self.state_history

        ax.plot3D(data[:, 0], data[:, 1], data[:, 2], lw=1, color='r')

        plt.show()

def visualize(data):
    fig = plt.figure("Dynamical System")
    ax = plt.axes(projection='3d')
   
    for d in data:
        ax.plot3D(d[:, 0], d[:, 1], d[:, 2], lw=1)

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

    init_state = np.array([1., 1., 1.], dtype=float)

    system = LorenzAttractor(params, init_state)
    soln1 = system.simulate_stochastic()

    perturbed = LorenzAttractor(params, init_state + np.array([0., 0., 0.001]))
    soln2 = perturbed.simulate_stochastic()
    visualize([soln1, soln2])
    #system.visualize()

if __name__ == "__main__":
    main()
