import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

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

    def evolve_dt(self):
        new_state = self.state +  np.array([
                self.dtx(self.state[0], self.state[1]),
                self.dty(self.state[0], self.state[1], self.state[2]),
                self.dtz(self.state[0], self.state[1], self.state[2])
            ])

        self.state_history.append(new_state)
        self.state = new_state

    def simulate(self):
        for i in range(self.params['n_steps'] - 1):
            self.evolve_dt()

        return self.state_history

def main():
    params = {
        'sigma'   : 0.9 ,
        'rho'     : 0.8 ,
        'beta'    : 1.1 ,
        'n_steps' : 1000
    }

    init_state = np.array([1., 1., 1.], dtype=float)

    system = LorenzAttractor(params, init_state)
    system.simulate()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
