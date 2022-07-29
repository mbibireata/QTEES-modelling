import numpy as np
import scipy as sp
import scipy.linalg

from dynamicmodels import DynamicModel

class LinearKelvinChain(DynamicModel) :
    N_state = None
    N_control = None

    def __init__(self, N=10, mass=1, stiffness=1, damping=1) :
        self.N = N
        self.N_state = 2*N
        self.N_control = N - 1

        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

        # useful matrices
        self.I = np.eye(N)                                      # identity matrix
        self.P = sp.linalg.circulant([0, 1] + (N - 2)*[0]).T    # circulant unit shift matrix
        self.D1 = self.P - self.I                               # circulant first difference matrix
        self.D2 = np.dot(self.D1, self.D1.T)                    # circulant second difference matrix

        # normal modes come in cosine, sine pairs with identical frequencies and damping ratios
        # this is a consequence of translation invariance!
        self.D2_eigenvalues = 4*np.sin(np.pi*np.arange(N)/N)**2
        self.D1_eigenvalues = np.sqrt(self.D2_eigenvalues)

        # complex normal mode pairs are contained within the N by N DFT matrix
        self.normal_modes = sp.linalg.dft(N, scale="sqrtn")

        # we want real normal modes -- so we take cosine part from first mode in the
        # pair, sine part from the second mode in the pair
        dividing_index = int(np.ceil((N - 1)/2)) + 1 # index splitting first and second pair of modes
        self.normal_modes[:dividing_index] = np.real(self.normal_modes[:dividing_index])
        self.normal_modes[dividing_index:] = np.imag(self.normal_modes[dividing_index:])

        # now we sort the eigenvalues and eigenvectors in ascending order
        eigenvalue_sort_indices = np.argsort(self.D2_eigenvalues)
        self.D1_eigenvalues = self.D1_eigenvalues[eigenvalue_sort_indices]
        self.D2_eigenvalues = self.D2_eigenvalues[eigenvalue_sort_indices]
        self.normal_modes = np.real(self.normal_modes[eigenvalue_sort_indices])

        # compute natural frequencies and damping ratios from the eigenvalues
        self.natural_frequency_scale = np.sqrt(stiffness/mass)
        self.damping_ratio_scale = damping/(2*np.sqrt(stiffness*mass))
        self.natural_frequencies = self.natural_frequency_scale*self.D1_eigenvalues
        self.damping_ratios = self.damping_ratio_scale*self.D1_eigenvalues


    def evolution_rule(self, time, state, tension_input=None) :
        if tension_input is None : tension_input = np.zeros(self.N)

        q = state[:self.N]      # slice out position
        v = state[self.N:]      # slice out velocity

        Dq = v
        Dv = - self.stiffness*np.dot(self.D2, q)\
             - self.damping*np.dot(self.D2, v)\
             - np.dot(self.D1.T, self.input.input_func(time) + tension_input)
        Dv = Dv/self.mass
        return np.concatenate([Dq, Dv])


