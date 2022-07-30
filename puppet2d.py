import numpy as np
import scipy as sp
import scipy.linalg

import matplotlib.pyplot as plt

from dynamicmodels import DynamicModel

# optional external module imports
try :
    from tqdm import trange
    pretty_range = trange
except ModuleNotFoundError :
    print("TQDM isn't installed. Progress bars will be disabled.")
    pretty_range = range


# helper functions for constructing useful finite difference matrices
D1 = lambda N : 0.5*sp.linalg.toeplitz([0, -1] + [0]*(N - 2), [0, 1] + [0]*(N - 2)) # first diff.
D1_c = lambda N : 0.5*sp.linalg.circulant([0, -1] + [0]*(N - 3) + [1])              # circ. first diff.
D2 = lambda N : sp.linalg.toeplitz([2, -1] + [0]*(N - 2), [2, -1] + [0]*(N - 2))    # second diff.
D2_c = lambda N : sp.linalg.circulant([2, -1] + [0]*(N - 3) + [-1])                 # circ. second diff.

class Puppet2d(DynamicModel) :
    N_state = None
    N_control = None

    # 14 postural free parameters -- peristaltic preset
    sign_a = 1          # sign of axial Hookean term
    n_a = 1             # axial damping coefficient 
    a_a = 1.1           # axial skew cooperative coupling
    b_a = 0             # axial sym cooperative coupling
    A_a = -1            # axial alpha nonlinearity
    B_a = 10            # axial beta nonlinearity
    g_a = 0             # axial noise strength
    
    sign_t = -0.1       # sign of transverse Hookean term
    n_t = 0.1           # transverse damping coefficient
    a_t = 0             # transverse skew coupling strength
    b_t = 0             # transverse symmetric coupling strength
    d_t = 0             # axial--transverse quadratic coupling strength
    B_t = 0             # transverse beta nonlinearity
    g_t = 0.01          # transverse noise strength
    
    # 2 environmental free parameters -- peristaltic preset
    f_t = 0.0       
    f_a = 0.02

    channel_radius = 10
    channel_width = None
    constraint_force = 1

    def __init__(self, N=10, channel_radius=10, channel_width=0.2) :
        self.N = N                              # number of points/vertices in discrete model
        self.N_vertices = N
        self.N_edges = N - 1                    # number of edges in discrete model
        self.N_internal_vertices = N - 2        # number of vertices in interior of domain (joints)
        self.N_joints = self.N_internal_vertices

        self.N_configuration = 2*N                      # 2 coordinates for each vertex
        self.N_state = 4*N                              # 2 coordinates and 2 momenta for each vertex
        self.N_control = self.N_edges + self.N_internal_vertices  # no. tensions + no. torques

        self.space_axis = np.linspace(0, 1, self.N_vertices)
        self.space_axis_vertices = self.space_axis
        self.space_axis_edges = np.linspace(0, 1, self.N_edges)
        self.space_axis_internal_vertices = np.linspace(0, 1, self.N_internal_vertices)

        self.channel_width = channel_width
        self.channel_radius = channel_radius
        self.inner_radius = channel_radius
        self.center_radius = channel_radius + self.channel_width/2
        self.outer_radius = channel_radius + self.channel_width
        self.channel_circumference = 2*np.pi*self.center_radius

        # store some useful matrices
        self.R = np.array([[0, 1], [-1, 0]])         # 90 degree clockwise rotation matrix
        self.D1_a = D1_c(self.N_edges)               # first difference matrix on edges
        self.D2_a = D2_c(self.N_edges)               # second difference matrix on edges
        self.D1_t = D1_c(self.N_joints)              # first difference matrix on internal vertices
        self.D2_t = D2_c(self.N_joints)              # second difference matrix on internal vertices
        
        self.M_av = sp.linalg.toeplitz([1, 1] + [0]*(self.N_edges - 2), [1] + [0]*(self.N_edges - 1))[1:]

    def evolution_rule(self, time, state, inputs=None) :
        coordinates = state[:self.N_configuration]      # flattened coordinate vector
        momenta = state[self.N_configuration:]          # flattened momentum vector

        Dcoordinates = momenta                          # coordinate dynamics
        Dmomenta = self.forces(time, state, inputs)     # momentum dynamics

        Dstate = np.concatenate([Dcoordinates, Dmomenta])
        return Dstate

    def analyse_trajectory(self) :
        print("analysing trajectory")

        kinematic_names = ["r", "p", "e", "l", "t", "n", "epsilon", "alpha",
                "theta", "kappa", "De", "Depsilon", "Dalpha", "Dtheta", "Dkappa"]

        # compute kinematics for every frame of the final trajectory; store as a list
        # of arrays
        kinematic_quantities = []
        for i in pretty_range(len(self.trajectory)) :
            state = self.trajectory[i]
            kinematic_quantities.append(self.kinematics(state))

        # unpack kinematics into member attributes
        for i in pretty_range(len(kinematic_names)) :
            name = kinematic_names[i]
            setattr(self, name, np.array([arr[i] for arr in kinematic_quantities]))

    def kinematics(self, x) :
        # x is state vector; decompose into position and momentum components
        r = x[:2*self.N].reshape(self.N, 2)
        p = x[2*self.N:].reshape(self.N, 2)
    
        # configuration space kinematics
        e = np.diff(r, axis=0)                          # compute edge vectors
        l = np.linalg.norm(e, axis=1)                   # edge lengths
        t = (e.T/l).T                                   # unit tangent vectors
        n = np.dot(self.R, t.T).T                       # unit normal vectors
    
        epsilon = l - 1                                 # strain (resting edge length = 1)
    
        alpha = np.arctan2(t[:, 1], t[:, 0])            # linkage angles (rad)
        theta = np.diff(alpha)                          # bending angles (rad)
        kappa = 2*np.tan(theta/2)                       # use DDG curvature
    
        # tangent space kinematics
        De = np.diff(p, axis=0)                                                 # edge velocity
        Depsilon = [np.dot(t_i, De_i) for t_i, De_i in zip(t, De)]              # stretch rate
        Depsilon = np.array(Depsilon)                                           #   typecast
        Dalpha = [np.dot(n_i, -De_i)/l_i for n_i, De_i, l_i in zip(n, De, l)]   # linkage angular velocity
        Dalpha = np.array(Dalpha)                                               #   typecast
        Dtheta = np.diff(Dalpha)                                                # bending velocity
        Dkappa = Dtheta/(np.cos(theta/2)**2)                                    # DDG curvature rate
    
        return r, p, e, l, t, n, epsilon, alpha, theta, kappa, De, Depsilon, Dalpha, Dtheta, Dkappa

    def forces(self, time, x, inputs=None) :
        """
        Compute force as a function of phase space coordinates x.
        """

        # decompose inputs into tensions and torques
        if inputs is None : inputs = np.zeros(self.N_control)
        input_tensions = inputs[:self.N_edges]
        input_torques = inputs[self.N_edges:]
    
        # compute kinematics from given phase space coordinate vector
        kinematic_quantities = self.kinematics(x)
        r, p, e, l, t, n, epsilon, alpha, theta, kappa, De, Depsilon, Dalpha, Dtheta, Dkappa = kinematic_quantities

        # compute generalised forces using useful kinematic variables
        tau_epsilon_local = self.sign_a*epsilon - self.A_a*epsilon**2 - self.B_a*epsilon**3 - self.n_a*Depsilon
        tau_epsilon_nonlocal = - self.a_a*np.dot(self.D1_a, epsilon) - self.b_a*np.dot(self.D2_a, epsilon)
    
        tau_kappa_local = (self.d_t*np.dot(self.M_av, epsilon) + self.sign_t)*kappa\
                            - self.B_t*kappa**3 - self.n_t*Dkappa
        tau_kappa_nonlocal = self.a_t*np.dot(self.D1_t, kappa) - self.b_t*np.dot(self.D2_t, kappa)
    
        tau_epsilon = tau_epsilon_local + tau_epsilon_nonlocal     # total axial tension
        tau_kappa = tau_kappa_local + tau_kappa_nonlocal           # total transverse torque
    
        # find force exerted on each vertex
        F_a = np.zeros((self.N, 2))                     # will hold axial forces
        for i in range(self.N_edges) :                  # loop through edges
            F_a[i] += -tau_epsilon[i]*t[i]              # edge force exerts force on previous vertex
            F_a[i + 1] += +tau_epsilon[i]*t[i]          # edge force exerts force on next vertex
    
        F_t = np.zeros((self.N, 2))                     # will hold transverse forces
        for i in range(self.N - 2) :                    # loop through joints
            # first couple
            F_t[i + 0] += -tau_kappa[i]*n[i]/l[i]             # torque produces force on previous vertex
            F_t[i + 1] += +tau_kappa[i]*n[i]/l[i]             # torque produces force on current vertex
            # second couple
            F_t[i + 1] += +tau_kappa[i]*n[i + 1]/l[i + 1]     # torque produces force on current vertex
            F_t[i + 2] += -tau_kappa[i]*n[i + 1]/l[i + 1]     # torque produces force on next vertex
    
        F_s = np.zeros((self.N, 2))                     # will hold substrate interaction forces
    
#       TODO remove this -- it's only good for nematode simulations!
#        # substrate interaction at head/tail
#        F_s[0] = -self.f_a*t[0]*np.dot(t[0], p[0]) - self.f_t*n[0]*np.dot(n[0], p[0])
#        F_s[-1] = -self.f_a*t[-1]*np.dot(t[-1], p[-1]) - self.f_t*n[-1]*np.dot(n[-1], p[-1])
#        for i in range(N_t) :           # loop through internal vertices
#            tt = (t[i] + t[i + 1])/np.linalg.norm(t[i] + t[i + 1])  # find tangent vector
#            nn = np.dot(R, tt)                                      # find normal vector
#            F_s[i + 1] = -self.f_a*tt*np.dot(tt, p[i + 1]) - self.f_t*nn*np.dot(nn, p[i + 1])
    
        # isotropic Coulomb substrate interaction
        p_unit = (p.T/np.linalg.norm(p, axis=1)).T
        F_s = -self.f_a*p_unit
    
        # compute confinement forces due to track boundaries
        ry = r[:, 1]    # y displacement of each point
        F_cy = - self.constraint_force*(ry > self.channel_width/2)\
               + self.constraint_force*(ry < -self.channel_width/2)
        F_cx = np.zeros(F_cy.shape)
        F_c = np.array([F_cx, F_cy]).T

        F_c = self.constraint_forces(kinematic_quantities)

        F_total = F_a + F_t + F_s + F_c
    
        return F_total.flatten()

    def constraint_forces(self, kinematic_quantities) :
        # kinematics
        r, p, e, l, t, n, epsilon, alpha, theta, kappa, De, Depsilon, Dalpha, Dtheta, Dkappa = kinematic_quantities
        segment_radii = np.linalg.norm(r, axis=1)
        unit_vectors = (r.T/segment_radii).T

        # forces
        inner_force = ((segment_radii < self.inner_radius)*self.constraint_force*unit_vectors.T).T
        outer_force = -((segment_radii > self.outer_radius)*self.constraint_force*unit_vectors.T).T
        total_constraint_force = inner_force + outer_force
        return total_constraint_force


    def generate_initial_state(self, coordinate_std=0, momentum_std=0) :
        # our default configuration has all segments equally spaced, wrapped onto the
        # circular track
        axial_displacements = np.arange(self.N_vertices)
        track_angles = 2*np.pi*axial_displacements/self.channel_circumference

        template_rx = self.center_radius*np.cos(track_angles)
        template_ry = self.center_radius*np.sin(track_angles)
        template_r = np.array([template_rx, template_ry]).T

        # add noise to the template configuration
        initial_r = template_r + coordinate_std*np.random.randn(*template_r.shape)

        # generate noisy initial momentum, near rest
        template_px = np.zeros(self.N_vertices)
        template_py = np.zeros(self.N_vertices)
        template_p = np.array([template_px, template_py]).T
        initial_p = template_p + momentum_std*np.random.randn(*template_p.shape)

        initial_x = np.concatenate([initial_r.flatten(), initial_p.flatten()])

        return initial_x

