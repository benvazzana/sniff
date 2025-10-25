from sniff.noise import RMT

import numpy as np
from scipy.integrate import solve_ivp


class Simulate:
    def __init__(self, A, time, time_step, x0):
        self.T = time
        self.dT = time_step
        self.x0 = x0
        self.rmo = RMT(A)

    def solve_consensus_dynamics_w_GOE_noise(self):
        """
        Solves the consensus dynamics
        """
        t_span = (0, self.T)
        t_eval = np.linspace(*t_span, int(self.T/self.dT))

        solution = solve_ivp(self._consensus_dynamics_w_GOE_noise,
                             t_span, self.x0, t_eval=t_eval, method='RK45')

        return solution

    def _consensus_dynamics_w_GOE_noise(self, t, x):
        """
        Consensus dynamics state-space model
        """
        L_noisy = self.rmo.noise_GOE()
        return -L_noisy @ x
