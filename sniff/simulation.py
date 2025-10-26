from sniff.noise import RMT
from sniff.graph_theory import Graph

import matplotlib.pyplot as plt
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

    def solve_consensus_setpoint_tracking_w_GOE_noise(self):
        """
        Solves the consensus dynamics for converging to a 2D
        setpoint
        """
        t_span = (0, self.T)
        t_eval = np.linspace(*t_span, int(self.T/self.dT))

        solution = solve_ivp(self._consensus_2D_setpoint_w_GOE_noise,
                             t_span, self.x0, t_eval=t_eval, method='RK45')

        return solution

    def _consensus_2D_setpoint_w_GOE_noise(self, t, x):
        L_noisy = self.rmo.noise_GOE(noise_strength=0.5)
        B, c = Graph.make_setpoint_transform_2D(
            L_noisy, alpha=1, beta=1, p_track=[0, 0])
        return B @ x + c

    def solve_consensus_formation_setpoint_tracking_w_GOE_noise(self):
        """
        Solves the consensus dynamics for converging to a 2D
        setpoint and arranges agents into a formation
        """
        t_span = (0, self.T)
        t_eval = np.linspace(*t_span, int(self.T/self.dT))

        solution = solve_ivp(self._consensus_formation_2D_setpoint_w_GOE_noise,
                             t_span, self.x0, t_eval=t_eval, method='RK45')

        return solution

    def _consensus_formation_2D_setpoint_w_GOE_noise(self, t, x):
        # triangle formation for test
        # formation = np.array([
        #     [0, 0],
        #     [0.05, 0],
        #     [0.025, 0.05]
        # ])
        L_noisy = self.rmo.noise_GOE(noise_strength=0.5)
        formation = Simulate.generate_formation(L_noisy.shape[0], 0.05)
        B, c = Graph.make_setpoint_formation_transform_2D(
            L_noisy, alpha=1, beta=1, p_track=[0, 0], formation_offsets=formation)
        return B @ x + c

    @staticmethod
    def generate_formation(n, spacing) -> np.array:
        """
        Generates a grid formation for a given number of agents
        and scales the formation according to the spacing
        """
        side = int(np.ceil(np.sqrt(n)))
        formation = []

        for i in range(side):
            for j in range(side):
                if len(formation) >= n:
                    break
                formation.append([i*spacing, j*spacing])

        return np.array(formation)
