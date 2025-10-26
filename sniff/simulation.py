from sniff.noise import RMT
from sniff.graph_theory import Graph

import numpy as np
from scipy.integrate import solve_ivp


class Simulate:
    def __init__(self, A, time, time_step, x0,
                 noise_model='piecewise', noise_update_rate=0.1):
        """
        The noise model options:
        'static' :  single GOE sample for fixed uncertain channels
        'piecewise' : update GOE sample at noise_update_rate intervals
        'white' : update GOE sample at every time step
        """
        self.T = time
        self.dT = time_step
        self.x0 = x0
        self.rmo = RMT(A)
        self.noise_model = noise_model
        self.noise_update_rate = noise_update_rate
        self.current_L_noisy = None
        self.last_noise_update = -np.inf

    def _get_noisy_laplacian(self, t, noise_strength):
        """
        Returns the proper noisy laplacian based on the simulation
        specifications
        """
        if self.noise_model == 'white':
            return self.rmo.noise_GOE(noise_strength)

        elif self.noise_model == 'static':
            if self.current_L_noisy is None:
                self.current_L_noisy = self.rmo.noise_GOE(noise_strength)
            return self.current_L_noisy

        elif self.noise_model == 'piecewise':
            if self.current_L_noisy is None:
                self.current_L_noisy = self.rmo.noise_GOE(noise_strength)
                self.last_noise_update = t
            elif t - self.last_noise_update >= self.noise_update_rate:
                self.current_L_noisy = self.rmo.noise_GOE(noise_strength)
                self.last_noise_update = t
            return self.current_L_noisy

        else:
            raise ValueError(f"Unknown noise_model: {self.noise_model}")

    def solve_consensus_dynamics_w_GOE_noise(self, noise_strength=0.1):
        """
        Solves the consensus dynamics
        """
        self.current_L_noisy = None
        self.last_noise_update = -np.inf

        t_span = (0, self.T)
        t_eval = np.linspace(*t_span, int(self.T/self.dT))

        solution = solve_ivp(self._consensus_dynamics_w_GOE_noise,
                             t_span, self.x0, t_eval=t_eval,
                             method='RK45', args=(noise_strength,))

        return solution

    def _consensus_dynamics_w_GOE_noise(self, t, x, noise_strength):
        """
        Consensus dynamics state-space model
        """
        L_noisy = self._get_noisy_laplacian(t, noise_strength)
        return -L_noisy @ x

    def solve_consensus_setpoint_tracking_w_GOE_noise(self, alpha=1, beta=1, p_track=[0, 0], noise_strength=0.1):
        """
        Solves the consensus dynamics for converging to a 2D
        setpoint
        """
        self.current_L_noisy = None
        self.last_noise_update = -np.inf

        t_span = (0, self.T)
        t_eval = np.linspace(*t_span, int(self.T/self.dT))

        solution = solve_ivp(self._consensus_2D_setpoint_w_GOE_noise,
                             t_span,
                             self.x0,
                             t_eval=t_eval,
                             args=(alpha, beta, p_track, noise_strength),
                             method='RK45')

        return solution

    def _consensus_2D_setpoint_w_GOE_noise(self, t, x, alpha, beta, p_track, noise_strength):
        L_noisy = self._get_noisy_laplacian(t, noise_strength)
        B, c = Graph.make_setpoint_transform_2D(
            L_noisy, alpha=alpha, beta=beta, p_track=p_track)
        return B @ x + c

    def solve_consensus_formation_setpoint_tracking_w_GOE_noise(self, alpha=1, beta=1, p_track=[0, 0], noise_strength=0.1, spacing=0.05):
        """
        Solves the consensus dynamics for converging to a 2D
        setpoint and arranges agents into a formation
        """
        self.current_L_noisy = None
        self.last_noise_update = -np.inf

        t_span = (0, self.T)
        t_eval = np.linspace(*t_span, int(self.T/self.dT))

        solution = solve_ivp(self._consensus_formation_2D_setpoint_w_GOE_noise,
                             t_span,
                             self.x0,
                             t_eval=t_eval,
                             method='RK45',
                             args=(alpha, beta, p_track, noise_strength, spacing))

        return solution

    def _consensus_formation_2D_setpoint_w_GOE_noise(self, t, x, alpha, beta, p_track, noise_strength, spacing):
        L_noisy = self._get_noisy_laplacian(t, noise_strength)
        formation = Simulate.generate_formation(L_noisy.shape[0], spacing)
        B, c = Graph.make_setpoint_formation_transform_2D(
            L_noisy, alpha=alpha, beta=beta, p_track=p_track, formation_offsets=formation)
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
