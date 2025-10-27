import numpy as np
from sniff.graph_theory import Graph
from sniff.simulation import Simulate


def test_consensus_dynamics_converges():
    g = Graph(5)
    A = g.erdos_renyi_adj(1.0)
    x0 = np.random.random(5)
    sim = Simulate(A, time=2, time_step=0.01, x0=x0)
    solution = sim.solve_consensus_dynamics_w_GOE_noise(noise_strength=0.0)

    final_state = solution.y[:, -1]
    assert np.std(final_state) < 1e-3
