import numpy as np
from sniff.graph_theory import Graph
from sniff.simulation import Simulate

SEED = 42


def test_deterministic_1D_matches_reference():
    rng = np.random.default_rng(SEED)

    g = Graph(4, seed=SEED)
    A = g.erdos_renyi_adj(1.0)
    x0 = rng.random(4)

    sim = Simulate(A, 2.0, 0.05, x0, seed=SEED)
    sol = sim.solve_consensus_dynamics_w_GOE_noise(noise_strength=0.0)

    ref = np.load("tests/reference_data/solution_1d.npy")
    np.testing.assert_allclose(sol.y, ref, rtol=1e-6, atol=1e-8)


# def test_deterministic_2D_matches_reference():
#     rng = np.random.default_rng(SEED)
#
#     g = Graph(4, seed=SEED)
#     A = g.erdos_renyi_adj(1.0)
#     x0 = rng.random((4, 2)).flatten()
#
#     sim = Simulate(A, 2.0, 0.05, x0, seed=SEED)
#     sol = sim.solve_consensus_formation_setpoint_tracking_w_GOE_noise(
#         alpha=1.0,
#         beta=1.0,
#         p_track=[0, 0],
#         noise_strength=0.0,
#         spacing=1.0,)
#
#     ref = np.load("tests/reference_data/solution_2d.npy")
#     np.testing.assert_allclose(sol.y, ref, rtol=1e-6, atol=1e-8)
