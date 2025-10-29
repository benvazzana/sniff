import numpy as np
from sniff.graph_theory import Graph
from sniff.simulation import Simulate

SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # 1D case
    g = Graph(4, seed=SEED)
    A = g.erdos_renyi_adj(1.0)

    x0 = rng.random(4)
    sim = Simulate(A, time=2.0, time_step=0.05, x0=x0, seed=SEED)
    sol = sim.solve_consensus_dynamics_w_GOE_noise(noise_strength=0.0)
    np.save("tests/reference_data/solution_1d.npy", sol.y)

    # 2D case
    x0_2d = rng.random((4, 2)).flatten()
    sim2 = Simulate(A, time=2.0, time_step=0.05, x0=x0_2d, seed=SEED)
    sol2 = sim2.solve_consensus_formation_setpoint_tracking_w_GOE_noise(
        alpha=1.0,
        beta=1.0,
        p_track=[0.0, 0.0],
        noise_strength=0.0,
        spacing=1.0,)
    np.save("tests/reference_data/solution_2d.npy", sol2.y)


if __name__ == "__main__":
    main()
