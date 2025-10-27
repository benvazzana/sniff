import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, solution):
        """
        The result returned by scipy.integrate.solve_ivp
        """
        self.sol = solution

    def plot_1D_convergence(self, save=None):
        """
        Plots state convergence of all agents in 1D.
        """
        plt.figure(figsize=(7, 4))
        for i, y in enumerate(self.sol.y):
            plt.plot(self.sol.t, y, lw=1.8, label=f"agent {i+1}")

        plt.title("1D Consensus Convergence", fontsize=13, pad=10)
        plt.xlabel("Time", fontsize=11)
        plt.ylabel("State Value", fontsize=11)
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_2D_paths(self, x0, save=None):
        """
        Plots the 2D trajectories of agents from their initial positions.
        """
        trajectories = self.sol.y.T.reshape(-1, len(x0) // 2, 2)
        x_coords = trajectories[:, :, 0]
        y_coords = trajectories[:, :, 1]

        n = x_coords.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n))

        plt.figure(figsize=(6, 6))
        for i, color in enumerate(colors):
            plt.plot(
                x_coords[:, i],
                y_coords[:, i],
                color=color,
                lw=2,
                alpha=0.9,
                label=f"agent {i+1}"
            )
            plt.scatter(
                x_coords[0, i],
                y_coords[0, i],
                color=color,
                marker="o",
                s=40,
                edgecolor="k",
                zorder=3,
            )

        plt.title("2D Consensus Trajectories", fontsize=13, pad=10)
        plt.xlabel("x", fontsize=11)
        plt.ylabel("y", fontsize=11)
        plt.axis("equal")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(frameon=False, fontsize=9, loc="best")
        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        else:
            plt.show()
