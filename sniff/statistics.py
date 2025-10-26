import matplotlib.pyplot as plt


class Plot:
    def __init__(self, solution):
        """
        Solution is the result of scipy integration
        """
        self.sol = solution

    def plot_1D_convergence(self):
        plt.plot(self.sol.t, self.sol.y.T)
        plt.show()

    def plot_2D_paths(self):
        trajectories = self.sol.y.T.reshape(-1, len(self.sol) // 2, 2)
        x_coords = trajectories[:, :, 0]
        y_coords = trajectories[:, :, 1]

        n = x_coords.shape[1]
        for i in range(n):
            plt.plot(x_coords[:, i], y_coords[:, i])

        plt.show()
