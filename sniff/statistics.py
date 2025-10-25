import matplotlib.pyplot as plt


class Plot:
    def __init__(self, solution):
        """
        Solution is the result of scipy integration
        """
        self.sol = solution

    def plot_convergence(self):
        plt.plot(self.sol.t, self.sol.y)
