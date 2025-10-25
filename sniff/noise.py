import numpy as np
from sniff.graph_theory import Graph


class RMT:
    def __init__(self, A):
        """
        Iniitalize Random Matrix Theory operations with
        the size of the given matrix and the matrix Laplacian
        """
        self.n = A.shape[0]
        self.A = A
        self.L = Graph.make_laplacian(A)

    def noise_GOE(self, noise_strength=0.1) -> np.ndarray:
        """
        Generates noise for an adjacency matrix in the form of a
        Gaussian Orthogonal Ensemble (GOE)

        Returns a modified matrix Laplacian mapped with GOE noise
        """

        W = np.random.randn(self.n, self.n)
        W = (W + W.T)/2
        np.fill_diagonal(W, W.diagonal()*np.sqrt(2))
        L_noisy = self.L + noise_strength*W

        return L_noisy
