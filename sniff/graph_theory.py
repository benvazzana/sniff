import numpy as np


class Graph:
    def __init__(self, n):
        """
        Initialize the graph with the number of agents
        """
        self.n = n

    def erdos_renyi_adj(self, p) -> np.ndarray:
        """
        Generates an adjacency matrix from an Erdos-Renyi
        graph G of size n with edge probability p
        """
        random_matrix = np.random.random((self.n, self.n))
        A = (random_matrix < p).astype(int)
        A = np.triu(A, 1)
        A = A + A.T

        return A

    @staticmethod
    def make_laplacian(A) -> np.ndarray:
        """
        Makes a graph Laplacian matrix L from an adjacency matrix A
        """
        D = np.diag(A.sum(axis=1))
        L = D - A
        return L
