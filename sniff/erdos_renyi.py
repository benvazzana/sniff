import numpy as np


def erdos_renyi_adj(n, p):
    """
    Generates an adjacency matrix from an Erdos-Renyi
    graph of size n with edge probability p
    """
    random_matrix = np.random.random((n, n))
    A = (random_matrix < p).astype(int)
    A = np.triu(A, 1)
    A = A + A.T

    return A


n = 10
p = 0.3
A = erdos_renyi_adj(n, p)
print(A)
