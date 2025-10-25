import numpy as np
import matplotlib.pyplot as plt


def erdos_renyi_adj(n, p) -> np.ndarray:
    """
    Generates an adjacency matrix from an Erdos-Renyi
    graph G of size n with edge probability p
    """
    random_matrix = np.random.random((n, n))
    A = (random_matrix < p).astype(int)
    A = np.triu(A, 1)
    A = A + A.T

    return A


def apply_RMT_noise(A) -> np.ndarray:
    pass


def make_laplacian(A) -> np.ndarray:
    """
    Makes a graph Laplacian matrix L from an adjacency matrix A
    """
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L


# why does it converge faster with more agents?
# look at the eigenvalues
n = 10
p = 0.3
A = erdos_renyi_adj(n, p)
print(A)
L = make_laplacian(A)
print(L)

# define starting states for our agents
x0 = np.random.random(n)
dt = 0.01
T = 5.0
steps = int(T/dt)

x = x0.copy()
logs = [x.copy()]
for _ in range(steps):
    xdot = -L @ x
    x = x + dt*xdot
    logs.append(x.copy())

logs = np.array(logs)
time = np.linspace(0, T, steps+1)

plt.plot(time, logs)
plt.show()
