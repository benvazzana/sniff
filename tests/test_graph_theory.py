import numpy as np
from sniff.graph_theory import Graph


def test_erdos_renyi_adj_shape():
    g = Graph(5)
    A = g.erdos_renyi_adj(0.5)
    assert A.shape == (5, 5)


def test_laplacian_properties():
    g = Graph(4)
    A = g.erdos_renyi_adj(1.0)
    L = Graph.make_laplacian(A)
    np.testing.assert_allclose(L.sum(axis=1), np.zeros(4), atol=1e-8)
    assert np.allclose(L, L.T), "Laplacian should be symmetric"
