from sniff.graph_theory import Graph
from sniff.simulation import Simulate
from sniff.statistics import Plot

import typer
import numpy as np

app = typer.Typer()


@app.command()
def protocol(agents: int, link: float):
    graph = Graph(agents)
    A = graph.erdos_renyi_adj(link)

    x0 = np.random.random(agents)
    simulation = Simulate(A, 8, 0.01, x0)
    logs = simulation.solve_consensus_dynamics_w_GOE_noise()

    plotter = Plot(logs)
    plotter.plot_convergence()


if __name__ == "__main__":
    app()
