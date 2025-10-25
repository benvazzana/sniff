from sniff.graph_theory import Graph
from sniff.noise import RMT

import typer

app = typer.Typer()


@app.command()
def protocol(agents: int, link: float):
    graph = Graph(agents)
    A = graph.erdos_renyi_adj(link)
    rmo = RMT(A)
    L_noisy = rmo.noise_GOE()
    print(L_noisy)


if __name__ == "__main__":
    app()
