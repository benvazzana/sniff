from sniff.graph_theory import Graph

import typer

app = typer.Typer()


@app.command()
def protocol(agents: int, link: float):
    graph = Graph(agents)
    adjacency_matrix = graph.erdos_renyi_adj(link)
    print(adjacency_matrix)


if __name__ == "__main__":
    app()
