use sniff_rust::graph::erdos_renyi_adj;
use sniff_rust::graph::make_laplacian;

fn main() {
    let n = 10;
    let p = 0.3;

    let adj = erdos_renyi_adj(n, p);
    print!("{:?}", adj);
    let l = make_laplacian(adj);
    print!("{:?}", l);
}
