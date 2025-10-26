use sniff_rust::graph::erdos_renyi_adj;

fn main() {
    let n = 10;
    let p = 0.3;

    let adj = erdos_renyi_adj(n, p);
    println!("{:?}", adj);
}
