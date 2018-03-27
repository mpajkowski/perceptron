mod neuron;

fn main() {
    println!("Hello, world!");
    let v = vec![1.0, 0.0, 1.0, 1.0];
    let n = neuron::Neuron::new(1, v.len(), false);
    //n.set_input(&v);
}
