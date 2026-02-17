use rand::SeedableRng; // Provides a seedable rng
use rand::prelude::SliceRandom; // Provides the shuffle trait
use rand::rngs::StdRng; // Provides a base rng
use std::fs; // Provides file handling

const FILEPATH: &str = "./input.txt";
// Let there be order among chaos
const SEED: u64 = 42;

// Let there be Autograd, to recursively apply the chain rule through a computation graph

#[derive(PartialEq, Clone)]
struct Value {
    pub data: f64,              // scalar value of this node calculated during forward pass
    pub grad: f64, // derivative of the loss w.r.t. this node, calculated in backward pass
    pub _children: Vec<Value>, // children of this node in the computation graph (TODO: Follow up if this recursion is correct)
    pub _local_grads: Vec<f64>, // local derivative of this node w.r.t. its children
}

impl Value {
    fn build_topo(&self, v: &Value, mut visited: &mut Vec<Value>, mut topo: &mut Vec<Value>) {
        if !visited.contains(&v) {
            visited.push(v.clone());
            for child in &self._children {
                self.build_topo(&child, &mut visited, &mut topo)
            }
            topo.push(v.clone())
        }
    }

    pub fn backward(&mut self) {
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: Vec<Value> = Vec::new();
        self.build_topo(&self, &mut visited, &mut topo);
        self.grad = 1.0 as f64;
        for v in topo.iter_mut().rev() {
            for (child, local_grad) in v._children.iter_mut().zip(v._local_grads.iter_mut()) {
                child.grad += local_grad.clone() * v.grad;
            }
        }
    }
}

fn main() {
    // Let there be an RNG
    let mut rng: StdRng = SeedableRng::seed_from_u64(SEED);

    // Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
    let all_docs = fs::read_to_string(FILEPATH).unwrap();
    let mut docs: Vec<&str> = all_docs.split('\n').collect();
    docs.shuffle(&mut rng);
    println!("num docs: {}", docs.len());

    // Initialize parameters to store the knowledge of the model
    let n_embd: u64 = 16; // embedding dimension
    let n_head: u64 = 4; // number of attention heads
    let n_layer: u64 = 1; // number of layers
    let block_size: u64 = 16; // maximum sequence length
    let head_dim: u64 = n_embd / n_head; // dimension of each head, rust int division is floor division by default
}
