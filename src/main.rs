use rand::RngExt; // Provides the random_range trait
use rand::SeedableRng; // Provides a seedable rng
use rand::prelude::SliceRandom; // Provides the shuffle trait
use rand::rngs::StdRng; // Provides a base rng
use std::collections::HashMap; // Provides Hashmaps
use std::fs; // Provides file handling
use std::ops::{Add, Mul}; // Allows us to add custom types
const FILEPATH: &str = "./input.txt";
// Let there be order among chaos
const SEED: u64 = 42;

// Let there be Autograd, to recursively apply the chain rule through a computation graph
#[derive(PartialEq, Clone, Debug)]
struct Value {
    pub data: f64,              // scalar value of this node calculated during forward pass
    pub grad: f64, // derivative of the loss w.r.t. this node, calculated in backward pass
    pub _children: Vec<Value>, // children of this node in the computation graph (TODO: Follow up if this recursion is correct)
    pub _local_grads: Vec<f64>, // local derivative of this node w.r.t. its children
}

impl Add for Value {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Value {
            data: self.data + other.data,
            grad: 0 as f64,
            _children: vec![self.clone(), other.clone()],
            _local_grads: vec![1 as f64, 1 as f64],
        }
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Value {
            data: self.data * other.data,
            grad: 0 as f64,
            _children: vec![self.clone(), other.clone()],
            _local_grads: vec![self.data, other.data],
        }
    }
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

/// Let there be a handy matrix constructor
fn matrix(nout: usize, nin: usize, rng: &mut StdRng) -> Vec<Vec<Value>> {
    let mut m = Vec::new();
    for _ in 0..nout {
        let mut inner_row: Vec<Value> = Vec::new();
        for _j in 0..nin {
            inner_row.push(Value {
                data: rng.random_range(-0.16..0.16),
                grad: 0.0 as f64,
                _children: Vec::new(),
                _local_grads: Vec::new(),
            });
        }
        m.push(inner_row)
    }
    m
}

fn main() {
    // Let there be an RNG
    let mut rng: StdRng = SeedableRng::seed_from_u64(SEED);

    // Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
    let all_docs = fs::read_to_string(FILEPATH).unwrap();
    let mut docs: Vec<&str> = all_docs.split('\n').collect();
    docs.shuffle(&mut rng);
    println!("num docs: {}", docs.len());

    // Let there be a Tokenizer to translate strings to discrete symbols and back
    let mut uchars = Vec::from_iter(docs.join("").to_string().to_lowercase().chars());
    uchars.sort();
    uchars.dedup(); // sort must come before dedup as dedup operates on consecutive elements
    let BOS: usize = uchars.len(); // token id for the special Beginning of Sequence (BOS) token
    let vocab_size: usize = uchars.len() + 1; // total number of unique tokens, +1 is for BOS
    println!("vocab size: {}", vocab_size);

    // Initialize parameters to store the knowledge of the model
    let n_embd: usize = 16; // embedding dimension
    let n_head: usize = 4; // number of attention heads
    let n_layer: usize = 1; // number of layers
    let block_size: usize = 16; // maximum sequence length
    let head_dim: usize = n_embd / n_head; // dimension of each head, rust int division is floor division by default

    let mut state_dict: HashMap<String, Vec<Vec<Value>>> = HashMap::new();

    state_dict.insert(String::from("wte"), matrix(vocab_size, n_embd, &mut rng));
    state_dict.insert(String::from("wpe"), matrix(block_size, n_embd, &mut rng));
    state_dict.insert(
        String::from("lm_head"),
        matrix(vocab_size, n_embd, &mut rng),
    );

    for i in 0..n_layer {
        state_dict.insert(
            format!("layer{i}.attn_wq"),
            matrix(n_embd, n_embd, &mut rng),
        );
        state_dict.insert(
            format!("layer{i}.attn_wk"),
            matrix(n_embd, n_embd, &mut rng),
        );
        state_dict.insert(
            format!("layer{i}.attn_wv"),
            matrix(n_embd, n_embd, &mut rng),
        );
        state_dict.insert(
            format!("layer{i}.attn_wo"),
            matrix(n_embd, n_embd, &mut rng),
        );
        state_dict.insert(
            format!("layer{i}.mlp_fc1"),
            matrix(4 * n_embd, n_embd, &mut rng),
        );
        state_dict.insert(
            format!("layer{i}.mlp_fc2"),
            matrix(n_embd, 4 * n_embd, &mut rng),
        );
    }

    let mut params: Vec<Value> = Vec::new();
    for key in state_dict.keys() {
        let m = state_dict.get(key).unwrap();
        let dim_1 = m.len();
        let dim_2 = m[0].len();
        for idx in 0..dim_1 {
            for jdx in 0..dim_2 {
                params.push(m[idx][jdx].clone())
            }
        }
    }
    println!("num params: {}", params.len());

    // Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
    // Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

    fn linear(x: Vec<Value>, w: Vec<Vec<Value>>) -> Vec<Value> {
        //return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
        let mut l: Vec<Value> = Vec::new();
        for wo in w {
            let inner_elements = wo.into_iter().zip(x.clone()).map(|(wi, xi)| wi * xi);

            let mut inner_sum = Value {
                data: 0 as f64,
                grad: 0 as f64,
                _children: Vec::new(),
                _local_grads: Vec::new(),
            };
            for elem in inner_elements {
                inner_sum = inner_sum + elem
            }
            l.push(inner_sum);
        }
        l
    }
}
