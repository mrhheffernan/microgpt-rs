use rand::SeedableRng; // Provides a seedable rng
use rand::prelude::SliceRandom; // Provides the shuffle trait
use rand::rngs::StdRng; // Provides a base rng
use std::fs; // Provides file handling

const FILEPATH: &str = "./input.txt";
// Let there be order among chaos
const SEED: u64 = 42;

fn main() {
    // Let there be an RNG
    let mut rng: StdRng = SeedableRng::seed_from_u64(SEED);

    // Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
    let all_docs = fs::read_to_string(FILEPATH).unwrap();
    let mut docs: Vec<&str> = all_docs.split('\n').collect();
    docs.shuffle(&mut rng);
    println!("{}", docs.len());
}
