use rand::SeedableRng; // Provides a seedable rng
use rand::prelude::SliceRandom; // Provides the shuffle trait
use rand::rngs::StdRng;
use std::fs; // Provides a base rng

const FILEPATH: &str = "./input.txt";
// A seed of your choice
const SEED: u64 = 42;

fn main() {
    // Let there be an RNG
    let mut rng: StdRng = SeedableRng::seed_from_u64(SEED);

    // Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
    let all_docs = fs::read_to_string(FILEPATH).unwrap();
    let mut docs: Vec<&str> = all_docs.split('\n').collect();
    docs.shuffle(&mut rng);
}
