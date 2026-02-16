use std::fs;

const FILEPATH: &str = "./input.txt";

fn main() {
    // Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
    let docs = fs::read_to_string(FILEPATH).unwrap();

    println!("{}", docs);
}
