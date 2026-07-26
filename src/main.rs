use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod score;
pub mod model;
pub mod sample;
pub mod bootstrap;
pub mod utils;

use crate::score::ScoringPolicy;
use crate::sample::Sample;


fn main() {
    // Initialize the RNG
    let seed: u64 = 48;
    let mut rng = StdRng::seed_from_u64(seed);

    // Create the base sample that we will bootstrap off of
    let sample_size: usize = 256;
    let base_sample = Sample::random(sample_size, &mut rng);

    // Set the policy we use to scrore
    let circumspection: f64 = 0.5;
    let policy = ScoringPolicy::new(circumspection).unwrap();

    // Run the bootstrap
    let num_resamples: usize = 100;
    let mut bootstrap_means: Vec<f64> = bootstrap::run(
        num_resamples,
        &base_sample,
        &policy,
        &mut rng
    );

    // Compute the CI
    let gamma: f64 = 0.95;
    let ci: [f64; 2] = utils::ci(&mut bootstrap_means, gamma);
    println!("{:.1} % Bootstrap CI: [{:.2}, {:.2}]", gamma * 100.0, ci[0], ci[1]);
}
