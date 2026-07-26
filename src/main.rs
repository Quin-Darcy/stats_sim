use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod score;
pub mod model;
pub mod sample;
pub mod bootstrap;
pub mod utils;

use crate::score::ScoringPolicy;
use crate::model::{Battery, Config};
use crate::sample::Sample;


fn main() {
    // Construct the Battery with some random values
    let battery_mean: [f64; 3] = [0.5, 0.3, 0.2];
    let concentration: f64 = 5.0;
    let battery = Battery::new(battery_mean, concentration).unwrap();

    // Create the configs
    let effect1: [f64; 3] = [10.0, 1.0, 1.0];
    let effect2: [f64; 3] = [2.6, 4.0, 0.44];
    let config1 = Config::new(effect1).unwrap();
    let config2 = Config::new(effect2).unwrap();

    // Initialize the RNG
    let mut rng = StdRng::seed_from_u64(48);

    // Create the base sample that we will bootstrap off of
    let sample_size: usize = 132;
    let base_sample = Sample::new(
        sample_size,
        &battery,
        &config1,
        &config2,
        &mut rng
    );

    // Set the policy we use to scrore
    let circumspection: f64 = 0.5;
    let policy = ScoringPolicy::new(circumspection).unwrap();

    // Run the bootstrap
    let num_resamples: usize = 10000;
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
