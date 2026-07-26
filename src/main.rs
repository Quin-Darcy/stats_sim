use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod score;
pub mod model;
pub mod sample;
pub mod world;
pub mod bootstrap;
pub mod utils;

use crate::score::ScoringPolicy;
use crate::world::World;
use crate::sample::Sample;


fn get_coverage(
    gamma: f64,
    num_resamples: usize,
    base_samples: &[Sample],
    policy: &ScoringPolicy,
    parameter: f64,
    rng: &mut impl Rng
) -> f64 {
    let mut ci: [f64; 2];
    let simulations: usize = base_samples.len();
    let mut coverage: f64 = 0.0;
    for i in 0..simulations {
        ci = bootstrap::ci(
            gamma,
            num_resamples,
            &base_samples[i],
            policy,
            rng
        );

        if ci[0] <= parameter && parameter <= ci[1] {
            coverage += 1.0;
        }
    }

    coverage / (simulations as f64)
}


fn main() {
    // Initialize the RNG
    let seed: u64 = 48;
    let mut rng = StdRng::seed_from_u64(seed);

    // Set the policy we use to scrore
    let circumspection: f64 = 0.5;
    let policy = ScoringPolicy::new(circumspection).unwrap();

    // Create a world from which we will sample
    let world = World::random(&mut rng);

    // Generate set of samples from this world
    let num_samples: usize = 500;
    let sample_size: usize = 132;
    let samples = world.sample(num_samples, sample_size, &policy, &mut rng);

    ////////////////////////////////////////////////////////////
    // Below this point is simply statistics exploring difference
    // between sample and bootstrap distributions
    ////////////////////////////////////////////////////////////

    // Here we use our set of fresh samples to compute an estimate
    // of the population parameter. Each sample carries its own mean
    // already and so we will simply get the average of the sample
    // means
    let sample_means: Vec<f64> = samples.iter().map(|a| a.mean).collect();
    let parameter: f64 = utils::mean(&sample_means);

    // Validate bootstrap CI coverage claim
    let gamma: f64 = 0.95;
    let num_resamples: usize = 10000;
    let coverage: f64 = get_coverage(
        gamma,
        num_resamples,
        &samples,
        &policy,
        parameter,
        &mut rng,
    );

    println!("Sample Size: {:?}", sample_size);
    println!("Sample Distribution Size: {:?}\n", num_samples);
    println!("CI Gamma: {:?}", gamma);
    println!("Bootstrap Resamples: {:?}", num_resamples);
    println!("Bootstrap Simulations: {:?}", num_samples);
    println!("Bootstrap Coverage: {:.2}%", coverage * 100.0);
}
