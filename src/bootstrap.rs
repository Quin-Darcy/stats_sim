use rand::Rng;

use crate::score::ScoringPolicy;
use crate::sample::Sample;

pub fn run(
    num_resamples: usize, 
    base_sample: &Sample, 
    policy: &ScoringPolicy, 
    rng: &mut impl Rng
) -> Vec<f64> {
    // Each round we will resample base_sample.len() many valuation pairs
    // from base_sample and compute the cumulative average to get the actual
    // mean of the discordant difference

    let mut index: usize;
    let mut ca: f64 = 0.0;
    let mut diff: f64;
    let sample_size: usize = base_sample.size;
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(num_resamples);

    for _ in 0..num_resamples {
        for j in 0..sample_size {
            index = rng.gen_range(0..sample_size);
            diff = policy.score(&base_sample.observations[index].0) - policy.score(&base_sample.observations[index].1);
            ca = ca + (diff - ca) / ((j + 1) as f64);
        }
        bootstrap_means.push(ca);
    }
    bootstrap_means
}
