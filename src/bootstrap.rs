use rand::Rng;

use crate::score::ScoringPolicy;
use crate::sample::Sample;
use crate::utils;

// This function takes in a single Sample, the scoring policy, and the number of 
// resamples to perform, along with an RNG.
//
// The function is simple in that, for each resample iteration, we create an artificial
// Sample by randomly selecting elements (observations) from the base sample (with replacement,
// which means those same elements can be selected again). 
//
// For effeciency, we don't actually instantiate a full Sample from the selected observations,
// but rather we compute the discordant difference of each observation on the fly and use that
// value to update the cumulative average of them across the length of one sample.
//
// Thus, for each resample iteration, we obtain a single f64 which represents the average
// discordant differences across all the sample-size many observations we randomly selected
// (with replacement) from the base sample.
//
// Doing this num_replicates times gives us a vector of these averages, which (to the degree
// the base sample is representative of the actual population - something influenced by sample
// size) is an approximation of what we would get if we had performed this process on 
// real fresh samples (the sample distribution) instead of these artificial ones, which itself
// is an approximation of the population parameter.
//
// The population parameter being a theoretical object that can be thought of as the mean
// discordant difference across a Battery consisting of all possible questions that exist to be
// asked and evaluated throught the two Configs being compared. It represents the "true" comparison
// of the Configs, no longer contigent on the questions we happened to select.
pub fn run(
    num_replicates: usize, 
    base_sample: &Sample, 
    policy: &ScoringPolicy, 
    rng: &mut impl Rng
) -> Vec<f64> {
    let mut index: usize;
    let mut ca: f64 = 0.0;
    let mut diff: f64;
    let sample_size: usize = base_sample.size;
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(num_replicates);

    for _ in 0..num_replicates {
        for j in 0..sample_size {
            index = rng.gen_range(0..sample_size);
            diff = utils::discord_diff(
                policy,
                &base_sample.observations[index].0,
                &base_sample.observations[index].1
            );
            ca = ca + (diff - ca) / ((j + 1) as f64);
        }
        bootstrap_means.push(ca);
    }
    bootstrap_means
}

// This function uses the ci utility to create a confidence interval by running the bootstrap
// procedure and using the vector of returned re-sample means to then create the interval which
// contains (100 * confidence_level)% of the mass. 
//
// It is important to keep in mind that for any CI, it is a statement about the procedure
// which generated it. A 95% CI is only saying that if you repeated the procedure which generated
// the CI 100 times, then on average, 95 of those intervals will contain the population parameter
pub fn ci(
    confidence_level: f64,
    num_replicates: usize,
    base_sample: &Sample,
    policy: &ScoringPolicy,
    rng: &mut impl Rng
) -> [f64; 2] {
    let mut bootstrap_means: Vec<f64> = run(
        num_replicates,
        base_sample,
        policy,
        rng
    );

    utils::ci(&mut bootstrap_means, confidence_level)
}
