use rand::Rng;
use rand::distributions::Distribution;
use statrs::distribution::Multinomial;
use nalgebra::DVector;

use crate::new_sample::NewSample;
use crate::new_score::NewScoringPolicy;
use crate::utils;

/*
pub fn run(
    num_replicates: usize,
    base_sample: &Sample,
    policy: &ScoringPolicy,
    rng: &mut impl Rng,
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
                &base_sample.observations[index].1,
            );
            ca = ca + (diff - ca) / ((j + 1) as f64);
        }
        bootstrap_means.push(ca);
    }
    bootstrap_means
}

pub fn ci(
    confidence_level: f64,
    num_replicates: usize,
    base_sample: &Sample,
    policy: &ScoringPolicy,
    rng: &mut impl Rng,
) -> [f64; 2] {
    let mut bootstrap_means: Vec<f64> = run(num_replicates, base_sample, policy, rng);

    utils::ci(&mut bootstrap_means, confidence_level)
}
*/

pub fn run(
    num_replicates: usize,
    base_sample: &NewSample,
    policy: &NewScoringPolicy,
    rng: &mut impl Rng,
) -> Vec<f64> {
    // The base sample contains the ratios of each cell and these ratios function
    // as a probability vector describing a multinomial distribution
    
    let mn = Multinomial::new(
        base_sample.observations.to_vec(),
        base_sample.size as u64
    ).unwrap();
    
    let mut replicate: DVector<u64>;
    let mut replicate_scores: Vec<f64> = Vec::with_capacity(base_sample.observations.len());
    let mut replicate_score_means: Vec<f64> = Vec::with_capacity(num_replicates);

    for _ in 0..num_replicates {
        replicate_scores.clear();
        replicate = mn.sample(rng);
        for i in 0..replicate.len() {
           replicate_scores.push((replicate[i] as f64 / base_sample.size as f64) * policy.outcome_scores[i])
        }
        replicate_score_means.push(replicate_scores.iter().sum());
    }
    replicate_score_means
}

pub fn ci(
    confidence_level: f64,
    num_replicates: usize,
    base_sample: &NewSample,
    policy: &NewScoringPolicy,
    rng: &mut impl Rng,
) -> [f64; 2] {
    let mut bootstrap_means: Vec<f64> = run(num_replicates, base_sample, policy, rng);
    utils::ci(&mut bootstrap_means, confidence_level)
}
