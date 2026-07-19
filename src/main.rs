use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod score;
pub mod model;
//pub mod sample;
//pub mod bootstrap;
pub mod utils;

use crate::score::ScoringPolicy;
use crate::model::{Battery, Config};
//use crate::sample::Sample;
//use crate::bootstrap::Bootstrap;
use crate::utils::{mean, sd, ci};


fn main() {
    let mut rng = StdRng::seed_from_u64(48);
    let battery_mean: [f64; 3] = [0.5, 0.3, 0.2];
    let concentration: f64 = 5.0;
    let num_questions: usize = 132;
    let battery = Battery::new(battery_mean, concentration).unwrap();

    let effect1: [f64; 3] = [10.0, 1.0, 1.0];
    let config1 = Config::new(effect1).unwrap();

    let effect2: [f64; 3] = [2.6, 4.0, 0.44];
    let config2 = Config::new(effect2).unwrap();

    let v = battery.get_valuation_pairs(num_questions, (&config1, &config2), &mut rng);

    /*
    let circumspection: f64 = 0.3;
    let policy = ScoringPolicy::new(circumspection).unwrap();

    // This is temporary and just to confirm coverage claim
    let gamma: f64 = 0.95;
    let mut temp_sample;
    let num_samples: usize = 10000;
    let mut sample_means: Vec<f64> = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        temp_sample = Sample::new(&battery, &config1, &config2, &mut rng);
        sample_means.push(
            temp_sample.get_mean(&policy)
        );
    }

    println!(
        "Sample Stats:\nMean: {:?} - SD: {:?} - {:?}% {:?}", 
        mean(&sample_means), 
        sd(&sample_means),
        100.0 * gamma,
        ci(&mut sample_means, gamma)
    );

    let sample_mean: f64 = mean(&sample_means);

    let mut sample = Sample::new(&battery, &config1, &config2, &mut rng);

    let num_resamples: usize = num_samples;
    let mut inclusion_count: usize = 0;
    let mut bootstrap = Bootstrap::new(num_resamples, &sample, &mut rng);
    let mut bootstrap_means: Vec<f64>;
    let mut tmp_ci: [f64; 2];
    
    let num_sims: usize = 10000;
    for _ in 0..num_sims {
        bootstrap_means = bootstrap.get_means(&policy);
        tmp_ci = ci(&mut bootstrap_means, gamma);
        if tmp_ci[0] <= sample_mean && sample_mean <= tmp_ci[1] {
            inclusion_count += 1;
        }
        sample = Sample::new(&battery, &config1, &config2, &mut rng);
        bootstrap = Bootstrap::new(num_resamples, &sample, &mut rng);
    }

    println!("Coverage: {:?} %", 100.0 * (inclusion_count as f64) / (num_sims as f64));
    ////////////////////////////////////////////////////////////////
    */

}
