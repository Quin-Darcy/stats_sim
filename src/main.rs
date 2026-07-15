use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod score;
pub mod model;
pub mod sample;
pub mod bootstrap;

use crate::score::ScoringPolicy;
use crate::model::{Battery, Config};
use crate::sample::Sample;
use crate::bootstrap::Bootstrap;


fn main() {
    let mut rng = StdRng::seed_from_u64(48);
    let battery_mean: [f64; 3] = [0.5, 0.3, 0.2];
    let concentration: f64 = 5.0;
    let num_questions: usize = 10;
    let battery = Battery::new(battery_mean, concentration, num_questions, &mut rng).unwrap();

    let effect1: [f64; 3] = [10.0, 1.0, 1.0];
    let config1 = Config::new(effect1).unwrap();

    let effect2: [f64; 3] = [2.6, 4.0, 0.44];
    let config2 = Config::new(effect2).unwrap();

    let circumspection: f64 = 0.3;
    let policy = ScoringPolicy::new(circumspection).unwrap();

    let sample = Sample::new(&battery, &config1, &config2, &mut rng);

    let num_resamples: usize = 100;
    let bootstrap = Bootstrap::new(num_resamples, &sample, &mut rng);
    let means: Vec<f64> = bootstrap.get_means(&policy);
    let agreement_ratios: Vec<f64> = bootstrap.get_agreement_ratios();

    println!("Bootstrapped Means:\n{:?}", means);
    println!("Agreement Ratios:\n{:?}", agreement_ratios);
}
