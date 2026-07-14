use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod score;
pub mod model;
pub mod sample;

use crate::score::ScoringPolicy;
use crate::model::{Battery, Config};
use crate::sample::Sample;

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
    let diffs: Vec<f64> = sample.get_differences(&policy);
    let mean: f64 = sample.get_mean(&policy);

    println!("{:?}", sample);
    println!("{:?}", diffs);
    println!("{:?}", mean);
}
