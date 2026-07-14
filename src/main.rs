pub mod score;
pub mod sample;

use crate::score::ScoringPolicy;
use crate::sample::{Battery, Config, BatteryEval};

fn main() {
    let battery_mean: [f64; 3] = [0.5, 0.3, 0.2];
    let concentration: f64 = 5.0;
    let num_questions: usize = 100;
    let battery = Battery::new(battery_mean, concentration, num_questions).unwrap();

    let effect: [f64; 3] = [10.0, 1.0, 1.0];
    let config = Config::new(effect).unwrap();

    let circumspection: f64 = 0.3;
    let policy = ScoringPolicy::new(circumspection).unwrap();

    let battery_eval = BatteryEval::new(&battery, &config, &policy);
}
