pub mod score;
pub mod sample;

use crate::score::ScoringPolicy;
use crate::sample::{Battery, Config, BatteryEval};

fn main() {
    let s: f64 = 12.0;
    let v: [f64; 3] = [0.5, 0.3, 0.2];
    let b: Battery = Battery::new(v, s, 100).unwrap();

    let effect: [f64; 3] = [10.0, 1.0, 1.0];
    let c: Config = Config::new(effect).unwrap();

    let circ: f64 = 0.3;
    let policy = ScoringPolicy::new(circ).unwrap();

    let battery_eval = BatteryEval::new(&b, &c, &policy);

    println!("battery eval: {:?}, {:?}", battery_eval.valuations, battery_eval.scores);
}
