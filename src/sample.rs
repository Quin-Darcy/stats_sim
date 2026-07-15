use rand::Rng;
use crate::model::{Battery, Config, eval_battery};
use crate::score::{Valuation, ScoringPolicy};


#[derive(Debug)]
pub struct Sample {
    // A sample consists of the paired evaluations of a
    // fixed battery under the effect of two different
    // configs.
    pub size: usize,
    pub observations: Vec<(Valuation, Valuation)>,
}

impl Sample {
    pub fn new(
        battery: &Battery,
        config1: &Config,
        config2: &Config,
        rng: &mut impl Rng
    ) -> Sample {
        let size: usize = battery.questions.len();
        let eval1: Vec<Valuation> = eval_battery(battery, config1, rng);
        let eval2: Vec<Valuation> = eval_battery(battery, config2, rng);
        let observations: Vec<(Valuation, Valuation)> = std::iter::zip(
            eval1,
            eval2
        ).collect();

        Sample{ size, observations }
    }

    pub fn from(size: usize, observations: Vec<(Valuation, Valuation)>) -> Sample {
        Sample { size, observations }
    }

    pub fn get_differences(&self, policy: &ScoringPolicy) -> Vec<f64> {
        let mut diffs: Vec<f64> = Vec::with_capacity(self.size);
        for i in 0..self.size {
            diffs.push(
                policy.score(&self.observations[i].0) - policy.score(&self.observations[i].1)
            );
        }
        diffs
    }
    
    pub fn get_mean(&self, policy: &ScoringPolicy) -> f64 {
        let diffs: Vec<f64> = self.get_differences(policy);
        diffs.iter().sum::<f64>() / (self.size as f64)
    }

    pub fn get_agreement_ratio(&self) -> f64 {
        let mut count = 0.0;
        for i in 0..self.size {
            if self.observations[i].0 == self.observations[i].1 {
                count += 1.0;
            }
        }
        count / (self.size as f64)
    }
}
