use rand::Rng;

use crate::score::{Valuation, ScoringPolicy};
use crate::sample::Sample;

pub struct Bootstrap {
    resamples: Vec<Vec<(Valuation, Valuation)>>
}

impl Bootstrap {
    pub fn new(
        num_resamples: usize,
        base_sample: &Sample,
        rng: &mut impl Rng
    ) -> Bootstrap {
        let sample_size: usize = base_sample.size;
        let mut resamples: Vec<Vec<(Valuation, Valuation)>> = Vec::with_capacity(num_resamples);
        
        let mut index: usize;
        for _ in 0..num_resamples {
            // Inner loop resamples with replacement
            let mut resample: Vec<(Valuation, Valuation)> = Vec::with_capacity(sample_size);
            for _ in 0..sample_size {
                index = rng.gen_range(0..sample_size);
                resample.push(
                    base_sample.observations[index].clone()
                );
            }
            resamples.push(resample.clone());
        }
        Bootstrap {
            resamples
        }
    }

    pub fn get_means(&self, policy: &ScoringPolicy) -> Vec<f64> {
        let num_resamples: usize = self.resamples.len();
        let mut means: Vec<f64> = Vec::with_capacity(num_resamples);

        for i in 0..num_resamples {
            let temp_sample = Sample::from(
                self.resamples[i].len(),
                self.resamples[i].clone()
            );
            means.push(temp_sample.get_mean(policy));
        }
        means
    }

    pub fn get_agreement_ratios(&self) -> Vec<f64> {
        let num_resamples: usize = self.resamples.len();
        let mut ratios: Vec<f64> = Vec::with_capacity(num_resamples);

        for i in 0..num_resamples {
            let temp_sample = Sample::from(
                self.resamples[i].len(),
                self.resamples[i].clone()
            );
            ratios.push(temp_sample.get_agreement_ratio());
        }
        ratios
    }
}
