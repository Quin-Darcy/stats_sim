use rand::Rng;

use crate::score::ScoringPolicy;
use crate::model::{Battery, Config};
use crate::sample::Sample;

// A World is a particular Battery and two Configs
pub struct World {
    pub battery: Battery,
    pub incumbent: Config,
    pub candidate: Config
}

impl World {
    pub fn random(rng: &mut impl Rng) -> Self {
        let battery = Battery::random(rng);
        let incumbent = Config::random(rng);
        let candidate = Config::random(rng);

        World {
            battery,
            incumbent,
            candidate
        }
    }

    pub fn sample(
        &self,
        num_samples: usize,
        sample_size: usize,
        policy: &ScoringPolicy,
        rng: &mut impl Rng
    ) -> Vec<Sample> {
        let mut samples: Vec<Sample> = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            samples.push(Sample::new(sample_size, self, policy, rng))
        }
        samples
    }
}
