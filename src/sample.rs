use rand::Rng;

use crate::score::ScoringPolicy;
use crate::model::Battery;
use crate::score::Valuation;
use crate::world::World;
use crate::utils;


#[derive(Debug, Clone)]
pub struct Sample {
    // A sample consists of the paired evaluations of a
    // fixed battery under the effect of two different
    // configs.
    pub size: usize,
    pub mean: f64,
    pub observations: Vec<(Valuation, Valuation)>,
}

impl Sample {
    pub fn new(
        size: usize,
        world: &World,
        policy: &ScoringPolicy,
        rng: &mut impl Rng
    ) -> Sample {
        // Use the Battery's own method to get the valuation pairs
        let battery: &Battery = &world.battery;
        let observations: Vec<(Valuation, Valuation)> = battery.get_valuation_pairs(
            size, 
            (&world.incumbent, &world.candidate),
            rng
        );

        // Compute the mean of the sample as cumulative average of discordant differences
        let mut ca: f64 = 0.0;
        let mut diff: f64;
        for i in 0..size {
            diff = utils::discord_diff(
                policy, 
                &observations[i].0, 
                &observations[i].1,
            );
            ca = ca + (diff - ca) / ((i + 1) as f64); 
        }
        Sample{ size, mean: ca, observations }
    }
}
