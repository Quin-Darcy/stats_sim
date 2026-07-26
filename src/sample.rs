use rand::Rng;
use crate::model::{Battery, Config};
use crate::score::Valuation;


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
        size: usize,
        battery: &Battery,
        config1: &Config,
        config2: &Config,
        rng: &mut impl Rng
    ) -> Sample {
        // Use the Battery's own method to get the valuation pairs
        let observations: Vec<(Valuation, Valuation)> = battery.get_valuation_pairs(
            size, 
            (config1, config2), 
            rng
        );

        Sample{ size, observations }
    }
}
