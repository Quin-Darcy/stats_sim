use crate::model::{Battery, BatteryEval, Config};
use crate::score::{Valuation, ScoringPolicy};

#[derive(Debug)]
pub struct Observation {
    // An observation is the type associated with 
    // the components of the sample. For a given Battery,
    // there is one observation for each of it's questions.
    //
    // The observation consists of the answer (Valuation) 
    // given by both configs along with the difference of
    // their respective scores
    result: (Valuation, Valuation),
    score: f64
}

#[derive(Debug)]
pub struct Sample {
    // A sample consists of the paired evaluations of a
    // fixed battery under the effect of two different
    // configs.
    size: usize,
    observations: Vec<Observation>,
}

impl Sample {
    pub fn new(
        battery: &Battery,
        config1: &Config,
        config2: &Config,
        policy: &ScoringPolicy
    ) -> Sample {
        let size: usize = battery.questions.len();
        let eval1 = BatteryEval::new(battery, config1, policy);
        let eval2 = BatteryEval::new(battery, config2, policy);

        let mut observations: Vec<Observation> = Vec::with_capacity(size);
        for i in 0..size {
            observations.push(
                Observation {
                    result: (
                        eval1.valuations[i].clone(), 
                        eval2.valuations[i].clone()
                    ),
                    score: eval1.scores[i] - eval2.scores[i]
                }
            );
        }

        Sample{ size, observations }
    }
}
