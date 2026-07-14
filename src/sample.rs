use rand::thread_rng;
use rand::distributions::Distribution;
use statrs::distribution::{Dirichlet, Categorical};

use crate::score::{Valuation, ScoringPolicy};


#[derive(Debug)]
pub struct Battery {
    // for a given set of questions, we let each question have an 
    // intrinsic "difficulty" which is described by the Categorical
    // distribution over the Valuation space {CORRECT, PARTIAL, INCORRECT}
    //
    // Imagine asking one question to many people and then create a
    // histogram plotting the outcomes of how many people got it right
    // (CORRECT), wrong (INCORRECT), or were unsure (PARTIAL). The
    // distribution you get is the categorical describing the "difficulty"
    // of the question.
    //
    // In this model, a question is determined by this difficulty profile.
    // Instead of actual questions, we use the probability vector that
    // defines the categorical as a stand-in for the question itself.
    //
    // A Battery consists of a vector of these probability vectors.
    //
    // Sampling from the distribution defined by one of these probability 
    // vectors gives you one Valuation back.
    //
    // We would say "most of the questions are difficult" when the categorical 
    // distribution for each question has high probability for the 
    // INCORRECT valuation. This means most of the questions most of the time
    // will be answered incorrectly. This is the "baseline" difficulty
    // *before* a config is introduced which may shift these distributions
    // in some way.
    questions: Vec<[f64; 3]>
}

impl Battery {
    pub fn new(
        battery_mean: [f64; 3],
        concentration: f64,
        num_questions: usize
    ) -> Option<Battery> {
        // A Battery is defined by the questions it contains. In this model,
        // there are no actual questions but Categoricals which describe the
        // way the way in which the questions are answered on average (e.g.,
        // 48% CORRECT, 12% PARTIAL, 20% INCORRECT -> [0.48, 0.12, 0.20]).
        //
        // This means each question is really just a probability vector and the
        // the way in which we obtain these vectors is through sampling from
        // a special distribution whose samples are probability vectors.
        //
        // The distribution the Battery's questions are sampled from is called
        // the Dirichlet distribution of order 3..
        //
        // The battery_mean is the point in the 2-simplex that the sampled questions
        // average out to.
        //
        // The concentration controls how dense or diffuse those questions are around 
        // the battery_mean. It does so by getting multiplied through the mean.
        //
        // Once the concentration has been applied to the mean, if any of the components
        // fall below 1, the mass diverges out from the mean toward the vertex of
        // the simplex.

        let mut sum: f64 = 0.0;
        for a in battery_mean.iter() {
            if *a <= 0.0 {
                return None
            }
            sum += *a;
        }

        if (sum - 1.0).abs() > 0.001 {
            return None;
        }

        if concentration <= 0.0 {
            return None;
        }

        // The alpha vector is created by multiplying the concentration
        // parameter by the probability vector component wise
        let alpha1: f64 = concentration * battery_mean[0];
        let alpha2: f64 = concentration * battery_mean[1];
        let alpha3: f64 = concentration * battery_mean[2];
        let alpha: Vec<f64> = Vec::from([alpha1, alpha2, alpha3]);

        // An Rng is needed 
        let mut rng = thread_rng();

        // Create the Dirichlet distribution object
        let dir_dist = Dirichlet::new(alpha).unwrap();

        let mut questions: Vec<[f64; 3]> = Vec::with_capacity(num_questions);
        let mut norm_sum;
        for _ in 0..num_questions {
            let v = dir_dist.sample(&mut rng);
            norm_sum = v[0] + v[1] + v[2];
            questions.push([v[0] / norm_sum, v[1] / norm_sum, v[2] / norm_sum]);
        }
        
        Some(Battery { questions })
    }
}

pub struct Config {
    // In the process we are modelling, we have a fixed set of questions
    // (Battery), and we have the RAG system answer. But the RAG system
    // can be configured in many ways and each of these configurations
    // will have some impact on how well it does on the Battery.
    //
    // The basic idea is that a Config models the fact that different 
    // configurations of the RAG shift how the questions are answered (e.g., 
    // some questions are answered correctly more of the time now while others
    // are answered incorrectly more of the time with this config).
    // 
    // If questions are really just Categoricals that describe the how the
    // answers are distributed, then the effect of a config is to reshape
    // this distribution in some way.
    //
    // Importantly, the Battery's questions are points in the 2-simplex and so
    // whatever "effect" we want to impose on them must be sure to keep the
    // result in the 2-simplex (ie probability vectors map to probability vectors)
    //
    // A Config is therefore defined by the effect (perturbation) it has
    // on the battery. This effect is represented by an "Aitchison perturbation"
    // which is a type of transformation that maps simplex elements
    // back into the simplex. 
    //
    // The perturbation shifts, compresses or expands 
    // all the simplex vectors across the whole battery in the same 
    // general direction. 
    effect: [f64; 3]
}

impl Config {
    pub fn new(effect: [f64; 3]) -> Option<Config> {
        for i in 0..3 {
            if effect[i] <= 0.0 {
                return None;
            }
        }

        Some(Config { effect })
    }

    pub fn apply(&self, vector: &[f64; 3]) -> [f64; 3] {
        let mut new_vec: [f64; 3] = [
            self.effect[0] * vector[0],
            self.effect[1] * vector[1],
            self.effect[2] * vector[2]
        ];

        let norm_sum: f64 = new_vec[0] + new_vec[1] + new_vec[2];
        
        new_vec[0] = new_vec[0] / norm_sum;
        new_vec[1] = new_vec[1] / norm_sum;
        new_vec[2] = new_vec[2] / norm_sum;

        new_vec
    }
}

#[derive(Debug)]
pub struct BatteryEval {
    // A BatteryEval represents the outcome of each answer to 
    // a set of questions in a Battery under the effect of a 
    // specific Config.
    // 
    // A BatteryEval is the end result of applying a config to a specific
    // Battery and evaluating each answer, determining if they were
    // CORRECT, INCORRECT, or PARTIAL.
    //
    // Concretely, since a Battery's questions are just distributions, 
    // the BatteryEval is what you get by first reshaping the distributions
    // with the Config's effect, and then sampling from each distribution.
    pub valuations: Vec<Valuation>,
    pub scores: Vec<f64>
}

impl BatteryEval {
    pub fn new(
        battery: &Battery,
        config: &Config,
        policy: &ScoringPolicy,
    ) -> Self {
        let num_answers: usize = battery.questions.len();
        let mut valuations: Vec<Valuation> = Vec::with_capacity(num_answers);
        let mut scores: Vec<f64> = Vec::with_capacity(num_answers);

        let mut val_vec: [f64; 3];
        let mut rng = thread_rng();
        for i in 0..num_answers {
            val_vec = config.apply(
                &battery.questions[i]
            );

            let answer: f64 = Categorical::new(&val_vec).unwrap().sample(&mut rng);

            match answer {
                0.0 => {
                    valuations.push(Valuation::CORRECT);
                    scores.push(policy.score(&Valuation::CORRECT));
                },
                1.0 => {
                    valuations.push(Valuation::PARTIAL);
                    scores.push(policy.score(&Valuation::PARTIAL));
                },
                2.0 => {
                    valuations.push(Valuation::INCORRECT);
                    scores.push(policy.score(&Valuation::INCORRECT));
                },
                _ => todo!()
            };
        }
        BatteryEval { valuations, scores }
    }
}
