use rand::Rng;
use rand::distributions::Distribution;
use statrs::distribution::{Dirichlet, Categorical};

use crate::score::{Valuation, get_valuation};


#[derive(Debug)]
pub struct Battery {
    // A Battery is identified by the questions it contains. However,
    // we don't actually model the questions themselves, but the valuation
    // of the question (i.e., correct, incorrect, etc).
    //
    // A Battery is only used in service of comparing two configs, and the
    // comparison only cares about which config got which questions right
    // or wrong.
    //
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
    // A Battery consists of the parameters that define the Dirichlet
    // distribution from which the probability vectors that represent the 
    // questions are sampled from..
    //
    // We would say "most of the questions are difficult" when the categorical 
    // distribution for each question has high probability for the 
    // INCORRECT valuation. This means most of the questions most of the time
    // will be answered incorrectly. This is the "baseline" difficulty
    // *before* a config is introduced which may shift these distributions
    // in some way.
    pub alpha: [f64; 3],
}

impl Battery {
    pub fn new(
        battery_mean: [f64; 3],
        concentration: f64,
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
        let alpha: [f64; 3] = [alpha1, alpha2, alpha3];

        Some(Battery { alpha })
    }

    pub fn get_valuation_pairs(
        &self, 
        num_questions: usize, 
        configs: (&Config, &Config),
        rng: &mut impl Rng,
    ) -> Vec<(Valuation, Valuation)> {
        let dir_dist = Dirichlet::new(Vec::from(self.alpha)).unwrap();
        let mut base_question: [f64; 3];
        let mut norm_sum: f64;

        // Pre-declare muts to store effected questions and valuations
        let mut question_after_config1: [f64; 3];
        let mut question_after_config2: [f64; 3];
        let mut valuation_pairs: Vec<(Valuation, Valuation)> = Vec::with_capacity(num_questions);

        for _ in 0..num_questions {
            // Create the question which will be processed by each config
            let v = dir_dist.sample(rng);
            norm_sum = v[0] + v[1] + v[2];
            base_question = [v[0] / norm_sum, v[1] / norm_sum, v[2] / norm_sum];

            // Perturb the question by the configs
            question_after_config1 = configs.0.apply(&base_question);
            question_after_config2 = configs.1.apply(&base_question);

            // Sample verdict for each question
            let categoricals: (f64, f64) = (
                Categorical::new(
                    &question_after_config1
                ).unwrap().sample(rng),
                Categorical::new(
                    &question_after_config2
                ).unwrap().sample(rng)
            );

            valuation_pairs.push((
                get_valuation(categoricals.0),
                get_valuation(categoricals.1)
            ));
        }
        valuation_pairs
    }
}

#[derive(Debug)]
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
