use rand::thread_rng;
use rand::distributions::Distribution;
use statrs::distribution::{Dirichlet, Categorical};

use crate::score::{Valuation, ScoringPolicy};


#[derive(Debug)]
pub struct Battery {
    // for a given set of questions, we let each question have an 
    // intrinsic "difficulty" which is descibed by the Categorical
    // distribution over the Valuation space {CORRECT, PARTIAL, INCORRECT}
    //
    // This represents the character of a paritcular question in terms
    // of how the valuation outcomes of the question are distributed.
    // "Easy questions" would have high probability on the CORRECT
    // valuation and low probability elsewhere.
    //
    // A battery is defined by this set of probability vectors. 
    //
    // Sampling from the distribution defined by one of these intrinsic
    // difficulty probability vectors gives you one Valuation back.
    // These distributions describe the probability of the particular answer
    // being CORRECT, INCORRECT, or PARTIAL
    //
    // We would say "most of the questions are difficult" when the categorical 
    // distribution for each question has high probability for the 
    // INCORRECT valuation. This means most of the questions most of the time
    // will be answered incorrectly. This is the "baseline" difficulty
    // *before* a config is introduced which may shift these distributions
    // in some way.
    intrinsic_difficulty_probability_vectors: Vec<[f64; 3]>
}

impl Battery {
    pub fn new(
        base_valuation_probability_vector: [f64; 3],
        base_valuation_concentration: f64,
        num_questions: usize
    ) -> Option<Battery> {
        // Because a Battery is defined by a set of probability vectors (i.e.,
        // non-negative components which all add to 1), then these vectors 
        // themselves need to be sampled from some distribution lest we wish to
        // manually hardcode them.
        //
        // The distribution the battery's probability vectors are sampled from
        // is called the Dirichlet distribution with 3 concentration parameters.
        //
        // The base_valuation_probability_vector is a probability vector which
        // represents the point in the 2-simplex around which the Battery's 
        // probability vectors are concentrated. The base_valuation_concentration
        // is a float that controls how tightly the vectors are concentrated around 
        // distribution's peak point. Higher values result in tighter clustering and 
        // lower values spread the mass out (think pointy peak vs fat rounded hill).
        // Values lower than one makes a kind of crater where the mass is pushed out
        // into the corners.
        //
        // This distribution is one layer above the probability vectors which 
        // define a given Battery. If you want most of your batteries to contain 
        // difficult questions, you can do this by setting the concentration parameters 
        // of this Dirichlet distribution to cluster in the area of the 2-simplex 
        // where the vectors in that area have higher component values corresponding
        // to the INCORRECT valuation probability.

        let mut sum: f64 = 0.0;
        for a in base_valuation_probability_vector.iter() {
            if *a <= 0.0 {
                return None
            }
            sum += *a;
        }

        if (sum - 1.0).abs() > 0.001 {
            return None;
        }

        if base_valuation_concentration <= 0.0 {
            return None;
        }

        // The alpha vector is created by multiplying the concentration
        // parameter by the probability vector component wise
        let alpha1: f64 = base_valuation_concentration * base_valuation_probability_vector[0];
        let alpha2: f64 = base_valuation_concentration * base_valuation_probability_vector[1];
        let alpha3: f64 = base_valuation_concentration * base_valuation_probability_vector[2];
        let alpha: Vec<f64> = Vec::from([alpha1, alpha2, alpha3]);

        // An Rng is needed 
        let mut rng = thread_rng();

        // Create the Dirichlet distribution object
        let dir_dist = Dirichlet::new(alpha).unwrap();

        let mut battery_vecs: Vec<[f64; 3]> = Vec::with_capacity(num_questions);
        let mut norm_sum;
        for _ in 0..num_questions {
            let v = dir_dist.sample(&mut rng);
            norm_sum = v[0] + v[1] + v[2];
            battery_vecs.push([v[0] / norm_sum, v[1] / norm_sum, v[2] / norm_sum]);
        }
        
        Some(Battery { intrinsic_difficulty_probability_vectors: battery_vecs })
    }
}

pub struct Config {
    // A config contains a 3 float vector which is used to act on
    // each per-question instrinsic difficulty probability vectors
    // that define the character of a battery question.
    //
    // The battery questions are represented by the intrinsic difficulty
    // probability vectors which are members of a simplex.
    //
    // A config is identified by the effect (perturbation) it has
    // on the battery. This effect is represented by an "Aitchison perturbation"
    // which is a type of transformation that maps simplex elements
    // back into the simplex. 
    //
    // The perturbation shifts, compresses or expands 
    // all the simplex vectors across the whole battery in the same 
    // general direction. 
    //
    // The idea is that given a set of questions in a Battery, each question
    // is characterized by how the the valuations of it are distributed (e.g.,
    // mostly INCORRECT, mostly CORRECT, etc) and this distribution is a 
    // probability vector that sits in the 2-simplex. A config which is results 
    // in most most answers being CORRECT can be thought of as having altered 
    // transformed the probability vectors for each question such that sampling
    // from the distributions they descibe now sees more CORRECT valuations than
    // previously.
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
    // CORRECT, INCORRECT, or PARTIAL
    //
    // It is constructed by taking one sample from each of the perturbed 
    // intrinsic difficulty distributions (categoricals) that correspond to each
    // question in the battery after having been transformed by the config
    //
    // It is important to remember that there is never actually a "question"
    // or an "answer" in this model. This comes from realizing that for a 
    // given battery and config, once those questions are answered, each answer
    // is either correct, incorrect, or parital and this model skips right to 
    // that part, to the part where the questions have already been answered and
    // we have already determined if they were right or wrong.
    //
    // For HPO, we care only about being able to compare two configs based on
    // how many questions were correct or not as a result of their effect.
    // The only informaiton we need to perform this comparison is the per-question
    // valuation for each config. We do not need to model the actual questions
    // or answers.
    pub valuations: Vec<Valuation>,
    pub scores: Vec<f64>
}

impl BatteryEval {
    pub fn new(
        questions: &Battery,
        config: &Config,
        policy: &ScoringPolicy,
    ) -> Self {
        let num_answers: usize = questions.intrinsic_difficulty_probability_vectors.len();
        let mut valuations: Vec<Valuation> = Vec::with_capacity(num_answers);
        let mut scores: Vec<f64> = Vec::with_capacity(num_answers);

        let mut rng = thread_rng();
        for i in 0..num_answers {
            let val_vec: [f64; 3] = config.apply(
                &questions.intrinsic_difficulty_probability_vectors[i]
            );

            let cat_samp: f64 = Categorical::new(&val_vec).unwrap().sample(&mut rng);

            match cat_samp {
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
