use crate::utils;

// A Valuation is what you get when you check the
// provided answer against the actual answer to 
// a question. There are three possible outcomes
// to this checking.
#[derive(Debug, Clone, PartialEq)]
pub enum Valuation {
    CORRECT,
    PARTIAL,
    INCORRECT
}

pub fn get_valuation(num: f64) -> Valuation {
    let epsilon: f64 = 0.001;
    if (num - 0.0).abs() < epsilon {
        return Valuation::CORRECT;
    }

    if (num - 1.0).abs() < epsilon {
        return Valuation::PARTIAL;
    }

    return Valuation::INCORRECT;
}

#[derive(Debug)]
pub struct ScoringPolicy {
    // This parameter represents how much we reward an abstention
    // If the AI answers "I don't know", then how much to we value
    // that answer. If 0, then its no better than a wrong answer
    // The idea is that a value greater than 0 allows room for the
    // LLM to express uncertainty when it is warranted instead of 
    // being forced to commit to a YES or NO, even with no evidence.
    
    pub q: f64, // Kept public for step size calculation in main
    circumspection: f64
}

impl ScoringPolicy {
    pub fn new(p_in: u64, q_in: u64) -> Option<Self> {
        if q_in == 0 {
            return None
        }

        let mut circumspection: f64 = (p_in as f64) / (q_in as f64);
        if circumspection < 0.0 || circumspection > 1.0 {
            return None
        }

        let gcd: u64 = utils::gcd(p_in, q_in);
        let p: f64 = (p_in as f64) / (gcd as f64);
        let q: f64 = (q_in as f64) / (gcd as f64);
        circumspection = p / q;

        Some(ScoringPolicy { q, circumspection })
    }

    pub fn score(&self, v: &Valuation) -> f64 {
        match v {
            Valuation::CORRECT => 1.0,
            Valuation::PARTIAL => self.circumspection,
            Valuation::INCORRECT => 0.0,
        }
    }
}
