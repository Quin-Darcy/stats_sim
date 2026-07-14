// A Valuation is what you get when you check the
// provided answer against the actual answer to 
// a question. There are three possible outcomes
// to this checking.
#[derive(Debug, Clone)]
pub enum Valuation {
    CORRECT,
    PARTIAL,
    INCORRECT
}

pub struct ScoringPolicy {
    // This parameter represents how much we reward an abstention
    // If the AI answers "I don't know", then how much to we value
    // that answer. If 0, then its no better than a wrong answer
    // The idea is that a value greater than 0 allows room for the
    // LLM to express uncertainty when it is warranted instead of 
    // being forced to commit to a YES or NO, even with no evidence.
    circumspection: f64
}

impl ScoringPolicy {
    pub fn new(circumspection: f64) -> Option<Self> {
        if circumspection < 0.0 || circumspection > 1.0 {
            return None
        }

        Some(ScoringPolicy { circumspection })
    }

    pub fn score(&self, v: &Valuation) -> f64 {
        match v {
            Valuation::CORRECT => 1.0,
            Valuation::PARTIAL => self.circumspection,
            Valuation::INCORRECT => 0.0,
        }
    }
}
