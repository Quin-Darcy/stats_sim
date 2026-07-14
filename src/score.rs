#[derive(Debug)]
pub enum Valuation {
    CORRECT,
    PARTIAL,
    INCORRECT
}

pub struct ScoringPolicy {
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
