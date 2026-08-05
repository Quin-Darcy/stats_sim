use crate::utils;

pub struct NewScoringPolicy {
    pub q: f64,
    pub outcome_scores: [f64; 9],
}

impl NewScoringPolicy {
    pub fn new(p_in: u64, q_in: u64) -> Option<Self> {
        if q_in == 0 {
            return None;
        }

        let mut circumspection: f64 = (p_in as f64) / (q_in as f64);
        if circumspection < 0.0 || circumspection > 1.0 {
            return None;
        }

        let gcd: u64 = utils::gcd(p_in, q_in);
        let p: f64 = (p_in as f64) / (gcd as f64);
        let q: f64 = (q_in as f64) / (gcd as f64);
        circumspection = p / q;

        let outcome_scores: [f64; 9] = [
            0.0,                    1.0 - circumspection,   1.0,
            circumspection - 1.0,   0.0,                    circumspection,
            -1.0,                   -circumspection,        0.0
        ];

        Some(NewScoringPolicy { q, outcome_scores })
    }
}
