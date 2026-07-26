use rand::Rng;
use crate::score::{ScoringPolicy, Valuation};


pub fn mean(vals: &[f64]) -> f64 {
    vals.iter().sum::<f64>() / (vals.len() as f64)
}

pub fn variance(vals: &[f64]) -> f64 {
    let mean: f64 = mean(vals);
    let mut sum: f64 = 0.0;
    for i in 0..vals.len() {
        sum += (vals[i] - mean).powf(2.0);
    }
    sum / (vals.len() as f64)
}

pub fn sd(vals: &[f64]) -> f64 {
    variance(&vals).sqrt()
}

pub fn ci(vals: &mut [f64], gamma: f64) -> [f64; 2] {
    // sort vals
    vals.sort_by(|a, b| a.total_cmp(b));
    let l: f64 = (1.0 - gamma) / 2.0;
    let u: f64 = 1.0 - l;
    let li: usize = (l * vals.len() as f64) as usize;
    let lu: usize = (u * vals.len() as f64) as usize;
    [vals[li], vals[lu]]
}

pub fn get_rand_pv(rng: &mut impl Rng) -> [f64; 3] {
    // Select 3 random numbers between 0 and 1
    // order them, and their widths form the components
    // of the probability vector
    let c1: f64 = rng.gen_range(0.0..1.0);
    let c2: f64 = rng.gen_range(0.0..1.0);

    // These will be the actual components
    let a: f64;
    let b: f64;
    let c: f64;

    if c1 <= c2 {
        a = c1;
        b = c2 - c1;
        c = 1.0 - c2;
    } else {
        a = c2;
        b = c1 - c2;
        c = 1.0 - c1;
    }

    [a, b, c]
}

pub fn get_rand_vec(rng: &mut impl Rng) -> [f64; 3] {
    [
        rng.gen_range(0.0..10.0),
        rng.gen_range(0.0..10.0),
        rng.gen_range(0.0..10.0)
    ]
}

pub fn discord_diff(policy: &ScoringPolicy, v1: &Valuation, v2: &Valuation) -> f64 {
    policy.score(v1) - policy.score(v2)
}
