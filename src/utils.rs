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
    let l: f64 = (1.0 - gamma) / 2.0;
    let u: f64 = 1.0 - l;
    let li: usize = (l * vals.len() as f64) as usize;
    let lu: usize = (u * vals.len() as f64) as usize;
    [vals[li], vals[lu]]
}

