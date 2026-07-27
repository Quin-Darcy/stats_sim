use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod score;
pub mod model;
pub mod sample;
pub mod world;
pub mod bootstrap;
pub mod utils;

use crate::score::ScoringPolicy;
use crate::world::World;
use crate::sample::Sample;


// This function is only used to verify the implicit claim made by the CIs
// we return from utils::ci().
//
// We generate some number of fresh samples (not the artifical ones created
// during bootstrapping) which gives us the sample distribution of the average
// discordant difference. This approximates the "real" mean discordant difference
// with respect to the given Battery and Config pair. The average discordant 
// difference of the sample distribution is then treated as our population parameter.
//
// We use each of these fresh samples (base_samples) as *the* base sample in one
// bootstrap "simulation". Recall, given one real sample, we can use bootstrapping
// to get a CI. The process from one base sample to a CI is one bootstrapping process.
//
// We perform the bootstrap process with each sample in base_samples, and for each 
// resultant CI, we can ask "Does the CI contain the population parameter?". If it does,
// we count it. The total number of CIs which contained the population parameter over
// the total number of simulations is the "coverage". 
//
// It *should* approximately match what gamma is. That is, a gamma CI returned from
// utils::ci() is a *promise* about the procedure and get_coverage() is the empircal
// way to verify the promise.
fn get_coverage(
    gamma: f64,
    num_resamples: usize,
    base_samples: &[Sample],
    policy: &ScoringPolicy,
    parameter: f64,
    rng: &mut impl Rng
) -> f64 {
    let mut ci: [f64; 2];
    let simulations: usize = base_samples.len();
    let mut coverage: f64 = 0.0;
    for i in 0..simulations {
        ci = bootstrap::ci(
            gamma,
            num_resamples,
            &base_samples[i],
            policy,
            rng
        );

        if ci[0] <= parameter && parameter <= ci[1] {
            coverage += 1.0;
        }
    }

    coverage / (simulations as f64)
}

// Very similar in structure to get_coverage() above, this function will
// run a bootstrapping simulation for each Sample in base_samples. However,
// instead of checking the returned CIs for the population parameter,
// we capture the width of each CI.
//
// After all simulations are complete, we are left with a vector of f64s
// each equaling the width of a CI returned during one of the simulations.
//
// We then order the widths from smallest to largest and then use alpha to
// determine the width which (100 * alpha)% of all the other widths are *smaller*
// than. So if alpha = 0.95, this function returns the CI width that 95% of
// all the simulated widths were less than and so the returned width is a
// quantifiably *conservative* estimate of the CI widths associated with the
// parameters under simulation (e.g., sample size, Battery, Configs, gamma, etc)
fn conservative_ci_width(
    alpha: f64,
    gamma: f64,
    num_resamples: usize,
    base_samples: &[Sample],
    policy: &ScoringPolicy,
    rng: &mut impl Rng
) -> f64 {
    let mut ci: [f64; 2];
    let simulations: usize = base_samples.len();
    let mut cis: Vec<f64> = Vec::with_capacity(simulations);
    for i in 0..simulations {
        ci = bootstrap::ci(
            gamma,
            num_resamples,
            &base_samples[i],
            policy,
            rng
        );
        cis.push((ci[1] - ci[0]).abs());
    }
    cis.sort_by(|a, b| a.total_cmp(b));
    let index: usize = (alpha * simulations as f64) as usize;
    cis[index]
}


fn main() {
    // Initialize the RNG
    let seed: u64 = 48;
    let mut rng = StdRng::seed_from_u64(seed);

    // Set the policy we use to scrore
    let circumspection: f64 = 0.5;
    let policy = ScoringPolicy::new(circumspection).unwrap();

    // Create a world from which we will sample
    let world = World::random(&mut rng);

    // Generate set of samples from this world
    let num_samples: usize = 1000;
    let sample_size: usize = 132;
    let samples = world.sample(num_samples, sample_size, &policy, &mut rng);

    ////////////////////////////////////////////////////////////
    // Below this point is simply statistics exploring difference
    // between sample and bootstrap distributions
    ////////////////////////////////////////////////////////////

    // Here we use our set of fresh samples to compute an estimate
    // of the population parameter. Each sample carries its own mean
    // already and so we will simply get the average of the sample
    // means
    let sample_means: Vec<f64> = samples.iter().map(|a| a.mean).collect();
    let parameter: f64 = utils::mean(&sample_means);

    // Validate bootstrap CI coverage claim
    let gamma: f64 = 0.95;
    let num_resamples: usize = 10000;
    let coverage: f64 = get_coverage(
        gamma,
        num_resamples,
        &samples,
        &policy,
        parameter,
        &mut rng,
    );

    let alpha: f64 = 0.95;
    let con_ci: f64 = conservative_ci(
        alpha,
        gamma,
        num_resamples,
        &samples,
        &policy,
        &mut rng,
    );

    println!("Sample Size: {:?}", sample_size);
    println!("Sample Distribution Size: {:?}\n", num_samples);
    println!("CI Gamma: {:?}", gamma);
    println!("Bootstrap Resamples: {:?}", num_resamples);
    println!("Bootstrap Simulations: {:?}", num_samples);
    println!("Bootstrap Coverage: {:.2}%", coverage * 100.0);
    println!("Over {} simulations, {:.2}% of CI widths fall below {:.2}", num_samples, alpha * 100.0, con_ci);
}
