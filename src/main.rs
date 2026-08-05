use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub mod utils;
pub mod new_sample;
pub mod new_bootstrap;
pub mod new_score;

use crate::new_sample::NewSample;
use crate::new_score::NewScoringPolicy;

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
// It *should* approximately match what confidence_level is. That is, a confidence_level CI returned from
// utils::ci() is a *promise* about the procedure and get_coverage() is the empircal
// way to verify the promise.
fn _get_coverage(
    confidence_level: f64,
    num_replicates: usize,
    base_samples: &[NewSample],
    policy: &NewScoringPolicy,
    parameter: f64,
    rng: &mut impl Rng,
) -> f64 {
    let mut ci: [f64; 2];
    let simulations: usize = base_samples.len();
    let mut coverage: f64 = 0.0;
    for i in 0..simulations {
        ci = new_bootstrap::ci(
            confidence_level,
            num_replicates,
            &base_samples[i],
            policy,
            rng,
        );

        if ci[0] <= parameter && parameter <= ci[1] {
            coverage += 1.0;
        }
    }

    coverage / (simulations as f64)
}

// Each bootstrap run will have some jitter in it due to which observations
// are selected from the base sample to create the replicates. This is true
// even if we hold the base sample fixed. So if we hold everything else fixed
// and run a bootstrap and get a CI on the statistic, the width of the CI
// would vary run-to-run due to this jitter.
//
// In the figures folder is a graph which plots the size of the CI width
// deltas as a function of num_replicates. It actually shows the more
// conservative 95th percentile delta across 100 simulations for each num_replicate
// The graph shows that the CI width delta is decreasing as num_replicates increases.
// Beyond that, it shows 1/sqrt(num_replicates) is always greater than the CI width
// delta. The hypothesis then is that there is some constant C for which
// C/sqrt(num_replicates) approximates the CI width delta for that given num_replicates.
//
// The function below sweeps through a range of num_replicates and runs 500 identical
// bootstraps can captures the deltas between each consecutive CI width and retains the
// the delta at the 95th percentile. This set represents a sample of (x, y) pairs
// and under the hypothesis we have that y = C / sqrt(x), where x is num_replicates and
// y is the CI width delta at the 95th percentile. We store the product
// y * sqrt(x) since our hypthosesis would suggest that these should be roughly the same
// value for all x, namely, C. We then take the mean of this set as our approximation of
// C and use it to then return (C / precision)^2 as the approximation of the particular
// num_replicates such that the resultant CI width will be less than our precision since
// if CI width delta ~ C / sqrt(num_replicates) < precision.
fn calibrate_num_replicates(
    precision: f64,
    confidence_level: f64,
    sample: &NewSample,
    policy: &NewScoringPolicy,
    rng: &mut impl Rng,
) -> usize {
    let loop_step: usize = 100;
    let max_replicates: usize = 10000;
    let simulations: usize = 10;
    let mut tmp_ci: [f64; 2];
    let mut last_ci_width: f64 = 0.0;
    let mut this_ci_width: f64;
    let mut percentile_delta: f64;
    let mut per_repl_deltas: Vec<f64> = Vec::with_capacity(simulations);
    let mut percentile_deltas: Vec<f64> = Vec::with_capacity(max_replicates / loop_step);

    for num_replicates in (2..max_replicates).step_by(loop_step) {
        // reset vectors
        per_repl_deltas.clear();

        for i in 0..simulations {
            tmp_ci = new_bootstrap::ci(confidence_level, num_replicates, sample, policy, rng);

            if i == 0 {
                last_ci_width = (tmp_ci[1] - tmp_ci[0]).abs();
            } else {
                this_ci_width = (tmp_ci[1] - tmp_ci[0]).abs();
                per_repl_deltas.push((this_ci_width - last_ci_width).abs());
                last_ci_width = this_ci_width;
            }
        }
        percentile_delta = utils::percentile(&mut per_repl_deltas, confidence_level);
        percentile_deltas.push(percentile_delta * (num_replicates as f64).sqrt());
    }
    let c: f64 = utils::mean(&percentile_deltas);

    // This is the num_replicates which should yield CI width deltas < precision
    (c / precision).powf(2.0) as usize
}

// Very similar in structure to get_coverage() above, this function will
// run a bootstrapping simulation for each Sample in base_samples. However,
// instead of checking the returned CIs for the population parameter,
// we capture the width of each CI.
//
// After all simulations are complete, we are left with a vector of f64s
// each equaling the width of a CI returned during one of the simulations.
//
// We then order the widths from smallest to largest and then use ci_safety_percentile to
// determine the width which (100 * ci_safety_percentile)% of all the other widths are *smaller*
// than. So if ci_safety_percentile = 0.95, this function returns the CI width that 95% of
// all the simulated widths were less than and so the returned width is a
// quantifiably *conservative* estimate of the CI widths associated with the
// parameters under simulation (e.g., sample size, Battery, Configs, confidence_level, etc)
fn conservative_ci_width(
    ci_safety_percentile: f64,
    confidence_level: f64,
    num_replicates: usize,
    base_samples: &[NewSample],
    policy: &NewScoringPolicy,
    rng: &mut impl Rng,
) -> f64 {
    let mut ci: [f64; 2];
    let num_simulations: usize = base_samples.len();
    let mut cis: Vec<f64> = Vec::with_capacity(num_simulations);

    for i in 0..num_simulations {
        ci = new_bootstrap::ci(
            confidence_level,
            num_replicates,
            &base_samples[i],
            policy,
            rng,
        );
        cis.push((ci[1] - ci[0]).abs());
    }
    cis.sort_by(|a, b| a.total_cmp(b));

    let percentile_index: usize = (ci_safety_percentile * num_simulations as f64) as usize;
    cis[percentile_index]
}

fn main() {
    // Initialize the RNG
    let seed: u64 = 48;
    let mut rng = StdRng::seed_from_u64(seed);

    // Set the policy we use to scrore. Circumspection will equal p / q
    let p: u64 = 1;
    let q: u64 = 2;
    let policy = NewScoringPolicy::new(p, q).unwrap();


    // Generate set of samples from this world
    let num_samples: usize = 1000;
    let sample_size: usize = 100;
    let samples: Vec<NewSample> = new_sample::sample_set(num_samples, sample_size,  &mut rng);

    // CI confidence level on the interval containing the
    // the population parameter (e.g., the mean of the
    // sample discordant difference across all samples
    let confidence_level: f64 = 0.95;

    // Each bootstrap run will produce a vector of means off
    // each of the replicates generated from the base sample.
    // A CI based off that vector is therefore determined by
    // the values in the vector which itself is determined in
    // part by the particular base sample we used.
    //
    // In order to make sure we didn't just get lucky and use
    // a forgiving base sample which resulted in an unusually
    // narrow CI, we will take a set of fresh samples, run the
    // bootstrapping process off each of them and then return
    // the CI whose width is bigger than some percentage of
    // all the other CI widths. That percentage is this variable.
    let ci_safety_percentile: f64 = 0.95;

    // The safety percentile above helps assure us we aren't
    // using a CI whose width is unusually narrow due to selecting
    // a lucky base sample. However, even if we held the base
    // sample and confidence interval fixed and called bootstrap::run()
    // multiple times, the CI resulting from each run would
    // *still* differ.
    //
    // The cause of this stems from the fact we create each
    // by populating it with randomly selected (with replacement)
    // values from the base sample. The values in the replicate
    // directly impact the statistic we compute and store in the
    // vector that bootstrap::run() returns. Thus, two consecutive
    // runs of bootstrap::run() with everything else equal will
    // result in a two runs, each with two different sets of replicates
    // and thus two different vectors of statistics and finally
    // two different CI widths.
    //
    // If we only created 1 replicate per bootstrap::run() call,
    // holding everything else fixed, then it should be clear that
    // the magnitude of the CI width difference between consecutive
    // runs is at its largest. The more replicates we generate for
    // each bootstrap::run() call, the more smaller the contribution
    // of jitter between runs is coming from the particular values
    // we randomly selected to create the replicates. This is because
    // more replicates means higher precision estimates of the
    // distribution contained in the base sample and the smaller
    // the step size between its samples.
    //
    // So the question becomes how much jitter are we willing to
    // tolerate, or what is the biggest step size between the samples
    // in the bootstrapped distribution. That value is the precision
    let desired_precision: f64 = 0.01;

    // For any single observation from a sample, its score is equal to the
    // difference between the two Valuations. This means that the difference
    // must be in the set {1, 1-a, a, 0, -a, a-1, -1}, where a is what the
    // ScoringPolicy's "circumspection" term is. By construction, a = p/q,
    // where p/q are share no common factors. With this, we can see the set
    // of possible differences has GCD of 1/q.
    //
    // The sample statistic we measure for each bootstrap run is the mean of
    // all the discordant differences over an entire sample. That is, we compute
    // the score for each Valuation pair in the replicate, sum them, and divide
    // by the sample size. For each replicate, we compute that mean and the set
    // of all the means is what bootstrap::run() returns.
    //
    // The mean discordant difference across a replicate is therefore some
    // multiple of 1/(sample_size * q). Thus, the vector returned from
    // bootstrap::run() can be thought of as containing elements from the lattice
    // with step size 1 / (sample_size * q). It then follows that the smallest non-zero
    // difference between any two of these values is equal to that step size.
    //
    // Therefore, the desired_precision can be no smaller than this step size
    // since it is not possible to have a non-zero difference between two bootstrap
    // means smaller than the step size and thus, since the CI bounds are themselves
    // bootstrap means, then the smallest non-zero difference in CI width is one
    // step size.
    let step_size: f64 = 1.0 / (policy.q * sample_size as f64);
    let precision: f64 = desired_precision.max(step_size);

    // Compute optimal num_replicates
    let num_replicates: usize =
        calibrate_num_replicates(precision, confidence_level, &samples[0], &policy, &mut rng);

    // This value represents a CI width whose trustworthiness comes from having
    // accounted for the two factors of chance which plays into how big it is
    // (i.e., (1) what base sample we happened to have used and (2) which
    // observations happened to be selected from the base sample for each replicate
    let con_ci_width: f64 = conservative_ci_width(
        ci_safety_percentile,
        confidence_level,
        num_replicates,
        &samples,
        &policy,
        &mut rng,
    );

    println!("Sample Size: {:?}", sample_size);
    println!("CI Confidence Level: {:?}", confidence_level);
    println!("Optimal Bootstrap Replicates: {:?}", num_replicates);
    println!(
        "Over {} simulations generating {:.1}% CIs, {:.2}% of the CI widths fell below {:.4}. Re-running any of these bootstraps with a different RNG seed changes the reported width by less than {:.4} in 95% of cases.",
        num_samples,
        confidence_level * 100.0,
        ci_safety_percentile * 100.0,
        con_ci_width,
        precision,
    );
}
