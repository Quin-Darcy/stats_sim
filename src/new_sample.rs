/*
 * Configs are being compared based on how they answer a set of questions.
 * The same set of questions will run through two RAG pipelines defined by 
 * the configs. Each answer provided is either correct, partial, or incorrect.
 * What we care about is the per-question comparison since that is what models
 * the fact that the same questions are asked through both pipelines. 
 *
 * We score an answer based on if it is correct, incorrect, or partial. What
 * each of those 3 possibilities are worth is determined by the ScoringPolicy.
 * Typically, it will be that correct is worth 1 point, partial is worth 0.5,
 * and incorrect is worth 0. 
 *
 * For each question, we compare the answers given by each pipeline by taking
 * the difference of their scores. So (correct, correct) would give a difference
 * of 0 since they agree. But (partial, correct) would give -0.5. If we assume
 * the "candidate's" answers are in the first component and the "incumbent's"
 * answers are in the second component, then it follows that if the candidate
 * is "better" than the incumbent, then more of the score differences should be
 * greater than 0. Since in each case where the incumbent's score is greater
 * than the candidate's, the difference will be negative. It then follows that
 * if we were to sum all the score differences, a sum which is greater than 0
 * should correspond to a candidate pipeline (config) which performs better
 * on the battery than the incumbent. 
 *
 * If we consider the two answers given, one from each pipeline, to a single
 * question, we can see there are 9 possible scenarios:
 *
 *  1. Candidate:   CORRECT,    Incumbent:  CORRECT
 *  2. Candidate:   CORRECT,    Incumbent:  PARTIAL
 *  3. Candidate:   CORRECT,    Incumbent:  INCORRECT
 *  4. Candidate:   PARTIAL,    Incumbent:  CORRECT
 *  5. Candidate:   PARTIAL,    Incumbent:  PARTIAL
 *  6. Candidate:   PARTIAL,    Incumbent:  INCORRECT
 *  7. Candidate:   INCORRECT,  Incumbent:  CORRECT
 *  8. Candidate:   INCORRECT,  Incumbent:  PARTIAL
 *  9. Candidate:   INCORRECT,  Incumbent:  INCORRECT
 *
 * If this is for a single question, then we can further imagine these 9 same
 * scenarios across all the questions in the battery. Now suppose that as we
 * processed each question, we kept track of which scenario the valuation fell
 * into. To formalize this, consider the following table:
 *
 * |-----------------------|---------|---------|-----------|
 * | Candidate \ Incumbent | CORRECT | PARTIAL | INCORRECT |
 * |-----------------------|---------|---------|-----------|
 * | CORRECT               |   t00   |   t01   |    t02    |
 * |-----------------------|---------|---------|-----------|
 * | PARTIAL               |   t10   |   t11   |    t12    |
 * |-----------------------|---------|---------|-----------|
 * | INCORRECT             |   t20   |   t21   |    t22    |
 * |-----------------------|---------|---------|-----------|
 *
 * We use tij to denote the particular scenario. This gives rise to a way to
 * write two quantities of interest:
 *
 *  1. n(tij): The ratio of the K answer pairs in scenario tij
 *  2. s(tij): The score difference of pair associated with scenario tij
 *
 *  It should be noted that while s(tij) can change depending on the questions
 *  or the configs defining the pipelines. However, s(tij) is fixed and determined
 *  only by the scoring policy. This means a similar table can be created to
 *  represent the possible score differences for each of the 9 scenarios. If we
 *  let c denote the amount we credit for a partial answer, where 0 <= c <= 1, then
 *  we could have:
 *
 * |-----------------------|---------|---------|-----------|
 * | Candidate \ Incumbent | CORRECT | PARTIAL | INCORRECT |
 * |-----------------------|---------|---------|-----------|
 * | CORRECT               |    0    |  1 - c  |     1     |
 * |-----------------------|---------|---------|-----------|
 * | PARTIAL               |  c - 1  |    0    |     c     |
 * |-----------------------|---------|---------|-----------|
 * | INCORRECT             |   -1    |   -c    |     0     |
 * |-----------------------|---------|---------|-----------|
 *
 *  From this it can be seen that s(tij) = 0 when i == j which corresponds
 *  to the main diagonal. 
 *  
 *  It should also be clear the values of n(tij) can also be expressed in a similar 
 *  table. Taking each the s(tij) and n(tij) instead as matrices, S and N, then taking
 *  the Frobenius inner product of N and S we get
 *  
 *  tr(N^T S) = n(t00) * s(00) + ... + n(22) * s(22)
 *
 *  This happens to be the same thing as the mean taken over the set of score differences
 *  across the battery.
 *
 *  This motivates how we end up defining what a Sample is. Namely, for a given set
 *  of questions, what matters for comparing them is the mean of score differences across
 *  all the questions. But importantly, the mean is just a single number and to get 
 *  insights into the distribution from which the mean is sampled, we need more information
 *  than just the mean. Specifically, we need n(tij) matrix since that is an object 
 *  which we can create replicates from by resampling with replacement.
 */


use rand::Rng;

use crate::utils;

#[derive(Debug)]
pub struct NewSample {
    pub size: usize,
    pub mean: f64,
    pub observations: [f64; 9]
}

impl NewSample {
    pub fn new(size: usize, rng: &mut impl Rng) -> NewSample {
        // Processing each question in the battery consists of determining
        // which of the 9 scenarios the answer pair falls in and assigning the
        // corrsponding score. Doing this for all the question means all answer
        // pairs fall into 1 of the 9 scenarios and thus we can represent this
        // by partitioning the set {1, 2, ..., K} (where K is the number of questions
        // in the battery), into a random partition of 9 pieces. 
        // 
        // We will randomly select 8 numbers from the interval [1, K] and
        // use them to define subintervals whose width determines the number
        // of answer pairs which fall into the assoicated scenario.
        //
        // |---+-+-----++----+----+------+--+----|
        //
        // Consider the example below where K = 15. The random numbers selected are:
        // 
        // | 1 2 | 3 | 4 5 || 6 7 || 8 9 | 10 11 12 13 14 | 15 |
        //
        // The randomly selected number are: {2, 3, 5, 5, 7, 7, 9, 14} which
        // produces widths: {2, 1, 2, 0, 2, 0, 2, 5, 1}, where
        // 
        // * s_0: 2 = 2 - 0
        // * s_1: 1 = 3 - 2
        // * s_2: 2 = 5 - 3
        // * s_3: 0 = 5 - 5
        // * s_4: 2 = 7 - 5
        // * s_5: 0 = 7 - 7
        // * s_6: 2 = 9 - 7
        // * s_7: 5 = 14 - 9
        // * s_8: 1 = 15 - 14
        //
        // And so we see that:
        //  
        //  2 + 1 + 2 + 0 + 2 + 0 + 2 + 5 + 1 = 15
        //
        // Which gives 
        //
        // |-----------------------|---------|---------|-----------|
        // | Candidate \ Incumbent | CORRECT | PARTIAL | INCORRECT |
        // |-----------------------|---------|---------|-----------|
        // | CORRECT               |    2    |    1    |     2     |
        // |-----------------------|---------|---------|-----------|
        // | PARTIAL               |    0    |    2    |     0     |
        // |-----------------------|---------|---------|-----------|
        // | INCORRECT             |    2    |    5    |     1     |
        // |-----------------------|---------|---------|-----------|
        //
        // And generally, if we let {s_i} denote the sequence of randomly generated
        // numbers from the interval [1, K], then the widths (scenario counts) are 
        // calculated as: w_0 = s_0, w_{i} = s_i - s_{i - 1}, w_{8} = K - s_{7}
        
        // Create and order the cuts
        let mut cuts: [usize; 8] = [0; 8];
        for i in 0..8 {
            cuts[i] = rng.gen_range(0..size);
        }
        cuts.sort_by(|a, b| a.cmp(b));

        // Use the cuts to populate the samples observations (i.e., ratios of scenario 
        // counts divided by sample size).
        let mut observations: [f64; 9] = [0.0; 9];
        for i in 0..9 {
            if i == 0 {
                observations[i] = cuts[0] as f64 / size as f64;
            } else if i == 8 {
                observations[i] = (size - cuts[i - 1]) as f64 / size as f64;
            } else {
                observations[i] = (cuts[i] - cuts[i - 1]) as f64 / size as f64;
            }
        }

        let mean: f64 = utils::mean(&observations);

        NewSample { size, mean, observations }
    }
}

pub fn sample_set(
    num_samples: usize, 
    sample_size: usize, 
    rng: &mut impl Rng
) -> Vec<NewSample> {
    let mut samples: Vec<NewSample> = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        samples.push(NewSample::new(sample_size, rng));
    }
    samples
}

