data {
    int<lower=1> N;                  // Number of observations
    int<lower=1> num_stims;           // Number of stimuli
    int<lower=1> num_subs;            // Number of subjects
    int<lower=1, upper=num_subs> sub_num[N]; // Subject indices
    matrix[N, num_stims] stim_data;   // Stimulus-related data
    vector[N] y;                      // Observed response
    vector[N] y_t_1;                  // Lag-1 response (if AR terms included)
    vector[N] y_t_2;                  // Lag-2 response (if AR terms included)
    int<lower=0, upper=1> ar_flag;    // Indicator for AR terms
}

parameters {
    vector[num_stims] b;                      // Fixed effects
    real<lower=0> sigma;                      // Observation noise
    real<lower=0> sigma_sub_A;                // Random slope variance A
    real<lower=0> sigma_sub_B;                // Random slope variance B
    vector[num_subs] z_u0;                    // Standard normal for random slopes A (non-centered)
    vector[num_subs] z_u1;                    // Standard normal for random slopes B (non-centered)
    real ar1;                                 // AR(1) coefficient
    real ar2;                                 // AR(2) coefficient
}

transformed parameters {
    vector[num_subs] u0 = z_u0 * sigma_sub_A; // Non-centered random slopes A
    vector[num_subs] u1 = z_u1 * sigma_sub_B; // Non-centered random slopes B
    vector[N] mu;

    mu = u0[sub_num] .* rowSums(stim_data[, 1:(num_stims/2)]) +
         u1[sub_num] .* rowSums(stim_data[, (num_stims/2+1):num_stims]) +
         stim_data * b;

    if (ar_flag == 1) {
        mu += ar1 * y_t_1 + ar2 * y_t_2;
    }
}

model {
    // Priors
    b ~ normal(0, 10);
    sigma ~ cauchy(0, 10);
    sigma_sub_A ~ cauchy(0, 10);
    sigma_sub_B ~ cauchy(0, 10);
    z_u0 ~ std_normal();  // Standard normal prior for non-centered parameterization
    z_u1 ~ std_normal();

    if (ar_flag == 1) {
        ar1 ~ cauchy(0, 1);
        ar2 ~ cauchy(0, 1);
    }

    // Likelihood
    y ~ normal(mu, sigma);
}

// Generated quantities for contrasts
generated quantities {
    real cond_diff;
    cond_diff = mean(b[(num_stims/2 + 1):num_stims]) - mean(b[1:(num_stims/2)]);
}
