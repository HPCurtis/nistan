// Programme identical to fixed_nc.stan but without + stim_data * b on the mu parameter. 
data {
    int<lower=1> N;                  // Number of observations
    int<lower=1> n_stims;           // Number of stimuli
    int<lower=1> n_subs;            // Number of subjects
    array[N] int<lower=1, upper=n_subs> sub_num; // Subject indices
    vector[N]  u0_stim_data;
    vector[N] u1_stim_data;
    matrix[N, n_stims] u2_stim_data;   // Stimulus-related data
    matrix[N, n_stims] u3_stim_data;   // Stimulus-related data
    vector[N] y;                      // Observed response
    vector[N] y_t_1;                  // Lag-1 response (if AR terms included)
    vector[N] y_t_2;                  // Lag-2 response (if AR terms included)
    int<lower=0, upper=1> ar_flag;    // Indicator for AR terms
}
parameters {
    vector[2] b;                      // Fixed effects
    real<lower=0> sigma;                      // Observation noise
    real<lower=0> sigma_sub_A;                // Random slope variance A
    real<lower=0> sigma_sub_B;                // Random slope variance B
    real<lower=0> sigma_stim_A;                // Random slope variance A
    real<lower=0> sigma_stim_B;                // Random slope variance B
    vector[n_subs] z_u0;                    // Standard normal for random slopes A (non-centered)
    vector[n_subs] z_u1;                    // Standard normal for random slopes B (non-centered)
    vector[n_stims] z_u2;                    // Standard normal for random slopes A (non-centered)
    vector[n_stims] z_u3;                    // Standard normal for random slopes B (non-centered)
    real ar1;                                 // AR(1) coefficient
    real ar2;                                 // AR(2) coefficient
}
model {
    // Priors
    b ~ normal(0, 10);
    sigma ~ cauchy(0, 10);
    sigma_sub_A ~ cauchy(0, 10);
    sigma_sub_B ~ cauchy(0, 10);
    z_u0 ~ std_normal();  // Standard normal prior for non-centered parameterization
    z_u1 ~ std_normal();
    z_u2 ~ std_normal();  // Standard normal prior for non-centered parameterization
    z_u3 ~ std_normal();
    if (ar_flag == 1) {
        ar1 ~ cauchy(0, 1);
        ar2 ~ cauchy(0, 1);
    }
    vector[n_subs] u0 = z_u0 * sigma_sub_A; // Non-centered random slopes A
    vector[n_subs] u1 = z_u1 * sigma_sub_B; // Non-centered random slopes B
    vector[n_stims] u2 = z_u2 * sigma_stim_A; // Non-centered random slopes A
    vector[n_stims] u3 = z_u3 * sigma_stim_B; // Non-centered random slopes B
    vector[N] mu = (b[1] + u0[sub_num] .* u0_stim_data) +
         (b[2] + u1[sub_num] .* u1_stim_data) +
         u2_stim_data * u2 + u3_stim_data * u3;
    if (ar_flag == 1) {
        mu += ar1 * y_t_1 + ar2 * y_t_2;
    }
    // Likelihood
    y ~ normal(mu, sigma);
}
generated quantities {
   real diff = b[2] - b[1];
}