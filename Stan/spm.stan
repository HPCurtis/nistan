data {
  int<lower=0> N;               // Number of observations
  int<lower=0> K;               // Number of predictors (including intercept)
  matrix[N, K] X;               // Design matrix
  vector[N] y;                  // Response vector
}

parameters {
  vector[K] beta;               // Regression coefficients
  real<lower=0> sigma;          // Error standard deviation
}

model {
  // Likelihood
  y ~ normal(X * beta, sigma);
}
generated quantities {
   real diff =  beta[2] - beta[1];
}