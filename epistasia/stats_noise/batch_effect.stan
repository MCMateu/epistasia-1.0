data {
  int<lower=1> M;               // number of states
  int<lower=1> R;               // number of replicates
  int<lower=1> K;               // number of observed entries (non-NaN)
  array[K] int<lower=1, upper=M> m;  // state index for each observation
  array[K] int<lower=1, upper=R> r;  // replicate index for each observation
  vector[K] y;                  // observed values (in model space)
}

parameters {
  vector[M] F_bar;

  vector[R-1] a_raw;
  vector<lower=-3, upper=3>[R] log_b_raw;

  real<lower=-5, upper=1> log_sigma;
}

transformed parameters {
  vector[R] a;
  vector[R] b;
  real sigma;
  real gm_log_b;

  for (j in 1:(R - 1)) a[j] = a_raw[j];
  a[R] = -sum(a_raw);

  gm_log_b = mean(log_b_raw);
  for (j in 1:R) b[j] = exp(log_b_raw[j] - gm_log_b);

  sigma = exp(log_sigma);
}

model {
  a_raw      ~ normal(0, 0.5);
  log_b_raw  ~ normal(0, 0.3);

  F_bar      ~ normal(0, 5);
  log_sigma  ~ normal(log(0.05), 0.5);

  for (k in 1:K) {
    y[k] ~ normal(F_bar[m[k]] + a[r[k]], sigma * b[r[k]]);
  }
}
