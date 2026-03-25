functions {
  matrix cholesky_cor_ar1(real ar, int nrows) {
    matrix[nrows, nrows] mat; 
    vector[nrows - 1] gamma; 
    mat = diag_matrix(rep_vector(1, nrows)); 
    for (i in 2:nrows) { 
      gamma[i - 1] = pow(ar, i - 1); 
      for (j in 1:(i - 1)) { 
        mat[i, j] = gamma[i - j]; 
        mat[j, i] = gamma[i - j]; 
      } 
    } 
    return cholesky_decompose(mat ./ (1 - ar^2));
  }
  matrix matrix_sqrt(matrix A) {
    int dim = rows(A);
    vector[dim] eival = eigenvalues_sym(A);
    matrix[dim, dim] eivec = eigenvectors_sym(A);
    return eivec * diag_matrix(sqrt(eival)) * inverse(eivec);
  }
}

data {
  int<lower=1> N;
  int<lower=1> n;
  int<lower=1> l;
  int<lower=1> q1;
  int njvec[n];
  vector[N] y;
  matrix[N, l] x;
  matrix[N, q1] z;
  int timevar[N];
  int<lower=1, upper=n> ind[N];
  real<lower=0> sdLP;
}

parameters {
  vector[l] beta;
  cholesky_factor_corr[q1] Lcorr;
  vector<lower=0>[q1] ddsqrt; 
  real<lower=0> sigmae;
  real<lower=-1, upper=1> phi1;
  matrix[q1, n] etavec;
  vector[q1] lambda;
  vector<lower=0>[n] t0vec;
  
  real<lower=0.001, upper=.999> nu1; // Prob of contamination
  real<lower=0.001, upper=.999> nu2; // Scale factor
}

transformed parameters {
  matrix[q1, n] b_raw; 
  cov_matrix[q1] D1; 
  vector[q1] Delta;
  matrix[q1, q1] dL;
  vector[q1] center_vec;
  
  D1 = quad_form_diag(multiply_lower_tri_self_transpose(Lcorr), ddsqrt);
  Delta = matrix_sqrt(D1) * (lambda / sqrt(1 + sum(square(lambda))));
  dL = cholesky_decompose(D1 - Delta * Delta');

  // Centering: E[U^(-1/2)] for SCN
  real mu_u = (1 - nu1) + nu1 * inv_sqrt(nu2);
  center_vec = Delta * (-sqrt(2.0 / pi()) * mu_u);

  for (j in 1:n) {
    // We define the random effect without the 'u' scaling here
    // b_raw = Delta * t0 + Gamma^1/2 * eta
    b_raw[,j] = Delta * t0vec[j] + dL * etavec[,j];
  }
}

model {
  // Priors
  beta ~ normal(0, 100);
  sigmae ~ student_t(4, 0, 5);
  ddsqrt ~ student_t(4, 0, 5);
  Lcorr ~ lkj_corr_cholesky(2.0);
  to_vector(etavec) ~ std_normal();
  lambda ~ normal(0, sdLP);
  t0vec ~ std_normal();
  nu1 ~ beta(1, 9); // Matches frequentist ~0.1
  nu2 ~ beta(1, 4); // Matches frequentist ~0.2

  {
    int njloc = 0;
    for (j in 1:n) {
      int idx[njvec[j]];
      for(k in 1:njvec[j]) idx[k] = njloc + k;
      
      matrix[njvec[j], njvec[j]] R_chol = cholesky_cor_ar1(phi1, max(timevar[idx]));
      matrix[njvec[j], njvec[j]] Sig_base = sigmae * R_chol[timevar[idx], timevar[idx]];
      
      vector[njvec[j]] subj_y = y[idx];
      vector[njvec[j]] fixed_mean;
      vector[njvec[j]] rand_mean_raw;

      for (k in 1:njvec[j]) {
        fixed_mean[k] = x[idx[k]] * beta + row(z, idx[k]) * center_vec;
        rand_mean_raw[k] = row(z, idx[k]) * col(b_raw, j);
      }

      // State 1: u = 1 (Normal)
      real lp1 = log1m(nu1) + multi_normal_cholesky_lpdf(subj_y | fixed_mean + rand_mean_raw, Sig_base);
      
      // State 2: u = nu2 (Contaminated)
      // Mean scales by inv_sqrt(nu2) for the random part, Var scales by 1/nu2
      real lp2 = log(nu1) + multi_normal_cholesky_lpdf(subj_y | fixed_mean + inv_sqrt(nu2) * rand_mean_raw, inv_sqrt(nu2) * Sig_base);

      target += log_sum_exp(lp1, lp2);
      njloc += njvec[j];
    }
  }
}

generated quantities {
  vector[n] prob_contaminated;
  {
    int njloc = 0;
    for (j in 1:n) {
      int idx[njvec[j]];
      for(k in 1:njvec[j]) idx[k] = njloc + k;
      matrix[njvec[j], njvec[j]] R_chol = cholesky_cor_ar1(phi1, max(timevar[idx]));
      matrix[njvec[j], njvec[j]] Sig_base = sigmae * R_chol[timevar[idx], timevar[idx]];
      vector[njvec[j]] fixed_mean;
      vector[njvec[j]] rand_mean_raw;
      for (k in 1:njvec[j]) {
        fixed_mean[k] = x[idx[k]] * beta + row(z, idx[k]) * center_vec;
        rand_mean_raw[k] = row(z, idx[k]) * col(b_raw, j);
      }
      real lp1 = log1m(nu1) + multi_normal_cholesky_lpdf(y[idx] | fixed_mean + rand_mean_raw, Sig_base);
      real lp2 = log(nu1) + multi_normal_cholesky_lpdf(y[idx] | fixed_mean + inv_sqrt(nu2) * rand_mean_raw, inv_sqrt(nu2) * Sig_base);
      prob_contaminated[j] = exp(lp2 - log_sum_exp(lp1, lp2));
      njloc += njvec[j];
    }
  }
}