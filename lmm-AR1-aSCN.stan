functions{//auxiliary functions
	matrix cholesky_cor_ar1(real ar, int nrows) { //from brms package
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
	   return eivec*diag_matrix(sqrt(eival))*inverse(eivec);
   }
}
data {
     int<lower=1> N;
     int<lower=1> n;
     int<lower=1> l;
     int<lower=1> q1;
     int njvec[n];
	 vector[N] y;
     matrix[N,l] x;
     matrix[N,q1] z;
     int timevar[N];
     int<lower=1,upper=n> ind[N];
	 real<lower=0> sdLP;
}
parameters {
       vector[l] beta;
		   cholesky_factor_corr[q1] Lcorr;// cholesky factor (L matrix for correlation decomposition of D)
		   vector<lower=0>[q1] ddsqrt; //scale vector for D
       real<lower=0> sigmae;//scale of random error
		   real<lower=-1, upper=1> phi1;//AR(1)
           matrix[q1, n] etavec;//standardized random effects
		   //
		   vector[q1] lambda;//skewness
		   vector<lower=0>[n] t0vec;//half-normal term
		   //
		   real<lower=0.001, upper=.49> nu1; // probability of contamination
		   real<lower=0.01, upper=.8> nu2; // variance scale factor
		   vector<lower=0>[n] uvec;    // mixing variable
}
transformed parameters {
  matrix[q1, n] bvec;
  cov_matrix[q1] D1; 
  vector[n] tvec;
  vector[q1] Delta;
  matrix[q1,q1] dL;
  //
  D1 = quad_form_diag(multiply_lower_tri_self_transpose(Lcorr), ddsqrt); // quad_form_diag: diag_matrix(ddsqrt) * L * L' * diag_matrix(ddsqrt)
  Delta = matrix_sqrt(D1)*(lambda/sqrt(1+sum(square(lambda))));
  dL = cholesky_decompose(D1 - Delta * Delta');
  //
  real mu_u = (1 - nu1) + nu1 * inv_sqrt(nu2); //k1 = E(1/sqrt(U)) = 1 - nu1 + nu1/sqrt(nu2)
  //
  {
  for (j in 1:n) {
    #conditional mean of t = c = -sqrt(2/pi)*k1
    tvec[j] = -sqrt(2 / pi()) * mu_u + inv_sqrt(uvec[j]) * t0vec[j];
    bvec[,j] = Delta * tvec[j] + dL * inv_sqrt(uvec[j]) * etavec[,j];
  }
  }
}
model {
  beta ~ normal(0, 100);
  sigmae ~ student_t(4, 0, 5);
  ddsqrt ~ student_t(4, 0, 5);
  Lcorr ~ lkj_corr_cholesky(2.0);
  to_vector(etavec) ~ std_normal();
  lambda ~ normal(0, sdLP);
  t0vec ~ std_normal();
  //
  nu1 ~ beta(1, 4); 
  nu2 ~ beta(1, 4); 

  // Prior for uvec: The "Two-Spike" Approximation
  for (j in 1:n) {
    target += log_sum_exp(log1m(nu1) + normal_lpdf(uvec[j] | 1.0, 0.15), 
                          log(nu1)   + normal_lpdf(uvec[j] | nu2, 0.15));
  }

  {
    vector[N] yhat;
    int njloc = 0;
    // Calculate yhat
    for (i in 1:N) {
      yhat[i] = x[i] * beta + row(z, i) * col(bvec, ind[i]);
    }
    // Likelihood
    for (j in 1:n) {
      int idx_start = njloc + 1;
      int idx_end = njvec[j] + njloc;
      int current_time_indices[njvec[j]] = timevar[idx_start:idx_end];
      
      matrix[njvec[j], njvec[j]] Sigmat = (sigmae * cholesky_cor_ar1(phi1, max(current_time_indices)))[current_time_indices, current_time_indices];
      
      y[idx_start:idx_end] ~ multi_normal_cholesky(yhat[idx_start:idx_end], inv_sqrt(uvec[j]) * Sigmat);
      
      njloc += njvec[j];
    }
  }
}
generated quantities {
  real sigma2 = pow(sigmae, 2);
  matrix[q1,q1] Dsqrt = matrix_sqrt(D1);
}

