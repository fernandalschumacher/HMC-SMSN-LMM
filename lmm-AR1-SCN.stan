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
		   //vector<lower=0>[n] uvec;//mixing variable
		   real<lower=0.0001, upper=.9999> nu1; // probability of contamination
       real<lower=0.0001, upper=.5> nu2; // variance scale factor, up to .5 to ensure contamination has high-variance
}
transformed parameters {
  matrix[q1, n] bvec_base;//without u
  cov_matrix[q1] D1; 
  //vector[n] tvec;
  vector[q1] Delta;
  cov_matrix[q1] Gamma;
  D1 = quad_form_diag(multiply_lower_tri_self_transpose(Lcorr), ddsqrt); // quad_form_diag: diag_matrix(ddsqrt) * L * L' * diag_matrix(ddsqrt)
  Delta = matrix_sqrt(D1)*(lambda/sqrt(1+sum(square(lambda))));
  Gamma = D1 - Delta*Delta';
  //
  real scn_const = sqrt(2.0 / pi()) * ((1 - nu1) + nu1 / sqrt(nu2));
  //real mu_u = 1 - nu1 + nu1 * sqrt(nu2); //k1
  //tvec = -sqrt(2/pi()) * mu_u + inv_sqrt(uvec) .* t0vec;
  //tvec = -sqrt(2/pi()) * mu_u * lambda/sqrt(1+sum(square(lambda))) + inv_sqrt(uvec) .* t0vec;
  //
  {
    matrix[q1,q1] dL = cholesky_decompose(Gamma);
	for (j in 1:n){
	 //bvec[,j] = Delta*tvec[j] + dL*inv_sqrt(uvec[j]) * etavec[,j];
	  bvec_base[,j] = Delta * (-scn_const + t0vec[j]) + dL * etavec[,j];//no u
    }
  }
}
model {
  beta ~ normal(0,100);
  sigmae ~ student_t(4,0,5);
  // 
  ddsqrt ~ student_t(4,0,5);
  Lcorr ~ lkj_corr_cholesky(2.0); // prior for cholesky factor of a correlation matrix
  to_vector(etavec) ~ std_normal();
  //
  nu1 ~ beta(.5, 3); // prior for probability
  nu2 ~ beta(.5, 3); // prior for scale factor
  //
  lambda ~ normal(0,sdLP);
  //
  //uvec ~ gamma(nu/2,nu/2);
  t0vec ~ std_normal();
  {
	int njloc = 0;
	for (j in 1:n) {
      // 1. Calculate the AR1 Correlation Matrix for this subject
      int current_indices[njvec[j]];
      for(k in 1:njvec[j]) current_indices[k] = njloc + k;
      
      matrix[njvec[j], njvec[j]] R = cholesky_cor_ar1(phi1, max(timevar[current_indices]));
      matrix[njvec[j], njvec[j]] Sig_chol = sigmae * R[timevar[current_indices], timevar[current_indices]];
      
      // 2. Calculate the mean (yhat) for this subject
      vector[njvec[j]] subject_y = y[current_indices];
      vector[njvec[j]] subject_yhat;
      for (k in 1:njvec[j]) {
          int row_idx = current_indices[k];
          subject_yhat[k] = x[row_idx] * beta + row(z, row_idx) * col(bvec_base, j);
      }

      // 3. MARGINALIZED LIKELIHOOD
      // Log-probability of State 1: u = 1 (Normal)
      real lp1 = log1m(nu1) + multi_normal_cholesky_lpdf(subject_y | subject_yhat, Sig_chol);
      
      // Log-probability of State 2: u = nu2 (Contaminated)
      vector[njvec[j]] subject_yhat_contam = subject_yhat * inv_sqrt(nu2); //so that b is also contaminated
      real lp2 = log(nu1) + multi_normal_cholesky_lpdf(subject_y | subject_yhat_contam, inv_sqrt(nu2) * Sig_chol);
      
      target += log_sum_exp(lp1, lp2);

      njloc += njvec[j];
    }
  }
}
generated quantities {
  real sigma2 = pow(sigmae, 2);
  matrix[q1,q1] Dsqrt = matrix_sqrt(D1);
  vector[n] prob_contaminated;
  {
    int njloc_gq = 0; // MUST initialize here
    for (j in 1:n) {
      int current_indices[njvec[j]];
      for(k in 1:njvec[j]) current_indices[k] = njloc_gq + k;
      
      matrix[njvec[j], njvec[j]] R = cholesky_cor_ar1(phi1, max(timevar[current_indices]));
      matrix[njvec[j], njvec[j]] Sig_chol = sigmae * R[timevar[current_indices], timevar[current_indices]];
      
      vector[njvec[j]] subject_y = y[current_indices];
      vector[njvec[j]] subject_yhat;
      for (k in 1:njvec[j]) {
          int row_idx = current_indices[k];
          subject_yhat[k] = x[row_idx] * beta + row(z, row_idx) * col(bvec_base, j);
      }

      real lp1 = log1m(nu1) + multi_normal_cholesky_lpdf(subject_y | subject_yhat, Sig_chol);
      vector[njvec[j]] subject_yhat_contam = subject_yhat * inv_sqrt(nu2); 
      real lp2 = log(nu1) + multi_normal_cholesky_lpdf(subject_y | subject_yhat_contam, inv_sqrt(nu2) * Sig_chol);
      
      prob_contaminated[j] = exp(lp2 - log_sum_exp(lp1, lp2)); 
      
      njloc_gq += njvec[j];
    }
  }
}

