functions{
	matrix vector_to_symmat(vector x, int n) {
	  matrix[n, n] m;
	  int k;
	  k = 1;
	  for (j in 1:n) {
		for (i in 1:j) {
		  m[i, j] = x[k];
		  if (i != j) {
			m[j, i] = m[i, j];
		  }
		  k = k + 1;
		}
	  }
	  return m;
	}
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
     return cholesky_decompose(mat ./ (1 - ar^2)); //
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
}
parameters {
           vector[l] beta;
		   cholesky_factor_corr[q1] Lcorr;// cholesky factor (L_u matrix for D1R)
		   vector<lower=0>[q1] ddsqrt; 
           real<lower=0> sigmae;
		   real<lower=-.9999, upper=.9999> phi1;
           matrix[q1, n] etavec;
}
transformed parameters {
  matrix[q1, n] bvec;
  cov_matrix[q1] D1; // VCV matrix
  D1 = quad_form_diag(multiply_lower_tri_self_transpose(Lcorr), ddsqrt); // quad_form_diag: diag_matrix(ddsqrt) * d1R * diag_matrix(ddsqrt)
  {
    matrix[q1,q1] dL = diag_pre_multiply(ddsqrt, Lcorr);
	for (j in 1:n){
	  bvec[,j] = dL * etavec[,j];
    }
  }
}
model {
  beta ~ normal(0,100);
  sigmae ~ student_t(4,0,5);//cauchy(0, 2.5);//normal(0, 1);
// 
  ddsqrt ~ student_t(4,0,5);//cauchy(0, 2.5);
  Lcorr ~ lkj_corr_cholesky(2.0); // prior for cholesky factor of a correlation matrix
  to_vector(etavec) ~ std_normal();
  phi1 ~ uniform(-1,1);
  {
	vector[N] yhat;
	int njloc = 0;
	for (i in 1:N){
	  yhat[i] = x[i]*beta+row(z,i)*col(bvec,ind[i]);
	}
	for (j in 1:n){
		{
		  //matrix[njvec[j],njvec[j]] Sigmat = sigmae*cholesky_cor_ar1(phi1, njvec[j]);//pow(sigmae,2)*cholesky_cor_ar1(phi1, njvec[j]);
		  matrix[njvec[j],njvec[j]] Sigmat = (sigmae*cholesky_cor_ar1(phi1, max(timevar[(njloc+1):(njvec[j]+njloc)])))[timevar[(njloc+1):(njvec[j]+njloc)],timevar[(njloc+1):(njvec[j]+njloc)]];
		  y[(njloc+1):(njvec[j]+njloc)] ~ multi_normal_cholesky(yhat[(njloc+1):(njvec[j]+njloc)], Sigmat); 
		  //target += multi_norm_lpdf( (y[(njloc+1):(njvec[j]+njloc)]) | (yhat[(njloc+1):(njvec[j]+njloc)]), Sigmat); 
		  njloc += njvec[j];
		}
	}
//  y ~ normal(yhat, sigmae);
  }
}
