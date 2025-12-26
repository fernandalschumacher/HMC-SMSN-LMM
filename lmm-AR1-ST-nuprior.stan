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
		   vector<lower=0>[n] uvec;//mixing variable
		   real<lower=2> nu;//degrees of freedom
}
transformed parameters {
  matrix[q1, n] bvec;
  cov_matrix[q1] D1; 
  vector[n] tvec;
  vector[q1] Delta;
  cov_matrix[q1] Gamma;
  D1 = quad_form_diag(multiply_lower_tri_self_transpose(Lcorr), ddsqrt); // quad_form_diag: diag_matrix(ddsqrt) * L * L' * diag_matrix(ddsqrt)
  Delta = matrix_sqrt(D1)*(lambda/sqrt(1+sum(square(lambda))));
  Gamma = D1 - Delta*Delta';
  //
  tvec = -sqrt(nu/pi())*tgamma((nu-1)/2)/tgamma(nu/2) + inv_sqrt(uvec).*t0vec;
  //
  {
    matrix[q1,q1] dL = cholesky_decompose(Gamma);
	for (j in 1:n){
	  bvec[,j] = Delta*tvec[j] + dL*inv_sqrt(uvec[j]) * etavec[,j];
    }
  }
}
model {
  beta ~ normal(0,100);
  sigmae ~ student_t(4,0,5);
  ddsqrt ~ student_t(4,0,5);
  Lcorr ~ lkj_corr_cholesky(2.0); // prior for cholesky factor of a correlation matrix
  to_vector(etavec) ~ std_normal();
  nu ~ gamma(2, .2);//cauchy(0, 2.5);
  //
  lambda ~ normal(0,sdLP);
  //
  uvec ~ gamma(nu/2,nu/2);
  t0vec ~ std_normal();
  {
	vector[N] yhat;
	int njloc = 0;
	for (i in 1:N){
	  yhat[i] = x[i]*beta+row(z,i)*col(bvec,ind[i]);
	}
	for (j in 1:n){
		{
		  matrix[njvec[j],njvec[j]] Sigmat = (sigmae*cholesky_cor_ar1(phi1, max(timevar[(njloc+1):(njvec[j]+njloc)])))[timevar[(njloc+1):(njvec[j]+njloc)],timevar[(njloc+1):(njvec[j]+njloc)]];
		  y[(njloc+1):(njvec[j]+njloc)] ~ multi_normal_cholesky(yhat[(njloc+1):(njvec[j]+njloc)], inv_sqrt(uvec[j])*Sigmat); 
		  njloc += njvec[j];
		}
	}
  }
}
generated quantities {
  real sigma2 = pow(sigmae, 2);
  matrix[q1,q1] Dsqrt = matrix_sqrt(D1);
}

