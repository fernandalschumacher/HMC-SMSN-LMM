# Code from https://github.com/fernandalschumacher/HMC-SMSN-LMM

# For information on installing Stan, please check
# https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started#installation-of-rstan
requireNamespace("skewlmm") ## skewlmm will be used for data generation
# Loading required packages
library(dplyr)
library(purrr)
library(reshape2)
library(ggplot2)
library(tictoc)
library(loo)
library(rstan, quietly = T)
library(shinystan)
library(HDInterval)
library(bayesplot)
library(tidyverse)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Generating a data set for this illustration - True model is AR(1)-ST-LMM
nj1 <- 5 #number of observations per subject
m <- 200 #number of subjects
D1 <- matrix(c(.5,.1,.1,1),ncol=2) #variance of random effects
set.seed(955)
gendatList <- map(rep(nj1, m), function(nj) skewlmm::rsmsn.lmm(1:nj, cbind(1, 1:nj), cbind(1, 1:nj), 
                                                               sigma2 = .25, D1 = D1, 
                                                               beta = c(1, 2), lambda = c(1,3),
                                                               depStruct = "ARp", phi = .5, 
                                                               distr = 'st', nu = 6))
gendat <- bind_rows(gendatList, .id = "ind")
# Plotting the simulated data set
ggplot(gendat, aes(x = x, y = y, group = ind)) + geom_line() +
  stat_summary(aes(group = 1), geom = "line", fun = mean, col = "blue", size = 2) + 
  theme_bw()

# Setup for running stan
N <- nrow(gendat)
n <- n_distinct(gendat$ind)
x <- model.matrix(y~x, data=gendat)
z <- model.matrix(~x, data=gendat)
l <- ncol(x)
q1<- ncol(z); q2 <- q1*(q1+1)/2
gendat$indnum <- as.numeric(gendat$ind)
gendat <- gendat[order(gendat$indnum),]
njvec <- as.numeric(table(gendat$indnum))
data_list <- list(y = gendat$y, x= x, z = z,
                  timevar = gendat$time, indnum  = gendat$indnum)

# Fitting ST-AR(1)
tic()
tbayes <- stan(file='lmm-AR1-ST.stan', 
               data = list(N=N, n=n, l=l, q1 = q1, njvec =njvec,y=gendat$y,
                           x=x,z=z, timevar = gendat$time, ind = gendat$indnum,
                           sdLP = 2), #sd of lambda's prior 
               thin = 5, chains = 3, iter = 5000, warmup = 1000, 
               seed = 9955, control = list(adapt_delta=.95))
toc()
print(tbayes,par=c("beta","sigmae","phi1","D1","lambda","nu"), 
      probs = c(.025,.975), digits=3)

# Checking acceptance probability
sampler_params <- get_sampler_params(tbayes, inc_warmup = FALSE)
mean(unlist(lapply(sampler_params, function(x) x[, "accept_stat__"])))

# Computing HDI
tsample <- extract(tbayes, pars = c("beta[1]","beta[2]","sigmae","phi1","D1[1,1]",
                                    "D1[1,2]","D1[2,2]","lambda[1]","lambda[2]","nu"))
HDI<- HDInterval::hdi(tsample) %>% unlist()
data.frame(lower = HDI[c(TRUE, FALSE)],upper = HDI[c(FALSE, TRUE)]) %>% knitr::kable()

# Looking at posterior distributions
bayesplot::color_scheme_set("red")
bayesplot::mcmc_rank_hist(tbayes, pars=c("beta[1]","beta[2]","phi1","lambda[1]","lambda[2]","nu"))
bayesplot::color_scheme_set("blue")
bayesplot::mcmc_dens_chains(tbayes, pars = c("beta[1]","beta[2]","phi1","lambda[1]","lambda[2]","nu")) + 
  theme_bw()

# Launch shinystan for further exploration
#launch_shinystan(tbayes)

# Loading auxiliary function to compute the model's information criteria
source("auxfunctions-bayes.R")

# Computing criteria
tic()
st_loo <- criteriaAR1(tbayes, data_list, distr = "st")
toc()
print(st_loo$loo)
plot(st_loo$loo)
print(st_loo$waic)

###############################################################################
# Fitting N-AR(1) for comparison
tic()
nbayes <- stan(file='lmm-AR1-N.stan', 
               data = list(N=N, n=n, l=l, q1 = q1, njvec =njvec,y=gendat$y,
                           x=x,z=z, timevar = gendat$time,ind = gendat$indnum), 
               thin = 5, chains = 3, iter = 5000, warmup = 1000, 
               seed = 9955, control = list(adapt_delta=.9))
toc()
print(nbayes,par=c("beta","sigmae","phi1","D1"), 
      probs = c(.025,.975), digits=3)

tic()
normal_loo <- criteriaAR1(nbayes, data_list, distr = "sn")
toc()
plot(normal_loo$loo)

# Comparing criteria for both models
comp_loo <- loo_compare(list(st = st_loo$loo, norm = normal_loo$loo))
print(comp_loo, simplify = FALSE)
#
comp_waic <- loo_compare(list(st = st_loo$waic, norm = normal_loo$waic))
print(comp_waic, simplify = FALSE)

###############################################################################
# Fitting SCN-AR(1) for comparison
tic()
cnbayes <- stan(file='lmm-AR1-SCN.stan', 
               data = list(N=N, n=n, l=l, q1 = q1, njvec =njvec,y=gendat$y,
                           x=x,z=z, timevar = gendat$time, ind = gendat$indnum,
                           sdLP = 2), #sd of lambda's prior 
               thin = 5, chains = 3, iter = 5000, warmup = 1000, 
               seed = 9955, control = list(adapt_delta=.98))
toc()
print(cnbayes,par=c("beta","sigmae","phi1","D1","lambda","nu1","nu2"), 
      probs = c(.025,.975), digits=3)

tic()
scn_loo <- criteriaAR1(cnbayes, data_list, distr = "scn")
toc()
plot(scn_loo$loo)

###############################################################################
# Fitting SSL-AR(1) for comparison
tic()
slbayes <- stan(file='lmm-AR1-SSL.stan', 
                data = list(N=N, n=n, l=l, q1 = q1, njvec =njvec,y=gendat$y,
                            x=x,z=z, timevar = gendat$time, ind = gendat$indnum,
                            sdLP = 2), #sd of lambda's prior 
                thin = 5, chains = 3, iter = 5000, warmup = 1000, 
                seed = 9955, control = list(adapt_delta=.95))
toc()
print(slbayes,par=c("beta","sigmae","phi1","D1","lambda","nu"), 
      probs = c(.025,.975), digits=3)

tic()
ssl_loo <- criteriaAR1(slbayes, data_list, distr = "ssl")
toc()
plot(ssl_loo$loo)

###############################################################################
# Comparing all models
comp_loo <- loo_compare(list(norm = normal_loo$loo,
                             st = st_loo$loo,
                             scn = scn_loo$loo,
                             ssl = ssl_loo$loo))
print(comp_loo, simplify = FALSE)
#
comp_waic <- loo_compare(list(norm = normal_loo$waic,
                              st = st_loo$waic, 
                              scn = scn_loo$waic,
                              ssl = ssl_loo$waic))
print(comp_waic, simplify = FALSE)

###############################################################################
## Sensitivity analysis: changing prior distribution for nu
tic()
tbayes_nu <- stan(file='lmm-AR1-ST-nuprior.stan', 
                  data = list(N=N, n=n, l=l, q1 = q1, njvec =njvec,y=gendat$y,
                              x=x,z=z, timevar = gendat$time, ind = gendat$indnum,
                              sdLP = 2), #sd of lambda's prior 
                  thin = 5, chains = 3, iter = 5000, warmup = 1000, 
                  seed = 9955, control = list(adapt_delta=.95))
toc()
print(tbayes_nu,par=c("beta","sigmae","phi1","D1","lambda","nu"), 
      probs = c(.025,.975), digits=3)

## Comparing with original model
posterior1 <- posterior::as_draws_matrix(tbayes)       
posterior2 <- posterior::as_draws_matrix(tbayes_nu)    

library(posterior)

d1 <- as_draws_df(posterior1) |>
  dplyr::select(nu, starts_with("beta")) |>
  pivot_longer(cols = everything(), names_to = "param",values_to="values") |>
  dplyr::mutate(Model = "Cauchy Prior")

d2 <- as_draws_df(posterior2) |>
  dplyr::select(nu, starts_with("beta")) |>
  pivot_longer(cols = everything(), names_to = "param",values_to="values") |>
  dplyr::mutate(Model = "Gamma Prior")

d <- dplyr::bind_rows(d1, d2)

ggplot(d, aes(x = values, fill = Model)) +
  geom_density(alpha = 0.4) +
  ggtitle("Sensitivity of nu to Prior Choice") +
  theme_minimal() + facet_wrap(~param, scale = "free")


###############################################################################
# Fitting a frequentist AR(1)-ST-LMM for comparison
library(skewlmm)
fit_EM <- smsn.lmm(data = gendat, formFixed = y~x, groupVar = 'ind', 
                   formRandom = ~x, depStruct = "ARp", timeVar = 'time',
                   distr = 'st', control = lmmControl(showCriterium = T))
cbind(fit_EM$theta,fit_EM$std.error) %>% knitr::kable(digits = 3, format = 'simple')
round(fit_EM$estimates$D,3)
round(fit_EM$estimates$sigma2 %>% sqrt,3)

