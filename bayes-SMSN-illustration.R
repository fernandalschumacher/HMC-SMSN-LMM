#https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started#installation-of-rstan
#setwd("C:\\Users\\schumacher.313\\Documents\\GitHub\\bayes-SMSN-LMM")
#library(skewlmm)
#rsmsn.lmm
library(dplyr)
library(purrr)
library(reshape2)
library(ggplot2)
nj1 <- 5
m <- 50
D1 <- matrix(c(.5,.1,.1,1),ncol=2)
set.seed(95955)
gendatList <- map(rep(nj1, m), function(nj) skewlmm::rsmsn.lmm(1:nj, cbind(1, 1:nj), cbind(1, 1:nj), #rep(1, nj),
                                                      sigma2 = .25, D1 = D1, 
                                                      beta = c(1, 2), lambda = c(1,3),
                                                      depStruct = "ARp", phi = -.3, 
                                                      distr = 'st', nu = 6))
gendat <- bind_rows(gendatList, .id = "ind")
ggplot(gendat, aes(x = x, y = y, group = ind)) + geom_line() +
  stat_summary(aes(group = 1), geom = "line", fun = mean, col = "blue", size = 2)
#
#gendat <- subset(gendat,!(ind%in%c("2","4":"10")&time==4))


######
##load libraries
library(rstan, quietly = T)
library(shinystan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

N <- nrow(gendat)
n <- n_distinct(gendat$ind)
x <- model.matrix(y~x, data=gendat)
z <- model.matrix(~x, data=gendat)
l <- ncol(x)
q1<- ncol(z); q2 <- q1*(q1+1)/2
gendat$indnum <- as.numeric(gendat$ind)
gendat <- gendat[order(gendat$indnum),]
njvec <- as.numeric(table(gendat$indnum))

# ST-AR(1)
tbayes <- try(stan(file='lmm-AR1-ST.stan', 
                   data = list(N=N, n=n, l=l, q1 = q1, njvec =njvec,y=gendat$y,
                               x=x,z=z, timevar = gendat$time,ind = gendat$indnum,
                               sdLP = 2), 
                   thin = 1, chains = 3, iter = 5000, warmup = 1000, 
                   seed = 9955, control = list(adapt_delta=.9)),
              silent = F)
print(tbayes,par=c("beta","sigmae","phi1","D1","lambda","nu","lp__"), 
      probs = c(.025,.975), digits=3)

library(HDInterval)
tsample <- extract(tbayes, pars = c("beta[1]","beta[2]","sigmae","phi1","D1[1,1]",
                                    "D1[1,2]","D1[2,2]","lambda[1]","lambda[2]","nu"))
#tsample %>% glimpse()
HDI<- hdi(tsample) %>% unlist()
cbind(HDI[c(TRUE, FALSE)],HDI[c(FALSE, TRUE)])

##
library(skewlmm)
fit_EM <- smsn.lmm(data = gendat, formFixed = y~x, groupVar = 'ind', 
                  formRandom = ~x, depStruct = "ARp", timeVar = 'time',
                  distr = 'st', control = lmmControl(showCriterium = T))
cbind(fit_EM$theta,fit_EM$std.error) %>% knitr::kable(digits = 3, format = 'simple')
round(fit_EM$estimates$D,3)
round(fit_EM$estimates$sigma2 %>% sqrt,3)

##### extra options for stan
postsig <- rstan::extract(fit_stan, pars=c("sigmae","d0"))
ref <- melt(postsig)
ggplot(data=ref,aes(x=value))+geom_density()+facet_wrap(~L1,scales="free")

postbeta <- rstan::extract(fit_stan, pars=c("beta")) %>% as.data.frame()
ref <- melt(postbeta)
ggplot(data=ref,aes(x=value))+geom_density()+facet_wrap(~variable,scales="free")

pairs(fit_stan, pars = c("D1","sigmae","lp__"))
#launch_shinystan(fit_stan)
