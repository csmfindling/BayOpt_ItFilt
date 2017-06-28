# source code : https://kingaa.github.io/pomp/vignettes/pompjss.pdf
library(pomp)

# create instance of dynamical system
gompertz.proc.sim <- function(x, t, params, delta.t,...) {
eps <- exp(rnorm(n = 1, mean = 0, sd = params["sigma"]))
S <- exp(-params["r"] * delta.t)
setNames(params["K"]^(1 - S) * x["X"]^S * eps, "X")
}

gompertz.meas.sim <- function(x, t, params, ...) {
setNames(rlnorm(n = 1, meanlog = log(x["X"]), sd = params["tau"]), "Y")
}

gompertz.meas.dens <- function(y, x, t, params, log, ...) {
dlnorm(x = y["Y"], meanlog = log(x["X"]), sdlog = params["tau"],
log = log)
}

gompertz <- pomp(data = data.frame(time = 1:100, Y = NA), times = "time",
rprocess = discrete.time.sim(step.fun = gompertz.proc.sim, delta.t = 1),
rmeasure = gompertz.meas.sim, t0 = 0)

# set parameters
theta <- c(r = 0.1, K = 1, sigma = 0.1, tau = 0.1, X.0 = 1)

############# SIMULATION #############

gompertz <- simulate(gompertz, params = theta)

# plot
plot(gompertz, variables = "Y")

############# INFERENCE #############

 gompertz <- pomp(gompertz, dmeasure = gompertz.meas.dens)

 # lkd estimator with pf
 pf <- pfilter(gompertz, params = theta, Np = 1000) # params = coef(gompertz)
 loglik.truth <- logLik(pf)
 loglik.truth

 # maxLkd in a vague way
 theta.guess <- theta.true <- coef(gompertz)
 theta.guess[c("r", "K", "sigma")] <- 1.5 * theta.true[c("r", "K", "sigma")]
 pf <- pfilter(gompertz, params = theta.guess, Np = 1000)
 loglik.guess <- logLik(pf)

# iterating filtering
gompertz.log.tf <- function(params, ...) log(params)
gompertz.exp.tf <- function(params, ...) exp(params)

gompertz <- pomp(gompertz, toEstimationScale = gompertz.log.tf,
fromEstimationScale = gompertz.exp.tf)

# estimating r, sigma and tau with iterated filtering
estpars <- c("r", "sigma", "tau")

library("foreach")

# mif1 <- foreach(i = 1:10, .combine = c) {
theta.guess <- theta.true
theta.guess[estpars] <- rlnorm(n = length(estpars),
	meanlog = log(theta.guess[estpars]), sdlog = 1)

mif1 <- mif(gompertz, Nmif = 200, start = theta.guess, transform = TRUE,
Np = 2000, var.factor = 2, cooling.fraction = 0.7,
rw.sd = c(r = 0.02, sigma = 0.02, tau = 0.02))

# estimating r, sigma and tau with bayesian optimisation
library(rBayesianOptimization)

Test_Fun <- function(r, sigma, tau) {
theta.guess <- coef(gompertz)
theta.guess["r"] <- r
theta.guess["sigma"] <- sigma
theta.guess["tau"] <- tau
list(Score = logLik(pfilter(gompertz, params = theta.guess, Np = 1000)),
Pred = 0)
}

OPT_Res <- BayesianOptimization(Test_Fun,
bounds = list(r = c(0, .5), sigma = c(0, .5), tau = c(0, .5)),
init_points = 50, n_iter = 50, #kernel=list(type = "matern", nu=(2*2+1)/2),
acq = "ei", kappa = 1., eps = 0., verbose = TRUE) # kappa = 2.576

# estimating r, sigma and tau with kalman filter
Y             <- gompertz@data
r             <- theta.true['r']
sigma         <- theta.true['sigma']
tau           <- theta.true['tau']
lkd_kalman    <- kalmanfilter(log(Y), r, sigma, tau)


