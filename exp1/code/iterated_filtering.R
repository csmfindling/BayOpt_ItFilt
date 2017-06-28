# librairies
library(pomp)

# arguments
args  <- commandArgs(trailingOnly=TRUE)
idx_Y <- args[1]
cat("R iterating filtering simulation index is ", idx_Y, "\n")

# create instance of gombertz
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

# load data
Y <- read.csv(paste("simulations/simulation", idx_Y, ".csv", sep=""), na="", sep=",", header = FALSE)
Y <- Y$V1

# set parameters
theta <- c(r = 0.1, K = 1, sigma = 0.1, tau = 0.1, X.0 = 1)

# add data
gompertz <- pomp(data = data.frame(time = 1:100, Y = Y), params = theta, times = "time",
rprocess = discrete.time.sim(step.fun = gompertz.proc.sim, delta.t = 1),
rmeasure = gompertz.meas.sim, t0 = 0)

############# INFERENCE #############
gompertz <- pomp(gompertz, dmeasure = gompertz.meas.dens)

 # lkd estimator with pf
pf <- pfilter(gompertz, params = theta, Np = 100)
loglik.truth <- logLik(pf)

cat("Rough likelihood estimate is ", loglik.truth, "\n")

# maxLkd in a vague way
theta.guess <- theta.true <- coef(gompertz)
theta.guess[c("r", "K", "sigma")] <- 1.5 * theta.true[c("r", "K", "sigma")]

# iterating filtering
gompertz.log.tf <- function(params, ...) log(params)
gompertz.exp.tf <- function(params, ...) exp(params)
gompertz <- pomp(gompertz, toEstimationScale = gompertz.log.tf,
fromEstimationScale = gompertz.exp.tf)

# estimating r, sigma and tau with iterated filtering
estpars <- c("r", "sigma", "tau")

library("foreach")

# mif1 <- foreach(i = 1:10, .combine = c) {
# theta.guess <- theta.true
# theta.guess[estpars] <- rlnorm(n = length(estpars), 
# 	meanlog = log(theta.guess[estpars]), sdlog = 1)

# mif1 <- mif(gompertz, Nmif = 100, start = theta.guess, transform = TRUE,
# Np = 2000, var.factor = 2, cooling.fraction = 0.7,
# rw.sd = c(r = 0.02, sigma = 0.02, tau = 0.02))
start.time <- Sys.time()
n_par <- 1
mif1 <- foreach(i = 1:n_par, .combine = c) %dopar% {
 theta.guess <- theta.true
 theta.guess[estpars] <- rlnorm(n = length(estpars),
 meanlog = log(theta.guess[estpars]), sdlog = 1)
 mif(gompertz, Nmif = 200, start = theta.guess, transform = TRUE,
 Np = 20, var.factor = 2, cooling.fraction = 0.7, save.params=TRUE,
 rw.sd = c(r = 0.02, sigma = 0.02, tau = 0.02))
}
elapsed_time      <- Sys.time() - start.time
mif1.elapsed_time <- elapsed_time

save(mif1, file = paste("results/iterated_filtering_4", idx_Y, sep=""))

# NOT : filter.mean(mif1). THIS : conv.rec(mif1)



