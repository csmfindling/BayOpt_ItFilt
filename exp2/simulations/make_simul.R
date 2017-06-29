# source code : https://kingaa.github.io/pomp/vignettes/pompjss.pdf
library(pomp)


# number of simulations
n_simulations <- 50

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

T_end <- 100
for (idx_simul in 1:n_simulations) {

	gompertz <- pomp(data = data.frame(time = 1:T_end, Y = NA), times = "time",
	rprocess = discrete.time.sim(step.fun = gompertz.proc.sim, delta.t = 1),
	rmeasure = gompertz.meas.sim, t0 = 0)

	# set parameters
	theta <- c(r = 0.1, K = 1, sigma = 0.1, tau = 0.1, X.0 = 1)

	############# SIMULATION #############
	gompertz <- simulate(gompertz, params = theta)

	# Write CSV in R
	write.table(gompertz@data[1:T_end], file = paste("simulation", idx_simul, ".csv", sep=""),row.names=FALSE, na="",col.names=FALSE, sep=",")

}


