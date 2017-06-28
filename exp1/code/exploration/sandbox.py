from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import csv
import warnings

utils = importr("pomp")


string = """
library(pomp)

square <- function(x) {
    return(x^2)
}

pf_r <- function(Y)
{
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

	gompertz <- pomp(data = data.frame(time = 1:100, Y = Y), times = "time",
	rprocess = discrete.time.sim(step.fun = gompertz.proc.sim, delta.t = 1),
	rmeasure = gompertz.meas.sim, t0 = 0)

	# set parameters
	theta <- c(r = 0.1, K = 1, sigma = 0.1, tau = 0.1, X.0 = 1)

	############# INFERENCE #############

	 gompertz <- pomp(gompertz, dmeasure = gompertz.meas.dens)

	 # lkd estimator with pf
	 pf           <- pfilter(gompertz, params = theta, Np = 1000)
	 loglik.truth <- logLik(pf)
	 return(loglik.truth)
}
"""

def get_Y(path='arguments.csv', T = 100):
	warnings.warn('Only one space for csv')
	output  = []
	csvfile = open(path, 'rb')
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	spamreader.next()
	for line in spamreader:
		output += line
	output = np.asarray(output, dtype=np.float).ravel()
	assert(len(output)==T)
	return output

powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")

Y   = get_Y()
lkd = powerpack.pf_r(robjects.FloatVector(Y))[0]
# rsum = robjects.r['square.R']
# r = robjects.r
# r.source()[0][0]
