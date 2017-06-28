from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import csv
import warnings
import numpy as np
import cPickle as pkl
import sys
utils = importr("pomp")
from scipy.stats import ttest_1samp, wilcoxon
from scipy.stats.mstats import normaltest

def get_Y(path='simulations/simulation1.csv', T = 100):
	warnings.warn('Only one space for csv')
	output  = []
	csvfile = open(path, 'rb')
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for line in spamreader:
		output += line
	output = np.asarray(output, dtype=np.float).ravel()
	assert(len(output)==T)
	return output

def get_lkd(path='res/python_bayesopt1.dat', nb_it = 100):
	max_lkd = -np.inf
	f       = open(path, 'rb')
	for i in range(26):
		f.next()
	line = f.next()
	return max(-np.asarray(line.split('(')[-1].split(')')[0].split(','), dtype=np.float))

def get_params(path='res/python_bayesopt1.dat', nb_it = 100):
	max_lkd = -np.inf
	f       = open(path, 'rb')
	for i in range(27):
		f.next()
	line = f.next()
	p    = np.asarray(line.split('(')[-1].split(')')[0].split(','), dtype=np.float)
	return p[np.arange(0,len(p),3)], p[np.arange(0,len(p),3) + 1], p[np.arange(0,len(p),3) + 2]

def get_traj(path='python_bayesopt1_2.dat', nb_it = 100):
	f       = open(path, 'rb')
	for i in range(26):
		f.next()
	line = f.next()
	return np.asarray([max(-np.asarray(line.split('(')[-1].split(')')[0].split(',')[0:i], dtype=np.float)) for i in range(1 , 2 * nb_it + 1)])

def get_traj_(path='python_bayesopt1_2.dat', nb_it = 100):
	f       = open(path, 'rb')
	for i in range(26):
		f.next()
	line = f.next()
	return -np.asarray(line.split('(')[-1].split(')')[0].split(','), dtype=np.float)


def get_traj_index(path='N_1000/python_bayesopt1_2.dat', nb_it = 100):
	f       = open(path, 'rb')
	for i in range(26):
		f.next()
	line = f.next()
	indexes = []
	for i in range(1, 2 * nb_it + 2):
		l = np.asarray(line.split('(')[-1].split(')')[0].split(',')[0:i], dtype=np.float)
		indexes.append(np.where(l==min(l))[0][0])
	return indexes

string = """
library(pomp)

square <- function(x) {
    return(x^2)
}

pf_r <- function(Y, r = 0.1, sigma = 0.1, tau = 0.1)
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
	theta <- c(r = r, K = 1, sigma = sigma, tau = tau, X.0 = 1)

	############# INFERENCE #############

	 gompertz <- pomp(gompertz, dmeasure = gompertz.meas.dens)

	 # lkd estimator with pf
	 pf           <- pfilter(gompertz, params = theta, Np = 1000)
	 loglik.truth <- logLik(pf)
	 return(loglik.truth)
}

pf_r_5000 <- function(Y, r = 0.1, sigma = 0.1, tau = 0.1)
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
	theta <- c(r = r, K = 1, sigma = sigma, tau = tau, X.0 = 1)

	############# INFERENCE #############

	 gompertz <- pomp(gompertz, dmeasure = gompertz.meas.dens)

	 # lkd estimator with pf
	 pf           <- pfilter(gompertz, params = theta, Np = 5000)
	 loglik.truth <- logLik(pf)
	 return(loglik.truth)
}

res_r <- function(path = "iterated_filtering_41")
{
	load(path)
	return(mif1@loglik)
}

traj_r <- function(path = "iterated_filtering_41")
{
	load(path)
	return(conv.rec(mif1))
}

"""
