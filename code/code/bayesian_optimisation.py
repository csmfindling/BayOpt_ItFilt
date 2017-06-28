# https://rmcantin.bitbucket.io/html/usemanual.html
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import csv
import warnings
import numpy as np
import cPickle as pkl
import sys
sys.path.append("../bayesopt/lib/")
sys.path.append("../bayesopt/python/")
import bayesopt
from bayesoptmodule import BayesOptContinuous
utils = importr("pomp")

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

class BayesOptParticleFiltering(BayesOptContinuous):

	def __init__(self, idx_simul, low_b, upp_b):
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
			 pf           <- pfilter(gompertz, params = theta, Np = 20)
			 loglik.truth <- logLik(pf)
			 return(loglik.truth)
		}
		"""
		self.n_dim     = 3
		self.idx_simul = idx_simul
		BayesOptContinuous.__init__(self, self.n_dim)
		self.powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")
		self.params    = {}
		self.params['n_iterations']       = 100 # 200
		self.params['n_init_samples']     = 100 # 100
		self.params['n_iter_relearn']     = 1
		self.params['verbose_level']      = 2
		self.params['noise']              = 1.
		self.params['l_type']             = "L_EMPIRICAL"
		self.params['mean.name']          = "mZero"
		self.params['l_all']              = True
		self.params['sc_type']            = "SC_MTL"
		self.params['surr_name']          = "sGaussianProcessML"
		self.params['load_save_flag']     = 2
		self.params['force_jump']         = 20
		self.params['save_filename']      = "results/python_bayesopt" + str(idx_simul) + "_4.dat"
		self.params['load_filename']      = "results/python_bayesopt" + str(idx_simul) + "_4.dat"
		self.params['kernel_name']        = "kMaternARD5"
		self.params['n_inner_iterations'] = 1000 # 500
		self.params['lower_bound']        = low_b
		self.params['upper_bound']        = upp_b
		self.lower_bound                  = self.params['lower_bound']
		self.upper_bound                  = self.params['upper_bound']
		self.ub                           = self.upper_bound
		self.lb                           = self.lower_bound
		self.Y                            = get_Y(path='simulations/simulation{0}.csv'.format(self.idx_simul), T = 100)
		print('Rough likelihood estimate is  {0}'.format(self.powerpack.pf_r(robjects.FloatVector(self.Y), .1, .1, .1)[0]))

	def evaluateSample(self, Xin):	
		r     = .5 * (10**Xin[0] - 10**0.)/(10**1. - 10**0.)
		sigma = .5 * (10**Xin[1] - 10**0.)/(10**1. - 10**0.)
		tau   = .5 * (10**Xin[2] - 10**0.)/(10**1. - 10**0.)
		print('parameters are : {0}, {1}, {2}'.format(r, sigma, tau))
		return -self.powerpack.pf_r(robjects.FloatVector(self.Y), r, sigma, tau)[0]

if __name__=='__main__':
	import time
	start                     = time.time()
	index                     = int(sys.argv[1])
	low_b                     = np.array([0., 0., 0.], dtype=np.double)
	upp_b                     = np.array([1., 1., 1.], dtype=np.double) 
	bayesOptParticleFiltering = BayesOptParticleFiltering(index, low_b, upp_b)
	mvalue, x_out, error      = bayesOptParticleFiltering.optimize()
	elapsed_time              = time.time() - start
	pkl.dump([mvalue, x_out, error, elapsed_time], open('results/python_bayesopt' + str(index) + '_4.pkl', 'wb'))


# powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")
 # Y   = get_Y()
 # l   = []
 # for i in range(100):
 # 	l += [bayesOptParticleFiltering.powerpack.pf_r(robjects.FloatVector(Y), .45, .45, .45)[0]]



