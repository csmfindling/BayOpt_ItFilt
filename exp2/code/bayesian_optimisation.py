# https://rmcantin.bitbucket.io/html/usemanual.html
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import csv
import warnings
import numpy as np
import cPickle as pkl
import sys
sys.path.append("../../bayesopt/lib/")
sys.path.append("../../bayesopt/python/")
import bayesopt
from bayesoptmodule import BayesOptContinuous
importr("pomp")
from utils.functions import get_Y, get_lkd, get_traj, string, get_params, get_traj_index, get_traj_

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

	def __init__(self, idx_simul, low_b, upp_b, nb_it, nb_ini, refine, load_save_flag, max_loglkd = None):
		if refine == True:
			assert(max_loglkd is not None)
			self.max_loglkd = max_loglkd

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
		self.refine    = refine
		self.idx_simul = idx_simul
		BayesOptContinuous.__init__(self, self.n_dim)
		self.powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")
		self.params    = {}
		self.params['n_iterations']       = nb_it # 200
		self.params['n_init_samples']     = nb_ini # 100
		self.params['n_iter_relearn']     = 1
		self.params['verbose_level']      = 2
		if self.refine:
			self.params['noise']          = .01
			self.params['crit_name']      = "cLCB"
		else:
			self.params['noise']          = 1
			self.params['crit_name']      = "cEI"
		self.params['l_type']             = "L_EMPIRICAL"
		self.params['mean.name']          = "mZero"
		self.params['l_all']              = True
		self.params['sc_type']            = "SC_MTL"
		self.params['surr_name']          = "sGaussianProcessML"
		self.params['load_save_flag']     = load_save_flag
		self.params['force_jump']         = 5
		if load_save_flag == 2 and refine == 1:
			self.params['save_filename']      = "logs/bayesopt_" + str(idx_simul) + "_N_20_refine{0}_tmp.dat".format(refine*1)
			self.params['load_filename']      = "logs/bayesopt_" + str(idx_simul) + "_N_20_refine{0}_tmp.dat".format(refine*1)			
		else:
			self.params['save_filename']      = "logs/bayesopt_" + str(idx_simul) + "_N_20_refine{0}.dat".format(refine*1)
			self.params['load_filename']      = "logs/bayesopt_" + str(idx_simul) + "_N_20_refine{0}.dat".format(refine*1)
		self.params['kernel_name']        = "kMaternARD5"
		self.params['n_inner_iterations'] = 1000 # 500
		self.params['lower_bound']        = low_b
		self.params['upper_bound']        = upp_b
		self.lower_bound                  = self.params['lower_bound']
		self.upper_bound                  = self.params['upper_bound']
		self.ub                           = self.upper_bound
		self.lb                           = self.lower_bound
		self.Y                            = get_Y(path='../simulations/simulation{0}.csv'.format(self.idx_simul), T = 100)
		print('Rough likelihood estimate is  {0}'.format(self.powerpack.pf_r(robjects.FloatVector(self.Y), .1, .1, .1)[0]))

	def evaluateSample(self, Xin):	
		r     = .5 * (10**Xin[0] - 10**0.)/(10**1. - 10**0.)
		sigma = .5 * (10**Xin[1] - 10**0.)/(10**1. - 10**0.)
		tau   = .5 * (10**Xin[2] - 10**0.)/(10**1. - 10**0.)
		print('parameters are : {0}, {1}, {2}'.format(r, sigma, tau))
		if self.refine:
			return -np.exp(self.powerpack.pf_r(robjects.FloatVector(self.Y), r, sigma, tau)[0] - self.max_loglkd)
		else:
			return -self.powerpack.pf_r(robjects.FloatVector(self.Y), r, sigma, tau)[0]

def to_normalized_weights(logWeights):
    b = np.max(logWeights)
    weights = [np.exp(logw - b) for logw in logWeights]
    return weights/sum(weights)

def create_intermediate(path, lkd, string_x, string_y):
	f      = open(path, 'rb') 
	writer = open(path[:-8] + '.dat', 'w')
	for line in f:
		if line.startswith('mY=['):
			line = 'mY=[101]' + line.split(']')[-1].split(')')[0] + ',' + string_y + '\n'
			writer.write(line)
		elif line.startswith('mX=['):
			line = 'mX=[101,3]' + line.split(']')[-1].split(')')[0] + ',' + string_x + '\n'
			writer.write(line)
		elif line.startswith('mCurrentIter'):
			writer.write('mCurrentIter=1\n')
		elif line.startswith('mParameters.n_iterations'):
			writer.write('mParameters.n_iterations=99\n')
		elif line.startswith('mYPrev'):
			writer.write('mYPrev={0}\n'.format(-lkd[-1]))
		elif line.startswith('mParameters.n_init_samples'):
			writer.write('mParameters.n_init_samples=101\n')
		else:
			writer.write(line)
	writer.write('\n')
	writer.close()

def to_string(lkd):
	lkd_str = ''
	for l in lkd:
		lkd_str += str(l) + ','
	return lkd_str

def write_strings(lkd, par):
	string_y = to_string(lkd)[:-1] + ')'
	string_x = ''
	for n in range(par.shape[1]):
		string_x += to_string(par[:,n]) #str(par[:,n])[1:-1].replace('  ',',')[1:] + ','
	string_x = string_x[:-1] + ')'
	return string_x, string_y

if __name__=='__main__':
	import time
	start                     = time.time()
	try:
		index                 = int(sys.argv[1])
	except:
		index                 = 10
	low_b                     = np.array([0., 0., 0.], dtype=np.double)
	upp_b                     = np.array([1., 1., 1.], dtype=np.double) 
	bayesOptParticleFiltering = BayesOptParticleFiltering(index, low_b, upp_b, nb_it=100, nb_ini=100, refine=False, load_save_flag=2)
	print('launching first optimization...')
	mvalue, x_out, error      = bayesOptParticleFiltering.optimize()
	elapsed_time              = time.time() - start
	pkl.dump([mvalue, x_out, error, elapsed_time], open('../results/res/N_20/bayesopt_' + str(index) + '_refine0.pkl', 'wb'))
	print('selecting parameters bound...')
	loglkd                    = get_traj_(bayesOptParticleFiltering.params['save_filename'])
	params                    = np.asarray(get_params(bayesOptParticleFiltering.params['save_filename']))
	w                         = to_normalized_weights(loglkd)
	i                         = np.asarray(sorted(range(len(w)), key=lambda k: w[k]), dtype=np.int)
	p_select                  = params[:,i[-10:]]
	print('saving intermediary results...')
	pkl.dump([loglkd, params, w[i[-10:]], p_select], open('../results/res/N_20/refined_params_' + str(index) + '.pkl', 'wb'))
	high_loglkd               = loglkd[i[-10:]]
	low_b                     = np.min(p_select, axis=1)
	upp_b                     = np.max(p_select, axis=1)
	#string_x, string_y        = write_strings(np.exp(high_loglkd - np.max(high_loglkd)), p_select)
	print('lower bounds found {0}'.format(low_b))
	print('upper bounds found {0}'.format(upp_b))
	print('launching second optimization...')
	bayesOptParticleFiltering = BayesOptParticleFiltering(index, low_b, upp_b, nb_it=100, nb_ini=100, refine=True, load_save_flag=2, max_loglkd=np.max(high_loglkd))
	start                     = time.time()
	mvalue, x_out, error      = bayesOptParticleFiltering.optimize()
	#create_intermediate(bayesOptParticleFiltering.params['save_filename'], np.exp(high_loglkd), string_x, string_y)
	#bayesOptParticleFiltering = BayesOptParticleFiltering(index, low_b, upp_b, nb_it=99, nb_ini=101, refine=True, load_save_flag=3, max_loglkd=np.max(high_loglkd))
	#start                     = time.time()
	#mvalue, x_out, error      = bayesOptParticleFiltering.optimize()
	elapsed_time              = time.time() - start
	print('saving final results...')
	print('ML found after first optimization is {0}'.format(np.max(loglkd)))
	print('ML found after second optimization is {0}'.format(np.log(-mvalue) + np.max(loglkd)))
	pkl.dump([mvalue, x_out, error, elapsed_time], open('../results/res/N_20/bayesopt_' + str(index) + '_refine1.pkl', 'wb'))
