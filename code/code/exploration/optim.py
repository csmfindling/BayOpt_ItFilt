import numpy as np
import cPickle as pkl
import sys
sys.path.append("../bayesopt/lib/")
sys.path.append("../bayesopt/python/")
import bayesopt
from bayesoptmodule import BayesOptContinuous
from scipy.stats import norm

class BayesOptProbe(BayesOptContinuous):

	def __init__(self, path=):
		if cat_optim < 2:
			self.n_dim = 1
		elif cat_optim < 3:
			self.n_dim = 3
		else:
			self.n_dim = 2
		BayesOptContinuous.__init__(self, self.n_dim)
		self.info_open     = pkl.load(open('../data/open_data_case_' + str(subj_idx) + '.pkl', 'rb'))
		self.info_recc     = pkl.load(open('../data/recurrent_data_case_' + str(subj_idx) + '.pkl', 'rb'))
		self.cat_optim     = cat_optim
		self.params   = {}
		if cat_optim < 2:
			self.params['n_iterations']       = 20
			self.params['n_init_samples']     = 10
			self.params['n_iter_relearn']     = 2
		else:
			self.params['n_iterations']       = 200
			self.params['n_init_samples']     = 50
			self.params['n_iter_relearn']     = 10
		self.params['verbose_level']      = 2
		self.params['noise']              = 5
		self.params['l_type']             = "L_EMPIRICAL"
		self.params['mean.name']          = "mZero"
		self.params['l_all']              = True
		self.params['sc_type']            = "SC_MTL"
		self.params['surr_name']          = "sGaussianProcessML"
		self.params['load_save_flag']     = 3
		self.params['force_jump']         = 0
		self.params['save_filename']      = "results_/fit_" + str(subj_idx) + "_" + str(cat_names[cat_optim]) + "_bayesopt_new5.dat"
		self.params['load_filename']      = "results_/fit_" + str(subj_idx) + "_" + str(cat_names[cat_optim]) + "_bayesopt_new5.dat"
		self.params['kernel_name']        = "kMaternARD5"
		self.params['n_inner_iterations'] = 500
		self.params['lower_bound']        = low_b
		self.params['upper_bound']        = upp_b
		self.lower_bound                  = self.params['lower_bound']
		self.upper_bound                  = self.params['upper_bound']
		self.ub                           = self.upper_bound
		self.lb                           = self.lower_bound

	def evaluateSample(self, Xin):
		nS           = 3
		nC           = 0
		nA           = 4
		nbSamples    = 500
		mon_size     = 3
		alpha        = norm.ppf(Xin[0])
		beta_softmax = np.exp(alpha)
		if self.cat_optim == 0:
			res = - bootstrap_cstvol.SMC2(self.info_open, beta_softmax=beta_softmax, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples) - bootstrap_cstvol.SMC2(self.info_recc, beta_softmax=beta_softmax, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples)
		elif self.cat_optim == 1:
			res = - bootstrap_varvol.SMC2(self.info_open, beta_softmax=beta_softmax, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples) - bootstrap_varvol.SMC2(self.info_recc, beta_softmax=beta_softmax, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples)
		elif self.cat_optim == 2:
			lambda_noise = .5*(10**Xin[1] - 10**0.)/(10**1. - 10**0.)
			eta_noise    = 0.19*(10**Xin[2] - 10**0.)/(10**1. - 10**0.) + .01
			res          = - bootstrap_noisy.SMC2(self.info_open, beta_softmax=beta_softmax, lambda_noise=lambda_noise, eta_noise=eta_noise, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples) - bootstrap_noisy.SMC2(self.info_recc, beta_softmax=beta_softmax, lambda_noise=lambda_noise, eta_noise=eta_noise, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples)
		elif self.cat_optim == 3:
			lambda_noise = .5*(10**Xin[1] - 10**.0)/(10**1. - 10**0.)
			eta_noise    = 0.
			res          = - bootstrap_noisy.SMC2(self.info_open, beta_softmax=beta_softmax, lambda_noise=lambda_noise, eta_noise=eta_noise, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples) - bootstrap_noisy.SMC2(self.info_recc, beta_softmax=beta_softmax, lambda_noise=lambda_noise, eta_noise=eta_noise, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples)
		elif self.cat_optim == 4:
			lambda_noise = 0.
			eta_noise    = 0.19*(10**Xin[1] - 10**0.)/(10**1 - 10**0.) + .01
			res          = - bootstrap_noisy.SMC2(self.info_open, beta_softmax=beta_softmax, lambda_noise=lambda_noise, eta_noise=eta_noise, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples) - bootstrap_noisy.SMC2(self.info_recc, beta_softmax=beta_softmax, lambda_noise=lambda_noise, eta_noise=eta_noise, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=nbSamples)
		else:
			raise SyntaxError 			
		return res

if __name__=='__main__':
	index                = int(sys.argv[1]) - 1
	nb_subj              = 62
	cat_optim            = index/nb_subj
	cat_names            = ["cstvol_forward", "varvol_forward", "noise_forward_both", "noise_forward_weber", "noise_forward_const"]
	if cat_optim < 2:
		low_b            = np.array([0.1], dtype=np.double)
		upp_b            = np.array([.99], dtype=np.double)
		print('volatility model')
	elif cat_optim < 3:
		low_b            = np.array([.1, 0., 0.], dtype=np.double)
		upp_b            = np.array([.9, 1., 1.], dtype=np.double) 
		print('double noisy model')
	else:
		low_b            = np.array([.1, 0.], dtype=np.double)
		upp_b            = np.array([.99, 1.], dtype=np.double) 
		print('single noisy model')
	index                = index%nb_subj
	bayesOptProbe        = BayesOptProbe(index, cat_optim, low_b, upp_b, cat_names)
	mvalue, x_out, error = bayesOptProbe.optimize()
	pkl.dump([mvalue, x_out, error], open('results_/fit_' + str(index) + '_' + str(cat_names[cat_optim]) +'.pkl', 'wb'))






