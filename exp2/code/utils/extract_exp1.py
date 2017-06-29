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
from functions import get_Y, get_lkd, get_traj, string, get_params, get_traj_index


# parameters
powerpack          = SignatureTranslatedAnonymousPackage(string, "powerpack")
nb_simul           = 50
nb_iterations      = 200

# folders
paths = ['res/N_20/', 'res/N_100/', 'res/N_1000/']


def extract_exp1(nb_simul=50, nb_iterations=200, path='res/N_20/'):
	likelihoods        = np.zeros([2, nb_simul])
	trajectories_lik   = np.zeros([2, nb_simul, nb_iterations])
	trajectories_r     = np.zeros([2, nb_simul, nb_iterations])
	trajectories_sig   = np.zeros([2, nb_simul, nb_iterations])
	trajectories_tau   = np.zeros([2, nb_simul, nb_iterations])
	for idx_simul in range(1, nb_simul + 1):
		#print(idx_simul)

		# bayesian optimisation
		Y    = get_Y(path='../simulations/simulation{0}.csv'.format(idx_simul), T = 100)
		info = pkl.load(open(path + 'python_bayesopt{0}.pkl'.format(idx_simul), 'rb'))
		r     = .5 * (10**info[1][0] - 10**0.)/(10**1. - 10**0.)
		sigma = .5 * (10**info[1][1] - 10**0.)/(10**1. - 10**0.)
		tau   = .5 * (10**info[1][2] - 10**0.)/(10**1. - 10**0.)
		trajectories_lik[0, idx_simul - 1] = get_traj(path + 'python_bayesopt{0}.dat'.format(idx_simul))
		indexes                            = np.asarray(get_traj_index(path + 'python_bayesopt{0}.dat'.format(idx_simul)))[:-1]
		parameters                         = get_params(path + 'python_bayesopt{0}.dat'.format(idx_simul))
		trajectories_r[0, idx_simul - 1]   = .5 * (10**parameters[0][indexes] - 10**0)/(10**1. - 10**0.)
		trajectories_sig[0, idx_simul - 1] = .5 * (10**parameters[1][indexes] - 10**0)/(10**1. - 10**0.)
		trajectories_tau[0, idx_simul - 1] = .5 * (10**parameters[2][indexes] - 10**0)/(10**1. - 10**0.)
		likelihoods[0, idx_simul - 1]      = powerpack.pf_r(robjects.FloatVector(Y), r, sigma, tau)[0] 


		# iterated filtering
		information_if                     = np.asarray(powerpack.traj_r(path + 'iterated_filtering_{0}'.format(idx_simul)))
		trajectories_lik[1, idx_simul - 1] = information_if[:,0][:-1]
		trajectories_r[1, idx_simul - 1]   = np.exp(information_if[:,2][:-1])
		trajectories_sig[1, idx_simul - 1] = np.exp(information_if[:,4][:-1])
		trajectories_tau[1, idx_simul - 1] = np.exp(information_if[:,5][:-1])
		r_, sig_, tau_                     = trajectories_r[1, idx_simul - 1, -1], trajectories_sig[1, idx_simul - 1, -1], trajectories_tau[1, idx_simul - 1, -1]
		likelihoods[1, idx_simul - 1]      = powerpack.pf_r(robjects.FloatVector(Y), r_, sig_, tau_)[0] 

	traj_lkd_means      = np.mean(trajectories_lik, axis=1)
	traj_lkd_err        = np.std(trajectories_lik, axis=1)/np.sqrt(nb_simul)
	lkd_means           = np.mean(likelihoods, axis=1)
	lkd_err             = np.std(likelihoods, axis=1)/np.sqrt(nb_simul)

	traj_params_means    = np.zeros([3, 2, nb_iterations])
	traj_params_means[0] = np.mean(trajectories_r, axis=1)
	traj_params_means[1] = np.mean(trajectories_sig, axis=1)
	traj_params_means[2] = np.mean(trajectories_tau, axis=1)

	traj_params_err    = np.zeros([3, 2, nb_iterations])
	traj_params_err[0] = np.std(trajectories_r, axis=1)/np.sqrt(nb_simul)
	traj_params_err[1] = np.std(trajectories_sig, axis=1)/np.sqrt(nb_simul)
	traj_params_err[2] = np.std(trajectories_tau, axis=1)/np.sqrt(nb_simul)

	return {'traj_lkd_means':traj_lkd_means, 'traj_lkd_err':traj_lkd_err, 'lkd_means':lkd_means, 'lkd_err':lkd_err, 'traj_params_means':traj_params_means, 'traj_params_err':traj_params_err}


