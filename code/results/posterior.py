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
from functions import get_Y, get_lkd, get_traj, string, get_params, get_traj_index, get_traj_

######################################################## N = 20 ########################################################

powerpack          = SignatureTranslatedAnonymousPackage(string, "powerpack")
nb_simul           = 50
nb_iterations      = 200
likelihoods        = np.zeros([2, nb_simul])
trajectories_lik   = np.zeros([2, nb_simul, nb_iterations])
trajectories_lik_  = np.zeros([2, nb_simul, nb_iterations])
trajectories_r     = np.zeros([2, nb_simul, nb_iterations])
trajectories_r_    = np.zeros([2, nb_simul, nb_iterations])
trajectories_sig   = np.zeros([2, nb_simul, nb_iterations])
trajectories_tau   = np.zeros([2, nb_simul, nb_iterations])
for idx_simul in range(1, nb_simul + 1):
	print(idx_simul)
	Y    = get_Y(path='../simulations/simulation{0}.csv'.format(idx_simul), T = 100)
	info = pkl.load(open('N_20/python_bayesopt{0}_4.pkl'.format(idx_simul), 'rb'))
	r     = .5 * (10**info[1][0] - 10**0.)/(10**1. - 10**0.)
	sigma = .5 * (10**info[1][1] - 10**0.)/(10**1. - 10**0.)
	tau   = .5 * (10**info[1][2] - 10**0.)/(10**1. - 10**0.)
	trajectories_lik[0, idx_simul - 1] = get_traj('N_20/python_bayesopt{0}_4.dat'.format(idx_simul))
	trajectories_lik_[0, idx_simul - 1]= get_traj_('N_20/python_bayesopt{0}_4.dat'.format(idx_simul))
	trajectories_lik[1, idx_simul - 1] = np.asarray(powerpack.traj_r('N_20/iterated_filtering_4{0}'.format(idx_simul)))[:,0][:-1]
	trajectories_r[1, idx_simul - 1]   = np.exp(np.asarray(powerpack.traj_r('N_20/iterated_filtering_4{0}'.format(idx_simul)))[:,2][:-1])
	trajectories_sig[1, idx_simul - 1] = np.exp(np.asarray(powerpack.traj_r('N_20/iterated_filtering_4{0}'.format(idx_simul)))[:,4][:-1])
	trajectories_tau[1, idx_simul - 1] = np.exp(np.asarray(powerpack.traj_r('N_20/iterated_filtering_4{0}'.format(idx_simul)))[:,5][:-1])
	#indexes                            = np.asarray(get_traj_index('N_20/python_bayesopt{0}_4.dat'.format(idx_simul)))[:-1]
	trajectories_r[0, idx_simul - 1]   = get_params('N_20/python_bayesopt{0}_4.dat'.format(idx_simul))[0]
	trajectories_sig[0, idx_simul - 1] = get_params('N_20/python_bayesopt{0}_4.dat'.format(idx_simul))[1]
	trajectories_tau[0, idx_simul - 1] = get_params('N_20/python_bayesopt{0}_4.dat'.format(idx_simul))[2]
	likelihoods[0, idx_simul - 1]      = powerpack.pf_r(robjects.FloatVector(Y), r, sigma, tau)[0] # -info[0] # get_lkd('python_bayesopt_1{0}.dat'.format(idx_simul)) #
	likelihoods[1, idx_simul - 1]      = powerpack.pf_r(robjects.FloatVector(Y), trajectories_r[1, idx_simul - 1, -1], 
																						trajectories_sig[1, idx_simul - 1, -1], 
																						trajectories_tau[1, idx_simul - 1, -1])[0] #powerpack.res_r(idx_simul)[0] # powerpack.res_r(idx_simul)[0] #

def to_normalized_weights(logWeights):
    b = np.max(logWeights)
    weights = [np.exp(logw - b) for logw in logWeights]
    return weights/sum(weights)

idx_sim  = 1
w        = to_normalized_weights(u)
i        = np.asarray(sorted(range(len(w)), key=lambda k: w[k]), dtype=np.int)
p_select = p[:,i[-10:]]
u_       = u[i[-10:]]
std      = np.sqrt(np.sum(w * (trajectories_r[0, idx_sim] - mean)**2)/np.sum(w))
indexes  = (trajectories_r[0, idx_sim] > (mean - std)) * (trajectories_r[0, idx_sim] < (mean + std))
val, bin = numpy.histogram(trajectories_r[0, idx_sim, indexes], bins=40, weights=w[indexes],density=True)

plt.plot(bin[:-1] + (bin[1:]-bin[:-1])/2, val)

