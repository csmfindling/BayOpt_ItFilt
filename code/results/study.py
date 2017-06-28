from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import csv
import warnings
import numpy as np
import cPickle as pkl
import sys
from scipy.stats import ttest_1samp, wilcoxon
from scipy.stats.mstats import normaltest
from utils.extract_exp1 import extract_exp1

# folders
paths = ['res/N_20/', 'res/N_100/', 'res/N_1000/']

######################################################## N = 20 ########################################################
dic = extract_exp1(nb_simul=50, nb_iterations=200, path='res/N_20/')

fig=plt.figure(figsize=(17,9))
fig.suptitle("N=20", fontsize=16)
ax = plt.subplot(232)
it_start = 0
ax.plot(dic['traj_lkd_means'][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][0][it_start:] - dic['traj_lkd_err'][0][it_start:], dic['traj_lkd_means'][0][it_start:] + dic['traj_lkd_err'][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_lkd_means'][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][1][it_start:] - dic['traj_lkd_err'][1][it_start:], dic['traj_lkd_means'][1][it_start:] + dic['traj_lkd_err'][1][it_start:], facecolor='red', alpha=.4)
ax.set_title('Lkd trajectories')
ax = plt.subplot(233)
it_start = 100
ax.plot(dic['traj_lkd_means'][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][0][it_start:] - dic['traj_lkd_err'][0][it_start:], dic['traj_lkd_means'][0][it_start:] + dic['traj_lkd_err'][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_lkd_means'][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][1][it_start:] - dic['traj_lkd_err'][1][it_start:], dic['traj_lkd_means'][1][it_start:] + dic['traj_lkd_err'][1][it_start:], facecolor='red', alpha=.4)
ax.set_title('Lkd trajectories')
ax = plt.subplot(231)
ax.bar(0., dic['lkd_means'][0], yerr=dic['lkd_err'][0], width=0.6, color='b', ecolor='k', align='center')
ax.bar(1., dic['lkd_means'][1], yerr=dic['lkd_err'][1],  width=0.6, color='r', ecolor='k', align='center')
label = ['', '', 'baysopt', '', '', '', '', 'itfilt']
ax.set_xticklabels(label)
ax.set_ylim((30,50))
ax.set_title('Maximum likelihoods')
ax = plt.subplot(234)
it_start = 100
ax.plot(dic['traj_params_means'][0][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][0][0][it_start:] - dic['traj_params_err'][0][0][it_start:], dic['traj_params_means'][0][0][it_start:] + dic['traj_params_err'][0][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][0][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][0][1][it_start:] - dic['traj_params_err'][0][1][it_start:], dic['traj_params_means'][0][1][it_start:] + dic['traj_params_err'][0][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories r')
ax = plt.subplot(235)
it_start = 100
ax.plot(dic['traj_params_means'][2][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][2][0][it_start:] - dic['traj_params_err'][2][0][it_start:], dic['traj_params_means'][2][0][it_start:] + dic['traj_params_err'][2][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][2][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][2][1][it_start:] - dic['traj_params_err'][2][1][it_start:], dic['traj_params_means'][2][1][it_start:] + dic['traj_params_err'][2][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories tau')
ax = plt.subplot(236)
it_start = 100
ax.plot(dic['traj_params_means'][1][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][1][0][it_start:] - dic['traj_params_err'][1][0][it_start:], dic['traj_params_means'][1][0][it_start:] + dic['traj_params_err'][1][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][1][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][1][1][it_start:] - dic['traj_params_err'][1][1][it_start:], dic['traj_params_means'][1][1][it_start:] + dic['traj_params_err'][1][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories sigma')


######################################################## N = 100 ########################################################

dic = extract_exp1(nb_simul=50, nb_iterations=200, path='res/N_100/')

fig=plt.figure(figsize=(17,9))
fig.suptitle("N=100", fontsize=16)
ax = plt.subplot(232)
it_start = 0
ax.plot(dic['traj_lkd_means'][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][0][it_start:] - dic['traj_lkd_err'][0][it_start:], dic['traj_lkd_means'][0][it_start:] + dic['traj_lkd_err'][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_lkd_means'][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][1][it_start:] - dic['traj_lkd_err'][1][it_start:], dic['traj_lkd_means'][1][it_start:] + dic['traj_lkd_err'][1][it_start:], facecolor='red', alpha=.4)
ax.set_title('Lkd trajectories')
ax = plt.subplot(233)
it_start = 100
ax.plot(dic['traj_lkd_means'][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][0][it_start:] - dic['traj_lkd_err'][0][it_start:], dic['traj_lkd_means'][0][it_start:] + dic['traj_lkd_err'][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_lkd_means'][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][1][it_start:] - dic['traj_lkd_err'][1][it_start:], dic['traj_lkd_means'][1][it_start:] + dic['traj_lkd_err'][1][it_start:], facecolor='red', alpha=.4)
ax.set_title('Lkd trajectories')
ax = plt.subplot(231)
ax.bar(0., dic['lkd_means'][0], yerr=dic['lkd_err'][0], width=0.6, color='b', ecolor='k', align='center')
ax.bar(1., dic['lkd_means'][1], yerr=dic['lkd_err'][1],  width=0.6, color='r', ecolor='k', align='center')
label = ['', '', 'baysopt', '', '', '', '', 'itfilt']
ax.set_xticklabels(label)
ax.set_ylim((30,50))
ax.set_title('Maximum likelihoods')
ax = plt.subplot(234)
it_start = 100
ax.plot(dic['traj_params_means'][0][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][0][0][it_start:] - dic['traj_params_err'][0][0][it_start:], dic['traj_params_means'][0][0][it_start:] + dic['traj_params_err'][0][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][0][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][0][1][it_start:] - dic['traj_params_err'][0][1][it_start:], dic['traj_params_means'][0][1][it_start:] + dic['traj_params_err'][0][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories r')
ax = plt.subplot(235)
it_start = 100
ax.plot(dic['traj_params_means'][2][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][2][0][it_start:] - dic['traj_params_err'][2][0][it_start:], dic['traj_params_means'][2][0][it_start:] + dic['traj_params_err'][2][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][2][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][2][1][it_start:] - dic['traj_params_err'][2][1][it_start:], dic['traj_params_means'][2][1][it_start:] + dic['traj_params_err'][2][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories tau')
ax = plt.subplot(236)
it_start = 100
ax.plot(dic['traj_params_means'][1][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][1][0][it_start:] - dic['traj_params_err'][1][0][it_start:], dic['traj_params_means'][1][0][it_start:] + dic['traj_params_err'][1][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][1][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][1][1][it_start:] - dic['traj_params_err'][1][1][it_start:], dic['traj_params_means'][1][1][it_start:] + dic['traj_params_err'][1][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories sigma')


######################################################## N = 1000 ########################################################

dic = extract_exp1(nb_simul=50, nb_iterations=200, path='res/N_1000/')

fig=plt.figure(figsize=(17,9))
fig.suptitle("N=1000", fontsize=16)
ax = plt.subplot(232)
it_start = 0
ax.plot(dic['traj_lkd_means'][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][0][it_start:] - dic['traj_lkd_err'][0][it_start:], dic['traj_lkd_means'][0][it_start:] + dic['traj_lkd_err'][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_lkd_means'][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][1][it_start:] - dic['traj_lkd_err'][1][it_start:], dic['traj_lkd_means'][1][it_start:] + dic['traj_lkd_err'][1][it_start:], facecolor='red', alpha=.4)
ax.set_title('Lkd trajectories')
ax = plt.subplot(233)
it_start = 100
ax.plot(dic['traj_lkd_means'][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][0][it_start:] - dic['traj_lkd_err'][0][it_start:], dic['traj_lkd_means'][0][it_start:] + dic['traj_lkd_err'][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_lkd_means'][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_lkd_means'][1][it_start:] - dic['traj_lkd_err'][1][it_start:], dic['traj_lkd_means'][1][it_start:] + dic['traj_lkd_err'][1][it_start:], facecolor='red', alpha=.4)
ax.set_title('Lkd trajectories')
ax = plt.subplot(231)
ax.bar(0., dic['lkd_means'][0], yerr=dic['lkd_err'][0], width=0.6, color='b', ecolor='k', align='center')
ax.bar(1., dic['lkd_means'][1], yerr=dic['lkd_err'][1],  width=0.6, color='r', ecolor='k', align='center')
label = ['', '', 'baysopt', '', '', '', '', 'itfilt']
ax.set_xticklabels(label)
ax.set_ylim((30,50))
ax.set_title('Maximum likelihoods')
ax = plt.subplot(234)
it_start = 100
ax.plot(dic['traj_params_means'][0][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][0][0][it_start:] - dic['traj_params_err'][0][0][it_start:], dic['traj_params_means'][0][0][it_start:] + dic['traj_params_err'][0][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][0][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][0][1][it_start:] - dic['traj_params_err'][0][1][it_start:], dic['traj_params_means'][0][1][it_start:] + dic['traj_params_err'][0][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories r')
ax = plt.subplot(235)
it_start = 100
ax.plot(dic['traj_params_means'][2][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][2][0][it_start:] - dic['traj_params_err'][2][0][it_start:], dic['traj_params_means'][2][0][it_start:] + dic['traj_params_err'][2][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][2][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][2][1][it_start:] - dic['traj_params_err'][2][1][it_start:], dic['traj_params_means'][2][1][it_start:] + dic['traj_params_err'][2][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories tau')
ax = plt.subplot(236)
it_start = 100
ax.plot(dic['traj_params_means'][1][0][it_start:], 'b')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][1][0][it_start:] - dic['traj_params_err'][1][0][it_start:], dic['traj_params_means'][1][0][it_start:] + dic['traj_params_err'][1][0][it_start:], facecolor='blue', alpha=.4)
ax.plot(dic['traj_params_means'][1][1][it_start:], 'r')
ax.fill_between(range(200 - it_start), dic['traj_params_means'][1][1][it_start:] - dic['traj_params_err'][1][1][it_start:], dic['traj_params_means'][1][1][it_start:] + dic['traj_params_err'][1][1][it_start:], facecolor='red', alpha=.4)
ax.plot(plt.gca().get_xlim(), [.1,.1], 'g--', linewidth=2);
ax.set_title('trajectories sigma')
