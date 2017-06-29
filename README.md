# BayOpt_ItFilt


 ## Gombertz model

All results are obtained from applying the different algorithms to the same setting. We assume an oracle theta, theta^* = (0.1, 0.1, 0.1). 50 simulations of 200 trials are simulated given theta^*. We obtain 50 'observation' arrays of 200 time steps. Each algorithm is applied of every simulation and averaging is done in a second time over these 50 simulations. 


### Exp 1: ###
Comparison of Gompertz model BO and IF. We compare BO and IF on the simulations. Results are :
https://github.com/csmfindling/BayOpt_ItFilt/blob/master/exp1/results/comparison_plots.ipynb

### Exp 2: ### 
Comparison of Gompertz model BO with a refined possibility. To do so, we launch a first BO, with loglikelihoods, then take the 10 best points and launch a second BO with the likelihoods. Results are :
https://github.com/csmfindling/BayOpt_ItFilt/blob/master/exp2/results/refined_BO.ipynb
