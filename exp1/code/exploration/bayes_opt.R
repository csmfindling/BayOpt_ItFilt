# source : https://cran.r-project.org/web/packages/rBayesianOptimization/rBayesianOptimization.pdf

library(rBayesianOptimization)

Test_Fun <- function(x, y) {
list(Score = exp(-(x - 2)^2) + exp(-(x - 6)^2/10) + 1/ (x^2 + 1),
Pred = 0)
}

OPT_Res <- BayesianOptimization(Test_Fun,
bounds = list(x = c(1, 3), y=c(0,1)),
init_points = 2, n_iter = 10,
acq = "ei", kappa = 2.576, eps = 0.0,
verbose = TRUE)