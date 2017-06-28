# source : http://perso.telecom-paristech.fr/~blanchet/UE_SI342/misc/initTRs.pdf


kalmanfilter <- function(logy, r, sigma, tau) 
{
T              <- length(logy)
loglkd         <- 0
logx           <- matrix(nrow=1, ncol=T)
logx[1]        <- 0
covariances    <- matrix(nrow=1, ncol=T)
covariances[1] <- 1
loglkd         <- loglkd + log(dnorm(logy[1], 0, tau))

for (t_idx in 2:T) 
	{
		pred               <- exp(-r)*logx[t_idx - 1]
		cov                <- covariances[t_idx - 1] * exp(-2*r) + sigma^2
		innov              <- logy[t_idx] - pred
		gain               <- cov / (cov + tau^2)
		logx[t_idx]        <- pred + gain * innov
		covariances[t_idx] <- (1 - gain) * cov
		loglkd             <- loglkd + log(dnorm(logy[t_idx], pred, sqrt(cov + tau^2)))
	}
return(loglkd)
}
