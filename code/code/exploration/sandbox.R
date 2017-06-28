args = commandArgs(trailingOnly=TRUE)

print(args)


n = 1000

the = matrix(data = NA, nrow = n, ncol = 3, byrow = FALSE, dimnames = NULL) 

for (i in 1:n){
  	theta.guess <- theta.true
	the[i,1:3] <- rlnorm(n = length(estpars),
							meanlog = log(theta.guess[estpars]), sdlog = 1)
}



theta.guess <- theta.true
theta.1 <- rlnorm(n = length(estpars),
	meanlog = log(theta.guess[estpars]), sdlog = 1)