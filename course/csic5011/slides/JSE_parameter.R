
L = 1000
res = as.data.frame(matrix(0,L,5))
names(res) = c('sigmasq=0.5','sigmasq=1','sigmasq=1.5','sigmasq=2','JSE')
sigmasq = c(0.5,1,1.5,2)
sigma0sq = 1
par(mfrow = c(1,2))
Nseq = c(10,100)
for (index in 1:2){
N = Nseq[index]

for (i in 1:L){
mu = rnorm(N,0,sigma0sq) #mu ~ N(0,sigma0sq)
z = rnorm(N,mu,1) #z|mu ~ N(mu,1)
for (ii in 1:length(sigmasq)){
mu_hat = (1-1/(sigmasq[ii]+1))*z
res[i,ii] = sum((mu_hat -mu)^2)/N
}
mu_JS = (1-(N-2)/(sum(z^2)))*z
res[i,5] = sum((mu_JS-mu)^2)/N
}

boxplot(res)
title(paste('N=',Nseq[index],sep=''))
}

