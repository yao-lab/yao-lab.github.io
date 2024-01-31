#####
# A simulation to show that JSE has smaller Mean Square Error than MLE
# 	

nrep = 100
err_MLE = rep(0,nrep)
err_JSE = rep(0,nrep)

# p = N in the following
N = 100
mu = runif(N)
for (i in 1:nrep){
#mu = rnorm(N)    #mu is generated from N(0,1)
z = rnorm(N,mu,1)
mu_MLE = z
mu_JSE = (1-(N-2)/sum(z^2))*z
err_MLE[i] = sum((mu_MLE-mu)^2)/N
err_JSE[i] = sum((mu_JSE-mu)^2)/N
}
err1 = as.data.frame(cbind(err_MLE,err_JSE))
names(err1) = c("err_MLE","err_JSE")


N = 100
for (i in 1:nrep){
mu = runif(N)#mu is generated from uniform [0,1]
z = rnorm(N,mu,1)
mu_MLE = z
mu_JSE = (1-(N-2)/sum(z^2))*z
err_MLE[i] = sum((mu_MLE-mu)^2)/N
err_JSE[i] = sum((mu_JSE-mu)^2)/N
}
err2 = as.data.frame(cbind(err_MLE,err_JSE))
names(err2) = c("err_MLE","err_JSE")

par(mfrow=c(1,2))
boxplot(err1,ylab = "Error: ||hat{mu}-mu||/N")
title(paste("mu_i is generated from Normal(0,1), sample size N=",N,sep=""))
boxplot(err1,ylab = "Error: ||hat{mu}-mu||/N")
title(paste("mu_i is generated from Uniform [0,1], sample size N=",N,sep=""))

#########
# Efron's Batting example
#
names<-c("Clemente","F.Robinson","F.Howard","Johnstone","Berry","Spencer","Kessinger","L.Alvarado","Santo","Swoboda","Unser","Williams","Scott","Petrocelli","E.Rodriguez","Campaneris","Munson","Alvis")
hits<-c(18,17,16,15,14,14,13,12,11,11,10,10,10,10,10,9,8,7)
n <- 45
mu<-c(.346,.298,.276,.222,.273,.270,.263,.210,.269,.230,.264,.256,.303,.264,.226,.286,.316,.200)
#p <- length(bat)
p <- length(hits)

#mu_mle<-bat/n
mu_mle<-hits/n

z<-mu_mle
z_bar = mean(z)
S = sum((z-z_bar)^2)
sigma02 = z_bar*(1-z_bar)/n
mu_js = (1-S/p*(p-2)/(t(z)%*%z))*z
mu_js1 = z_bar + (1 - (p-3)*sigma02/S) * (z-z_bar)

err_js = sum((round(mu_js,digits=3)-mu)^2)
err_js1 = sum((round(mu_js1,digits=3)-mu)^2)
err_mle = sum((round(mu_mle,digits=3)-mu)^2)

err_js/err_mle

err_js1/err_mle

# Generate Table 
X<-as.data.frame(cbind(names,hits,round(mu_mle,digits=3),mu,round(mu_js,digits=3),round(mu_js1,digits=3)))

write.table(X,file="efron_bat.txt",sep="&",row.names=FALSE,quote=FALSE)