
library('paran')
X<-read.csv('snp452-data.csv')
head(X, 6)

Y<-log(X)
head(Y, 6)

dY<-apply(Y, 2, diff)
head(dY, 6)

n = dim(dY)[1]
dYt = t(dY)
S <- 1/n * (dYt%*%dY)

y <- eigen(S)

eigen_values = y$val
eigen_vectors = y$vec

de_eigen_values = sort(eigen_values, decreasing=TRUE)

#Horn's Parallel Analysis
de_eigen_values[1:10]


snp.paran<-paran(dY, iterations=500)

snp.paran1<-paran(dY, iterations=50, centile=95)