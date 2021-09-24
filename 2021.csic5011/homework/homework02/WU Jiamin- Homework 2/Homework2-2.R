# Read data
x=as.matrix(stockdata[1])
x=t(x[[1]])
dim(x)

# 2(a)
y=log(x)

# 2(b)
delta_y=matrix(0, 452, 1257)
for (t in 2:1258)
{
  for (i in 1:452)
  {
    delta_y[i,t-1]=y[i,t]-y[i,t-1]
  }
}
dim(delta_y)

# 2(c)
cov=matrix(0,452,452)
for (i in 1:452)
{
  for (j in 1:452)
  {
    for (t in 1:1257)
    {
      cov[i,j]=cov[i,j]+delta_y[i,t]*delta_y[j,t]
    }
    cov[i,j]=1/1257*cov[i,j]
  }
}
dim(cov)

# 2(d)
eigval=eigen(cov)$values

# 2(e) 
snp.pca<-prcomp(y, center = TRUE)
summary(snp.pca)
plot(snp.pca,type='l')
library('paran')
snp.paran<-paran(y,iterations=100)

"If we use the Horn's Parallel Analysis, only 6 components reamins after
100 iterations such that p<0.05. Those 6 signals are strong."
