# load Horn's Parallel Analysis package
library('paran')

# USArrests dataset, contains crime rates of 4 crimes (columns) in 50 states (rows)
dim(USArrests)
USArrests

# ## The variances of the variables in the
## USArrests data vary by orders of magnitude, so scaling is appropriate
# apply PCA - scale. = TRUE is highly 
# advisable, but default is FALSE. 
usa.pca <- prcomp(USArrests,
                 center = TRUE,
                 scale. = TRUE)


# The first component dominates:
summary(usa.pca)
#Importance of components:
#                           PC1      PC2    PC3     PC4
#Standard deviation     83.7324 14.21240 6.4894 2.48279
#Proportion of Variance  0.9655  0.02782 0.0058 0.00085
#Cumulative Proportion   0.9655  0.99335 0.9991 1.00000

print(usa.pca)
plot(usa.pca,type='l')
biplot(usa.pca)

# Horn's parallel analysis, which only picks up the PC1
## perform a standard parallel analysis on the US Arrest data
usa.paran<-paran(USArrests, iterations=5000)
names(usa.paran)

## a conservative analysis with different result! 
usa.paran1<-paran(USArrests, iterations=5000, centile=95)
names(usa.paran1)

# PCA of S&P 500 stock data 
load('snp500.Rda')
X<-diff(log(stockdata$data),1);
names<-stockdata$info;
dim(X)
snp.pca<-prcomp(X, center = TRUE)
summary(snp.pca)
plot(snp.pca,type='l')

# 14 PCs are kept using Horn's parallel analysis
snp.paran<-paran(X,iterations=100)

# Further reading: Permutation Parallel Analysis
## Buja A and Eyuboglu N. (1992) Remarks on parrallel analysis. Multivariate Behavioral Research, 27(4), 509-540

