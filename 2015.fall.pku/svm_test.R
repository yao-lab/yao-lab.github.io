# 9.6.1 in the book
# An Introduction to Statistical Learning

# Construct a linearly seperable dataset on 2-D plane
set.seed (1)
x=matrix(rnorm(20*2), ncol=2)
y=c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1
plot(x, col=(3-y))
dat=data.frame(x=x, y=as.factor(y))

# Load the libsvm R interface
# use LiblineaR for very large problem
library('e1071')

svmfit=svm(y ~ ., data=dat, kernel='linear', cost=10, scale=FALSE)
plot(svmfit,dat)
svmfit$index
summary(svmfit)

svmfit=svm(y ~ ., data=dat, kernel='linear', cost=0.1, scale=FALSE)
plot(svmfit,dat)
svmfit$index
summary(svmfit)

# Find optimal tuning parameter
set.seed (1)
tune.out=tune(svm,y ~ .,data=dat,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
bestmod=tune.out$best.model
summary(bestmod)

# Construct the test data
xtest=matrix(rnorm(20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]=xtest[ytest==1,] + 1
testdat=data.frame(x=xtest, y=as.factor(ytest))

ypred=predict(bestmod ,testdat)
# Confusion matrix
table(predict=ypred, truth=testdat$y)

# Change Cost parameter to be 0.01
svmfit=svm(y~., data=dat, kernel="linear", cost=.01, scale=FALSE)
ypred=predict(svmfit,testdat)
table(predict=ypred, truth=testdat$y)

x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch=19)

dat=data.frame(x=x,y=as.factor(y))
svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
summary(svmfit)
plot(svmfit , dat)

svmfit=svm(y~., data=dat, kernel="linear", cost=1)
summary(svmfit)
plot(svmfit , dat)

# 9.6.2 Nonlinear kernel with radial basis
set.seed (1) 
x=matrix(rnorm(200*2), ncol=2) 
x[1:100,]=x[1:100,]+2 
x[101:150,]=x[101:150,]-2 
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y))
plot(x, col=y)
train=sample(200,100)
svmfit=svm(y~., data=dat[train,], kernel="radial", gamma=1, cost =1)
plot(svmfit , dat[train ,])
summary(svmfit)
svmfit=svm(y~., data=dat[train,], kernel="radial",gamma=1, cost=1e5)
plot(svmfit ,dat[train ,])
tune.out=tune(svm, y~., data=dat[train,], kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4) ))
table(true=dat[-train,"y"], pred=predict(tune.out$best.model, newx=dat[-train ,]))

# 9.6.3 ROC curves
library('ROCR')
rocplot=function(pred, truth, ...){
	predob = prediction(pred, truth)
	perf = performance(predob, "tpr", "fpr")
	plot(perf,...)}
svmfit.opt=svm(y~., data=dat[train,], kernel="radial", gamma=2, cost=1,decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=TRUE))$decision.values
par(mfrow=c(1,2))
rocplot(fitted,dat[train,"y"],main="Training Data")
svmfit.flex=svm(y~., data=dat[train,], kernel="radial", gamma=50, cost=1, decision.values=T)
fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values
rocplot(fitted ,dat[train ,"y"],add=T,col="red")
fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],main="Test Data")
fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],add=T,col="red")

# 9.6.4 Multiclass SVM
set.seed (1)
x=rbind(x, matrix(rnorm(50*2), ncol=2))
y=c(y, rep(0,50))
x[y==0,2]=x[y==0,2]+2
dat=data.frame(x=x, y=as.factor(y))
par(mfrow=c(1,1))
plot(x,col=(y+1))
svmfit=svm(y~., data=dat, kernel="radial", cost=10, gamma=1)
plot(svmfit, dat)

# 9.6.5 Gene Expression data 'Khan'
library(ISLR)
names(Khan)
#[1] "xtrain" "xtest"  "ytrain" "ytest" 
dim(Khan$xtrain )
#[1]   63 2308
dim(Khan$xtest )
#[1]   20 2308
length(Khan$ytrain )
#[1] 63
length(Khan$ytest )
#[1] 20
table(Khan$ytrain )
table(Khan$ytest )

#training error
dat=data.frame(x=Khan$xtrain, y=as.factor(Khan$ytrain))
out=svm(y~., data=dat, kernel="linear",cost=10)
summary(out)
table(out$fitted, dat$y)

# test error
dat.te=data.frame(x=Khan$xtest, y=as.factor(Khan$ytest ))
pred.te=predict(out, newdata=dat.te)
table(pred.te, dat.te$y)