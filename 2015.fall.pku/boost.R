# 8.3.4 in the book
# An Introduction to Statistical Learning

library("MASS")
library("gbm")

set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)

# Gradient Boosted Methods for regression
boost.boston = gbm(medv~., data=Boston[train,], distribution="gaussian", n.trees=5000, interaction.depth=4)
summary(boost.boston)

par(mfrow=c(1,2))
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")

# Test Error in MSE, 11.8
yhat.boost=predict(boost.boston,newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost -boston.test)^2)

# Use shrinkage to do regularization 
boost.boston = gbm(medv~., data=Boston[train,], distribution="gaussian", n.trees=5000, interaction.depth=4, shrinkage=0.2, verbose =F)

# New Test Error in MSE, 11.5
yhat.boost = predict(boost.boston,newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost -boston.test)^2)

# 8.3.3 in the book
# An Introduction to Statistical Learning

library(randomForest)

# Bagging with m=13=p
set.seed (1)
bag.boston = randomForest(medv~.,data=Boston, subset=train, mtry=13, importance=TRUE)
bag.boston

yhat.bag = predict(bag.boston, newdata=Boston[-train ,])
plot(yhat.bag, boston.test)
abline(0 ,1)
mean((yhat.bag-boston.test)^2)

bag.boston = randomForest(medv~., data=Boston, subset=train, mtry=13, ntree=25)
yhat.bag = predict(bag.boston, newdata=Boston[-train ,])
mean((yhat.bag-boston.test)^2)

# Random Forest with m=6, with an improvement
set.seed (1)
rf.boston = randomForest(medv~.,data=Boston, subset=train, mtry=6, importance=TRUE)
yhat.rf = predict(rf.boston, newdata=Boston[-train ,])
mean((yhat.rf-boston.test)^2)

# Variable importance
importance(rf.boston)
varImpPlot(rf.boston)