# 8.3 in the book
# An Introduction to Statistical Learning

# Remember to install.packages("ISLR") before load it
library("ISLR")
attach(Carseats)

# Construct a binary response variable "High"
High = ifelse(Sales<=8,"No","Yes")
Carseats = data.frame(Carseats,High)

# Remember to install.packages("tree") before load it
library("tree")

tree.carseats=tree(High~.-Sales,Carseats)
summary(tree.carseats)

# Plot the tree and show the node labels
plot(tree.carseats)
text(tree.carseats, pretty=0)

# Print the tree on the screen. 
tree.carseats 
# Split the dataset into a training set and a test set. 
set.seed (2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats [-train ,]
High.test=High[-train]

# Train a model, get the test error, 
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")

# Get the confusion matrix
table(tree.pred, High.test)
# Compute the prediction accuracy, which is 71.5%
sum(diag(table(tree.pred, High.test)))/200
# Use CV with misclassification error minimization to prune the tree 
set.seed(3)
cv.carseats =cv.tree(tree.carseats ,FUN=prune.misclass )
names(cv.carseats )

# Note: cv.carseats$dev gives the misclassification rates
cv.carseats

# Error Plots against "size" and "k"
par(mfrow=c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type="b")
plot(cv.carseats$k, cv.carseats$dev, type="b")

# Get the optimally pruned tree and plot it
prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats)
text(prune.carseats, pretty=0)

# New test performance, which is 77%
table(tree.pred, High.test)
sum(diag(table(tree.pred,High.test)))/200

# Regression tree
library("MASS")

# training set
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)

# regression tree
tree.boston=tree(medv~.,data=Boston,subset=train)
summary(tree.boston)

plot(tree.boston)
text(tree.boston, pretty=0)

# CV for optimal tuning
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type="b")

# Choose the optimal tuned tree
prune.boston=prune.tree(tree.boston, best=5)
plot(prune.boston)
text(prune.boston, pretty=0)

# Test Error in MSE
yhat = predict(tree.boston, newdata=Boston[-train ,])
boston.test = Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat-boston.test)^2)
