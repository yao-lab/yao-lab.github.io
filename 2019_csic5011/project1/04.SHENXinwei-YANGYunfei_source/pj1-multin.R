
########## Data ########## 
setwd("/Users/xinwei/HKUST/1-2/CSIC5011-code/pj1-digits")
#setwd("~/R/digits")
train <- read.table(gzfile("zip.train"))
test <- read.table(gzfile("zip.test"))
#train$V1 <- relevel(train$V1, ref = "0")
data <- rbind(train, test)

## Separate training and testing data 80%:20%
index <- sample(nrow(data), floor(nrow(data) * 0.8))
train <- data[index,]
test <- data[-index,]

#### Multinomial Logistic regression ####
library(nnet)
multin <- multinom(V1 ~ ., data = train, family = "multinomial", MaxNWts = 10000, maxit = 500)
#str(multin)
#coef_logit <- summary(multin)$coefficients
pred_multin <- predict(multin, test, type = "class")
# Building classification table
print(tab <- table(test$V1, pred_multin))
# pred_multin
# 0   1   2   3   4   5   6   7   8   9
# 0 332   0   2   4   3   0   5   2   6   5
# 1   0 242   2   1   5   1   7   1   3   2
# 2   6   1 148   9   8   3   4   3  14   2
# 3   2   3   5 128   0  11   3   1   7   6
# 4   4   3   5   4 149   5  10   4   6  10
# 5   2   4   0  10   8 116   4   1  10   5
# 6   1   1   4   0   6   8 144   1   4   1
# 7   2   2   3   3   7   0   0 125   0   5
# 8   7   0   3   2   5  10   4   3 131   1
# 9   2   2   0   4   2   1   1   4   3 158
# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(tab))/sum(tab))*100, 2)
# [1] 83.36



#### Logistic + Lasso / Elastic net ####
library(glmnet)
x <- as.matrix(train[,-1])
y <- as.factor(train[,1])
lasso <- cv.glmnet(x, y, type.measure = "class", alpha = 1, family = "multinomial")
x_test <- as.matrix(test[,-1])
y_test <- as.factor(test[,1])
pred_lasso <- predict(lasso, newx = x_test, s = "lambda.min", type = "class") ## think of some way to illustrate the sparse structure
print(tab <- table(test$V1, pred_lasso))
# pred_lasso
# 0   1   2   3   4   5   6   7   8   9
# 0 346   0   2   3   3   0   2   0   2   1
# 1   0 251   0   3   3   0   4   0   1   2
# 2   2   0 173   4   6   2   2   1   7   1
# 3   1   0   4 145   1   8   0   2   3   2
# 4   1   2   7   0 181   1   1   1   0   6
# 5   4   0   0   8   2 140   1   1   1   3
# 6   2   0   4   0   2   3 158   0   1   0
# 7   0   1   1   1   7   0   0 131   1   5
# 8   7   0   4   4   2   6   1   0 140   2
# 9   0   0   0   0   2   1   0   3   3 168
round((sum(diag(tab))/sum(tab))*100, 2)
# [1] 91.33

elas <- cv.glmnet(x, y, type.measure = "class", alpha = 0.5, family = "multinomial")
pred_elas <- predict(elas, newx = x_test, s = "lambda.min", type = "class")
print(tab <- table(test$V1, pred_elas))
# pred_elas
# 0   1   2   3   4   5   6   7   8   9
# 0 348   0   1   3   3   0   1   0   2   1
# 1   0 252   0   3   3   0   4   0   1   1
# 2   2   0 175   4   6   2   1   1   7   0
# 3   1   0   4 145   1   9   0   2   2   2
# 4   1   2   7   0 181   1   1   1   0   6
# 5   5   0   0   8   2 140   1   1   0   3
# 6   1   0   4   0   2   2 160   0   1   0
# 7   0   1   1   2   6   0   0 132   0   5
# 8   7   0   5   4   1   6   1   0 140   2
# 9   0   1   1   0   2   1   0   3   1 168
round((sum(diag(tab))/sum(tab))*100, 2)
# [1] 91.73

ridge <- cv.glmnet(x, y, type.measure = "class", alpha = 0, family = "multinomial")
pred_ridge <- predict(ridge, newx = x_test, s = "lambda.min", type = "class")
print(tab <- table(test$V1, pred_ridge))
# pred_ridge
# 0   1   2   3   4   5   6   7   8   9
# 0 348   0   2   2   3   0   2   0   1   1
# 1   0 252   0   2   4   0   4   0   1   1
# 2   5   0 166   4  10   2   3   1   7   0
# 3   4   0   3 142   1  11   0   2   2   1
# 4   2   2   5   0 179   0   4   1   1   6
# 5   7   0   0   7   2 139   0   0   1   4
# 6   2   0   3   0   3   3 158   0   1   0
# 7   0   0   1   1   5   0   0 134   1   5
# 8   6   0   4   5   2   4   1   1 141   2
# 9   0   2   1   0   3   1   0   1   1 168
round((sum(diag(tab))/sum(tab))*100, 2)
# 90+

#### PCA + Logistic ####
pr <- prcomp(x, center = TRUE, scale. = TRUE)
varex <- pr$sdev^2
head(cumsum(varex) / sum(varex), 100)
plot(varex / sum(varex))
pc <- predict(pr)[,1:50]
train_pc <- as.data.frame(cbind(y, pc))
multin_pc <- multinom(y ~ ., data = train_pc, family = "multinomial", MaxNWts = 10000, maxit = 1000)
pc_pred <- predict(pr, x_test)
test_pc <- as.data.frame(cbind(y_test, pc_pred))
pred_multin_pc <- predict(multin_pc, test_pc, type = "class")
print(tab <- table(test$V1, pred_multin_pc))
# pred_multin_pc
# 1   2   3   4   5   6   7   8   9  10
# 0 344   0   3   3   3   0   1   0   4   1
# 1   0 249   0   3   5   1   3   0   2   1
# 2   2   0 169   8   8   2   0   2   6   1
# 3   1   0   4 146   1   8   0   2   3   1
# 4   2   1   6   0 173   1   4   4   2   7
# 5   5   0   2   9   2 130   0   3   5   4
# 6   2   0   3   1   2   0 160   0   2   0
# 7   0   0   0   2   5   2   0 132   0   6
# 8   4   0   3   5   0   7   0   1 145   1
# 9   0   1   0   1   0   0   0   4   1 170
round((sum(diag(tab))/sum(tab))*100, 2)
# [1] 90.58



#### PCA adjustment + Logistic + Lasso ####
pc5 <- predict(pr)[,1:5]
train_pcadj <- as.data.frame(cbind(y, pc5, x))
multin_pcadj <- multinom(y ~ ., data = train_pcadj, family = "multinomial", MaxNWts = 10000, maxit = 1000)
test_pcadj <- as.data.frame(cbind(y_test, pc_pred, x_test))
pred_multin_pcadj <- predict(multin_pcadj, test_pcadj, type = "class")
print(tab <- table(test$V1, pred_multin_pcadj))
# pred_multin_pcadj
# 1   2   3   4   5   6   7   8   9  10
# 0 332   1   3   2   5   0   3   6   4   3
# 1   1 238   3   5   4   1   6   2   3   1
# 2   5   6 140   9   8   4   9   3  10   4
# 3   1   2   4 130   1  14   2   4   4   4
# 4   5   4   9   2 146   1   6   4   9  14
# 5   5   4   2   8   4 128   3   1   1   4
# 6   2   4   5   0   4   6 145   1   3   0
# 7   3   1   1   5   8   1   1 120   0   7
# 8   9   2   7   8   5   8   2   3 117   5
# 9   1   0   1   2   1   2   1   4   5 160
round((sum(diag(tab))/sum(tab))*100, 2)
# [1] 82.51

### + Lasso
x_pcadj <- cbind(pc5, x)
lasso_pcadj <- cv.glmnet(x_pcadj, y, type.measure = "class", alpha = 1, family = "multinomial")
x_test_pcadj <- cbind(pc_pred[,1:5], as.matrix(test[,-1]))
pred_lasso_pcadj <- predict(lasso_pcadj, newx = x_test_pcadj, s = "lambda.min", type = "class") ## think of some way to illustrate the sparse structure
print(tab <- table(test$V1, pred_lasso_pcadj))
# pred_lasso_pcadj
# 0   1   2   3   4   5   6   7   8   9
# 0 348   0   1   3   3   0   1   0   2   1
# 1   0 250   0   4   3   0   4   0   1   2
# 2   2   0 173   4   6   2   1   2   7   1
# 3   1   0   4 143   1  10   0   2   3   2
# 4   1   2   7   1 180   2   1   1   0   5
# 5   4   0   0   8   2 140   1   1   1   3
# 6   3   0   4   0   2   2 158   0   1   0
# 7   0   0   1   1   7   0   0 132   0   6
# 8   7   0   3   4   2   6   1   0 142   1
# 9   0   0   0   0   0   1   0   4   3 169
round((sum(diag(tab))/sum(tab))*100, 2)
# [1] 91.43
