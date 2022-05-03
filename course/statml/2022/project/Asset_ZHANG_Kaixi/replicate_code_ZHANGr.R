#################### Final project of MATH 5470 #########################
## Replicate the results: Empirical Asset Pricing via Machine Learning ##
## Written by: Kaixi ZHANG, Economics department, HKUST #################
##########################################################################
rm(list = ls())
library(data.table)
library(dplyr)
library(fastDummies)
library(lubridate)
library(purrr)
library(MASS)
library(pls)
library(hqreg)
library(glmnet)
library(grpreg)
library(randomForest) 
library(gbm)
library(ggplot2)
#open data
df <- fread(file = "/Users/zhangkaixi/Desktop/HKUST_MPhil1/machine_learning/Final_project/datashare/GKX.csv")
totaldata <- (df$DATE >= 19570329)&(df$DATE <= 20161231)
df <- df[totaldata, ]
##########################################################################
# Firm Characteristics Dataset Description:
# Columns:
# permno: CRSP Permanent Company Number
# DATE: The end day of each month (YYYYMMDD) 
# RET: CRSP Holding Returns with dividends in that month (can used as response variable) 
# 94 Lagged Firm Characteristics (Details are in the appendix)
# sic2: The first two digits of the Standard Industrial Classification code on DATE
# (e.g. When DATE=19570329 in our dataset, you can use the monthly RET at 195703 as the response variable.) 
##########################################################################
######################## Define 920 features #############################
##########################################################################
#### 94 characteristics of predictors
char94 <- df %>%
  mutate(., 
         DATE = as.numeric(substr(DATE, start = 1, stop = 6))) %>%
  dplyr::select(., RET, DATE, mvel1, beta : bm_ia) %>%
  dplyr::select(., -c(mve0, sic2)) %>%
  cbind(., 1)
#### create eight macroeconomic predictors
monthly <- read.csv2("/Users/zhangkaixi/Desktop/HKUST_MPhil1/machine_learning/Final_project/datashare/monthly_2021.csv", 
                     sep = ",", na.strings = "NaN", stringsAsFactors = FALSE)
# convert the class of character to the numeric
mm <- matrix(as.numeric(unlist(monthly)), ncol = ncol(monthly))
colnames(mm) = c(names(monthly))
# calculate the relevant variables 
macro <-  mm %>%
  as.data.frame() %>%
  mutate(.,
         dp = log(D12) - log(Index),
         ep_m = log(E12) - log(Index),
         tms = lty - tbl,
         dfy = BAA - AAA) %>%
  dplyr::select(.,
         yyyymm, dp, ep_m, b.m, ntis, tbl, tms, dfy, svar)
colnames(macro)[1] = "DATE"
totaldata <- (macro$DATE >= 195703)&(macro$DATE <= 201612)
macropred <- macro[totaldata, ]
# merge two data sets
pred <- data.table(char94, key = "DATE")[data.table(macropred, key = "DATE")]
cit <- pred[ , 3:96]
cit_3 <- dplyr::select(cit, 
                mvel1, bm, mom12m, mom1m, mom36m, mom6m)
xt <- pred[ , 97:105]

kronecker_product <- function(a){a*xt} # kronecker product: %x%
zit <- cit %>%
  apply(., MARGIN = 2, kronecker_product) %>%
  unlist() %>%
  as.numeric() %>%
  matrix(., nrow = nrow(cit))
z3it <- cit_3 %>%
  apply(., MARGIN = 2, kronecker_product) %>%
  unlist() %>%
  as.numeric() %>%
  matrix(., nrow = nrow(cit_3))

## drop the missing value
pred_n <- na.omit(pred)
cit_n <- pred_n[ , 3:96]
xt_n <- pred_n[ , 97:105]
cit_3_n <- dplyr::select(cit_n, 
                       mvel1, bm, mom12m, mom1m, mom36m, mom6m)
kronecker_product <- function(a){a*xt_n}
z3it_n <- cit_3_n %>%
  apply(., MARGIN = 2, kronecker_product) %>%
  unlist() %>%
  as.numeric() %>%
  matrix(., nrow = nrow(cit_3_n))
train_n <- (pred_n$DATE <= 198612)

#### create the industry dummies 
indsdummy <- df$sic2 %>%
  as.character() %>%
  dummy_cols(., 
             remove_most_frequent_dummy = TRUE, ignore_na = TRUE)
fullfeatures <- cbind(zit, indsdummy)
ols3features <- cbind(z3it, indsdummy)
pred <- data.frame(pred$DATE, response, fullfeatures)
# define the response variable 
response <- df$RET
response_n <- pred_n$RET
## Sample splitting
train <- (pred$DATE <= 198612)
test <- !train
##########################################################################
############################## Modeling ##################################
##########################################################################

################### Ordinary least square (OLS) ##########################
# using Huber loss instead of the l2 loss
ols_Huber <- rlm(fullfeatures[train, ], response[train])
ols3_Huber <- rlm(ols3features[train, ], response[train])
## Default S3 method:
##rlm(x, y, weights, ..., w = rep(1, nrow(x)),
    #init = "ls", psi = psi.huber,
    #scale.est = c("MAD", "Huber", "proposal 2"), k2 = 1.345,
    #method = c("M", "MM"), wt.method = c("inv.var", "case"),
    #maxit = 20, acc = 1e-4, test.vec = "resid", lqs.control = NULL)
pred.ols_Huber <- as.matrix(fullfeatures[!train, ]) %*% as.matrix(ols_Huber$coefficients) 
pred.ols3_Huber <- as.matrix(ols3features[!train, ]) %*% as.matrix(ols3_Huber$coefficients) 

## drop the missing 
ols_Huber_n <- rlm(fullfeatures_n[train_n, ], response_n[train_n])
pred.ols_Huber_n <- as.matrix(fullfeatures_n[!train_n, ]) %*% as.matrix(ols_Huber_n$coefficients) 
ols3_Huber_n <- rlm(cit_3_n[train_n, ], response_n[train_n])
pred.ols3_Huber_n <- as.matrix(cit_3_n[!train_n, ]) %*% as.matrix(ols3_Huber_n$coefficients) 
###################### Partial least square (PLS) ########################
set.seed(1)
pls.fit <- plsr(response ~ . -DATE, data = pred, 
               subset = train, scale = T, validation = "CV")
summary(pls.fit)
pred.pls <- predict(pls.fit, fullfeatures[!train, ], ncomp = 30)
################# Principal components regression (PCR) #################
set.seed(1)
pcr.fit <- pcr(response ~ . -DATE, data = pred, 
               subset = train, scale = T, validation = "CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP")
pred.pcr <- predict(pcr.fit, fullfeatures[!train, ], ncomp = 30)
######################### Elastic net (ENet) ############################
ENet.fit <- hqreg(fullfeatures[train, ], response[train], method = "huber")
pred.ENet <- predict(ENet.fit, fullfeatures[!train, ])
############ Generalized linear model with group lasso(GLM) #############
# lasso
lasso.fit <- glmnet(fullfeatures[train, ], response[train], alpha = 1,
                lambda = grid)
pred.lasso <- predict(lasso.fit, s = bestlam, newx = fullfeatures[!train, ])
# group lasso
glm.fit <- grpreg(fullfeatures[train, ], response[train],
                   group=1:ncol(X), penalty = "grLasso")
######################### Random forest (RF) ############################
set.seed(1)
rf.fit <- randomForest(response ~ . -DATE, data = pred, 
                           subset = train, mtry = 30, importance = T)
importance(rf.fit)
varImpPlot(rf.fit)
pred.rf <- predict(rf.fit, newdata = fullfeatures[!train, ])
################ gradient boosted regression trees (GBRT) ################
set.seed(0)
boost.fit <- gbm(response ~ . -DATE, data = pred[train, ],
                 distribution = "gaussian", n.trees = 500,
                 interaction.depth = 4) # l2 norm
summary(boost.fit)
plot(boost.fit, i = "mom1m")
plot(boost.fit, i = "dy")
pred.boost <- predict(boost.fit, newdata = fullfeatures[!train, ], n.trees = 500)
##########################################################################
##################### Assess predictive performance ######################
##########################################################################
R2_oos <- function(pred){
  y.test <- response[!train]
  1- sum((pred - y.test)^2, na.rm = T)/sum(y.test^2, na.rm = T)
}
R2_oos(pred.ols_Huber)
R2_oos(pred.ols3_Huber)
R2_oos(pred.pls)
R2_oos(pred.pcr)
R2_oos(pred.ENet)
R2_oos(pred.lasso)
R2_oos(pred.rf)
R2_oos(pred.boost)

# drop the missing
R2_oos_n <- function(pred){
  y.test <- response_n[!train_n]
  1- sum((pred - y.test)^2, na.rm = T)/sum(y.test^2, na.rm = T)
}
R2_oos_n(pred.ols3_Huber_n)
































