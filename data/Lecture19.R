library(lattice)
library(MASS)

myDir <- './Figures'

# Read in the data -- your csv file will be different.
sleep <- read.csv('sleep1.csv', row.names = 1)

dim(sleep)
names(sleep)

# Change variable name 'sleepExposure' to 'exposure' for simplicity
names(sleep)[names(sleep) == 'sleepExposure'] <-
                                          'exposure'
names(sleep)

# Count NAs
nNAs <- sapply(is.na(sleep), sum)
nNAs <- apply(is.na(sleep), 2, sum)

# Remove variables 'slowWaveSleep' and 'dreamSleep'
keepNames <- setdiff(names(sleep),
                     c('dreamSleep', 'slowWaveSleep'))
# Remove all samples (rows) with NAs
keepRow <- apply(!is.na(sleep[, keepNames]), 1, all)
sleep <- sleep[keepRow, keepNames]
dim(sleep)

# Show the first 20 samples (species) in 5 variables, with 1 digit round-off 
round(sleep[1:20, c('sleep', 'body', 'brain', 'predation', 'danger')], 1)

# Compute quantiles of some variables in newX and allX
newX <- c('brain', 'life', 'gestation', 'predation','exposure')
allX <- c(newX[1:3], 'body', newX[4:5], 'danger')
round(sapply(sleep[, allX], quantile, na.rm = TRUE), 1)


# Histogram of sleep and QQ plot for normality check
png(file = paste(myDir,'sleepNoX.png', sep ='/'),width = 400, height = 400)
#histogram(~sSleep, data = sleep, subset = !is.na(totalSleep),
#          breaks = 12,
#          xlab = 'Hours Per Day', main = 'Sleep For 58 Mammal Species')
p1 <- histogram(~sleep, data = sleep, subset = !is.na(sleep),
		breaks = 12,
                xlab = 'Hours Per Day', main = 'Sleep For 58 Mammal Species')
p2 <- qqmath(~sleep, data = sleep, subset = !is.na(sleep), distribution = qnorm,
       pch = 16, cex = 1.25, xlab = 'NORMAL QUANTILES', ylab = 'Sleep Hours Per Day',
       panel = function(x, ...) {
         panel.qqmathline(x, ...)
         panel.qqmath(x, ...)
         panel.grid()
       }
       )
print(p1, position = c(0, .49, 1, 1), more = TRUE)
print(p2, position = c(0, 0, 1, .51))
graphics.off()

# Compute mean and variance for variable sleep
b0 <- mean(sleep$sleep)
s2 <- var(sleep$sleep)


# Plot of sleep against body weight: far from being linear relationship.
png(file = paste(myDir,'sleepBody.png', sep ='/'), width = 500, height = 500)
xyplot(sleep ~ body, data = sleep, xlab = 'BODY WEIGHT (kg)', ylab = 'SLEEP',
       pch = 16, cex = 1.2)
graphics.off()

# Regression of sleep on body weight, despite of nonlinearityS.
z0 <- lm(sleep ~ body, data = sleep)
summary(z0)
# Note that the coefficient of body weight is statistically significant.

# Plot of sleep against log(body weight).
png(file = paste(myDir,'sleepLogBody.png', sep='/'), width = 500, height = 500)
xyplot(sleep ~ log(body), data = sleep, xlab = 'LOG(BODY WEIGHT)', ylab = 'SLEEP',
       pch = 16, cex = 1.25)
graphics.off()


# Regress sleep on log(body weight).
z1 <- lm(sleep ~ log(body), data = sleep)
summary(z1)
# Estimate the mean sleep for body weights of 1 and 100 kg. 
# Column 'lwr' and 'upr' gives 0.95 confidence interval in t-statistics
predict(z1, interval = 'prediction', newdata = data.frame(body = c(1, 100)))

# Plot sleep vs log(body weight) with regression line and 'residuals'.
png(file = paste(myDir,'sleepLogBodyWeight.png', sep='/'),
    width = 450, height = 450)
xyplot(sleep ~ log(body), data = sleep, xlab = 'LOG(BODY WEIGHT)', ylab = 'SLEEP',
       panel = function(x, y, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, pch = 16, cex = 1.25)
         panel.lmline(x, y)
         z1 <- lm(sleep ~ log(body), data = sleep)
         ind = order(x)[1:3]
         panel.segments(x, y, x, z1$fitted.values,
                        col = 'cyan4')
       }
       )
graphics.off()

# Simulate to evaluate the uncertainty in the regression
# function b0 + b1 * X
# First get covariance matrix of regression coefficients
# for sleep vs log(body).
z1Summary <- summary(z1)
z1Correlation <- z1Summary$sigma^2 *
                 z1Summary$cov.unscaled

# Simulate 30 regression lines from the approximate
# distrib of the coefficients.
simEstimates <- mvrnorm(30, z1$coefficients,
                        z1Correlation)

# Plot the simulated regression lines.
png(file = paste(myDir,'sleepBodyRegSim.png',sep='/'),
    width = 350, height = 350)
xyplot(sleep ~ log(body), data = sleep, xlab = 'LOG(BODY WEIGHT)', ylab = 'SLEEP',
       panel = function(x, y, ...) {
         panel.grid()
         panel.xyplot(x, y, cex = 1.25, pch = 16)
          for (i in 1:nrow(simEstimates)) {
           panel.abline(simEstimates[i, ], col = gray(.9))
         }
         panel.abline(z1$coefficients, col = 'red', lwd = 1.5)
 
       }
       )
graphics.off()

# Plot sleep against danger.
png(file = paste(myDir,'sleepDanger.png',sep='/'), width = 400, height = 400)
xyplot(sleep ~ danger, data = sleep, xlab = 'DANGER', ylab = 'SLEEP',
       pch = 16, cex = 1.25)
graphics.off()

# Regress sleep on both log(body weight) and danger.
z2 <- lm(sleep ~ log(body) + danger, data = sleep)
summary(z2)

# Plot residuals vs fitted.
png(file = paste(myDir,'sleepBodyDangerResVFit.png', sep='/'))
xyplot(residuals ~ fitted.values, data = z2, xlab = 'FITTED', ylab = 'RESIDUALS',
       pch = 16, cex = 1.25,
       panel = function(x, y, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         panel.abline(h = 0)
       }
       )
graphics.off()


# Check normality of residuals.
png(file = paste(myDir,'sleepBodyDangerResiduals.png', sep='/'),
    width = 350, height = 650)
p1 <- histogram(~residuals, data = z2,
                xlab = 'RESIDUALS', ylab = 'PERCENT OF TOTAL')
p2 <- qqmath(~residuals, data = z2, distribution = qnorm,
       pch = 16, cex = 1.25, xlab = 'NORMAL QUANTILES', ylab = 'RESIDUALS',
       panel = function(x, ...) {
         panel.qqmathline(x, ...)
         panel.qqmath(x, ...)
         panel.grid()
       }
       )
print(p1, position = c(0, .49, 1, 1), more = TRUE)
print(p2, position = c(0, 0, 1, .51))
graphics.off()

# Get estimated coefficients for regression of sleep on log(body) and danger.
summary(z2)


# Get confidence interval for the mean
newBody <- rep(c(.1, 1, 10, 100), 2)
newDanger <- rep(c(1, 5), each = 4)
ciMean <- predict(z2, interval = 'confidence',
                  newdata = data.frame(body = newBody, danger = newDanger))
ciMean <- cbind(newBody, newDanger, ciMean)
colnames(ciMean) <- c('body', 'danger', 'estimate', 'lower', 'upper')
round(ciMean, 1)

# Get confidence interval to predict the sleep for a species not in the data.
ciPred <- predict(z2, interval = 'prediction',
                  newdata = data.frame(body = newBody, danger = newDanger))
ciPred <- cbind(newBody, newDanger, ciPred)
colnames(ciPred) <- c('body', 'danger', 'estimate', 'lower', 'upper')
round(ciPred, 1)
ci <- cbind(ciMean, ciPred[, 4:5])
round(ci, 1)


# Plotting y vs more than one predictor.
png(file = paste(myDir,'sleepBodyDanger.png',sep='/'),
    width = 850, height = 250)
xyplot(sleep ~ log(body) | factor(danger), data = sleep,
       pch = 16, cex = 1.25,
              xlab = 'LOG(BODY)', ylab = 'SLEEP',
      main = 'SLEEP VS Log(BODY) As A Function Of DANGER',
       layout = c(5, 1),
       panel = function(x, y, subscripts, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
       })
graphics.off()

# Adding the original regression line in each plot.
png(file = paste(myDir,'sleepBodyDangerNoInt.png',sep='/'),
    width = 850, height = 250)
xyplot(sleep ~ log(body) | factor(danger), data = sleep,
       pch = 16, cex = 1.25,
       xlab = 'LOG(BODY)', ylab = 'SLEEP',
      main = 'Regression of Sleep on Log Body and Danger',
       layout = c(5, 1),
       panel = function(x, y, subscripts, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         panel.abline(z2$coef[1] +
                 z2$coef[3] * sleep$danger[subscripts][1],
                      z2$coef[2])
       })
graphics.off()

# Show the uncertainty when there are more predictors.
nsim <- 30
z2Summary <- summary(z2)
z2Correlation <-  z2Summary$sigma^2 *
                    z2Summary$cov.unscaled
sim2Estimates <- mvrnorm(nsim, z2$coefficients,
                         z2Correlation)

png(file = paste(myDir,'sleepBodyDangerUncertainty.png',sep='/'),
    width = 850, height = 250)
xyplot(sleep ~ log(body) | factor(danger), data = sleep,
       pch = 16, cex = 1.25,
       xlab = 'LOG(BODY)', ylab = 'SLEEP',
      main = 'Regression of Sleep on Log Body and Danger',
       layout = c(5, 1),
       panel = function(x, y, subscripts, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         for (i in 1:nsim) {
           panel.abline(sim2Estimates[i, 1] +
        sim2Estimates[i, 3] * sleep$danger[subscripts][1],
            sim2Estimates[i, 2],                          
                        col = gray(.9))
         }
         panel.abline(z2$coef[1] +
                 z2$coef[3] * sleep$danger[subscripts][1],
                      z2$coef[2], col =2, lwd = 1.5)
         
       })
graphics.off()

# Adding the original regression line and interaction
# lines in each panel.
# Interactions: slopes can be different for each level of danger
# Be cautious: adding interactions lead to a danger of overfitting 
png(file = paste(myDir,'sleepBodyDangerInt.png',sep='/'),
    width = 850, height = 250)
xyplot(sleep ~ log(body) | factor(danger), data = sleep,
       pch = 16, cex = 1.25,
       xlab = 'LOG(BODY)', ylab = 'SLEEP',
      main = 'Regression of Sleep on Log Body and Danger',
       layout = c(5, 1),
       panel = function(x, y, subscripts, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         panel.abline(z2$coef[1] +
                 z2$coef[3] * sleep$danger[subscripts][1],
                      z2$coef[2])
         panel.lmline(x, y, col = 'magenta')
       })
graphics.off()

# Danger is nonparametric; additive effect on sleep.
sleep$dangerFactor <- factor(sleep$danger)
z2Nonp <- lm(sleep ~ log(body) + dangerFactor,
             data = sleep)
summary(z2Nonp)
z2Intercepts <- z2Nonp$coefficients[1] +
                c(0, z2Nonp$coefficients[3:6])

png(file = paste(myDir,'sleepBodyFactorDanger.png',sep='/'),
    width = 850, height = 250)
xyplot(sleep ~ log(body) | factor(danger), data = sleep,
       pch = 16, cex = 1.25,
       xlab = 'LOG(BODY)', ylab = 'SLEEP',
        layout = c(5, 1),
       panel = function(x, y, subscripts, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         panel.abline(z2Intercepts[panel.number()],
                      z2$coefficients[2],
                      lwd = 1.25)         
       })
graphics.off()

# Danger as numeric: regression of sleep on log(body) and danger.
summary(z2)

# Danger as a factor: regression of sleep on log(body) and factor(danger)
summary(z2Nonp)

# Nonparametric danger; interaction effect with log(body).
z2NonpInt <- lm(sleep ~ log(body) * dangerFactor,
                data = sleep)
coefs <- summary(z2NonpInt)$coefficients
round(coefs, 2)

z2Intercepts <- z2Nonp$coefficients[1] +
                c(0, z2Nonp$coefficients[3:6])

png(file = paste(myDir,'sleepBodyFactorDangerInt.png',sep='/'),
    width = 850, height = 250)
xyplot(sleep ~ log(body) | dangerFactor, data = sleep,
       pch = 16, cex = 1.25,
       xlab = 'LOG(BODY)', ylab = 'SLEEP',
        layout = c(5, 1),
       panel = function(x, y, subscripts, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         panel.lmline(x,y, ...)         
       })
graphics.off()

