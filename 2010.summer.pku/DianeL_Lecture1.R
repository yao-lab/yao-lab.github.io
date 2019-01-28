library(lattice)
library(MASS)

# Read in the data -- your csv file will be different.
sleep <- read.csv('/Users/dlambert/CN/Data/sleep1.csv', row.names = 1)
keepRow <- apply(!is.na(sleep[, -(1:2)]), 1, all)
sleep <- sleep[keepRow, -c(1,2)]
round(sleep[1:20, c('sleep', 'body', 'brain', 'predation', 'danger')], 1)

# Histogram of sleep
png(file = '/Users/dlambert/CN/Figures/sleepNoX.png', width = 400, height = 400)
histogram(~totalSleep, data = sleep, subset = !is.na(totalSleep),
          breaks = 12,
          xlab = 'Hours Per Day', main = 'Sleep For 58 Mammal Species')
graphics.off()

b0 <- mean(sleep$sleep)
s2 <- var(sleep$sleep)

# Plot of sleep against body weight.
png(file = '/Users/dlambert/CN/Figures/sleepBody.png', width = 500, height = 500)
xyplot(sleep ~ body, data = sleep, xlab = 'BODY WEIGHT (kg)', ylab = 'SLEEP',
       pch = 16, cex = 1.2)
graphics.off()

# Regression of sleep on body weight.
z0 <- lm(sleep ~ body, data = sleep)
summary(z0)
# Note that the coefficient of body weight is statistically significant.

# Plot of sleep against log(body weight).
png(file = '/Users/dlambert/CN/Figures/sleepLogBody.png', width = 500, height = 500)
xyplot(sleep ~ log(body), data = sleep, xlab = 'LOG(BODY WEIGHT)', ylab = 'SLEEP',
       pch = 16, cex = 1.25)
graphics.off()


# Plot of sleep against log(body weight) with regression line.
png(file = '/Users/dlambert/CN/Figures/sleepLogBodyRegNoErrors.png',
    width = 450, height = 450)
xyplot(sleep ~ log(body), data = sleep, xlab = 'LOG(BODY WEIGHT)', ylab = 'SLEEP',
       panel = function(x, y, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, pch = 16, cex = 1.25)
         panel.lmline(x, y)
        }
       )
graphics.off()

# Regress sleep on log(body weight).
z1 <- lm(sleep ~ log(body), data = sleep)
# Estimate the mean sleep for body weights of 1 and 100 kg.
predict(z1, newdata = data.frame(body = c(1, 100)))

# Plot sleep vs log(body weight) with regression line and 'residuals'.
png(file = '/Users/dlambert/CN/Figures/sleepLogBodyWeight.png',
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

# Plot sleep against danger.
png(file = '/Users/dlambert/CN/Figures/sleepDanger.png', width = 400, height = 400)
xyplot(sleep ~ danger, data = sleep, xlab = 'DANGER', ylab = 'SLEEP',
       pch = 16, cex = 1.25)
graphics.off()

# Regress sleep on both log(body weight) and danger.
z2 <- lm(sleep ~ log(body) + danger, data = sleep)

# Plot residuals vs fitted.
png(file = '/Users/dlambert/CN/Figures/sleepBodyDangerResVFit.png')
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
png(file = '/Users/dlambert/CN/Figures/sleepBodyDangerResiduals.png',
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


# Get covariance matrix of regression coefficients for sleep vs log(body).
z1Summary <- summary(z1)
z1Correlation <- z1Summary$sigma^2 * z1Summary$cov.unscaled
# Simulate 30 regression lines from the approximate distrib of the coefficients.
simEstimates <- mvrnorm(30, z1$coefficients, z1Correlation)

# Plot the simulated regression lines.
png(file = '/Users/dlambert/CN/Figures/sleepBodyRegSim.png',
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

