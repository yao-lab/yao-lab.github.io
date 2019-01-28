library(lattice)
library(MASS)

# If you quit R without saving the data, you'll have to
# read it in again.

# Regress sleep on log(body weight).
z1 <- lm(sleep ~ log(body), data = sleep)

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
png(file =
    '/Users/dlambert/CN/Figures/sleepBodyRegSim.png',
    width = 350, height = 350)
xyplot(sleep ~ log(body), data = sleep,
       xlab = 'LOG(BODY WEIGHT)', ylab = 'SLEEP',
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

# Plotting y vs more than one predictor.
png(file =
    '/Users/dlambert/CN/Figures/sleepBodyDanger.png',
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
png(file =
    '/Users/dlambert/CN/Figures/sleepBodyDangerNoInt.png',
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


# Adding the original regression line and interaction
# lines in each panel.
png(file =
    '/Users/dlambert/CN/Figures/sleepBodyDangerInt.png',
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



# Show the uncertainty when there are more predictors.
nsim <- 30
z2 <- lm(sleep ~ log(body) + danger, data = sleep)
z2Summary <- summary(z2)
z2Correlation <-  z2Summary$sigma^2 *
                    z2Summary$cov.unscaled
sim2Estimates <- mvrnorm(nsim, z2$coefficients,
                         z2Correlation)

png(file =
'/Users/dlambert/CN/Figures/sleepBodyDangerUncertainty.png',
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

# Danger is nonparametric; additive effect on sleep.
sleep$dangerFactor <- factor(sleep$danger)
z2Nonp <- lm(sleep ~ log(body) + dangerFactor,
             data = sleep)
summary(z2Nonp)
z2Intercepts <- z2Nonp$coefficients[1] +
                c(0, z2Nonp$coefficients[3:6])

png(file =
'/Users/dlambert/CN/Figures/sleepBodyFactorDanger.png',
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


# Nonparametric danger; interaction effect with log(body).
z2NonpInt <- lm(sleep ~ log(body) * dangerFactor,
                data = sleep)
coefs <- summary(z2NonpInt)$coefficients
round(coefs, 2)

z2Intercepts <- z2Nonp$coefficients[1] +
                c(0, z2Nonp$coefficients[3:6])

png(file =
'/Users/dlambert/CN/Figures/sleepBodyFactorDangerInt.png',
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

install.packages('leaps')
library(leaps)

