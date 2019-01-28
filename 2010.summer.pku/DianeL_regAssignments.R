library(lattice)
library(MASS)

sleep <- read.csv('/Users/dlambert/CN/Data/sleep1.csv')
names(sleep)[names(sleep) == 'sleepExposure'] <-
                                          'exposure'
sleep[seq(1, nrow(sleep), 5), ]

# Count NAs
nNAs <- sapply(is.na(sleep), sum)
nNAs <- apply(is.na(sleep), 2, sum)

keepNames <- setdiff(names(sleep),
                     c('dreamSleep', 'slowWaveSleep'))
naRows <- (1:nrow(sleep))[apply(is.na(sleep[, keepNames]), 1, any)]

newX <- c('brain', 'life', 'gestation', 'predation',
          'exposure')
allX <- c(newX[1:3], 'body', newX[4:5], 'danger')

round(sapply(sleep[, newX], quantile, na.rm = TRUE), 1)

sleepR2 <- rep(NA, 7)
names(sleepR2) <- allX
for (i in allX[1:4]) {
  z <- lm(sleep$sleep ~ log(sleep[[i]]))
  sleepR2[i] <- summary(z)$r.squared
}
for (i in allX[5:7]) {
  z <- lm(sleep$sleep ~ sleep[[i]])
  sleepR2[i] <- summary(z)$r.squared
}
sleepLog <- sleep
for (i in c('brain', 'gestation', 'body')) {
  sleepLog[[i]] <- log(sleep[[i]])
}
newX <- c('brain', 'gestation', 'body', 'exposure', 'danger')
plotList <- list()
for (i in newX) {
  plotList[[i]] <- xyplot(sleepLog$sleep ~ sleepLog[[i]],
                          xlab = i, ylab = 'sleep',
                          pch = 16, cex = 1.25)
}
png('/Users/dlambert/CN/Figures/onePredictor.png', height = 250, width = 1000)
for (i in 1:4) {
  print(plotList[[i]], position = c(.25 * (i-1), 0, .25*i, 1), more = TRUE)
}
    print(plotList[[i]], position = c(.75, 0, 1, 1))
graphics.off()

z <- lm(sleep ~ gestation , data = sleepLog)
z <- lm(sleep ~ gestation + body + danger, data = sleepLog)
summary(z)

# Get fitted values for species with NA for sleep$sleep
indx <- which(is.na(sleepLog$sleep))
predNA <- predict(z, newdata = sleepLog[indx, ])
predAll <- predict(z, newdata = sleepLog)

# Plot fitted values against each of the predictors
plotList <- list()
for (i in c('gestation', 'body', 'danger')) {
  plotList[[i]] <- xyplot(predAll ~ sleepLog[[i]],
                          ylab = 'sleep',
                          xlab = i,
                          pch =  16, cex = 1.25,
                          panel = function(x, y, ...) {
                            panel.grid(h=-1, v=-1)
                            panel.xyplot(x, y, ...)
  panel.points(sleepLog[[i]][indx], predNA,
               col = 'red', pch = 16, cex = 1.25)
                          })
}
png('/Users/dlambert/CN/Figures/fittedNA.png', height = 250, width = 800)
print(plotList[[1]], position = c(0, 0, 1/3, 1), more = T)
print(plotList[[2]], position = c(1/3, 0, 2/3, 1), more=T)
print(plotList[[3]], position = c(2/3, 0, 1, 1))
graphics.off()

# Plot observed values against each of the predictors, but
# use the fitted values for the outcomes with NA.
plotList <- list()
for (i in c('gestation', 'body', 'danger')) {
  plotList[[i]] <- xyplot(sleep$sleep ~ sleepLog[[i]],
                          ylab = 'sleep',
                          xlab = i,
                          pch =  16, cex = 1.25,
                          panel = function(x, y, ...) {
                            panel.grid(h=-1, v=-1)
                            panel.xyplot(x, y, ...)
  panel.points(sleepLog[[i]][indx], predNA,
               col = 'red', pch = 16, cex = 1.25)
                          })
}
png('/Users/dlambert/CN/Figures/observedNA.png', height = 250, width = 800)
print(plotList[[1]], position = c(0, 0, 1/3, 1), more = T)
print(plotList[[2]], position = c(1/3, 0, 2/3, 1), more=T)
print(plotList[[3]], position = c(2/3, 0, 1, 1))
graphics.off()


# Simulating uncertainty in the model.
z <- lm(sleep ~ gestation, data = sleepLog)
zSumm <- summary(z)
zCorr <- zSumm$sigma^2 * zSumm$cov.unscaled
simEstimates <- mvrnorm(30, z$coefficients,zCorr)
png(file = '/Users/dlambert/CN/Figures/gestationUnc.png')
xyplot(sleep ~ gestation, data = sleepLog,
       pch = 16, cex = 1.25,
       panel = function(x,y, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         for (i in 1:30) {
           panel.abline(simEstimates[i, ], col = gray(.7))
         }
         panel.abline(z$coefficients, col = 'red', lwd = 1.25)
       })
dev.off()


# Repeat for log(body)
z <- lm(sleep ~ body, data = sleepLog)
zSumm <- summary(z)
zCorr <- zSumm$sigma^2 * zSumm$cov.unscaled
simEstimates <- mvrnorm(30, z$coefficients,zCorr)

png(file = '/Users/dlambert/CN/Figures/bodyUnc.png')
xyplot(sleep ~ body, data = sleepLog,
       pch = 16, cex = 1.25,
       panel = function(x,y, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         for (i in 1:30) {
           panel.abline(simEstimates[i, ], col = gray(.7))
         }
         panel.abline(z$coefficients, col = 'red', lwd = 1.25)
       })
dev.off()

# To compare to a model with two predictors, plot against # both.
z <- lm(sleep ~ gestation, data = sleepLog)
zSumm <- summary(z)
zCorr <- zSumm$sigma^2 * zSumm$cov.unscaled
simEstimates <- mvrnorm(30, z$coefficients,zCorr)
sleepLog$dangerFact <- factor(sleepLog$danger)
png(file =
    '/Users/dlambert/CN/Figures/gestationUncDanger.png',
    width = 850, height = 250)
xyplot(sleep ~ gestation | dangerFact, data = sleepLog,
       pch = 16, cex = 1.25,
       xlab = 'LOG(Gestation)', ylab = 'SLEEP',
      main = 'Regression of Sleep on Log Gestation',
       layout = c(5, 1),
       panel = function(x, y,  ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
                  for (i in 1:30) {
           panel.abline(simEstimates[i, ], col = gray(.7))
         }

         panel.abline(z$coef, col='red', lwd = 1.25)
       })
graphics.off()

# Add a discrete variable to the model with danger

sleepGestR2 <- rep(NA, 3)
names(sleepGestR2) <- c('predation', 'danger', 'exposure')
for (i in names(sleepGestR2)) {
  z <- lm(sleep$sleep ~ sleepLog$gestation + sleepLog[[i]])
  print(summary(z))
  sleepGestR2[i] <- summary(z)$r.squared
}

# danger linearly and additively
zL <- lm(sleep ~ gestation + danger, data = sleepLog)
zA <- lm(sleep ~ gestation + dangerFact, data = sleepLog)
aIntercept <- zA$coef[1] + c(0, zA$coef[3:6])
LIntercept <- zL$coef[1] + seq(5) * zL$coef[2]

png(file = '/Users/dlambert/CN/Figures/gestDangerLA.png',
    width = 800, height = 300)
xyplot(sleep ~ gestation | dangerFact, data = sleepLog,
       cex = 1.25, pch = 16,
       layout = c(5, 1),
       panel = function(x, y, ...) {
         panel.grid(h = -1, v = -1)
         pno <- panel.number()
         panel.xyplot(x, y, ...)
         panel.abline(LIntercept[pno], zL$coef[2],
                      col = 'red', lwd = 1.4)
         panel.abline(aIntercept[pno], zA$coef[2],
                      lwd = 1.4, col = 'forest green')
       })
graphics.off()

zI <- lm(sleep ~ gestation * dangerFact, data = sleepLog)

# Interaction of numeric variables.
z <- lm(sleep ~ gestation * danger, data = sleepLog)
z <- lm(sleep ~ gestation + danger + I(gestation * danger), data = sleepLog)

# Uncertainty of the model with an interaction of danger
# as a factor and log(gestation).

zISumm <- summary(zI)
zICorr <- zISumm$sigma^2 * zISumm$cov.unscaled
simEstimates <- mvrnorm(500, zI$coefficients,zICorr)

newX <- sapply(split(sleepLog$gestation, sleep$danger),
               sample, 2)
newX <- data.frame(gestation = c(newX),
                   danger = rep(1:5, each = 2))
# Get intercepts and slopes under each simulated model
intercepts <- array(NA, c(500, 10))
slopes <- array(NA, c(500, 10))
for (i in 1:10) {
  iDang <- newX$danger[i] 
  intercepts[, i] <- simEstimates[, 1]
  slopes[, i] <- simEstimates[, 2]
  if (iDang > 1) {
    intercepts[, i] <-
      intercepts[,i] + simEstimates[, 1 + iDang]
    slopes[,i] <- slopes[,i] + simEstimates[, 5 + iDang]
  }
}
 
# Get new means,
newMeans <- array(NA, c(500, 10))
for (i in 1:10) {
  newMeans[, i] <- intercepts[, i] +
    slopes[, i] * newX$gestation[i]
}
# Get confidence intervals.
names(newX) <- c('gestation', 'dangerFact')
newX$dangerFact <- factor(newX$dangerFact)
newPredInt <- predict(zI, newdata = newX,
                      interval = 'confidence', level = .9)
round(cbind(newX[,1], as.numeric(newX[,2]), round(newPredInt,1)))

# Find how many simulated means are in, below and above
# the confidence intervals. 
nIn <- nBelow <- nAbove <- rep(NA, 10)
for (i in 1:10) {
  nBelow[i] <- sum(newMeans[,i] < newPredInt[i, 'lwr'])
  nAbove[i] <- sum(newMeans[,i] > newPredInt[i,'upr'])
}
nIn <- 500 - nBelow - nAbove

simInt <- apply(newMeans, 2, quantile, c(.05, .95))
x <- round(cbind(simInt[,1], newPredInt[, 'lwr'],
            simInt[,2], newPredInt[,'upr']), 2)
colnames(x) <- c('simLow', 'ciLow', 'simHi', 'ciHi')
