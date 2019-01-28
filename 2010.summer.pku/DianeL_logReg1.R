library(lattice)
library(MASS)

myDir <- '/Users/dlambert/CN'

# Plot of the log-odds function.
p <- seq(-10, 10, length = 500)
logOddsp <- plogis(p)
png(file = paste(myDir, 'Figures/logodds.png', sep = '/'),
      height = 300, width = 300)
xyplot(logOddsp ~ p,
       xlab = 'LOG-ODDS(P)', ylab = 'P',
       panel = function(x, y, ...) {
         panel.grid(h=-1, v=-1)
         panel.lines(x, y, lwd = 1.5, col = 'red')
       })
graphics.off()

# Plot of logit link vs probit link. 
png(file = paste(myDir,'nlQ.png', sep ='/'), w=300, h=300)
x <- seq(1e-6, 1-1e-6, len =1000)
y1 <- qlogis(x)
y2 <- qnorm(x)
xyplot(y1 ~ y2, type = 'l',
       xlab = 'NORMAL QUANTILES',
       ylab = 'LOGISTIC QUANTILES',
       panel = function(x, y, ...) {
         panel.grid(h=-1, v=-1)
         panel.lines(x, y,lwd = 2.5)
         z <- lm(y ~ x, subset = (x > -2 & x < 2))
         panel.abline(z$coef, col = 'dark green',
                      lwd = 1.2)
         })
graphics.off()

# Read in the data
wells <- read.csv(paste(myDir,
                        'Labs/wells.csv', sep = '/'))

myDir <- paste(myDir, 'Figures', sep = '/')

# Create the arsenic maps.
wells$arsenicLevel <- cut(wells$arsenic,
                          c(0, 10, 50, 1e3),
                          include.lowest = TRUE)
# All 3 levels of arsenic on one plot.
png(file = paste(myDir, 'arsenicMap.png', sep = '/'),
    height = 600, width = 600)
xyplot(y ~ x, groups = arsenicLevel, data = wells,
       col = c('blue3', 'forest green', 'magenta3'),
       pch = 16, cex = c(.65, .55, .4))
graphics.off()

# Different levels of arsenic separated.
png(file = paste(myDir, 'arsenicMap3.png', sep = '/'),
    height = 600, width = 350)
xyplot(y ~ x | arsenicLevel, groups = arsenicLevel,
       data = wells,
       col = c('blue3', 'forest green', 'magenta3'),
       cex = .25, layout = c(1,3)) #blue, magenta, green
graphics.off()

histogram(~log(arsenic), data = wells)
histogram(~log(distance), data = wells)

# Assume people won't walk more than 10 km.
wells$walkDistance <- pmin(wells$distance/1000, 10)
zArDist <- glm(switch ~ walkDistance + log(arsenic),
               data = wells, subset = unsafe,
               family = binomial)


calibrationPlot <- function(z, nCuts=10, alpha = .05) {
# z is the output from a glm.
  fitted <- z$fitted.values
  fitBreaks <- quantile(fitted, seq(0, nCuts)/nCuts)
  fitCut <- cut(fitted, fitBreaks, include.lowest = TRUE)
  expCut <- sapply(split(fitted, fitCut), mean)
  obsCut <- sapply(split(z$y, fitCut), mean)
  nCut <- table(fitCut)
  obsSE <- sqrt(expCut * (1-expCut)/nCut)
  p <- xyplot(obsCut ~ expCut,
              xlab = 'PREDICTED FRACTION',
              ylab = 'OBSERVED FRACTION',
              panel = function(x, y, ...) {
                panel.grid(h=-1, v=-1)
                panel.abline(0,1, col = gray(.6))
                qa <- qnorm(1 - alpha/2)
                panel.segments(expCut, expCut-qa*obsSE,
                               expCut, expCut+qa*obsSE,
                               col = gray(.7))
                panel.points(x, y, pch=16, cex = 1.25)              })
  return(p)
}

# Example calibration plot.
png(file = paste(myDir, 'calib1.png', sep = '/'),
      width = 300, height = 300)
p1 <- calibrationPlot(zArDist, nCuts = 50, alpha = .05)
print(p1)
graphics.off()


# Plot of the predictions as a function of arsenic level
# for given values of distance. 
newArs <- seq(50, 750, length = 200)
newWalkDist <- c(.5, 1, 2, 5, 8, 10)
myX <- data.frame(arsenic =
                  rep(newArs, each = length(newWalkDist)),
                  walkDistance =
                  rep(newWalkDist, length(newArs)))
newPred <- predict(zArDist, newdata = myX,
                       type = 'response')
distFact <- factor(myX$walkDistance)
png(file = paste(myDir, 'pred2.png', sep = '/'),
    height = 500, width = 500)
xyplot(newPred ~ myX$arsenic, groups = distFact,
       type = 'l', lwd = 1.5,
       xlab = 'ARSENIC', ylab = 'PREDICTED',
       sub = list(
     'lines represent different distances to a safe well',
                  cex = 1.5),
       auto.key = list(text =
                         paste(levels(distFact), 'km'),
                       lines = TRUE, points = FALSE,
                       cex = 1.5,
                       lwd = 6, columns = 3),
       panel = function(x, y, groups,  ...) {
         panel.grid(h=-1, v=-1)
         panel.xyplot(x, y, groups, ...) 
        })
graphics.off()


# Uncertainty in predictions as a function of arsenic for
# distance = .5 km and distance = nKM
nsim <- 30
npoints <- 1000
bCov <- summary(zArDist)$cov.scaled
bNew <- mvrnorm(nsim, zArDist$coef, bCov)
newArsenic <- seq(50, 700, length = npoints)
nKM <- 1
xNew1 <- cbind(1, nKM, log(newArsenic))
muPred1 <- plogis(xNew1 %*% array(zArDist$coef, c(3,1)))
# Get a matrix where each column is 100 points on the
# probability curve for one of the simulated models.
muNew1 <- plogis(xNew1 %*% t(bNew)) 
p1 <- xyplot(c(muNew1) ~ rep(newArsenic, nsim),
       groups = factor(rep(seq(nsim), each = npoints)),
       pch = 16, cex = .1, type = 'l',
       col = gray(.7),
       ylim = c(.35, .95),
       ylab = 'PROBABILITY',
       xlab = 'ARSENIC',
       main = paste(nKM, 'km to NEAREST SAFE WELL'),
       panel = function(x, y, groups, ...){
            panel.grid(h=-1, v=-1)
            panel.xyplot(x, y, groups, ...)
            panel.lines(newArsenic, muPred1,
                        col = 'red', lwd = 1.5)
             })
png(file = paste(myDir, 'uncerLog.png', sep = '/'),
    width = 300, height = 600)
plot(p1, more = TRUE, position = c(0, 0, 1, .5))
plot(p5, position = c(0, .5, 1, 1))
graphics.off()
